# -*- coding: utf-8 -*-
"""
Хранение результатов в SQLite.

Храним только агрегаты: занятия, участников и 30-секундные интервалы
с вероятностями эмоций и индексом. Ни кадры, ни вырезанные лица в БД
не попадают (ТЗ, п. 4.1.8; 152-ФЗ).
"""

import sqlite3
from datetime import datetime

from .config import EMOTIONS, settings

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS lessons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    source        TEXT,                 -- 'video' или 'live'
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL,        -- queued / processing / done / error
    progress      REAL DEFAULT 0,
    error         TEXT,
    duration_sec  REAL,
    frames        INTEGER,
    processing_sec REAL,
    device        TEXT,
    detector      TEXT,
    engagement    REAL
);
CREATE TABLE IF NOT EXISTS participants (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id     INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    track_id      TEXT NOT NULL,        -- P1, P2, ... (ID внутри занятия)
    student_id    TEXT,                 -- идентификатор слушателя в LMS (назначается вручную)
    first_seen    REAL,
    frames_total  INTEGER,
    frames_with_face INTEGER,
    engagement    REAL,
    level         TEXT,
    risk          INTEGER,
    {", ".join(f"p_{e} REAL" for e in EMOTIONS)}
);
CREATE TABLE IF NOT EXISTS intervals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id     INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    participant_id INTEGER NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    t_start       REAL,
    t_end         REAL,
    n_frames      INTEGER,
    n_face        INTEGER,
    q             REAL,
    engagement    REAL,
    {", ".join(f"p_{e} REAL" for e in EMOTIONS)}
);
CREATE INDEX IF NOT EXISTS idx_intervals_lesson ON intervals(lesson_id);
"""


def connect(db_path=None):
    path = db_path or settings.db_path
    if str(path) != ":memory:":
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def now():
    return datetime.now().isoformat(timespec="seconds")


def create_lesson(conn, title, source="video", status="queued"):
    cur = conn.execute(
        "INSERT INTO lessons (title, source, created_at, status) VALUES (?, ?, ?, ?)",
        (title, source, now(), status))
    conn.commit()
    return cur.lastrowid


def update_lesson(conn, lesson_id, **fields):
    if not fields:
        return
    keys = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE lessons SET {keys} WHERE id = ?", (*fields.values(), lesson_id))
    conn.commit()


def _probs_values(probs):
    return [probs[e] if probs else None for e in EMOTIONS]


def save_result(conn, lesson_id, participants, summary):
    """Сохраняет результат анализа (перезаписывая прежний, если был)."""
    conn.execute("DELETE FROM participants WHERE lesson_id = ?", (lesson_id,))
    p_cols = ", ".join(f"p_{e}" for e in EMOTIONS)
    marks = ", ".join("?" * len(EMOTIONS))
    for p in participants:
        cur = conn.execute(
            f"INSERT INTO participants (lesson_id, track_id, first_seen, frames_total, "
            f"frames_with_face, engagement, level, risk, {p_cols}) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, {marks})",
            (lesson_id, p["track_id"], p["first_seen"], p["frames_total"], p["frames_with_face"],
             p["engagement"], p["level"], int(p["risk"]), *_probs_values(p["emotions"])))
        pid = cur.lastrowid
        conn.executemany(
            f"INSERT INTO intervals (lesson_id, participant_id, t_start, t_end, n_frames, "
            f"n_face, q, engagement, {p_cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, {marks})",
            [(lesson_id, pid, it["t_start"], it["t_end"], it["n_frames"], it["n_face"],
              it["q"], it["engagement"], *_probs_values(it["probs"])) for it in p["intervals"]])
    update_lesson(conn, lesson_id,
                  engagement=summary.get("engagement"),
                  frames=summary.get("frames_processed"),
                  duration_sec=summary.get("duration_sec"),
                  processing_sec=summary.get("processing_sec"),
                  device=summary.get("device"),
                  detector=summary.get("detector"))
    conn.commit()


def _probs(row):
    if row[f"p_{EMOTIONS[0]}"] is None:
        return None
    return {e: row[f"p_{e}"] for e in EMOTIONS}


def list_lessons(conn):
    rows = conn.execute("""
        SELECT l.*, COUNT(p.id) AS participants, COALESCE(SUM(p.risk), 0) AS at_risk
        FROM lessons l LEFT JOIN participants p ON p.lesson_id = l.id
        GROUP BY l.id ORDER BY l.id DESC""").fetchall()
    return [dict(r) for r in rows]


def get_lesson(conn, lesson_id):
    row = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
    return dict(row) if row else None


def delete_lesson(conn, lesson_id):
    conn.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
    conn.commit()


def get_participants(conn, lesson_id):
    rows = conn.execute("SELECT * FROM participants WHERE lesson_id = ? ORDER BY id",
                        (lesson_id,)).fetchall()
    result = []
    for r in rows:
        d = {k: r[k] for k in ("id", "track_id", "student_id", "first_seen", "frames_total",
                               "frames_with_face", "engagement", "level")}
        d["risk"] = bool(r["risk"])
        d["emotions"] = _probs(r)
        result.append(d)
    return result


def get_intervals(conn, lesson_id):
    rows = conn.execute("""
        SELECT i.*, p.track_id FROM intervals i JOIN participants p ON p.id = i.participant_id
        WHERE i.lesson_id = ? ORDER BY i.t_start, p.id""", (lesson_id,)).fetchall()
    result = []
    for r in rows:
        d = {k: r[k] for k in ("participant_id", "track_id", "t_start", "t_end", "n_frames",
                               "n_face", "q", "engagement")}
        d["probs"] = _probs(r)
        result.append(d)
    return result


def set_student_id(conn, participant_id, student_id):
    cur = conn.execute("UPDATE participants SET student_id = ? WHERE id = ?",
                       (student_id, participant_id))
    conn.commit()
    return cur.rowcount > 0


def student_history(conn):
    """Индексы слушателей (с назначенным student_id) по занятиям в хронологическом порядке."""
    rows = conn.execute("""
        SELECT p.student_id, p.lesson_id, l.title, l.created_at, p.engagement
        FROM participants p JOIN lessons l ON l.id = p.lesson_id
        WHERE p.student_id IS NOT NULL AND l.status = 'done'
        ORDER BY l.created_at, l.id""").fetchall()
    history = {}
    for r in rows:
        history.setdefault(r["student_id"], []).append(dict(r))
    return history
