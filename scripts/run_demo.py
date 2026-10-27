# -*- coding: utf-8 -*-
"""
Прогон демонстрационного видео через конвейер и сравнение со сценарием.

1. Анализирует data/demo/demo_lesson.mp4 (конвейер emotrack.pipeline).
2. Сохраняет результат в БД — занятие сразу видно в дашборде.
3. По сценарию (demo_lesson.json) считает «ожидаемый» индекс каждого
   30-секундного интервала: те же формула и веса, но вместо вероятностей
   модели — истинные эмоции сценария, вместо q — доля кадров, где лицо было.
   Сравнивает ожидаемый и измеренный индексы (корреляция Пирсона, MAE).

Запуск:
    python scripts/run_demo.py
Результат: docs/demo_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emotrack import engagement as eng, storage  # noqa: E402
from emotrack.classifier import EmotionClassifier  # noqa: E402
from emotrack.config import EMOTIONS, settings  # noqa: E402
from emotrack.pipeline import analyze_video  # noqa: E402


def expected_intervals(truth, step, interval, duration):
    """Ожидаемый индекс по сценарию для кадров, которые выбирает конвейер."""
    segment = truth[1]["t"] - truth[0]["t"]
    result = {}
    for k in range(int(np.ceil(duration / interval))):
        times = [t for t in np.arange(k * interval, (k + 1) * interval, step) if t < duration]
        emotions = [truth[int(t // segment)]["emotion"] for t in times]
        present = [e for e in emotions if e != "absent"]
        q = len(present) / len(times)
        probs = {e: present.count(e) / len(present) for e in EMOTIONS} if present else None
        result[k * interval] = eng.engagement_index(probs, q) if probs else 0.0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "data" / "demo" / "demo_lesson.mp4"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--title", default="Демо: вебинар «Машинное обучение», лекция 5")
    args = parser.parse_args()

    video = Path(args.video)
    meta = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    classifier = EmotionClassifier(device=args.device)
    participants, summary = analyze_video(video, classifier)

    conn = storage.connect()
    lesson_id = storage.create_lesson(conn, args.title, status="processing")
    storage.save_result(conn, lesson_id, participants, summary)
    storage.update_lesson(conn, lesson_id, status="done", progress=1.0)
    conn.close()

    # ID участников (P1, P2, ...) идут в порядке, в котором детектор нашёл лица,
    # поэтому сопоставляем их со «слушателями» сценария по положению плитки
    names = list(meta["participants"])
    cols, rows_n = meta["grid"]["cols"], meta["grid"]["rows"]
    mapping = {}
    for p in participants:
        x, y, w, h = p["box"]
        col = int((x + w / 2) / meta["width"] * cols)
        row = int((y + h / 2) / meta["height"] * rows_n)
        mapping[p["track_id"]] = names[row * cols + col]

    rows, all_exp, all_meas = [], [], []
    for p in participants:
        name = mapping.get(p["track_id"], "?")
        exp = expected_intervals(meta["truth"][name], settings.frame_step_sec,
                                 settings.interval_sec, meta["duration_sec"]) if name in meta["truth"] else {}
        pairs = [(exp[it["t_start"]], it["engagement"]) for it in p["intervals"] if it["t_start"] in exp]
        all_exp += [a for a, _ in pairs]
        all_meas += [b for _, b in pairs]
        exp_index = eng.participant_index([a for a, _ in pairs])
        rows.append({
            "track_id": p["track_id"], "name": name, "scenario": meta["participants"].get(name),
            "engagement": p["engagement"], "level": p["level"], "risk": p["risk"],
            "expected_engagement": exp_index,
            "face_share": round(p["frames_with_face"] / p["frames_total"], 3),
            "first_seen_sec": p["first_seen"],
            "intervals": [{"t": it["t_start"], "measured": it["engagement"], "expected": exp.get(it["t_start"]),
                           "q": it["q"]} for it in p["intervals"]],
        })

    a, b = np.array(all_exp), np.array(all_meas)
    result = {
        "lesson_id": lesson_id,
        "video": meta["video"], "duration_sec": summary["duration_sec"],
        "device": summary["device"], "detector": summary["detector"],
        "processing_sec": summary["processing_sec"],
        "frames_processed": summary["frames_processed"],
        "faces_classified": summary["faces_classified"], "faces_rejected": summary["faces_rejected"],
        "lesson_engagement": summary["engagement"], "lesson_level": summary["level"],
        "at_risk": summary["at_risk"],
        "minutes": summary["timeline"],
        "intervals_compared": len(a),
        "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "mae": round(float(np.abs(a - b).mean()), 2),
        "participants": rows,
    }
    out = ROOT / "docs" / "demo_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nЗанятие №{lesson_id}: индекс {summary['engagement']} ({summary['level']}), "
          f"в зоне риска: {summary['at_risk']}, обработка {summary['processing_sec']} с на {summary['device']}")
    print(f"Сравнение со сценарием: {len(a)} интервалов, r = {result['pearson_r']}, MAE = {result['mae']}")
    for r in rows:
        print(f"  {r['track_id']:<3} {r['name']:<12} индекс {r['engagement']:>6} (ожид. {r['expected_engagement']:>6}) "
              f"{r['level']:<6} {'РИСК' if r['risk'] else '':<5} лицо в кадре {r['face_share']:.0%}  — {r['scenario']}")
    print("По минутам:", [m["engagement"] for m in summary["timeline"]])


if __name__ == "__main__":
    main()
