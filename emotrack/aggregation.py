# -*- coding: utf-8 -*-
"""
Агрегация наблюдений в 30-секундные интервалы и итоги по занятию.

Наблюдение — это результат по одному участнику на одном выбранном кадре:
  * вектор вероятностей 7 эмоций — лицо найдено и прошло проверку качества;
  * None — лицо найдено, но кадр отбракован (темно, размыто, отвернулся).
Если участника на кадре нет совсем, наблюдения нет.
"""

import math

from . import engagement as eng
from .config import EMOTIONS, settings


def mean_probs(vectors):
    """Средние вероятности эмоций по списку векторов (порядок EMOTIONS)."""
    if not vectors:
        return None
    n = len(vectors)
    return {e: round(sum(float(v[i]) for v in vectors) / n, 4) for i, e in enumerate(EMOTIONS)}


def is_real_participant(frames_seen, frames_possible):
    """Отсекаем ложные срабатывания детектора: «участник», который
    мелькнул на 1–2 кадрах, скорее всего не человек, а артефакт."""
    if frames_seen < settings.min_track_frames:
        return False
    return frames_possible == 0 or frames_seen / frames_possible >= settings.min_track_share


def aggregate(frame_times, observations, interval_sec=None):
    """Строит интервалы и итоги по участникам.

    frame_times  — отсортированные моменты (с) всех обработанных кадров;
    observations — {participant_id: {t: probs | None}}.
    """
    interval_sec = interval_sec or settings.interval_sec
    if not frame_times:
        return []
    last_bucket = int(frame_times[-1] // interval_sec)

    participants = []
    for pid, obs in observations.items():
        if not obs:
            continue
        first_seen = min(obs)
        frames_possible = sum(1 for t in frame_times if t >= first_seen)
        if not is_real_participant(len(obs), frames_possible):
            continue

        intervals = []
        for k in range(int(first_seen // interval_sec), last_bucket + 1):
            t0, t1 = k * interval_sec, (k + 1) * interval_sec
            times = [t for t in frame_times if t0 <= t < t1 and t >= first_seen]
            if not times:
                continue
            good = [obs[t] for t in times if obs.get(t) is not None]
            q = len(good) / len(times)
            probs = mean_probs(good)
            index = eng.engagement_index(probs, q) if probs else 0.0
            intervals.append({
                "t_start": t0, "t_end": t1,
                "n_frames": len(times), "n_face": len(good),
                "q": round(q, 4), "probs": probs, "engagement": index,
            })

        all_good = [p for p in obs.values() if p is not None]
        index = eng.participant_index([it["engagement"] for it in intervals])
        participants.append({
            "track_id": pid,
            "first_seen": first_seen,
            "last_seen": max(obs),
            "frames_total": frames_possible,
            "frames_detected": len(obs),
            "frames_with_face": len(all_good),
            "emotions": mean_probs(all_good),
            "engagement": index,
            "level": eng.level(index),
            "risk": eng.is_risk(index),
            "intervals": intervals,
        })
    participants.sort(key=lambda p: int(p["track_id"][1:]) if p["track_id"][1:].isdigit() else math.inf)
    return participants


def summarize(participants):
    """Итог по занятию: индекс, динамика по минутам, распределение эмоций."""
    all_intervals = [it for p in participants for it in p["intervals"]]
    index = eng.lesson_index([p["engagement"] for p in participants])
    vectors = [[p["emotions"][e] for e in EMOTIONS] for p in participants if p["emotions"]]
    return {
        "engagement": index,
        "level": eng.level(index),
        "participants": len(participants),
        "at_risk": sum(1 for p in participants if p["risk"]),
        "emotions": mean_probs(vectors),
        "timeline": eng.lesson_timeline(all_intervals),
    }
