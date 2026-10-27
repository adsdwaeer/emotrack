# -*- coding: utf-8 -*-
"""
Расчёт индекса вовлечённости (ТЗ, п. 4.2).

    E = 100 · (w1·p(радость) + w2·p(удивление) + w3·p(нейтрально)
               − w4·p(грусть) − w5·p(злость) + c) · q

p(·) — средняя вероятность эмоции по кадрам интервала, в которых лицо
найдено и прошло проверку качества; q — доля кадров интервала, в которых
лицо обнаружено и направлено на экран. Результат обрезается в [0, 100].
"""

from collections import defaultdict

from .config import EngagementWeights, settings

LEVEL_RU = {"green": "высокий", "yellow": "средний", "red": "низкий"}


def engagement_index(probs, q, weights: EngagementWeights | None = None):
    """Индекс вовлечённости за интервал. probs — словарь {эмоция: вероятность}."""
    w = weights or settings.weights
    if q <= 0:
        return 0.0
    raw = (w.happy * probs.get("happy", 0.0)
           + w.surprise * probs.get("surprise", 0.0)
           + w.neutral * probs.get("neutral", 0.0)
           - w.sad * probs.get("sad", 0.0)
           - w.angry * probs.get("angry", 0.0)
           + w.bias)
    value = 100.0 * raw * q
    return round(min(100.0, max(0.0, value)), 2)


def level(index):
    """Цветовой уровень: ≥70 зелёный, 40–69 жёлтый, <40 красный (ТЗ, п. 4.1.6)."""
    if index is None:
        return None
    if index >= settings.green_threshold:
        return "green"
    if index >= settings.risk_threshold:
        return "yellow"
    return "red"


def is_risk(index):
    """Флаг «зона риска» — индекс ниже 40."""
    return index is not None and index < settings.risk_threshold


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def participant_index(interval_indices):
    """Индекс слушателя за занятие — среднее по его 30-секундным интервалам.

    Интервалы, где лица не было (q = 0), входят со значением 0:
    отсутствие у экрана снижает вовлечённость.
    """
    return mean(interval_indices)


def lesson_timeline(intervals, bucket_sec=60):
    """Динамика индекса занятия по минутам.

    intervals — список словарей с ключами t_start и engagement
    (по всем участникам). Для каждой минуты берётся среднее
    индексов всех участников во всех интервалах этой минуты.
    """
    buckets = defaultdict(list)
    for it in intervals:
        minute = int(it["t_start"] // bucket_sec)
        buckets[minute].append(it["engagement"])
    return [
        {"minute": m, "t_start": m * bucket_sec, "engagement": mean(vals),
         "level": level(mean(vals))}
        for m, vals in sorted(buckets.items())
    ]


def lesson_index(participant_indices):
    """Индекс занятия — среднее индексов слушателей."""
    return mean(participant_indices)


def dropout_risk(history, n=None):
    """Риск отчисления: индекс < 40 на n занятиях подряд (ТЗ, табл. 4).

    history — индексы слушателя по занятиям в хронологическом порядке.
    Смотрим на последние n занятий: если слушатель «вернулся»
    (последнее занятие выше порога), сигнал снимается.
    """
    n = n or settings.risk_lessons_in_row
    last = list(history)[-n:]
    return len(last) == n and all(is_risk(v) for v in last)
