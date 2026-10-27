# -*- coding: utf-8 -*-
import numpy as np
import pytest

from emotrack.aggregation import aggregate, mean_probs, summarize
from emotrack.config import EMOTIONS


def vec(emotion):
    v = np.zeros(len(EMOTIONS), dtype=np.float32)
    v[EMOTIONS.index(emotion)] = 1.0
    return v


def test_mean_probs():
    m = mean_probs([vec("happy"), vec("neutral")])
    assert m["happy"] == 0.5 and m["neutral"] == 0.5 and m["sad"] == 0.0
    assert mean_probs([]) is None


def test_intervals_of_30_seconds_with_q():
    times = [float(t) for t in range(0, 60, 2)]           # 30 кадров, 2 интервала по 15
    # Во втором интервале лицо есть только на 5 кадрах из 15, ещё на 3 — отбраковано
    obs = {"P1": {t: vec("neutral") for t in times if t < 40}}
    for t in (40.0, 42.0, 44.0):
        obs["P1"][t] = None
    parts = aggregate(times, obs, interval_sec=30)
    assert len(parts) == 1
    first, second = parts[0]["intervals"]
    assert (first["t_start"], first["n_frames"], first["n_face"], first["q"]) == (0, 15, 15, 1.0)
    assert (second["t_start"], second["n_frames"], second["n_face"]) == (30, 15, 5)
    assert second["q"] == pytest.approx(5 / 15, abs=1e-4)
    assert first["engagement"] == pytest.approx(75.0)
    assert second["engagement"] == pytest.approx(25.0)
    assert parts[0]["engagement"] == pytest.approx(50.0)
    assert parts[0]["level"] == "yellow" and not parts[0]["risk"]


def test_participant_counted_only_after_first_appearance():
    times = [float(t) for t in range(0, 90, 2)]
    obs = {"P1": {t: vec("happy") for t in times},
           "P2": {t: vec("happy") for t in times if t >= 60}}   # подключился на 60-й секунде
    parts = {p["track_id"]: p for p in aggregate(times, obs, interval_sec=30)}
    assert len(parts["P2"]["intervals"]) == 1
    assert parts["P2"]["intervals"][0]["q"] == 1.0
    assert parts["P2"]["frames_total"] == 15


def test_short_false_detections_are_dropped():
    times = [float(t) for t in range(0, 120, 2)]
    obs = {"P1": {t: vec("happy") for t in times}, "P2": {10.0: vec("sad")}}
    parts = aggregate(times, obs, interval_sec=30)
    assert [p["track_id"] for p in parts] == ["P1"]


def test_absent_listener_is_at_risk_and_summary():
    times = [float(t) for t in range(0, 120, 2)]
    obs = {"P1": {t: vec("happy") for t in times},
           "P2": {t: vec("neutral") for t in times if t < 20}}   # ушёл через 20 с
    parts = aggregate(times, obs, interval_sec=30)
    p2 = [p for p in parts if p["track_id"] == "P2"][0]
    assert p2["risk"] and p2["level"] == "red"
    s = summarize(parts)
    assert s["participants"] == 2 and s["at_risk"] == 1
    assert [m["minute"] for m in s["timeline"]] == [0, 1]
    assert s["emotions"]["happy"] == pytest.approx(0.5)


def test_empty_input():
    assert aggregate([], {}) == []
    assert summarize([])["engagement"] is None
