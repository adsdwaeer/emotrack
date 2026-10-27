# -*- coding: utf-8 -*-
import pytest

from emotrack import engagement as eng
from emotrack.config import EngagementWeights


def probs(**kw):
    base = {e: 0.0 for e in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]}
    base.update(kw)
    return base


def test_formula_matches_tz():
    w = EngagementWeights(happy=0.7, surprise=0.6, neutral=0.5, sad=0.5, angry=0.4, bias=0.25)
    p = probs(happy=0.2, surprise=0.1, neutral=0.4, sad=0.2, angry=0.1)
    expected = 100 * (0.7 * 0.2 + 0.6 * 0.1 + 0.5 * 0.4 - 0.5 * 0.2 - 0.4 * 0.1 + 0.25) * 0.8
    assert eng.engagement_index(p, 0.8, w) == pytest.approx(expected, abs=0.01)


def test_neutral_listener_is_green_and_sad_is_red():
    assert eng.engagement_index(probs(neutral=1.0), 1.0) == pytest.approx(75.0)
    assert eng.level(eng.engagement_index(probs(neutral=1.0), 1.0)) == "green"
    assert eng.engagement_index(probs(sad=1.0), 1.0) == 0.0


def test_index_is_clipped_to_0_100():
    big = EngagementWeights(happy=2.0, bias=0.5)
    assert eng.engagement_index(probs(happy=1.0), 1.0, big) == 100.0
    assert eng.engagement_index(probs(angry=1.0, sad=1.0), 1.0) == 0.0


def test_q_scales_index_and_zero_q_gives_zero():
    full = eng.engagement_index(probs(neutral=1.0), 1.0)
    half = eng.engagement_index(probs(neutral=1.0), 0.5)
    assert half == pytest.approx(full / 2)
    assert eng.engagement_index(probs(happy=1.0), 0.0) == 0.0


@pytest.mark.parametrize("value, lvl", [(100, "green"), (70, "green"), (69.99, "yellow"),
                                        (40, "yellow"), (39.9, "red"), (0, "red"), (None, None)])
def test_levels(value, lvl):
    assert eng.level(value) == lvl


def test_risk_flag():
    assert eng.is_risk(39.9)
    assert not eng.is_risk(40)
    assert not eng.is_risk(None)


def test_lesson_timeline_by_minutes():
    intervals = [
        {"t_start": 0, "engagement": 80}, {"t_start": 30, "engagement": 60},   # минута 0
        {"t_start": 0, "engagement": 40}, {"t_start": 30, "engagement": 20},
        {"t_start": 60, "engagement": 50},                                      # минута 1
    ]
    tl = eng.lesson_timeline(intervals)
    assert [m["minute"] for m in tl] == [0, 1]
    assert tl[0]["engagement"] == 50.0
    assert tl[1]["engagement"] == 50.0 and tl[1]["level"] == "yellow"


def test_participant_and_lesson_index():
    assert eng.participant_index([80, 60, 0]) == pytest.approx(46.67)
    assert eng.lesson_index([]) is None


def test_dropout_risk_three_lessons_in_row():
    assert eng.dropout_risk([35, 20, 39])
    assert not eng.dropout_risk([35, 20])
    assert not eng.dropout_risk([35, 20, 39, 75])      # последнее занятие — выше порога
    assert eng.dropout_risk([80, 30, 30, 30])
