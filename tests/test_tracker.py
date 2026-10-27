# -*- coding: utf-8 -*-
import pytest

from emotrack.tracker import FaceTracker, center_distance, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_center_distance_relative_to_face_size():
    assert center_distance((0, 0, 100, 100), (50, 0, 100, 100)) == pytest.approx(0.5)


def test_new_faces_get_sequential_ids():
    tr = FaceTracker()
    assert tr.update([(0, 0, 50, 50), (200, 0, 50, 50)], 0) == ["P1", "P2"]


def test_ids_are_kept_when_faces_move_slightly_and_order_changes():
    tr = FaceTracker()
    tr.update([(0, 0, 50, 50), (200, 0, 50, 50)], 0)
    assert tr.update([(205, 3, 50, 50), (4, 2, 52, 50)], 2) == ["P2", "P1"]


def test_returning_face_gets_old_id_and_far_face_gets_new_id():
    tr = FaceTracker()
    tr.update([(0, 0, 50, 50), (200, 0, 50, 50)], 0)
    assert tr.update([(0, 0, 50, 50)], 2) == ["P1"]                       # P2 пропал
    assert tr.update([(0, 0, 50, 50), (198, 1, 50, 50)], 4) == ["P1", "P2"]  # вернулся
    assert tr.update([(400, 300, 50, 50)], 6) == ["P3"]


def test_one_face_is_not_matched_to_two_tracks():
    tr = FaceTracker()
    tr.update([(0, 0, 50, 50), (30, 0, 50, 50)], 0)
    ids = tr.update([(10, 0, 50, 50)], 2)
    assert len(ids) == 1 and ids[0] in ("P1", "P2")
    assert len(tr.tracks) == 2
