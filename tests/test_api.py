# -*- coding: utf-8 -*-
import time

import cv2
import pytest
from fastapi.testclient import TestClient

from emotrack.api import create_app
from emotrack.config import settings
from emotrack.pipeline import analyze_video

from .conftest import StubClassifier, StubDetector, make_frame, write_video


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    app = create_app(classifier=StubClassifier(), db_path=tmp_path / "test.db",
                     detector_factory=StubDetector)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "lesson.avi"
    write_video(path, seconds=60)
    return path


def wait_done(client, lesson_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/lessons/{lesson_id}/status").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.2)
    raise TimeoutError


def test_pipeline_on_synthetic_video(video):
    participants, summary = analyze_video(video, StubClassifier(), StubDetector())
    assert summary["frames_processed"] == 30          # 60 с, 1 кадр в 2 с
    by_id = {p["track_id"]: p for p in participants}
    assert set(by_id) == {"P1", "P2"}
    assert by_id["P1"]["engagement"] == pytest.approx(95.0)   # радость: 100·(0.7+0.25)
    assert by_id["P2"]["engagement"] == 0.0 and by_id["P2"]["risk"]
    assert len(by_id["P1"]["intervals"]) == 2


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_process_and_read_results(client, video, tmp_path):
    with open(video, "rb") as f:
        r = client.post("/api/lessons", files={"file": ("lesson.avi", f, "video/x-msvideo")},
                        data={"title": "Тестовое занятие"})
    assert r.status_code == 202
    lesson_id = r.json()["id"]
    assert wait_done(client, lesson_id)["status"] == "done"
    assert not any((tmp_path / "uploads").iterdir()), "видео должно удаляться после обработки"

    lessons = client.get("/api/lessons").json()
    assert lessons[0]["title"] == "Тестовое занятие" and lessons[0]["participants"] == 2

    s = client.get(f"/api/lessons/{lesson_id}").json()
    assert s["participants"] == 2 and s["at_risk"] == 1
    assert s["engagement"] == pytest.approx(47.5)
    assert len(s["minutes"]) == 1

    parts = client.get(f"/api/lessons/{lesson_id}/participants").json()
    assert [p["risk"] for p in parts] == [False, True]

    tl = client.get(f"/api/lessons/{lesson_id}/timeline").json()
    assert len(tl["intervals"]) == 4 and tl["interval_sec"] == 30

    csv = client.get(f"/api/lessons/{lesson_id}/export.csv")
    assert csv.status_code == 200 and "text/csv" in csv.headers["content-type"]
    lines = csv.text.strip().splitlines()
    assert lines[0].lstrip("﻿").startswith("участник;") and len(lines) == 5

    assert client.get(f"/lessons/{lesson_id}").status_code == 200
    assert client.get("/").status_code == 200


def test_upload_rejects_non_video(client):
    r = client.post("/api/lessons", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_unknown_lesson_404(client):
    assert client.get("/api/lessons/999").status_code == 404
    assert client.get("/api/lessons/999/participants").status_code == 404


def test_analyze_single_frame(client):
    ok, png = cv2.imencode(".png", make_frame([(40, 70, 180), (320, 70, 90)]))
    r = client.post("/api/analyze-frame", files={"file": ("f.png", png.tobytes(), "image/png")})
    assert r.status_code == 200
    faces = r.json()["faces"]
    assert len(faces) == 2
    assert {f["top_emotion"] for f in faces} == {"happy", "sad"}
    assert client.post("/api/analyze-frame",
                       files={"file": ("f.png", b"not an image", "image/png")}).status_code == 400


def test_live_session_and_student_risk(client):
    _, png = cv2.imencode(".png", make_frame([(40, 70, 90)]))
    lesson_ids = []
    for _ in range(3):
        lesson_id = client.post("/api/live", data={"title": "Живое"}).json()["id"]
        for t in range(0, 20, 2):
            r = client.post(f"/api/live/{lesson_id}/frame",
                            files={"file": ("f.png", png.tobytes(), "image/png")}, data={"t": t})
            assert r.status_code == 200
        assert r.json()["participants"][0]["risk"]
        assert client.post(f"/api/live/{lesson_id}/finish").json()["status"] == "done"
        lesson_ids.append(lesson_id)
        pid = client.get(f"/api/lessons/{lesson_id}/participants").json()[0]["id"]
        assert client.put(f"/api/participants/{pid}", json={"student_id": "st-001"}).status_code == 200

    at_risk = client.get("/api/students/at-risk").json()
    assert [s["student_id"] for s in at_risk] == ["st-001"]
    assert client.delete(f"/api/lessons/{lesson_ids[0]}").status_code == 200
    assert client.get("/api/students/at-risk").json() == []


def test_swagger_available(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/lessons/{lesson_id}/timeline" in schema["paths"]
