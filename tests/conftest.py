# -*- coding: utf-8 -*-
"""Общие заглушки для тестов: модель и детектор не загружаются."""

import cv2
import numpy as np
import pytest

from emotrack.config import EMOTIONS
from emotrack.detector import Face


class StubClassifier:
    """Светлое «лицо» — радость, тёмное — грусть (порог с учётом чёрных полей кропа)."""
    device = "stub"

    def predict(self, crops):
        result = []
        for crop in crops:
            vec = np.full(len(EMOTIONS), 0.0, dtype=np.float32)
            if crop.mean() > 100:
                vec[EMOTIONS.index("happy")] = 1.0
            else:
                vec[EMOTIONS.index("sad")] = 1.0
            result.append(vec)
        return np.array(result).reshape(-1, len(EMOTIONS))


class StubDetector:
    """Находит на кадре однотонные прямоугольники («лица»)."""
    name = "stub"

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = (gray > 30).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 40 and h >= 40:
                faces.append(Face((float(x), float(y), float(w), float(h)), 1.0))
        return sorted(faces, key=lambda f: f.box[0])   # слева направо — стабильные ID


def make_frame(faces, size=(240, 480)):
    """Кадр с «лицами»: faces — список (x, y, яркость). Шум даёт резкость."""
    frame = np.zeros((*size, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for x, y, value in faces:
        patch = np.clip(rng.normal(value, 25, (100, 100)), 0, 255).astype(np.uint8)
        frame[y:y + 100, x:x + 100] = patch[..., None]
    return frame


def write_video(path, seconds=60, fps=5):
    """Тестовое видео: участник слева весь урок «радуется»,
    участник справа «грустит» и во второй половине пропадает из кадра."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (480, 240))
    for i in range(seconds * fps):
        t = i / fps
        faces = [(40, 70, 180)]
        if t < seconds / 2:
            faces.append((320, 70, 90))
        writer.write(make_frame(faces))
    writer.release()


@pytest.fixture
def stub_classifier():
    return StubClassifier()


@pytest.fixture
def stub_detector():
    return StubDetector()
