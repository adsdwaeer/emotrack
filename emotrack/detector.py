# -*- coding: utf-8 -*-
"""
Детекция лиц и оценка качества кадра.

Основной детектор — YuNet (OpenCV Zoo, лицензия MIT, файл весов 230 КБ,
встроен в OpenCV как cv2.FaceDetectorYN). Почему он, а не Haar:
  * на плитках «вебинара» из FER-2013 YuNet находит лицо в 95–97 %
    случаев против 76–78 % у Haar-каскада (evaluation/detector_comparison.py);
  * YuNet возвращает 5 ключевых точек (глаза, нос, уголки рта) — по ним
    оцениваем, смотрит ли слушатель в экран (множитель q в формуле индекса);
  * работает на CPU в реальном времени, GPU не нужен.
Если файла весов нет, используется Haar-каскад из cv2.data — он идёт
вместе с OpenCV и не требует скачивания (запасной вариант).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import settings


@dataclass
class Face:
    box: tuple                 # (x, y, w, h) в пикселях кадра
    score: float
    landmarks: np.ndarray | None = None   # 5 точек (x, y) или None для Haar


class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path, score=0.6):
        self.net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, 0.3, 5000)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self.net.setInputSize((w, h))
        _, raw = self.net.detect(frame_bgr)
        faces = []
        if raw is None:
            return faces
        for row in raw:
            x, y, bw, bh = row[:4]
            landmarks = row[4:14].reshape(5, 2)
            faces.append(Face((float(x), float(y), float(bw), float(bh)), float(row[14]), landmarks))
        return faces


class HaarDetector:
    name = "haar"

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(settings.min_face_size, settings.min_face_size))
        return [Face(tuple(float(v) for v in b), 1.0) for b in boxes]


def create_detector(kind=None):
    kind = kind or settings.detector
    if kind in ("auto", "yunet") and settings.yunet_path.exists():
        return YuNetDetector(settings.yunet_path, settings.detector_score)
    if kind == "yunet":
        raise FileNotFoundError(f"Нет файла весов YuNet: {settings.yunet_path}")
    return HaarDetector()


def crop_face(frame_bgr, box, margin=0.1):
    """Вырезает лицо с небольшим запасом по краям (с учётом границ кадра)."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box
    mx, my = bw * margin, bh * margin
    x1, y1 = max(0, int(x - mx)), max(0, int(y - my))
    x2, y2 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def yaw_ratio(landmarks):
    """Грубая оценка поворота головы по ключевым точкам.

    Смещение носа от середины между глазами, делённое на межглазное
    расстояние: 0 — смотрит прямо, ~0.5 и больше — сильно отвернулся.
    """
    if landmarks is None:
        return 0.0
    right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < 1:
        return 1.0
    mid = (left_eye + right_eye) / 2
    return float(abs(nose[0] - mid[0]) / eye_dist)


def face_quality(crop_bgr, face: Face):
    """Проверка качества лица: размер, освещённость, резкость, поворот.

    Возвращает (ok, details). Кадры, не прошедшие проверку, не идут
    в классификатор и не учитываются в q.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    yaw = yaw_ratio(face.landmarks)
    size = min(face.box[2], face.box[3])
    reasons = []
    if size < settings.min_face_size:
        reasons.append("маленькое лицо")
    if brightness < settings.min_brightness:
        reasons.append("темно")
    if brightness > settings.max_brightness:
        reasons.append("пересвет")
    if sharpness < settings.min_sharpness:
        reasons.append("размыто")
    if yaw > settings.max_yaw:
        reasons.append("не смотрит в экран")
    details = {"brightness": round(brightness, 1), "sharpness": round(sharpness, 1),
               "yaw": round(yaw, 2), "reasons": reasons}
    return not reasons, details
