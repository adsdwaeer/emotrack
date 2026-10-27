# -*- coding: utf-8 -*-
"""
Конвейер анализа занятия:

    видео / кадры с веб-камеры
      -> выборка кадров (1 кадр в 2 с)
      -> детекция лиц (YuNet / Haar)
      -> проверка качества лица
      -> сопровождение лиц, ID участников (P1, P2, ...)
      -> классификация эмоций ViT батчами
      -> агрегация в 30-секундные интервалы и индекс вовлечённости

Кадры и вырезанные лица живут только в оперативной памяти: на диск
ничего не пишется, наружу отдаются только вероятности и индексы (152-ФЗ).
"""

import time
from collections import defaultdict

import cv2

from .aggregation import aggregate, summarize
from .config import EMOTIONS, settings
from .detector import create_detector, crop_face, face_quality
from .engagement import engagement_index, level
from .tracker import FaceTracker


class LessonAnalyzer:
    """Накопитель результатов по одному занятию.

    Используется и для видеофайла (кадры подаются пачками),
    и в живом режиме (кадры приходят по одному с веб-камеры).
    """

    def __init__(self, classifier, detector=None):
        self.classifier = classifier
        self.detector = detector or create_detector()
        self.tracker = FaceTracker(settings.track_iou, settings.track_max_dist)
        self.frame_times = []
        self.observations = defaultdict(dict)   # {pid: {t: probs | None}}
        self.timings = {"detect": 0.0, "classify": 0.0}
        self.faces_classified = 0
        self.faces_rejected = 0

    def process_frames(self, frames):
        """frames — список (t, кадр BGR). Возвращает лица по каждому кадру."""
        crops, refs, per_frame = [], [], []
        for t, frame in frames:
            start = time.perf_counter()
            faces = [f for f in self.detector.detect(frame)
                     if min(f.box[2], f.box[3]) >= settings.min_face_size / 2]
            ids = self.tracker.update([f.box for f in faces], t)
            self.timings["detect"] += time.perf_counter() - start
            self.frame_times.append(t)

            results = []
            for face, pid in zip(faces, ids):
                crop = crop_face(frame, face.box)
                ok, quality = (False, {"reasons": ["лицо у края кадра"]}) if crop is None \
                    else face_quality(crop, face)
                item = {"participant": pid, "box": [round(v, 1) for v in face.box],
                        "score": round(face.score, 3), "quality_ok": ok, "quality": quality,
                        "probs": None}
                self.observations[pid][t] = None
                if ok:
                    crops.append(crop)
                    refs.append((pid, t, item))
                else:
                    self.faces_rejected += 1
                results.append(item)
            per_frame.append({"t": t, "faces": results})

        # Все лица из пачки кадров — одним батчем через ViT
        start = time.perf_counter()
        probs = self.classifier.predict(crops)
        self.timings["classify"] += time.perf_counter() - start
        self.faces_classified += len(crops)
        for (pid, t, item), vec in zip(refs, probs):
            self.observations[pid][t] = vec
            p = {e: round(float(v), 4) for e, v in zip(EMOTIONS, vec)}
            item["probs"] = p
            item["top_emotion"] = max(p, key=p.get)
            item["engagement"] = engagement_index(p, 1.0)
            item["level"] = level(item["engagement"])
        return per_frame

    def result(self):
        participants = aggregate(sorted(self.frame_times), self.observations)
        # Последняя рамка лица — только для отображения/отладки, в БД не пишется
        boxes = {tr.track_id: [round(v, 1) for v in tr.box] for tr in self.tracker.tracks}
        for p in participants:
            p["box"] = boxes.get(p["track_id"])
        summary = summarize(participants)
        summary["frames_processed"] = len(self.frame_times)
        summary["faces_classified"] = self.faces_classified
        summary["faces_rejected"] = self.faces_rejected
        summary["timings"] = {k: round(v, 3) for k, v in self.timings.items()}
        return participants, summary


def iter_video_frames(path, step_sec=None):
    """Выбирает из видео кадры с шагом step_sec. Отдаёт (t, кадр, длительность)."""
    step_sec = step_sec or settings.frame_step_sec
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0.0
    next_t, idx = 0.0, 0
    try:
        while True:
            # grab() без декодирования в картинку — быстро пропускаем лишние кадры
            if not cap.grab():
                break
            t = idx / fps
            idx += 1
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield round(t, 3), frame, duration
            next_t += step_sec
    finally:
        cap.release()


def analyze_video(path, classifier, detector=None, step_sec=None, chunk=8, progress=None):
    """Полный анализ видеофайла занятия.

    progress(доля 0..1) — необязательный колбэк для статуса обработки.
    Возвращает (participants, summary).
    """
    analyzer = LessonAnalyzer(classifier, detector)
    started = time.perf_counter()
    buffer, duration = [], 0.0
    for t, frame, duration in iter_video_frames(path, step_sec):
        buffer.append((t, frame))
        if len(buffer) >= chunk:
            analyzer.process_frames(buffer)
            buffer = []
            if progress and duration:
                progress(min(0.99, t / duration))
    if buffer:
        analyzer.process_frames(buffer)
    participants, summary = analyzer.result()
    summary["duration_sec"] = round(duration or (analyzer.frame_times[-1] if analyzer.frame_times else 0), 2)
    summary["processing_sec"] = round(time.perf_counter() - started, 2)
    summary["detector"] = analyzer.detector.name
    summary["device"] = getattr(classifier, "device", "stub")
    if progress:
        progress(1.0)
    return participants, summary
