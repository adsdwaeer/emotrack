# -*- coding: utf-8 -*-
"""
Настройки прототипа «ЭмоТрек».

Все параметры собраны в одном месте, чтобы их было удобно менять
и описывать в отчёте. Часть параметров можно переопределить
переменными окружения (EMOTRACK_DB, EMOTRACK_DEVICE, ...).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Порядок эмоций во всём проекте (в БД, API, CSV) — фиксированный
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_RU = {
    "angry":    "Злость",
    "disgust":  "Отвращение",
    "fear":     "Страх",
    "happy":    "Радость",
    "neutral":  "Нейтрально",
    "sad":      "Грусть",
    "surprise": "Удивление",
}


@dataclass
class EngagementWeights:
    """
    Веса формулы индекса вовлечённости (ТЗ, п. 4.2):

        E = 100 · (w1·p(радость) + w2·p(удивление) + w3·p(нейтрально)
                   − w4·p(грусть) − w5·p(злость) + c) · q

    Значения ниже — экспертные значения «по умолчанию» для прототипа.
    Логика выбора: внимательный слушатель чаще всего выглядит нейтрально,
    поэтому чистая «нейтральность» даёт 75 (зелёная зона), радость и
    удивление поднимают индекс, грусть и злость опускают. Страх и отвращение
    в формулу ТЗ не входят (их вклад равен нулю, остаётся только c).

    Калибровка весов по экспертной разметке кураторов (линейная регрессия
    с ограничениями на знаки коэффициентов) — задача ВКР (этап 3 по ТЗ).
    """
    happy: float = 0.70      # w1
    surprise: float = 0.60   # w2
    neutral: float = 0.50    # w3
    sad: float = 0.50        # w4 (со знаком минус)
    angry: float = 0.40      # w5 (со знаком минус)
    bias: float = 0.25       # c


@dataclass
class Settings:
    # --- модель ---
    model_name: str = "dima806/facial_emotions_image_detection"
    device: str = os.environ.get("EMOTRACK_DEVICE", "auto")  # auto / cuda / cpu
    batch_size: int = 32
    # Модель обучалась на чёрно-белых снимках FER-2013, поэтому
    # цветные кадры с камеры переводим в оттенки серого
    grayscale: bool = True

    # --- выборка кадров и интервалы ---
    frame_step_sec: float = 2.0      # 1 кадр в 2 с (ТЗ, табл. 4)
    interval_sec: float = 30.0       # агрегаты за 30-секундные интервалы (ТЗ, п. 4.3.2)

    # --- детекция лиц ---
    detector: str = os.environ.get("EMOTRACK_DETECTOR", "auto")  # auto / yunet / haar
    yunet_path: Path = PROJECT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
    detector_score: float = 0.6
    min_face_size: int = 40          # лица меньше 40 px отбрасываем

    # --- качество кадра ---
    min_brightness: float = 40.0     # средняя яркость лица (0..255)
    max_brightness: float = 220.0
    min_sharpness: float = 15.0      # дисперсия лапласиана: меньше — размыто
    max_yaw: float = 0.35            # смещение носа от центра глаз / межглазное расстояние

    # --- сопровождение лиц ---
    track_iou: float = 0.3           # минимальный IoU для сопоставления
    track_max_dist: float = 0.6      # макс. смещение центра (в долях размера лица)
    min_track_frames: int = 3        # участник должен быть замечен хотя бы на 3 кадрах
    min_track_share: float = 0.1     # ... и хотя бы на 10 % кадров после появления

    # --- индекс вовлечённости ---
    weights: EngagementWeights = field(default_factory=EngagementWeights)
    risk_threshold: float = 40.0     # «зона риска»: индекс < 40
    green_threshold: float = 70.0    # зелёный уровень: индекс ≥ 70
    risk_lessons_in_row: int = 3     # риск отчисления: < 40 на 3 занятиях подряд

    # --- хранение ---
    db_path: Path = Path(os.environ.get("EMOTRACK_DB", PROJECT_DIR / "data" / "emotrack.db"))
    upload_dir: Path = PROJECT_DIR / "data" / "uploads"


settings = Settings()
