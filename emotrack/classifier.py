# -*- coding: utf-8 -*-
"""
Классификатор эмоций — та же модель, что в прототипе с 1 курса:
dima806/facial_emotions_image_detection (ViT-Base/16, 7 эмоций).

В отличие от прототипа, здесь модель вызывается напрямую (без pipeline),
чтобы обрабатывать лица батчами на GPU и получать полный вектор
вероятностей в фиксированном порядке EMOTIONS.
"""

import cv2
import numpy as np

from .config import EMOTIONS, settings


def pick_device(device="auto"):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class EmotionClassifier:
    def __init__(self, model_name=None, device=None, batch_size=None):
        import torch
        from transformers import AutoModelForImageClassification, ViTImageProcessorPil

        self.torch = torch
        self.model_name = model_name or settings.model_name
        self.device = pick_device(device or settings.device)
        self.batch_size = batch_size or settings.batch_size
        print(f"Загрузка модели {self.model_name} на {self.device}...")
        # PIL-вариант препроцессора: не требует torchvision (resize 224, нормализация)
        self.processor = ViTImageProcessorPil.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        # Индексы выходов модели в нашем порядке EMOTIONS
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.order = [label2id[e] for e in EMOTIONS]
        print("Модель готова!")

    def prepare(self, crop_bgr):
        """BGR-кроп лица -> RGB-массив для модели (при необходимости — серый)."""
        if settings.grayscale:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def predict(self, crops_bgr):
        """Список BGR-кропов -> массив (N, 7) вероятностей в порядке EMOTIONS."""
        if not crops_bgr:
            return np.zeros((0, len(EMOTIONS)), dtype=np.float32)
        result = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch = [self.prepare(c) for c in crops_bgr[i:i + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            result.append(probs[:, self.order])
        return np.concatenate(result).astype(np.float32)
