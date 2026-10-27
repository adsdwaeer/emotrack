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
        # Параметры препроцессинга берём из конфигурации модели (224×224, mean/std),
        # а само преобразование делаем в OpenCV/NumPy — это в несколько раз быстрее
        # PIL-препроцессора и не требует torchvision
        processor = ViTImageProcessorPil.from_pretrained(self.model_name)
        self.size = (processor.size["width"], processor.size["height"])
        self.mean = np.array(processor.image_mean, dtype=np.float32)
        self.std = np.array(processor.image_std, dtype=np.float32)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        # На GPU считаем в половинной точности (fp16): в ~3 раза быстрее,
        # на метрики FER-2013 не влияет (см. README)
        self.fp16 = settings.fp16 and self.device.startswith("cuda")
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

    def to_tensor(self, crops_bgr):
        """Кропы -> тензор (N, 3, 224, 224): resize, [0, 1], нормализация mean/std."""
        batch = []
        for crop in crops_bgr:
            rgb = self.prepare(crop)
            upscale = rgb.shape[0] < self.size[1] or rgb.shape[1] < self.size[0]
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_LINEAR if upscale else cv2.INTER_AREA)
            batch.append((rgb.astype(np.float32) / 255.0 - self.mean) / self.std)
        array = np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2))
        return self.torch.from_numpy(array).to(self.device)

    def predict(self, crops_bgr):
        """Список BGR-кропов -> массив (N, 7) вероятностей в порядке EMOTIONS."""
        if not crops_bgr:
            return np.zeros((0, len(EMOTIONS)), dtype=np.float32)
        result = []
        for i in range(0, len(crops_bgr), self.batch_size):
            pixels = self.to_tensor(crops_bgr[i:i + self.batch_size])
            with self.torch.no_grad(), self.torch.autocast("cuda", self.torch.float16, enabled=self.fp16):
                logits = self.model(pixel_values=pixels).logits
            probs = self.torch.softmax(logits.float(), dim=-1).cpu().numpy()
            result.append(probs[:, self.order])
        return np.concatenate(result).astype(np.float32)
