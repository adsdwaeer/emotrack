# -*- coding: utf-8 -*-
"""
Сравнение детекторов лиц: YuNet (OpenCV DNN) и Haar-каскад.

Изображения FER-2013 (48×48, лицо крупным планом) увеличиваются до
размера лица на плитке вебинара и размещаются на однотонном фоне —
так, как в демонстрационном видео. Считаем долю изображений, на которых
найдено хотя бы одно лицо, и среднее время детекции на плитку.

Запуск:
    python -m evaluation.detector_comparison
"""

import io
import json
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from emotrack.config import PROJECT_DIR, settings
from emotrack.detector import HaarDetector, YuNetDetector

OUT = PROJECT_DIR / "evaluation" / "results" / "detector_comparison.json"


def make_tile(png_bytes, face_size, tile=(340, 420)):
    rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    face = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (face_size, face_size),
                      interpolation=cv2.INTER_CUBIC)
    canvas = np.full((*tile, 3), (60, 50, 45), np.uint8)
    y0, x0 = (tile[0] - face_size) // 2, (tile[1] - face_size) // 2
    canvas[y0:y0 + face_size, x0:x0 + face_size] = face
    return canvas


def main(n=500, seed=0):
    df = pq.read_table(PROJECT_DIR / "data" / "fer2013" / "privateTest.parquet").to_pandas()
    sample = df.sample(n, random_state=seed)
    detectors = {"yunet": YuNetDetector(settings.yunet_path, settings.detector_score),
                 "haar": HaarDetector()}
    results = {"n": n, "sizes": {}}
    for size in (120, 170, 220):
        tiles = [make_tile(row["bytes"], size) for row in sample["image"]]
        row = {}
        for name, det in detectors.items():
            start = time.perf_counter()
            found = sum(1 for t in tiles if det.detect(t))
            elapsed = time.perf_counter() - start
            row[name] = {"detection_rate": round(found / n, 4),
                         "ms_per_tile": round(1000 * elapsed / n, 2)}
        results["sizes"][size] = row
        print(f"лицо {size}px: " + ", ".join(
            f"{k}: {v['detection_rate']:.1%} ({v['ms_per_tile']} мс)" for k, v in row.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
