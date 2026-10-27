# -*- coding: utf-8 -*-
"""
Скачивание внешних файлов, которые не хранятся в репозитории:

  * веса детектора лиц YuNet (OpenCV Zoo, MIT) -> models/
  * FER-2013 (Hugging Face Hub, набор Aaryan333/fer2013_train_publicTest_privateTest,
    формат parquet) -> data/fer2013/

Скачанное — недоверенные данные: это только веса ONNX и таблицы parquet,
которые читаются библиотеками как данные; никакой код из них не исполняется.

Запуск:
    python scripts/download_data.py            # YuNet + FER-2013 privateTest
    python scripts/download_data.py --train    # + обучающая часть FER-2013 (для демо-видео)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

FER_BASE = ("https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/"
            "resolve/main/data/")
FER_FILES = {
    "privateTest.parquet": "privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
    "train.parquet": "train-00000-of-00001-5eab84e1c6a2fc27.parquet",
}


def download(url, path, sha256=None):
    if path.exists():
        print(f"уже есть: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256:
            tmp.unlink()
            sys.exit(f"Неверная контрольная сумма {path.name}: {digest}")
    tmp.rename(path)
    print(f"сохранено: {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="скачать и обучающую часть FER-2013")
    args = parser.parse_args()

    download(YUNET_URL, ROOT / "models" / "face_detection_yunet_2023mar.onnx", YUNET_SHA256)
    names = ["privateTest.parquet"] + (["train.parquet"] if args.train else [])
    for name in names:
        download(FER_BASE + FER_FILES[name], ROOT / "data" / "fer2013" / name)


if __name__ == "__main__":
    main()
