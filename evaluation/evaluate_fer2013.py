# -*- coding: utf-8 -*-
"""
Оценка качества классификатора эмоций на FER-2013.

Основная оценка — стратифицированная выборка из тестовой части PrivateTest
(по умолчанию 2000 изображений из 3589). Для контроля «утечки» та же
процедура повторяется на выборке из обучающей части FER-2013 (Training):
если точность на тесте почти такая же, как на обучающих данных, и сильно
выше известных результатов для FER-2013 (~73–76 % у лучших моделей),
значит, тестовые изображения, скорее всего, были в обучении модели.

Запуск (после scripts/download_data.py --train):
    python -m evaluation.evaluate_fer2013
Результаты: evaluation/results/
"""

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emotrack.classifier import EmotionClassifier  # noqa: E402
from emotrack.config import EMOTION_RU, EMOTIONS, PROJECT_DIR  # noqa: E402

DATA_DIR = PROJECT_DIR / "data" / "fer2013"
OUT_DIR = PROJECT_DIR / "evaluation" / "results"

# Метки FER-2013 в наборе на HF: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_split(name):
    df = pq.read_table(DATA_DIR / f"{name}.parquet").to_pandas()
    df["emotion"] = df["label"].map(lambda i: FER_LABELS[i])
    df["bytes"] = df["image"].map(lambda d: d["bytes"])
    return df[["emotion", "bytes"]]


def decode(png_bytes):
    rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pixel_hash(png_bytes):
    gray = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    return hashlib.md5(gray.tobytes()).hexdigest()


def evaluate(classifier, df, name):
    started = time.perf_counter()
    probs = classifier.predict([decode(b) for b in df["bytes"]])
    elapsed = time.perf_counter() - started
    y_true = df["emotion"].tolist()
    y_pred = [EMOTIONS[i] for i in probs.argmax(axis=1)]
    report = classification_report(y_true, y_pred, labels=EMOTIONS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    return {
        "split": name,
        "n": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=EMOTIONS, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=EMOTIONS, average="weighted"), 4),
        "per_class": {e: {k: round(v, 4) for k, v in report[e].items()} for e in EMOTIONS},
        "confusion_matrix": cm.tolist(),
        "images_per_sec": round(len(df) / elapsed, 1),
    }


def plot_confusion(result, path):
    cm = np.array(result["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [EMOTION_RU[e] for e in EMOTIONS]
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            color = "white" if norm[i, j] > 0.55 else "#17171a"
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Предсказанная эмоция")
    ax.set_ylabel("Истинная эмоция")
    ax.set_title(f"Матрица ошибок, FER-2013 {result['split']} (n = {result['n']})\n"
                 f"accuracy = {result['accuracy']:.3f}, macro-F1 = {result['macro_f1']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="доля от истинного класса")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_class_csv(result, path):
    lines = ["эмоция;emotion;precision;recall;f1;support"]
    for e in EMOTIONS:
        r = result["per_class"][e]
        lines.append(f"{EMOTION_RU[e]};{e};{r['precision']};{r['recall']};{r['f1-score']};{int(r['support'])}")
    lines.append(f"accuracy;;;;{result['accuracy']};{result['n']}")
    lines.append(f"macro-F1;;;;{result['macro_f1']};{result['n']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="размер стратифицированной выборки")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test = load_split("privateTest")
    test_sample, _ = train_test_split(test, train_size=args.n, stratify=test["emotion"],
                                      random_state=args.seed)
    classifier = EmotionClassifier(device=args.device)

    results = {"model": classifier.model_name, "device": classifier.device, "seed": args.seed}
    main_result = evaluate(classifier, test_sample, "PrivateTest")
    results["test"] = main_result
    plot_confusion(main_result, OUT_DIR / "confusion_matrix_privateTest.png")
    save_per_class_csv(main_result, OUT_DIR / "per_class_privateTest.csv")

    # Контроль утечки: та же процедура на обучающей части FER-2013
    train_path = DATA_DIR / "train.parquet"
    if train_path.exists():
        train = load_split("train")
        train_sample, _ = train_test_split(train, train_size=args.n, stratify=train["emotion"],
                                           random_state=args.seed)
        train_result = evaluate(classifier, train_sample, "Training")
        results["train_check"] = train_result
        plot_confusion(train_result, OUT_DIR / "confusion_matrix_training.png")

        # Точные дубликаты изображений между PrivateTest и Training
        train_hashes = set(train["bytes"].map(pixel_hash))
        dup = test["bytes"].map(pixel_hash).isin(train_hashes)
        results["duplicates_test_in_train"] = {"count": int(dup.sum()), "of": len(test)}

    # Результат по всему PrivateTest (3589) — для сравнения с выборкой
    results["test_full"] = {k: v for k, v in evaluate(classifier, test, "PrivateTest (полностью)").items()
                            if k in ("split", "n", "accuracy", "macro_f1", "weighted_f1")}

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: (v if not isinstance(v, dict) or "per_class" not in v else
                          {kk: v[kk] for kk in ("n", "accuracy", "macro_f1", "weighted_f1")})
                      for k, v in results.items()}, ensure_ascii=False, indent=2))
    print("\nПо классам (PrivateTest):")
    for e in EMOTIONS:
        r = main_result["per_class"][e]
        print(f"  {EMOTION_RU[e]:<11} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1-score']:.3f} n={int(r['support'])}")


if __name__ == "__main__":
    main()
