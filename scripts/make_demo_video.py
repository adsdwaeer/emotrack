# -*- coding: utf-8 -*-
"""
Демонстрационное видео «вебинара» без изображений реальных людей из интернета.

Сетка 3×2 из шести «слушателей». Лица — изображения открытого набора FER-2013
(обучающая часть, 48×48, увеличены до ~200 px). Каждые 4 секунды на плитке
появляется новое изображение с эмоцией, выбранной по сценарию участника.
Сценарии заданы заранее (SCENARIOS) и сохраняются вместе с видео в JSON —
по ним потом проверяем, отражает ли индекс то, что «происходило» на занятии.

Ограничение: в FER-2013 нет идентификаторов людей, поэтому у одного
«слушателя» каждые 4 с меняется внешность — для конвейера это не важно
(сопровождение идёт по положению плитки), но выглядит условно.

Запуск (после scripts/download_data.py --train):
    python scripts/make_demo_video.py            # data/demo/demo_lesson.mp4, 8 минут
"""

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emotrack.config import settings  # noqa: E402
from emotrack.detector import YuNetDetector  # noqa: E402

FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
W, H = 1280, 720
COLS, ROWS, GAP, TOP = 3, 2, 10, 50
TILE_W = (W - GAP * (COLS + 1)) // COLS
TILE_H = (H - TOP - GAP * (ROWS + 1)) // ROWS
FACE = 200
SEGMENT = 4.0   # секунд на одно изображение

# Сценарии: список (до какой секунды, {эмоция: вероятность}); "absent" — лица нет
SCENARIOS = {
    "Слушатель 1": {
        "role": "активный: радость, интерес, удивление на протяжении всего занятия",
        "plan": [(480, {"happy": 0.45, "neutral": 0.35, "surprise": 0.2})]},
    "Слушатель 2": {
        "role": "спокойный: в основном нейтральное выражение",
        "plan": [(480, {"neutral": 0.8, "happy": 0.1, "sad": 0.1})]},
    "Слушатель 3": {
        "role": "теряет интерес: нейтрально/радость -> грусть -> уходит от камеры",
        "plan": [(150, {"neutral": 0.6, "happy": 0.3, "surprise": 0.1}),
                 (300, {"neutral": 0.5, "sad": 0.5}),
                 (400, {"sad": 0.7, "neutral": 0.2, "absent": 0.1}),
                 (480, {"absent": 0.6, "sad": 0.4})]},
    "Слушатель 4": {
        "role": "отходит от камеры на 2-4-й минутах, остальное время внимателен",
        "plan": [(120, {"neutral": 0.7, "happy": 0.3}),
                 (240, {"absent": 1.0}),
                 (480, {"neutral": 0.7, "happy": 0.3})]},
    "Слушатель 5": {
        "role": "раздражение в середине занятия (3-5,5 мин), потом успокаивается",
        "plan": [(180, {"neutral": 0.8, "happy": 0.2}),
                 (330, {"angry": 0.6, "sad": 0.2, "neutral": 0.2}),
                 (480, {"neutral": 0.7, "happy": 0.3})]},
    "Слушатель 6": {
        "role": "подключился с опозданием (на 1,5 мин), дальше вовлечён",
        "plan": [(90, {"absent": 1.0}),
                 (480, {"happy": 0.4, "neutral": 0.4, "surprise": 0.2})]},
}


def load_pools(seed):
    """Изображения FER-2013 по эмоциям; оставляем только те, где YuNet видит лицо."""
    df = pq.read_table(ROOT / "data" / "fer2013" / "train.parquet").to_pandas()
    df = df.sample(frac=1.0, random_state=seed)
    detector = YuNetDetector(settings.yunet_path, settings.detector_score)
    pools = {}
    for label_id, name in enumerate(FER_LABELS):
        pool = []
        for item in df[df["label"] == label_id]["image"]:
            face = to_face(item["bytes"])
            canvas = np.full((TILE_H, TILE_W, 3), 60, np.uint8)
            y0, x0 = (TILE_H - FACE) // 2, (TILE_W - FACE) // 2
            canvas[y0:y0 + FACE, x0:x0 + FACE] = face
            if detector.detect(canvas):
                pool.append(face)
            if len(pool) >= 120:
                break
        pools[name] = pool
    return pools


def to_face(png_bytes):
    rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (FACE, FACE), interpolation=cv2.INTER_CUBIC)


def emotion_at(plan, t, rng):
    for until, dist in plan:
        if t < until:
            names = list(dist)
            return rng.choice(names, p=np.array([dist[n] for n in names]) / sum(dist.values()))
    return "absent"


def get_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_texts(frame, items):
    """Надписи кириллицей (cv2.putText её не умеет) — через PIL."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for xy, text, size, color in items:
        draw.text(xy, text, font=get_font(size), fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(ROOT / "data" / "demo" / "demo_lesson.mp4"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pools = load_pools(args.seed)
    print("изображений по эмоциям:", {k: len(v) for k, v in pools.items()})

    duration = args.minutes * 60
    names = list(SCENARIOS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    truth = {n: [] for n in names}
    n_segments = int(np.ceil(duration / SEGMENT))
    for seg in range(n_segments):
        t0 = seg * SEGMENT
        frame = np.full((H, W, 3), (36, 33, 32), np.uint8)
        labels = [((GAP, 12), f"Вебинар «Машинное обучение», лекция 5 · {int(t0 // 60):02d}:{int(t0 % 60):02d}",
                   22, (230, 230, 230))]
        for i, name in enumerate(names):
            col, row = i % COLS, i // COLS
            x = GAP + col * (TILE_W + GAP)
            y = TOP + GAP + row * (TILE_H + GAP)
            cv2.rectangle(frame, (x, y), (x + TILE_W, y + TILE_H), (67, 64, 60), -1)
            emotion = emotion_at(SCENARIOS[name]["plan"], t0, rng)
            truth[name].append({"t": t0, "emotion": str(emotion)})
            if emotion != "absent":
                face = pools[emotion][rng.integers(len(pools[emotion]))]
                # небольшое смещение — как будто человек немного двигается
                dx, dy = rng.integers(-12, 13), rng.integers(-8, 9)
                fx = x + (TILE_W - FACE) // 2 + dx
                fy = y + (TILE_H - FACE) // 2 + dy
                frame[fy:fy + FACE, fx:fx + FACE] = face
            else:
                cx, cy = x + TILE_W // 2, y + TILE_H // 2
                cv2.circle(frame, (cx, cy), 46, (110, 105, 100), -1)   # «аватар» без лица
            labels.append(((x + 10, y + TILE_H - 30), name, 18, (240, 240, 240)))
        frame = draw_texts(frame, labels)
        for _ in range(int(SEGMENT * args.fps)):
            writer.write(frame)
    writer.release()

    meta = {"video": out.name, "duration_sec": duration, "fps": args.fps, "segment_sec": SEGMENT,
            "width": W, "height": H, "grid": {"cols": COLS, "rows": ROWS},
            "seed": args.seed, "participants": {n: SCENARIOS[n]["role"] for n in names},
            "truth": truth}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {out} ({duration:.0f} с), сценарий: {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
