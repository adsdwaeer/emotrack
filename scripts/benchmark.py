# -*- coding: utf-8 -*-
"""
Замер производительности конвейера на демонстрационном видео (GPU и CPU).

Считаем: полное время обработки, выбранных кадров в секунду, лиц в секунду,
время обработки одной минуты видео, а также раскладку по этапам
(чтение/декодирование видео, детекция, классификация).

Запуск:
    python scripts/benchmark.py                 # cuda (если есть) и cpu
Результат: docs/benchmark.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from emotrack.classifier import EmotionClassifier  # noqa: E402
from emotrack.pipeline import analyze_video, iter_video_frames  # noqa: E402


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "data" / "demo" / "demo_lesson.mp4"))
    parser.add_argument("--devices", default="cuda,cpu")
    args = parser.parse_args()

    # Только чтение видео и выборка кадров (без анализа)
    start = time.perf_counter()
    n = sum(1 for _ in iter_video_frames(args.video))
    decode_sec = time.perf_counter() - start

    result = {
        "video": Path(args.video).name,
        "cpu": cpu_name(), "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "decode_only_sec": round(decode_sec, 2), "sampled_frames": n,
        "runs": {},
    }
    for device in args.devices.split(","):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        clf = EmotionClassifier(device=device)
        clf.predict([__import__("numpy").zeros((64, 64, 3), "uint8")] * 4)   # прогрев
        participants, s = analyze_video(args.video, clf)
        minutes = s["duration_sec"] / 60
        faces = s["faces_classified"]
        run = {
            "processing_sec": s["processing_sec"],
            "video_sec": s["duration_sec"],
            "sampled_frames": s["frames_processed"],
            "faces_classified": faces,
            "frames_per_sec": round(s["frames_processed"] / s["processing_sec"], 2),
            "faces_per_sec": round(faces / s["timings"]["classify"], 1),
            "sec_per_video_minute": round(s["processing_sec"] / minutes, 2),
            "realtime_factor": round(s["duration_sec"] / s["processing_sec"], 1),
            "detect_sec": s["timings"]["detect"],
            "classify_sec": s["timings"]["classify"],
            "other_sec": round(s["processing_sec"] - s["timings"]["detect"] - s["timings"]["classify"], 2),
        }
        result["runs"][device] = run
        print(device, json.dumps(run, ensure_ascii=False))
        del clf
        if device == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "docs" / "benchmark.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
