# -*- coding: utf-8 -*-
"""
Скриншоты дашборда и Swagger для отчёта (headless Chromium через Playwright).

Сервер должен быть запущен:  uvicorn emotrack.api:app --port 8000
Для живого режима Chromium получает «веб-камеру» из файла MJPEG, нарезанного
из демонстрационного видео (флаг --use-file-for-fake-video-capture).

Запуск:
    python scripts/screenshots.py --lesson 1 --lesson2 2
Результат: docs/screenshots/*.png
"""

import argparse
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"


def make_fake_camera(video, path, seconds=90, fps=10, start=150):
    """MJPEG-файл (поток JPEG-кадров) для подстановки вместо веб-камеры."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / fps))
    with open(path, "wb") as out:
        for i in range(int(seconds * src_fps)):
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out.write(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--lesson", type=int, default=1)
    parser.add_argument("--lesson2", type=int, default=None)
    parser.add_argument("--live-seconds", type=int, default=75)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fake_cam = ROOT / "data" / "demo" / "fake_camera.mjpeg"
    make_fake_camera(ROOT / "data" / "demo" / "demo_lesson.mp4", fake_cam)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={fake_cam}"])
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                                locale="ru-RU")

        def shot(url, name, wait=1500, full=True):
            page.goto(args.base + url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(wait)
            page.screenshot(path=str(OUT / name), full_page=full)
            print("сохранено:", OUT / name)

        shot(f"/lessons/{args.lesson}", "02_lesson_demo.png")
        if args.lesson2:
            shot(f"/lessons/{args.lesson2}", "03_lesson_upload.png")

        # Живой режим: «камера» показывает демонстрационную сетку слушателей
        page.goto(args.base + "/live")
        page.click("#start")
        page.wait_for_timeout(args.live_seconds * 1000)
        page.screenshot(path=str(OUT / "04_live.png"), full_page=True)
        print("сохранено:", OUT / "04_live.png")
        page.click("#stop")
        page.wait_for_url("**/lessons/*")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "05_lesson_live.png"), full_page=True)
        print("сохранено:", OUT / "05_lesson_live.png")

        shot("/", "01_lessons.png")

        # Swagger: список эндпоинтов и выполненный запрос сводки по занятию
        shot("/docs", "06_swagger.png", wait=2500)
        page.click("#operations-Занятия-lesson_detail_api_lessons__lesson_id__get button.opblock-summary-control")
        page.click("button.try-out__btn")
        page.fill("input[placeholder='lesson_id']", str(args.lesson))
        page.click("button.execute")
        page.wait_for_selector(".live-responses-table .response-col_status")
        page.wait_for_timeout(1000)
        block = page.locator("#operations-Занятия-lesson_detail_api_lessons__lesson_id__get")
        block.screenshot(path=str(OUT / "07_swagger_lesson_summary.png"))
        print("сохранено:", OUT / "07_swagger_lesson_summary.png")
        browser.close()


if __name__ == "__main__":
    main()
