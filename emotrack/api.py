# -*- coding: utf-8 -*-
"""
FastAPI — REST API и дашборд прототипа «ЭмоТрек».
Автор: Кузнецов Никита

Запуск:
    uvicorn emotrack.api:app --port 8000

Дашборд:    http://127.0.0.1:8000/
Swagger UI: http://127.0.0.1:8000/docs
"""

import csv
import io
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import engagement as eng
from . import storage
from .aggregation import mean_probs
from .config import EMOTION_RU, EMOTIONS, settings
from .detector import create_detector
from .pipeline import LessonAnalyzer, analyze_video

BASE_DIR = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class StudentIn(BaseModel):
    student_id: str | None = None


def create_app(classifier=None, db_path=None, detector_factory=create_detector):
    """Фабрика приложения. В тестах сюда передаются заглушки классификатора
    и детектора и временная БД, чтобы не загружать модель."""
    state = {"classifier": classifier, "live": {}}
    gpu_lock = threading.Lock()   # одна модель на GPU — обрабатываем по очереди

    @asynccontextmanager
    async def lifespan(app):
        if state["classifier"] is None:
            from .classifier import EmotionClassifier
            state["classifier"] = EmotionClassifier()
        yield

    app = FastAPI(
        title="ЭмоТрек API",
        description="Оценка вовлечённости слушателей онлайн-занятий по эмоциям на лицах. "
                    "Модель: dima806/facial_emotions_image_detection (ViT). "
                    "Изображения не сохраняются — только агрегаты (152-ФЗ).",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    def db():
        return storage.connect(db_path)

    def lesson_or_404(conn, lesson_id):
        lesson = storage.get_lesson(conn, lesson_id)
        if not lesson:
            raise HTTPException(404, "Занятие не найдено")
        return lesson

    def read_image(data):
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Не удалось прочитать изображение")
        return frame

    # ---------- фоновая обработка видео ----------

    def process_video(lesson_id, path):
        conn = db()
        try:
            storage.update_lesson(conn, lesson_id, status="processing")
            with gpu_lock:
                participants, summary = analyze_video(
                    path, state["classifier"], detector_factory(),
                    progress=lambda p: storage.update_lesson(conn, lesson_id, progress=round(p, 3)))
            storage.save_result(conn, lesson_id, participants, summary)
            storage.update_lesson(conn, lesson_id, status="done", progress=1.0)
        except Exception as exc:  # статус ошибки виден через API
            storage.update_lesson(conn, lesson_id, status="error", error=str(exc))
        finally:
            # Видео занятия — персональные данные: удаляем сразу после обработки
            Path(path).unlink(missing_ok=True)
            conn.close()

    def lesson_summary(conn, lesson_id):
        lesson = lesson_or_404(conn, lesson_id)
        participants = storage.get_participants(conn, lesson_id)
        intervals = storage.get_intervals(conn, lesson_id)
        vectors = [[p["emotions"][e] for e in EMOTIONS] for p in participants if p["emotions"]]
        index = lesson["engagement"]
        return {
            "lesson": lesson,
            "engagement": index,
            "level": eng.level(index),
            "participants": len(participants),
            "at_risk": sum(p["risk"] for p in participants),
            "emotions": mean_probs(vectors),
            "minutes": eng.lesson_timeline(intervals),
        }

    # ---------- служебное ----------

    @app.get("/api/health", tags=["Служебное"], summary="Проверка работоспособности")
    def health():
        clf = state["classifier"]
        return {"status": "ok", "model": settings.model_name,
                "model_loaded": clf is not None, "device": getattr(clf, "device", None)}

    # ---------- занятия (видеозапись) ----------

    @app.post("/api/lessons", status_code=202, tags=["Занятия"],
              summary="Загрузить видеозапись занятия (обработка в фоне)")
    async def upload_lesson(background: BackgroundTasks,
                            file: UploadFile = File(..., description="Видео занятия: mp4, avi, mov, mkv, webm"),
                            title: str = Form("", description="Название занятия")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"Поддерживаются видео: {', '.join(sorted(VIDEO_EXT))}")
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        path = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        conn = db()
        lesson_id = storage.create_lesson(conn, title or Path(file.filename).stem)
        conn.close()
        background.add_task(process_video, lesson_id, path)
        return {"id": lesson_id, "status": "queued"}

    @app.get("/api/lessons", tags=["Занятия"], summary="Список занятий")
    def lessons():
        conn = db()
        rows = storage.list_lessons(conn)
        conn.close()
        for r in rows:
            r["level"] = eng.level(r["engagement"])
        return rows

    @app.get("/api/lessons/{lesson_id}/status", tags=["Занятия"], summary="Статус обработки")
    def lesson_status(lesson_id: int):
        conn = db()
        lesson = lesson_or_404(conn, lesson_id)
        conn.close()
        return {k: lesson[k] for k in ("id", "status", "progress", "error")}

    @app.get("/api/lessons/{lesson_id}", tags=["Занятия"], summary="Сводка по занятию")
    def lesson_detail(lesson_id: int):
        conn = db()
        try:
            return lesson_summary(conn, lesson_id)
        finally:
            conn.close()

    @app.get("/api/lessons/{lesson_id}/timeline", tags=["Занятия"],
             summary="Динамика индекса по 30-секундным интервалам и по минутам")
    def lesson_timeline(lesson_id: int):
        conn = db()
        lesson_or_404(conn, lesson_id)
        intervals = storage.get_intervals(conn, lesson_id)
        conn.close()
        return {"interval_sec": settings.interval_sec,
                "minutes": eng.lesson_timeline(intervals),
                "intervals": intervals}

    @app.get("/api/lessons/{lesson_id}/participants", tags=["Занятия"],
             summary="Участники занятия с индексами и флагами риска")
    def lesson_participants(lesson_id: int):
        conn = db()
        lesson_or_404(conn, lesson_id)
        rows = storage.get_participants(conn, lesson_id)
        conn.close()
        return rows

    @app.get("/api/lessons/{lesson_id}/export.csv", tags=["Занятия"],
             summary="Экспорт интервалов занятия в CSV")
    def export_csv(lesson_id: int):
        conn = db()
        lesson_or_404(conn, lesson_id)
        intervals = storage.get_intervals(conn, lesson_id)
        conn.close()
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=";")
        writer.writerow(["участник", "начало_с", "конец_с", "кадров", "кадров_с_лицом", "q",
                         *[EMOTION_RU[e] for e in EMOTIONS], "индекс", "уровень", "зона_риска"])
        for it in intervals:
            probs = it["probs"] or {}
            writer.writerow([it["track_id"], it["t_start"], it["t_end"], it["n_frames"], it["n_face"],
                             it["q"], *[probs.get(e, "") for e in EMOTIONS], it["engagement"],
                             eng.LEVEL_RU[eng.level(it["engagement"])],
                             "да" if eng.is_risk(it["engagement"]) else "нет"])
        # BOM — чтобы Excel правильно открыл кириллицу
        data = "﻿" + buf.getvalue()
        return StreamingResponse(iter([data]), media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="lesson_{lesson_id}.csv"'})

    @app.delete("/api/lessons/{lesson_id}", tags=["Занятия"], summary="Удалить занятие")
    def remove_lesson(lesson_id: int):
        conn = db()
        lesson_or_404(conn, lesson_id)
        storage.delete_lesson(conn, lesson_id)
        conn.close()
        return {"deleted": lesson_id}

    # ---------- слушатели ----------

    @app.put("/api/participants/{participant_id}", tags=["Слушатели"],
             summary="Связать участника занятия со слушателем LMS")
    def set_student(participant_id: int, body: StudentIn):
        conn = db()
        ok = storage.set_student_id(conn, participant_id, body.student_id)
        conn.close()
        if not ok:
            raise HTTPException(404, "Участник не найден")
        return {"id": participant_id, "student_id": body.student_id}

    @app.get("/api/students/at-risk", tags=["Слушатели"],
             summary="Слушатели с индексом < 40 на трёх занятиях подряд")
    def students_at_risk():
        conn = db()
        history = storage.student_history(conn)
        conn.close()
        result = []
        for student, rows in history.items():
            values = [r["engagement"] for r in rows]
            if eng.dropout_risk(values):
                result.append({"student_id": student,
                               "last_lessons": rows[-settings.risk_lessons_in_row:]})
        return result

    # ---------- живой режим (веб-камера) ----------

    @app.post("/api/analyze-frame", tags=["Живой режим"],
              summary="Анализ одиночного кадра (без сохранения)")
    async def analyze_frame(file: UploadFile = File(..., description="Кадр: jpg, png, webp")):
        frame = read_image(await file.read())
        analyzer = LessonAnalyzer(state["classifier"], detector_factory())
        with gpu_lock:
            faces = analyzer.process_frames([(0.0, frame)])[0]["faces"]
        return {"faces": faces, "count": len(faces)}

    @app.post("/api/live", tags=["Живой режим"], summary="Начать живое занятие")
    def live_start(title: str = Form("Живое занятие")):
        conn = db()
        lesson_id = storage.create_lesson(conn, title, source="live", status="processing")
        conn.close()
        state["live"][lesson_id] = {
            "analyzer": LessonAnalyzer(state["classifier"], detector_factory()),
            "started": time.time(), "lock": threading.Lock()}
        return {"id": lesson_id}

    @app.post("/api/live/{lesson_id}/frame", tags=["Живой режим"],
              summary="Отправить кадр с веб-камеры в живое занятие")
    async def live_frame(lesson_id: int, file: UploadFile = File(...),
                         t: float | None = Form(None, description="Время кадра от начала, с")):
        session = state["live"].get(lesson_id)
        if not session:
            raise HTTPException(404, "Живое занятие не найдено или уже завершено")
        frame = read_image(await file.read())
        if t is None:
            t = round(time.time() - session["started"], 2)
        with session["lock"], gpu_lock:
            faces = session["analyzer"].process_frames([(t, frame)])[0]["faces"]
            participants, summary = session["analyzer"].result()
        summary["duration_sec"] = t
        conn = db()
        storage.save_result(conn, lesson_id, participants, summary)
        conn.close()
        return {"t": t, "faces": faces, "engagement": summary["engagement"],
                "level": summary["level"],
                "participants": [{k: p[k] for k in ("track_id", "engagement", "level", "risk")}
                                 for p in participants]}

    @app.post("/api/live/{lesson_id}/finish", tags=["Живой режим"], summary="Завершить живое занятие")
    def live_finish(lesson_id: int):
        session = state["live"].pop(lesson_id, None)
        if not session:
            raise HTTPException(404, "Живое занятие не найдено или уже завершено")
        conn = db()
        storage.update_lesson(conn, lesson_id, status="done", progress=1.0)
        conn.close()
        return {"id": lesson_id, "status": "done"}

    # ---------- страницы дашборда ----------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page_index(request: Request):
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/lessons/{lesson_id}", response_class=HTMLResponse, include_in_schema=False)
    def page_lesson(request: Request, lesson_id: int):
        conn = db()
        lesson = lesson_or_404(conn, lesson_id)
        conn.close()
        return templates.TemplateResponse(request, "lesson.html", {"lesson": lesson})

    @app.get("/live", response_class=HTMLResponse, include_in_schema=False)
    def page_live(request: Request):
        return templates.TemplateResponse(request, "live.html", {})

    return app


app = create_app()
