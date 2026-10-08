"""HTTP API cho kính / điện thoại.

Chạy:  uvicorn xiangqi_assistant.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .analysis import Analyzer, decode_image
from .board import Position
from .vision.geometry import BoardNotFound

app = FastAPI(title="Trợ lý cờ tướng", version="0.1.0")
analyzer = Analyzer()
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _parse_corners(corners: Optional[str]):
    if not corners:
        return None
    try:
        pts = json.loads(corners)
        assert len(pts) == 4 and all(len(p) == 2 for p in pts)
        return [[float(x), float(y)] for x, y in pts]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "corners phải là JSON [[x,y],[x,y],[x,y],[x,y]]") from exc


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/health")
def health():
    from .vision.recognize import get_classifier

    return {"ok": True, "engine": analyzer.engine.name, "classifier": get_classifier().source}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    turn: str = Form("auto", description="'w' (Đỏ), 'b' (Đen) hoặc 'auto'"),
    session_id: Optional[str] = Form(None, description="ID ván cờ để tự suy ra lượt đi"),
    movetime_ms: int = Form(1000, ge=50, le=30000),
    corners: Optional[str] = Form(None, description="4 góc bàn cờ JSON nếu tự động sai"),
    debug: bool = Form(False),
):
    if turn not in ("w", "b", "auto"):
        raise HTTPException(400, "turn phải là w, b hoặc auto")
    try:
        img = decode_image(await image.read())
        return analyzer.analyze_image(img, turn=turn, session_id=session_id, corners=_parse_corners(corners),
                                      movetime_ms=movetime_ms, debug=debug)
    except BoardNotFound as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class FenRequest(BaseModel):
    fen: str
    movetime_ms: int = 1000


@app.post("/api/analyze_fen")
def analyze_fen(req: FenRequest):
    try:
        pos = Position.from_fen(req.fen)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"fen": pos.fen(), "board": pos.ascii(), "suggestion": analyzer.suggest(pos, req.movetime_ms)}


@app.post("/api/calibrate")
async def calibrate(image: UploadFile = File(...), corners: Optional[str] = Form(None)):
    """Ảnh thế cờ ban đầu -> lưu 32 ảnh quân thật có nhãn để tinh chỉnh CNN."""
    from .vision.recognize import collect_crops

    try:
        return collect_crops(decode_image(await image.read()), corners=_parse_corners(corners))
    except BoardNotFound as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/session/{session_id}/reset")
def reset_session(session_id: str):
    analyzer.sessions.pop(session_id, None)
    return {"ok": True}
