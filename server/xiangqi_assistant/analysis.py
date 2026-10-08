"""Ghép các phần lại: ảnh -> thế cờ -> lượt đi -> engine -> gợi ý bằng tiếng Việt."""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .board import BLACK, RED, Position, move_to_uci, opponent, uci_to_move, validate
from .engine import Engine, EngineResult, create_engine
from .notation import describe, describe_uci
from .rules import apply_move, find_move_between, in_check, legal_moves

SIDE_NAME = {RED: "Đỏ", BLACK: "Đen"}


@dataclass
class GameSession:
    """Theo dõi một ván: nhớ thế cờ trước để tự suy ra ai vừa đi và đến lượt ai."""
    last: Optional[Position] = None
    history: list[str] = field(default_factory=list)


class Analyzer:
    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine
        self._lock = threading.Lock()
        self.sessions: dict[str, GameSession] = {}

    @property
    def engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = create_engine()
            return self._engine

    # ------------------------------------------------------------------ lượt đi
    def _decide_turn(self, pos: Position, turn: str, session: Optional[GameSession]) -> tuple[str, Optional[dict], list[str]]:
        notes: list[str] = []
        last_move = None
        if session and session.last is not None and session.last.grid != pos.grid:
            found = find_move_between(session.last, pos)
            if found:
                side, mv = found
                before = session.last.copy()
                before.turn = side
                last_move = {"side": side, "uci": move_to_uci(mv), **describe(before, mv)}
            else:
                notes.append("Thế cờ thay đổi nhiều hơn một nước so với ảnh trước (hoặc nhận dạng sai).")
        if turn in (RED, BLACK):
            return turn, last_move, notes
        if last_move:
            return opponent(last_move["side"]), last_move, notes
        for side in (RED, BLACK):
            if in_check(pos, side) and not in_check(pos, opponent(side)):
                return side, last_move, notes + [f"{SIDE_NAME[side]} đang bị chiếu nên đến lượt {SIDE_NAME[side]}."]
        if session and session.last is not None and session.last.grid == pos.grid:
            return session.last.turn, None, notes
        notes.append("Không xác định được lượt đi, mặc định Đỏ đi. Hãy chọn lượt đi nếu sai.")
        return RED, last_move, notes

    # ------------------------------------------------------------------ engine
    def suggest(self, pos: Position, movetime_ms: int = 1000) -> dict:
        errors = validate(pos)
        if errors:
            return {"ok": False, "errors": errors}
        if not legal_moves(pos):
            return {"ok": True, "best_move": None, "game_over": True,
                    "display": f"{SIDE_NAME[pos.turn]} hết nước đi (thua)."}
        res: EngineResult = self.engine.analyse(pos, movetime_ms)
        out: dict = {"ok": True, "engine": res.engine, "depth": res.depth}
        if res.best_move is None:
            out.update(best_move=None, game_over=True, display=f"{SIDE_NAME[pos.turn]} hết nước đi.")
            return out
        if uci_to_move(res.best_move) not in legal_moves(pos):
            return {"ok": False, "errors": [f"Engine trả về nước không hợp lệ: {res.best_move}"]}
        desc = describe_uci(pos, res.best_move)
        # Điểm theo góc nhìn bên Đỏ để hiển thị nhất quán
        sign = 1 if pos.turn == RED else -1
        score_red = res.score_cp * sign if res.score_cp is not None else None
        mate_red = res.mate_in * sign if res.mate_in is not None else None
        pv_text = []
        cur = pos
        for u in res.pv[:6]:
            try:
                mv = uci_to_move(u)
                if mv not in legal_moves(cur):
                    break
                pv_text.append(describe(cur, mv)["short"])
                cur = apply_move(cur, mv)
            except ValueError:
                break
        out.update(best_move=res.best_move, move_text=desc["text"], move_short=desc["short"],
                   capture=desc["capture"], score_cp_red=score_red, mate_in_red=mate_red, pv=pv_text,
                   display=_display(pos.turn, desc, score_red, mate_red))
        return out

    # ------------------------------------------------------------------ ảnh
    def analyze_image(self, img: np.ndarray, turn: str = "auto", session_id: Optional[str] = None,
                      corners: Optional[Sequence[Sequence[float]]] = None, movetime_ms: int = 1000,
                      debug: bool = False) -> dict:
        from .vision.recognize import recognize

        rec = recognize(img, turn=RED, corners=corners)
        pos = rec.position
        session = self.sessions.setdefault(session_id, GameSession()) if session_id else None
        side, last_move, notes = self._decide_turn(pos, turn, session)
        pos.turn = side
        result = {
            "fen": pos.fen(),
            "board": pos.ascii(),
            "turn": side,
            "last_move": last_move,
            "recognition": {
                "classifier": rec.classifier,
                "low_confidence": rec.low_confidence_squares(),
                "warnings": rec.warnings,
                "grid_score": round(rec.grid.score, 3),
            },
            "notes": notes,
        }
        result["suggestion"] = self.suggest(pos, movetime_ms)
        if session is not None and result["suggestion"].get("ok"):
            session.last = pos.copy()
            if last_move:
                session.history.append(last_move["short"])
        if debug:
            ok, buf = cv2.imencode(".jpg", rec.debug_image(img), [cv2.IMWRITE_JPEG_QUALITY, 80])
            result["debug_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
        return result


def _display(turn: str, desc: dict, score_red: Optional[int], mate_red: Optional[int]) -> str:
    """Chuỗi ngắn hiển thị trên kính."""
    text = f"{SIDE_NAME[turn]}: {desc['text']}"
    if mate_red is not None:
        winner = "Đỏ" if mate_red > 0 else "Đen"
        text += f" | {winner} chiếu bí sau {abs(mate_red)} nước"
    elif score_red is not None:
        text += f" | {score_red / 100:+.1f}"
    return text


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được ảnh")
    return img
