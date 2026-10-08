"""Engine gợi ý nước đi.

Ưu tiên Pikafish (rất mạnh, chạy qua giao thức UCI). Nếu chưa cài Pikafish,
dùng engine Python đơn giản để hệ thống vẫn chạy được (yếu hơn nhiều).
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..board import Position

log = logging.getLogger(__name__)


@dataclass
class EngineResult:
    best_move: Optional[str]           # UCI, None nếu hết nước (thua)
    score_cp: Optional[int] = None     # điểm theo góc nhìn bên đi, đơn vị 1/100 tốt
    mate_in: Optional[int] = None      # >0: bên đi chiếu bí sau n nước; <0: bị bí
    depth: int = 0
    pv: list[str] = field(default_factory=list)
    engine: str = ""


class Engine(Protocol):
    name: str

    def analyse(self, pos: Position, movetime_ms: int = 1000) -> EngineResult: ...

    def close(self) -> None: ...


def find_pikafish() -> Optional[str]:
    path = os.environ.get("PIKAFISH_PATH")
    if path and os.path.isfile(path):
        return path
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for cand in (os.path.join(here, "engines", "pikafish"), shutil.which("pikafish")):
        if cand and os.path.isfile(cand):
            return cand
    return None


def create_engine() -> Engine:
    path = find_pikafish()
    if path:
        from .uci import UciEngine

        try:
            return UciEngine(path, eval_file=os.environ.get("PIKAFISH_EVALFILE"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Không khởi động được Pikafish (%s), dùng engine dự phòng", exc)
    else:
        log.warning("Không tìm thấy Pikafish, dùng engine Python dự phòng (yếu)")
    from .simple import SimpleEngine

    return SimpleEngine()
