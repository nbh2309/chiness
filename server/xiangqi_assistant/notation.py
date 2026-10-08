"""Chuyển nước đi UCI (vd. h2e2) sang ký hiệu cờ tướng tiếng Việt (vd. "Pháo 2 bình 5").

Quy tắc:
  - Cột đánh số 1..9 từ phải sang trái theo góc nhìn của bên đi.
  - "tấn" = tiến, "thoái" = lui, "bình" = đi ngang.
  - Quân đi thẳng (Xe, Pháo, Tốt, Tướng): số sau tấn/thoái là số bước.
  - Quân đi chéo (Mã, Tượng, Sĩ): số sau tấn/thoái là cột đích.
  - Hai quân cùng loại trên một cột: dùng "trước"/"sau" (3 quân: thêm "giữa").
"""
from __future__ import annotations

from .board import RED, Move, Position, side_of, uci_to_move

NAMES = {"K": "Tướng", "A": "Sĩ", "B": "Tượng", "N": "Mã", "R": "Xe", "C": "Pháo", "P": "Tốt"}
SHORT = {"K": "Tg", "A": "S", "B": "T", "N": "M", "R": "X", "C": "P", "P": "B"}
DIAGONAL_PIECES = {"N", "B", "A"}


def file_number(f: int, side: str) -> int:
    return 9 - f if side == RED else f + 1


def _rank_order(squares: list, side: str) -> list:
    """Sắp các ô từ trước ra sau theo hướng tiến của bên `side`."""
    return sorted(squares, key=lambda sq: -sq[1] if side == RED else sq[1])


def describe(pos: Position, move: Move) -> dict:
    src, dst = move
    piece = pos[src]
    if piece is None:
        raise ValueError(f"Không có quân ở ô xuất phát của nước {move}")
    side = side_of(piece)
    t = piece.upper()
    fwd = 1 if side == RED else -1

    # Phần định danh quân: "Pháo 2" hoặc "Xe trước"
    same_file = [sq for sq, p in pos.pieces() if p == piece and sq[0] == src[0]]
    if len(same_file) >= 2 and t not in ("K",):
        order = _rank_order(same_file, side)
        idx = order.index(src)
        if len(order) == 2:
            tags = ["trước", "sau"]
            short_tags = ["t", "s"]
        elif len(order) == 3:
            tags = ["trước", "giữa", "sau"]
            short_tags = ["t", "g", "s"]
        else:
            tags = [str(i + 1) for i in range(len(order))]
            short_tags = tags
        who = f"{NAMES[t]} {tags[idx]}"
        who_short = f"{SHORT[t]}{short_tags[idx]}"
    else:
        n = file_number(src[0], side)
        who = f"{NAMES[t]} {n}"
        who_short = f"{SHORT[t]}{n}"

    dr = (dst[1] - src[1]) * fwd
    if dr == 0:
        action, sym, num = "bình", "-", file_number(dst[0], side)
    else:
        action, sym = ("tấn", ".") if dr > 0 else ("thoái", "/")
        num = file_number(dst[0], side) if t in DIAGONAL_PIECES else abs(dr)

    return {
        "text": f"{who} {action} {num}",
        "short": f"{who_short}{sym}{num}",
        "piece": piece,
        "capture": pos[dst],
    }


def describe_uci(pos: Position, uci: str) -> dict:
    return describe(pos, uci_to_move(uci))
