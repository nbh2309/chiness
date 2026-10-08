"""Sinh nước đi hợp lệ theo luật cờ tướng.

Dùng cho: kiểm tra nước đi engine trả về, đoán lượt đi khi theo dõi ván cờ,
và engine dự phòng khi chưa cài Pikafish.
"""
from __future__ import annotations

from typing import Optional

from .board import (
    BLACK,
    FILES,
    RANKS,
    RED,
    Move,
    Position,
    Square,
    in_palace,
    opponent,
    own_half,
    side_of,
)

ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
# (bước chân mã, đích)
KNIGHT = (
    ((0, 1), (1, 2)), ((0, 1), (-1, 2)),
    ((0, -1), (1, -2)), ((0, -1), (-1, -2)),
    ((1, 0), (2, 1)), ((1, 0), (2, -1)),
    ((-1, 0), (-2, 1)), ((-1, 0), (-2, -1)),
)


def on_board(f: int, r: int) -> bool:
    return 0 <= f < FILES and 0 <= r < RANKS


def _forward(side: str) -> int:
    return 1 if side == RED else -1


def pseudo_moves_from(pos: Position, sq: Square) -> list[Move]:
    piece = pos[sq]
    if piece is None:
        return []
    side = side_of(piece)
    t = piece.upper()
    f, r = sq
    out: list[Move] = []

    def add(nf: int, nr: int) -> None:
        target = pos.grid[nr][nf]
        if target is None or side_of(target) != side:
            out.append((sq, (nf, nr)))

    if t == "K":
        for df, dr in ORTHO:
            nf, nr = f + df, r + dr
            if on_board(nf, nr) and in_palace((nf, nr), side):
                add(nf, nr)
    elif t == "A":
        for df, dr in DIAG:
            nf, nr = f + df, r + dr
            if on_board(nf, nr) and in_palace((nf, nr), side):
                add(nf, nr)
    elif t == "B":
        for df, dr in DIAG:
            nf, nr = f + 2 * df, r + 2 * dr
            if on_board(nf, nr) and own_half((nf, nr), side) and pos.grid[r + dr][f + df] is None:
                add(nf, nr)
    elif t == "N":
        for (lf, lr), (df, dr) in KNIGHT:
            nf, nr = f + df, r + dr
            if on_board(nf, nr) and pos.grid[r + lr][f + lf] is None:
                add(nf, nr)
    elif t == "R":
        for df, dr in ORTHO:
            nf, nr = f + df, r + dr
            while on_board(nf, nr):
                target = pos.grid[nr][nf]
                if target is None:
                    out.append((sq, (nf, nr)))
                else:
                    if side_of(target) != side:
                        out.append((sq, (nf, nr)))
                    break
                nf, nr = nf + df, nr + dr
    elif t == "C":
        for df, dr in ORTHO:
            nf, nr = f + df, r + dr
            jumped = False
            while on_board(nf, nr):
                target = pos.grid[nr][nf]
                if not jumped:
                    if target is None:
                        out.append((sq, (nf, nr)))
                    else:
                        jumped = True
                elif target is not None:
                    if side_of(target) != side:
                        out.append((sq, (nf, nr)))
                    break
                nf, nr = nf + df, nr + dr
    elif t == "P":
        fwd = _forward(side)
        if on_board(f, r + fwd):
            add(f, r + fwd)
        if not own_half(sq, side):
            for df in (-1, 1):
                if on_board(f + df, r):
                    add(f + df, r)
    return out


def pseudo_moves(pos: Position, side: str) -> list[Move]:
    out: list[Move] = []
    for sq, p in pos.pieces():
        if side_of(p) == side:
            out.extend(pseudo_moves_from(pos, sq))
    return out


def king_square(pos: Position, side: str) -> Optional[Square]:
    k = "K" if side == RED else "k"
    found = pos.find(k)
    return found[0] if found else None


def kings_facing(pos: Position) -> bool:
    rk, bk = king_square(pos, RED), king_square(pos, BLACK)
    if rk is None or bk is None or rk[0] != bk[0]:
        return False
    f = rk[0]
    return all(pos.grid[r][f] is None for r in range(rk[1] + 1, bk[1]))


def is_attacked(pos: Position, sq: Square, by_side: str) -> bool:
    return any(to == sq for _, to in pseudo_moves(pos, by_side))


def in_check(pos: Position, side: str) -> bool:
    """Tướng của `side` có đang bị chiếu (kể cả lộ mặt tướng) không.

    Dò ngược từ ô tướng thay vì sinh toàn bộ nước của đối phương cho nhanh.
    """
    k = king_square(pos, side)
    if k is None:
        return True
    enemy = opponent(side)
    up = (lambda p: p.upper()) if enemy == RED else (lambda p: p.lower())
    kf, kr = k
    grid = pos.grid
    # Xe, Pháo, Tướng đối mặt
    for df, dr in ORTHO:
        f, r = kf + df, kr + dr
        screens = 0
        while on_board(f, r):
            p = grid[r][f]
            if p is not None:
                if screens == 0:
                    if p == up("R") or (df == 0 and p == up("K")):
                        return True
                    screens = 1
                else:
                    if p == up("C"):
                        return True
                    break
            f, r = f + df, r + dr
    # Mã (chân mã tính từ phía con mã)
    for (lf, lr), (df, dr) in KNIGHT:
        af, ar = kf - df, kr - dr
        if on_board(af, ar) and grid[ar][af] == up("N") and grid[ar + lr][af + lf] is None:
            return True
    # Tốt
    efwd = _forward(enemy)
    if on_board(kf, kr - efwd) and grid[kr - efwd][kf] == up("P"):
        return True
    for df in (-1, 1):
        if on_board(kf + df, kr) and grid[kr][kf + df] == up("P"):
            return True
    return False


def apply_move(pos: Position, move: Move) -> Position:
    nxt = pos.copy()
    nxt[move[1]] = nxt[move[0]]
    nxt[move[0]] = None
    nxt.turn = opponent(pos.turn)
    return nxt


def legal_moves(pos: Position, side: Optional[str] = None) -> list[Move]:
    side = side or pos.turn
    out = []
    for mv in pseudo_moves(pos, side):
        nxt = apply_move(pos, mv)
        if not in_check(nxt, side):
            out.append(mv)
    return out


def find_move_between(before: Position, after: Position) -> Optional[tuple[str, Move]]:
    """Tìm nước đi (của bên nào) biến `before` thành `after`.

    Dùng để tự đoán lượt đi khi người dùng chụp liên tục trong một ván.
    """
    for side in (before.turn, opponent(before.turn)):
        for mv in legal_moves(before, side):
            nxt = apply_move(before, mv)
            if nxt.grid == after.grid:
                return side, mv
    return None
