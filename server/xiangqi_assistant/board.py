"""Biểu diễn thế cờ tướng và chuyển đổi FEN.

Quy ước toạ độ (giống Pikafish / UCI cờ tướng):
  - file (cột) 0..8 tương ứng chữ a..i, nhìn từ phía quân Đỏ, a ở bên trái.
  - rank (hàng) 0..9, hàng 0 là hàng cuối của Đỏ, hàng 9 là hàng cuối của Đen.
  - Quân Đỏ viết hoa, quân Đen viết thường:
      K/k Tướng, A/a Sĩ, B/b Tượng, N/n Mã, R/r Xe, C/c Pháo, P/p Tốt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

FILES = 9
RANKS = 10
RED = "w"    # Pikafish dùng 'w' cho bên Đỏ
BLACK = "b"

PIECE_TYPES = "KABNRCP"
MAX_COUNT = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}

# Một số phần mềm dùng E (elephant) / H (horse) thay cho B / N.
_ALIASES = {"E": "B", "H": "N", "e": "b", "h": "n"}

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

Square = tuple[int, int]  # (file, rank)


def is_red(piece: str) -> bool:
    return piece.isupper()


def side_of(piece: str) -> str:
    return RED if piece.isupper() else BLACK


def opponent(side: str) -> str:
    return BLACK if side == RED else RED


def square_name(sq: Square) -> str:
    return f"{chr(ord('a') + sq[0])}{sq[1]}"


def parse_square(name: str) -> Square:
    if len(name) != 2 or not ("a" <= name[0] <= "i") or not name[1].isdigit():
        raise ValueError(f"Ô không hợp lệ: {name!r}")
    return ord(name[0]) - ord("a"), int(name[1])


Move = tuple[Square, Square]


def move_to_uci(move: Move) -> str:
    return square_name(move[0]) + square_name(move[1])


def uci_to_move(text: str) -> Move:
    text = text.strip()
    if len(text) != 4:
        raise ValueError(f"Nước đi UCI không hợp lệ: {text!r}")
    return parse_square(text[:2]), parse_square(text[2:])


@dataclass
class Position:
    # grid[rank][file] -> ký tự quân hoặc None
    grid: list[list[Optional[str]]] = field(
        default_factory=lambda: [[None] * FILES for _ in range(RANKS)]
    )
    turn: str = RED

    # ------------------------------------------------------------------ FEN
    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        parts = fen.strip().split()
        if not parts:
            raise ValueError("FEN rỗng")
        rows = parts[0].split("/")
        if len(rows) != RANKS:
            raise ValueError(f"FEN phải có {RANKS} hàng, nhận được {len(rows)}")
        pos = cls()
        for i, row in enumerate(rows):
            rank = RANKS - 1 - i
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                    continue
                ch = _ALIASES.get(ch, ch)
                if ch.upper() not in PIECE_TYPES:
                    raise ValueError(f"Ký tự quân không hợp lệ trong FEN: {ch!r}")
                if f >= FILES:
                    raise ValueError(f"Hàng FEN quá dài: {row!r}")
                pos.grid[rank][f] = ch
                f += 1
            if f != FILES:
                raise ValueError(f"Hàng FEN phải đủ {FILES} cột: {row!r}")
        if len(parts) > 1:
            t = parts[1].lower()
            if t in ("w", "r"):
                pos.turn = RED
            elif t == "b":
                pos.turn = BLACK
            else:
                raise ValueError(f"Lượt đi không hợp lệ: {parts[1]!r}")
        return pos

    def placement_fen(self) -> str:
        rows = []
        for rank in range(RANKS - 1, -1, -1):
            row, empty = "", 0
            for f in range(FILES):
                p = self.grid[rank][f]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p
            if empty:
                row += str(empty)
            rows.append(row)
        return "/".join(rows)

    def fen(self) -> str:
        return f"{self.placement_fen()} {self.turn} - - 0 1"

    # -------------------------------------------------------------- helpers
    def copy(self) -> "Position":
        return Position([row[:] for row in self.grid], self.turn)

    def __getitem__(self, sq: Square) -> Optional[str]:
        return self.grid[sq[1]][sq[0]]

    def __setitem__(self, sq: Square, piece: Optional[str]) -> None:
        self.grid[sq[1]][sq[0]] = piece

    def pieces(self) -> Iterator[tuple[Square, str]]:
        for rank in range(RANKS):
            for f in range(FILES):
                p = self.grid[rank][f]
                if p is not None:
                    yield (f, rank), p

    def find(self, piece: str) -> list[Square]:
        return [sq for sq, p in self.pieces() if p == piece]

    def ascii(self) -> str:
        lines = []
        for rank in range(RANKS - 1, -1, -1):
            cells = " ".join(self.grid[rank][f] or "." for f in range(FILES))
            lines.append(f"{rank} {cells}")
            if rank == 5:
                lines.append("  ~~~~~ sông ~~~~~")
        lines.append("  a b c d e f g h i")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Position) and self.grid == other.grid and self.turn == other.turn


# ---------------------------------------------------------------- vị trí hợp lệ
def in_palace(sq: Square, side: str) -> bool:
    f, r = sq
    if not 3 <= f <= 5:
        return False
    return 0 <= r <= 2 if side == RED else 7 <= r <= 9


def own_half(sq: Square, side: str) -> bool:
    return sq[1] <= 4 if side == RED else sq[1] >= 5


def allowed_square(piece: str, sq: Square) -> bool:
    """Quân `piece` có thể đứng ở ô `sq` trong một ván cờ hợp lệ không."""
    side = side_of(piece)
    t = piece.upper()
    f, r = sq
    if t == "K":
        return in_palace(sq, side)
    if t == "A":
        if not in_palace(sq, side):
            return False
        rr = r if side == RED else 9 - r
        return (f, rr) in {(3, 0), (5, 0), (4, 1), (3, 2), (5, 2)}
    if t == "B":
        rr = r if side == RED else 9 - r
        return (f, rr) in {(2, 0), (6, 0), (0, 2), (4, 2), (8, 2), (2, 4), (6, 4)}
    if t == "P":
        rr = r if side == RED else 9 - r
        if rr < 3:
            return False
        if rr <= 4:
            return f % 2 == 0
        return True
    return True


def validate(pos: Position) -> list[str]:
    """Trả về danh sách lỗi (rỗng nếu thế cờ có vẻ hợp lệ)."""
    errors = []
    counts: dict[str, int] = {}
    for sq, p in pos.pieces():
        counts[p] = counts.get(p, 0) + 1
        if not allowed_square(p, sq):
            errors.append(f"Quân {p} không thể đứng ở {square_name(sq)}")
    for side_upper in (True, False):
        for t, mx in MAX_COUNT.items():
            p = t if side_upper else t.lower()
            n = counts.get(p, 0)
            if n > mx:
                errors.append(f"Có {n} quân {p}, tối đa {mx}")
        k = "K" if side_upper else "k"
        if counts.get(k, 0) != 1:
            errors.append(f"Phải có đúng 1 quân {k}, đang có {counts.get(k, 0)}")
    return errors
