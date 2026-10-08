"""Engine dự phòng viết bằng Python thuần: alpha-beta + đánh giá vật chất.

Chỉ đủ để thấy ăn quân / chiếu bí ngắn. Muốn gợi ý mạnh thật sự hãy cài Pikafish.
"""
from __future__ import annotations

import time

from ..board import RED, Position, move_to_uci, side_of
from ..rules import apply_move, in_check, legal_moves
from . import EngineResult

VALUE = {"K": 0, "R": 900, "C": 450, "N": 400, "B": 200, "A": 200, "P": 100}
MATE = 100000


class _Timeout(Exception):
    pass


def evaluate(pos: Position) -> int:
    """Điểm theo góc nhìn bên đang đi."""
    score = 0
    for (f, r), p in pos.pieces():
        t = p.upper()
        v = VALUE[t]
        if t == "P":
            rr = r if p.isupper() else 9 - r
            if rr >= 5:
                v += 100 - abs(f - 4) * 10 + (20 if rr < 9 else 0)
        elif t in ("N", "C", "R"):
            v += 10 - abs(f - 4) * 2
        score += v if side_of(p) == RED else -v
    return score if pos.turn == RED else -score


class SimpleEngine:
    name = "SimpleEngine (Python, yếu)"

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def analyse(self, pos: Position, movetime_ms: int = 1000) -> EngineResult:
        self._deadline = time.monotonic() + movetime_ms / 1000
        self._nodes = 0
        moves = self._ordered(pos, legal_moves(pos))
        if not moves:
            return EngineResult(best_move=None, mate_in=0, engine=self.name)
        best = EngineResult(best_move=move_to_uci(moves[0]), score_cp=0, engine=self.name)
        for depth in range(1, self.max_depth + 1):
            try:
                score, mv = self._root(pos, moves, depth)
            except _Timeout:
                break
            best = EngineResult(best_move=move_to_uci(mv), depth=depth, pv=[move_to_uci(mv)],
                                engine=self.name)
            if abs(score) >= MATE - 100:
                plies = MATE - abs(score)
                best.mate_in = (plies + 1) // 2 * (1 if score > 0 else -1)
                break
            best.score_cp = score
            moves.remove(mv)
            moves.insert(0, mv)
        return best

    def close(self) -> None:
        pass

    def _ordered(self, pos: Position, moves):
        def key(mv):
            victim = pos[mv[1]]
            return -(VALUE[victim.upper()] * 10 - VALUE[pos[mv[0]].upper()] // 10) if victim else 0
        return sorted(moves, key=key)

    def _root(self, pos, moves, depth):
        alpha, best_mv = -MATE - 1, moves[0]
        for mv in moves:
            score = -self._search(apply_move(pos, mv), depth - 1, -MATE - 1, -alpha, 1)
            if score > alpha:
                alpha, best_mv = score, mv
        return alpha, best_mv

    def _search(self, pos, depth, alpha, beta, ply):
        self._nodes += 1
        if self._nodes & 255 == 0 and time.monotonic() > self._deadline:
            raise _Timeout
        if depth <= 0:
            return self._quiesce(pos, alpha, beta, ply, 0)
        moves = legal_moves(pos)
        if not moves:
            return -(MATE - ply)  # bị chiếu bí hoặc hết nước đều thua
        for mv in self._ordered(pos, moves):
            score = -self._search(apply_move(pos, mv), depth - 1, -beta, -alpha, ply + 1)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _quiesce(self, pos, alpha, beta, ply, qdepth):
        if in_check(pos, pos.turn) and qdepth == 0:
            moves = legal_moves(pos)
            if not moves:
                return -(MATE - ply)
        stand = evaluate(pos)
        if stand >= beta or qdepth >= 4:
            return stand
        alpha = max(alpha, stand)
        captures = [mv for mv in legal_moves(pos) if pos[mv[1]] is not None]
        for mv in self._ordered(pos, captures):
            score = -self._quiesce(apply_move(pos, mv), -beta, -alpha, ply + 1, qdepth + 1)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha
