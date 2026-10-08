"""Điều khiển Pikafish (hoặc engine cờ tướng UCI bất kỳ) qua stdin/stdout."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Optional

from ..board import Position
from . import EngineResult


class UciEngine:
    def __init__(self, path: str, eval_file: Optional[str] = None, threads: Optional[int] = None,
                 hash_mb: int = 128, startup_timeout: float = 10.0):
        self.path = path
        self.name = os.path.basename(path)
        self._lock = threading.Lock()
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._proc = subprocess.Popen(
            [path],
            cwd=os.path.dirname(os.path.abspath(path)),  # Pikafish tìm pikafish.nnue cạnh file chạy
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("uci")
        for line in self._wait_for("uciok", startup_timeout):
            if line.startswith("id name "):
                self.name = line[len("id name "):]
        if eval_file:
            self._send(f"setoption name EvalFile value {eval_file}")
        self._send(f"setoption name Threads value {threads or max(1, (os.cpu_count() or 2) - 1)}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._ready(startup_timeout)

    def _reader(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line.strip())
        self._lines.put("\x00eof")

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, prefix: str, timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        seen = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Engine không trả lời '{prefix}' sau {timeout}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line == "\x00eof":
                raise RuntimeError("Engine đã thoát")
            seen.append(line)
            if line.startswith(prefix):
                return seen

    def _ready(self, timeout: float = 10.0) -> None:
        self._send("isready")
        self._wait_for("readyok", timeout)

    def analyse(self, pos: Position, movetime_ms: int = 1000) -> EngineResult:
        with self._lock:
            self._ready()
            self._send(f"position fen {pos.fen()}")
            self._send(f"go movetime {int(movetime_ms)}")
            lines = self._wait_for("bestmove", movetime_ms / 1000 + 30)
        result = EngineResult(best_move=None, engine=self.name)
        for line in lines:
            if line.startswith("info") and " pv " in line:
                _parse_info(line, result)
        best = lines[-1].split()
        if len(best) >= 2 and best[1] not in ("(none)", "0000"):
            result.best_move = best[1]
        return result

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self._proc.kill()


def _parse_info(line: str, result: EngineResult) -> None:
    tok = line.split()
    i = 0
    while i < len(tok):
        t = tok[i]
        if t == "depth" and i + 1 < len(tok):
            result.depth = int(tok[i + 1])
            i += 2
        elif t == "score" and i + 2 < len(tok):
            if tok[i + 1] == "cp":
                result.score_cp, result.mate_in = int(tok[i + 2]), None
            elif tok[i + 1] == "mate":
                result.mate_in, result.score_cp = int(tok[i + 2]), None
            i += 3
        elif t == "pv":
            result.pv = tok[i + 1:]
            break
        else:
            i += 1
