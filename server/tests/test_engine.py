import os
import sys
import textwrap

from xiangqi_assistant.analysis import Analyzer
from xiangqi_assistant.board import Position, uci_to_move
from xiangqi_assistant.engine.simple import SimpleEngine
from xiangqi_assistant.engine.uci import UciEngine
from xiangqi_assistant.rules import apply_move, legal_moves

MATE_IN_1 = "3k5/R8/9/9/9/9/9/9/1R7/4K4 w - - 0 1"


def test_simple_engine_finds_mate():
    pos = Position.from_fen(MATE_IN_1)
    res = SimpleEngine().analyse(pos, 3000)
    assert res.mate_in == 1
    assert legal_moves(apply_move(pos, uci_to_move(res.best_move))) == []


def test_analyzer_suggestion_text():
    s = Analyzer(engine=SimpleEngine()).suggest(Position.from_fen(MATE_IN_1), 3000)
    assert s["ok"] and s["move_text"] in ("Xe 8 tấn 8", "Xe 8 bình 6")  # cả hai đều chiếu bí
    assert "chiếu bí" in s["display"]


def test_uci_wrapper_with_fake_engine(tmp_path):
    script = tmp_path / "fake_engine.py"
    script.write_text(textwrap.dedent("""
        import sys
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "uci":
                print("id name FakeFish"); print("uciok")
            elif cmd == "isready":
                print("readyok")
            elif cmd.startswith("go"):
                print("info depth 12 score cp 35 nodes 1000 pv h2e2 h9g7")
                print("bestmove h2e2 ponder h9g7")
            elif cmd == "quit":
                break
            sys.stdout.flush()
    """))
    launcher = tmp_path / "engine.sh"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    os.chmod(launcher, 0o755)
    eng = UciEngine(str(launcher))
    try:
        res = eng.analyse(Position.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"), 100)
    finally:
        eng.close()
    assert eng.name == "FakeFish"
    assert (res.best_move, res.score_cp, res.depth, res.pv) == ("h2e2", 35, 12, ["h2e2", "h9g7"])
