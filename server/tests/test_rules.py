import pytest

from xiangqi_assistant.board import START_FEN, Position, uci_to_move, validate
from xiangqi_assistant.notation import describe_uci
from xiangqi_assistant.rules import apply_move, find_move_between, in_check, legal_moves


def perft(pos, depth):
    if depth == 0:
        return 1
    return sum(perft(apply_move(pos, m), depth - 1) for m in legal_moves(pos))


def test_perft_start():
    pos = Position.from_fen(START_FEN)
    assert [perft(pos, d) for d in (1, 2, 3)] == [44, 1920, 79666]


def test_perft_midgame():
    pos = Position.from_fen("r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 0 1")
    assert [perft(pos, d) for d in (1, 2)] == [38, 1128]


def test_fen_roundtrip():
    for fen in (START_FEN, "3k5/R8/9/9/9/9/9/9/1R7/4K4 b - - 0 1"):
        assert Position.from_fen(fen).fen() == fen


@pytest.mark.parametrize("fen,uci,text,short", [
    (START_FEN, "h2e2", "Pháo 2 bình 5", "P2-5"),
    (START_FEN, "h0g2", "Mã 2 tấn 3", "M2.3"),
    (START_FEN, "c0e2", "Tượng 7 tấn 5", "T7.5"),
    (START_FEN, "f0e1", "Sĩ 4 tấn 5", "S4.5"),
    (START_FEN, "g3g4", "Tốt 3 tấn 1", "B3.1"),
    (START_FEN.replace(" w ", " b "), "h7e7", "Pháo 8 bình 5", "P8-5"),
    (START_FEN.replace(" w ", " b "), "b9c7", "Mã 2 tấn 3", "M2.3"),
    ("4k4/9/9/9/9/4R4/9/4R4/9/3K5 w - - 0 1", "e4e7", "Xe trước tấn 3", "Xt.3"),
    ("4k4/9/9/9/9/4R4/9/4R4/9/3K5 w - - 0 1", "e2e1", "Xe sau thoái 1", "Xs/1"),
])
def test_notation(fen, uci, text, short):
    d = describe_uci(Position.from_fen(fen), uci)
    assert (d["text"], d["short"]) == (text, short)


def test_flying_general_is_illegal():
    pos = Position.from_fen("4k4/9/9/9/9/9/9/9/4A4/3K5 w - - 0 1")
    # Sĩ e1 không được rời cột e vì sẽ làm hai tướng đối mặt? Không: tướng đỏ ở d0, nên được.
    assert any(m == uci_to_move("e1d2") for m in legal_moves(pos))
    pos = Position.from_fen("4k4/9/9/9/9/9/9/9/4A4/4K4 w - - 0 1")
    assert uci_to_move("e1d2") not in legal_moves(pos)


def test_check_detection():
    assert in_check(Position.from_fen("4k4/9/9/9/9/9/9/9/9/3K1R3 b - - 0 1"), "b") is False
    assert in_check(Position.from_fen("4k4/9/9/9/9/9/9/9/9/3K1R3 b - - 0 1"), "b") is False
    assert in_check(Position.from_fen("4k4/9/9/9/9/4R4/9/9/9/3K5 b - - 0 1"), "b") is True
    # Pháo cần đúng một ngòi
    assert in_check(Position.from_fen("4k4/9/9/9/4p4/9/9/4C4/9/3K5 b - - 0 1"), "b") is True
    assert in_check(Position.from_fen("4k4/9/9/9/9/9/9/4C4/9/3K5 b - - 0 1"), "b") is False
    # Mã bị cản chân
    assert in_check(Position.from_fen("4k4/9/3N5/9/9/9/9/9/9/3K5 b - - 0 1"), "b") is True
    assert in_check(Position.from_fen("4k4/3p5/3N5/9/9/9/9/9/9/3K5 b - - 0 1"), "b") is False


def test_validate():
    assert validate(Position.from_fen(START_FEN)) == []
    errs = validate(Position.from_fen("4k4/9/9/9/9/9/9/9/9/A3K4 w - - 0 1"))
    assert any("a0" in e for e in errs)
    assert validate(Position.from_fen("9/9/9/9/9/9/9/9/9/4K4 w - - 0 1"))


def test_find_move_between():
    before = Position.from_fen(START_FEN)
    after = apply_move(before, uci_to_move("h2e2"))
    side, mv = find_move_between(before, after)
    assert side == "w" and mv == uci_to_move("h2e2")
