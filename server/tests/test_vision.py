import cv2
import numpy as np
import pytest

from xiangqi_assistant.board import START_FEN, Position
from xiangqi_assistant.vision.glyphs import find_cjk_fonts

pytestmark = pytest.mark.skipif(not find_cjk_fonts(), reason="cần font chữ Hán để vẽ ảnh giả")

POSITIONS = [
    START_FEN,
    "r1bakab1r/9/1cn4cn/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R b - - 0 1",
    "3k5/4a4/4ba3/p3p3p/2b3n2/6P2/P3c3P/4B4/4A4/2BAK4 w - - 0 1",
    "r2akab2/9/2n1b1n2/p1C1p3p/6p2/2p3P2/P3P3P/2N1C1N2/4A4/R1B1KAB2 w - - 0 1",
]


def _expected_grid(corners, view):
    bw, bh = 8 * 80 + 140, 9 * 80 + 140
    src = np.roll(np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]), view, axis=0)
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    pts = np.float32([[70 + f * 80, 70 + r * 80] for r in range(10) for f in range(9)])
    return cv2.perspectiveTransform(pts[None], M)[0].reshape(10, 9, 2)


@pytest.mark.parametrize("seed", range(8))
def test_grid_detection(seed):
    from xiangqi_assistant.vision.geometry import detect_grid
    from xiangqi_assistant.vision.synth import render_photo

    pos = Position.from_fen(POSITIONS[seed % len(POSITIONS)])
    img, corners = render_photo(pos, seed=seed, view=seed % 4)
    pts = detect_grid(img).grid_points_img()
    exp = _expected_grid(corners, seed % 4)
    err = min(np.linalg.norm(pts - exp, axis=2).max(), np.linalg.norm(pts[::-1, ::-1] - exp, axis=2).max())
    assert err < 4.0
    # không bị lật gương: hướng (cột -> hàng) cùng chiều với ảnh
    v1, v2 = pts[0, 1] - pts[0, 0], pts[1, 0] - pts[0, 0]
    assert v1[0] * v2[1] - v1[1] * v2[0] > 0


def test_piece_detection_and_color():
    from xiangqi_assistant.vision.geometry import detect_grid
    from xiangqi_assistant.vision.recognize import _detect
    from xiangqi_assistant.vision.synth import render_photo

    for i, fen in enumerate(POSITIONS):
        pos = Position.from_fen(fen)
        img, _ = render_photo(pos, seed=100 + i, view=i % 4)
        _, _, dets = _detect(img, None)
        assert len(dets) == len(list(pos.pieces()))
        assert sum(d.is_red for d in dets) == sum(p.isupper() for _, p in pos.pieces())


def test_full_recognition_with_cnn():
    from xiangqi_assistant.vision.cnn import default_model_path
    from xiangqi_assistant.vision.recognize import recognize
    from xiangqi_assistant.vision.synth import render_photo
    import os

    if not os.path.isfile(default_model_path()):
        pytest.skip("chưa có models/piece_cnn.onnx")
    ok = total = 0
    for i, fen in enumerate(POSITIONS):
        for view in range(4):
            pos = Position.from_fen(fen)
            img, _ = render_photo(pos, seed=500 + 10 * i + view, view=view)
            rec = recognize(img, turn=pos.turn)
            ok += rec.position.grid == pos.grid
            total += 1
    assert ok / total >= 0.9


def test_collect_crops(tmp_path):
    from xiangqi_assistant.vision.recognize import collect_crops
    from xiangqi_assistant.vision.synth import render_photo

    img, _ = render_photo(Position.from_fen(START_FEN), seed=7, view=2)
    res = collect_crops(img, directory=str(tmp_path))
    assert res["saved"] == 32
    assert len(list((tmp_path / "P").glob("*.png"))) == 10
    assert len(list((tmp_path / "K").glob("*.png"))) == 2
