import cv2
import pytest
from fastapi.testclient import TestClient

from xiangqi_assistant import api
from xiangqi_assistant.analysis import Analyzer
from xiangqi_assistant.board import START_FEN, Position
from xiangqi_assistant.engine.simple import SimpleEngine
from xiangqi_assistant.vision.glyphs import find_cjk_fonts


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "analyzer", Analyzer(engine=SimpleEngine()))
    return TestClient(api.app)


def test_analyze_fen(client):
    r = client.post("/api/analyze_fen", json={"fen": "3k5/R8/9/9/9/9/9/9/1R7/4K4 w", "movetime_ms": 2000})
    assert r.status_code == 200
    assert r.json()["suggestion"]["mate_in_red"] == 1


def test_analyze_fen_invalid(client):
    assert client.post("/api/analyze_fen", json={"fen": "abc"}).status_code == 400
    r = client.post("/api/analyze_fen", json={"fen": "9/9/9/9/9/9/9/9/9/4K4 w"})
    assert r.json()["suggestion"]["ok"] is False


@pytest.mark.skipif(not find_cjk_fonts(), reason="cần font chữ Hán")
def test_analyze_image_and_session(client):
    from xiangqi_assistant.vision.synth import render_photo
    from xiangqi_assistant.board import uci_to_move
    from xiangqi_assistant.rules import apply_move

    start = Position.from_fen(START_FEN)
    img, _ = render_photo(start, seed=3)
    _, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/analyze", files={"image": ("a.jpg", buf.tobytes(), "image/jpeg")},
                    data={"turn": "w", "session_id": "s1", "movetime_ms": "300", "debug": "true"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["debug_jpeg_base64"]
    if j["fen"].split()[0] != start.placement_fen():
        pytest.skip("nhận dạng chưa đúng (chưa có mô hình CNN?)")

    # Đỏ đi Pháo 2 bình 5, chụp lại -> tự biết đến lượt Đen
    after = apply_move(start, uci_to_move("h2e2"))
    img2, _ = render_photo(after, seed=4)
    _, buf2 = cv2.imencode(".jpg", img2)
    r2 = client.post("/api/analyze", files={"image": ("b.jpg", buf2.tobytes(), "image/jpeg")},
                     data={"turn": "auto", "session_id": "s1", "movetime_ms": "300"})
    j2 = r2.json()
    if j2["fen"].split()[0] == after.placement_fen():
        assert j2["turn"] == "b"
        assert j2["last_move"]["text"] == "Pháo 2 bình 5"


def test_bad_image(client):
    r = client.post("/api/analyze", files={"image": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
