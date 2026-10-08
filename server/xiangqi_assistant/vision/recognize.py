"""Pipeline đầy đủ: ảnh chụp -> thế cờ (Position)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from ..board import MAX_COUNT, PIECE_TYPES, START_FEN, Position, allowed_square, square_name, validate
from .cnn import CnnClassifier, default_model_path, piece_crop
from .geometry import GridFit, detect_grid
from .pieces import Detection, TemplateClassifier, detect_pieces, extract_ink, glyph_radius

RED_RATIO_THRESHOLD = 0.3


def data_dir() -> str:
    return os.environ.get("XIANGQI_DATA") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def crops_dir() -> str:
    return os.path.join(data_dir(), "crops")


class PieceClassifier(Protocol):
    source: str
    low_conf: float  # dưới ngưỡng này coi là chưa chắc chắn

    def scores_for(self, canon: np.ndarray, dets: list[Detection]) -> list[dict[str, float]]: ...


class _Cnn:
    low_conf = 0.6

    def __init__(self, clf: CnnClassifier):
        self.clf = clf
        self.source = clf.source

    def scores_for(self, canon, dets):
        return self.clf.scores_for(canon, dets)


class _FontTemplates:
    """Dự phòng khi chưa có mô hình CNN: so khớp với chữ vẽ từ font (kém chính xác)."""
    low_conf = 0.5

    def __init__(self):
        self.clf = TemplateClassifier.from_fonts()
        self.source = self.clf.source

    def scores_for(self, canon, dets):
        return [self.clf.scores(d.ink_mask, d.is_red) for d in dets]


_cache: dict[str, PieceClassifier] = {}


def get_classifier() -> PieceClassifier:
    path = default_model_path()
    key = f"{path}:{os.path.getmtime(path)}" if os.path.isfile(path) else "fonts"
    if key not in _cache:
        _cache.clear()
        _cache[key] = _Cnn(CnnClassifier(path)) if key != "fonts" else _FontTemplates()
    return _cache[key]


@dataclass
class Recognition:
    position: Position
    grid: GridFit
    detections: list[Detection]
    red_bottom: bool                       # Đỏ ở phía dưới ảnh chuẩn
    confidence: dict[str, float] = field(default_factory=dict)  # ô -> độ tin cậy loại quân
    warnings: list[str] = field(default_factory=list)
    classifier: str = ""

    def square_of(self, det: Detection) -> tuple[int, int]:
        return _square(det, self.red_bottom)

    def debug_image(self, img: np.ndarray) -> np.ndarray:
        """Vẽ lưới và quân nhận dạng được lên ảnh gốc (vòng vàng = chưa chắc chắn)."""
        out = img.copy()
        pts = self.grid.grid_points_img()
        scale = max(1, int(round(max(img.shape[:2]) / 900)))
        for r in range(10):
            for c in range(9):
                x, y = pts[r, c]
                cv2.circle(out, (int(x), int(y)), 2 * scale, (0, 255, 0), -1)
        low = set(self.low_confidence_squares())
        for det in self.detections:
            sq = self.square_of(det)
            p = self.position[sq] or "?"
            x, y = pts[det.row, det.col]
            org = (int(x) - 8 * scale, int(y) + 8 * scale)
            color = (0, 0, 255) if p.isupper() else (255, 80, 0)
            cv2.putText(out, p, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), 4 * scale)
            cv2.putText(out, p, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color, 2 * scale)
            if square_name(sq) in low:
                cv2.circle(out, (int(x), int(y)), 18 * scale, (0, 255, 255), 2 * scale)
        cv2.polylines(out, [self.grid.quad.astype(np.int32)], True, (255, 0, 255), 2 * scale)
        return out

    def low_confidence_squares(self) -> list[str]:
        thr = get_classifier().low_conf
        return sorted(s for s, c in self.confidence.items() if c < thr)


def _square(det: Detection, red_bottom: bool) -> tuple[int, int]:
    return (det.col, 9 - det.row) if red_bottom else (8 - det.col, det.row)


def _resolve(dets: list[Detection], red_bottom: bool) -> tuple[list[str], list[float], float]:
    """Gán loại quân thoả luật (vị trí hợp lệ, số lượng tối đa), ưu tiên điểm cao.

    Ví dụ: quân đỏ nhận dạng là "Sĩ" nhưng đứng ngoài cung thì không thể là Sĩ ->
    lấy loại có điểm cao kế tiếp mà hợp lệ."""
    cands = []
    for i, d in enumerate(dets):
        sq = _square(d, red_bottom)
        for t in PIECE_TYPES:
            letter = t if d.is_red else t.lower()
            if allowed_square(letter, sq):
                cands.append((d.type_scores.get(t, 0.0), i, letter))
    cands.sort(reverse=True)
    assigned: list[Optional[str]] = [None] * len(dets)
    conf = [0.0] * len(dets)
    counts: dict[str, int] = {}
    total = 0.0
    for score, i, letter in cands:
        if assigned[i] is not None or counts.get(letter, 0) >= MAX_COUNT[letter.upper()]:
            continue
        assigned[i] = letter
        counts[letter] = counts.get(letter, 0) + 1
        conf[i] = score
        total += score
    for i, d in enumerate(dets):
        if assigned[i] is None:  # không gán hợp lệ được -> lấy loại điểm cao nhất, phạt điểm
            t = max(PIECE_TYPES, key=lambda k: d.type_scores.get(k, 0.0))
            assigned[i] = t if d.is_red else t.lower()
            total -= 1.0
    for k in ("K", "k"):
        if counts.get(k, 0) == 0:
            total -= 2.0
    return [a or "?" for a in assigned], conf, total


def _detect(img: np.ndarray, corners) -> tuple[GridFit, np.ndarray, list[Detection]]:
    grid = detect_grid(img, corners)
    canon = grid.warp_canonical(img)
    dets = detect_pieces(canon)
    frame_r = glyph_radius(dets)
    for d in dets:
        d.ink_mask, d.red_ratio = extract_ink(canon, d, frame_r)
        d.is_red = d.red_ratio > RED_RATIO_THRESHOLD
    return grid, canon, dets


def recognize(img: np.ndarray, turn: str = "w", corners: Optional[Sequence[Sequence[float]]] = None,
              classifier: Optional[PieceClassifier] = None) -> Recognition:
    clf = classifier or get_classifier()
    grid, canon, dets = _detect(img, corners)
    for d, scores in zip(dets, clf.scores_for(canon, dets)):
        d.type_scores = scores

    # Chưa biết Đỏ ở phía nào của ảnh: thử cả hai, chọn cách gán hợp luật hơn
    options = [(rb, *_resolve(dets, rb)) for rb in (True, False)]
    red_bottom, letters, conf, _ = max(options, key=lambda o: o[3])

    pos = Position(turn=turn)
    confidence = {}
    for d, letter, c in zip(dets, letters, conf):
        sq = _square(d, red_bottom)
        pos[sq] = letter
        confidence[square_name(sq)] = round(c, 3)

    rec = Recognition(position=pos, grid=grid, detections=dets, red_bottom=red_bottom,
                      confidence=confidence, classifier=clf.source)
    rec.warnings = validate(pos)
    low = [s for s, c in confidence.items() if c < clf.low_conf]
    if low:
        rec.warnings.append("Nhận dạng chưa chắc chắn ở: " + ", ".join(sorted(low)))
    if isinstance(clf, _FontTemplates):
        rec.warnings.append("Chưa có mô hình CNN (models/piece_cnn.onnx), đang so khớp chữ theo font: kém chính xác.")
    return rec


def collect_crops(img: np.ndarray, corners: Optional[Sequence[Sequence[float]]] = None,
                  directory: Optional[str] = None) -> dict:
    """Thu ảnh quân thật có nhãn từ một ảnh thế cờ ban đầu (32 quân, nhãn biết trước).

    Ảnh lưu vào <directory>/<loại>/*.png để tinh chỉnh CNN cho đúng bộ cờ của bạn:
        python tools/train_cnn.py --real data/crops --init models/piece_cnn.pt
    """
    grid, canon, dets = _detect(img, corners)
    if len(dets) != 32:
        raise ValueError(f"Ảnh phải là thế cờ ban đầu đủ 32 quân, phát hiện được {len(dets)} quân")
    reds = [d.row for d in dets if d.is_red]
    red_bottom = bool(reds) and float(np.mean(reds)) > 4.5
    start = Position.from_fen(START_FEN)
    problems = []
    for d in dets:
        sq = _square(d, red_bottom)
        exp = start[sq]
        if exp is None:
            problems.append(square_name(sq))
        elif exp.isupper() != d.is_red:
            problems.append(f"{square_name(sq)} (màu)")
    if problems:
        raise ValueError("Ảnh không khớp thế cờ ban đầu, kiểm tra các ô: " + ", ".join(problems))

    directory = directory or crops_dir()
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for d in dets:
        sq = _square(d, red_bottom)
        piece = start[sq]
        out = os.path.join(directory, piece.upper())
        os.makedirs(out, exist_ok=True)
        side = "r" if piece.isupper() else "b"
        cv2.imwrite(os.path.join(out, f"{stamp}_{side}_{square_name(sq)}.png"), piece_crop(gray, d))
    return {"saved": len(dets), "directory": directory}
