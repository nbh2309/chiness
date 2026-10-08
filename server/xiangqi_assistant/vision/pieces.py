"""Phát hiện và phân loại quân cờ trên ảnh bàn cờ đã nắn thẳng (ảnh chuẩn).

  - Có quân hay không: tìm đường viền tròn quanh mỗi giao điểm (gradient hướng tâm).
  - Màu quân: tỉ lệ điểm mực màu đỏ trong lòng quân.
  - Loại quân: chính là CNN (vision/cnn.py). Ở đây chỉ có bộ so khớp chữ theo font
    làm phương án dự phòng khi chưa có mô hình (kém chính xác).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..board import PIECE_TYPES
from .geometry import MARGIN, SPACING
from .glyphs import GLYPHS, find_cjk_fonts, render_glyph

DESC = 32               # kích thước vector đặc trưng chữ (DESC x DESC)
ROT_STEP = 10           # bước xoay mẫu (độ)


@dataclass
class Detection:
    row: int                    # hàng trên ảnh chuẩn (0 = trên)
    col: int
    center: tuple[float, float]  # pixel ảnh chuẩn
    radius: float
    ring_score: float
    red_ratio: float = 0.0
    is_red: bool = False
    type_scores: dict[str, float] = field(default_factory=dict)
    ink_mask: Optional[np.ndarray] = None


def grid_center(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * SPACING, MARGIN + row * SPACING


# ------------------------------------------------------------------ phát hiện quân
def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    g = cv2.GaussianBlur(gray, (5, 5), 1.2).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag, max(float(np.percentile(mag, 80)), 1e-3)


def _circle_scores(grads, cx: float, cy: float, offs: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Tỉ lệ các hướng có biên hướng tâm trên vòng tròn, cho mọi (dx, dy, bán kính)."""
    gx, gy, mag, thr = grads
    H, W = mag.shape
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    band = np.array([-1, 0, 1])
    dx = offs[:, None, None, None, None]
    dy = offs[None, :, None, None, None]
    rad = radii[None, None, :, None, None] + band[None, None, None, None, :]
    px = np.clip(np.rint(cx + dx + rad * ct[None, None, None, :, None]), 0, W - 1).astype(int)
    py = np.clip(np.rint(cy + dy + rad * st[None, None, None, :, None]), 0, H - 1).astype(int)
    m = mag[py, px]
    radial = np.abs(gx[py, px] * ct[None, None, None, :, None] + gy[py, px] * st[None, None, None, :, None])
    ok = (m > thr) & (radial > 0.85 * m)
    return ok.any(-1).mean(-1)  # (offs, offs, radii)


def _ring_scores(grads) -> np.ndarray:
    """Với mỗi giao điểm: (điểm, dx, dy, bán kính) của vòng tròn khớp nhất."""
    offs = np.arange(-6, 7, 2)
    radii = np.arange(int(SPACING * 0.34), int(SPACING * 0.52) + 1, 2)
    out = np.zeros((10, 9, 4), np.float32)
    for r in range(10):
        for c in range(9):
            cx, cy = grid_center(r, c)
            score = _circle_scores(grads, cx, cy, offs, radii)
            best = score.max()
            # Nhiều quân có vòng trang trí bên trong: chọn vòng lớn nhất (viền quân) có điểm gần tốt nhất
            per_radius = score.reshape(-1, len(radii)).max(0)
            ri = int(np.nonzero(per_radius >= 0.85 * best)[0].max())
            i = np.unravel_index(np.argmax(score[..., ri]), score.shape[:2])
            out[r, c] = (best, offs[i[0]], offs[i[1]], radii[ri])
    return out


def detect_pieces(canon: np.ndarray, threshold: Optional[float] = None) -> list[Detection]:
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    grads = _gradients(gray)
    rs = _ring_scores(grads)
    scores = rs[..., 0].ravel()
    if threshold is None:
        # Ngưỡng tự động: chia 2 cụm, nhưng không thấp hơn 0.45 / cao hơn 0.7
        lo, hi = scores.min(), scores.max()
        t = (lo + hi) / 2
        for _ in range(20):
            a, b = scores[scores <= t], scores[scores > t]
            if len(a) == 0 or len(b) == 0:
                break
            t = (a.mean() + b.mean()) / 2
        threshold = float(np.clip(t, 0.45, 0.7))
    occupied = [(r, c) for r in range(10) for c in range(9) if rs[r, c, 0] >= threshold]
    if not occupied:
        return []

    # Mọi quân trong một bộ cờ cùng cỡ: lấy bán kính chung (trung vị) rồi tinh chỉnh từng tâm
    radius = float(np.median([rs[r, c, 3] for r, c in occupied]))
    radii = np.arange(radius - 1, radius + 1.01, 1.0)
    fine = np.arange(-3, 4, 1)
    dets = []
    for r, c in occupied:
        s, dx, dy, _ = rs[r, c]
        cx, cy = grid_center(r, c)
        cx, cy = cx + dx, cy + dy
        score = _circle_scores(grads, cx, cy, fine, radii).mean(-1)
        i = np.unravel_index(np.argmax(score), score.shape)
        dets.append(Detection(row=r, col=c, center=(cx + fine[i[0]], cy + fine[i[1]]), radius=radius,
                              ring_score=float(s)))
    return dets


# ------------------------------------------------------------------ vùng chữ
# Vùng chữ = GLYPH_FRAC x bán kính quân. Chữ trên quân thường nằm trong ~0.7R, còn vòng
# tròn in trang trí ở ~0.8-0.85R: vòng này giống nhau ở mọi loại quân, để lọt vào sẽ lấn
# át phần chữ khi so khớp. Bộ cờ có chữ to/vòng sát thì chỉnh qua biến XIANGQI_GLYPH_FRAC.
GLYPH_FRAC = float(os.environ.get("XIANGQI_GLYPH_FRAC", "0.72"))


def glyph_radius(dets: list[Detection]) -> float:
    """Bán kính vùng chữ, dùng chung cho cả bàn (mọi quân cùng cỡ)."""
    if not dets:
        return SPACING * 0.3
    return GLYPH_FRAC * float(np.median([d.radius for d in dets]))


def extract_ink(canon: np.ndarray, det: Detection, frame_r: float) -> tuple[np.ndarray, float]:
    """Mặt nạ mực mềm (ảnh vuông 2F+1, tâm = tâm quân, F = bán kính vùng chữ)
    và tỉ lệ mực màu đỏ."""
    F = int(round(frame_r))
    cx, cy = int(round(det.center[0])), int(round(det.center[1]))
    padded = cv2.copyMakeBorder(canon, F, F, F, F, cv2.BORDER_REPLICATE)
    patch = padded[cy:cy + 2 * F + 1, cx:cx + 2 * F + 1]
    yy, xx = np.mgrid[-F:F + 1, -F:F + 1]
    inner = np.hypot(xx, yy) <= F
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(gray[inner].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < t) & inner
    # Mặt nạ mềm: 0 = mặt quân, 1 = mực đậm (ít nhạy với độ dày nét do ảnh mờ)
    face_px = gray[inner & ~ink]
    face = float(np.median(face_px)) if face_px.size else 255.0
    dark = float(np.percentile(gray[inner], 3))
    soft = np.clip((face - gray.astype(np.float32)) / max(face - dark, 1.0), 0, 1)
    soft[~inner] = 0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    redish = ((h <= 10) | (h >= 160)) & (sat >= 90) & (val >= 50)
    n_ink = int(ink.sum())
    red_ratio = float((redish & ink).sum() / n_ink) if n_ink else 0.0
    return (soft * 255).astype(np.uint8), red_ratio


# ------------------------------------------------------------------ đặc trưng chữ
def descriptor(mask: np.ndarray, scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)
               ) -> Optional[np.ndarray]:
    """Vector đặc trưng của vùng chữ, thu về DESC x DESC. `scale`/`offset` để thử các
    giả thuyết lệch nhỏ của tâm/bán kính khi so khớp."""
    if np.count_nonzero(mask > 127) < 8:
        return None
    F = (mask.shape[0] - 1) / 2
    side = int(round(2 * F * scale)) + 1
    patch = cv2.getRectSubPix(mask.astype(np.float32) / 255.0, (side, side), (F + offset[0], F + offset[1]))
    v = cv2.resize(patch, (DESC, DESC), interpolation=cv2.INTER_AREA)
    v = cv2.GaussianBlur(v, (5, 5), 1.0).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


# Các giả thuyết (tỉ lệ, độ lệch tâm theo đơn vị bán kính vùng chữ) thử khi so khớp
_HYPOTHESES = [(s, (dx, dy)) for s in (0.92, 1.0, 1.08)
               for dx in (-0.06, 0.0, 0.06) for dy in (-0.06, 0.0, 0.06)]
# Cỡ chữ so với bán kính vùng chữ khi dùng mẫu từ font
GLYPH_SCALES = (1.1, 1.25, 1.4, 1.55)


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    F = (mask.shape[0] - 1) / 2
    out = []
    for ang in range(0, 360, ROT_STEP):
        M = cv2.getRotationMatrix2D((F, F), ang, 1.0)
        d = descriptor(cv2.warpAffine(mask, M, mask.shape[::-1], flags=cv2.INTER_LINEAR))
        if d is not None:
            out.append(d)
    return out


def font_template(char: str, font: str, scale: float, F: int = 50) -> np.ndarray:
    """Mặt nạ chữ vẽ từ font trong vùng chữ bán kính F, cỡ chữ = scale * F."""
    size = int(round(scale * F / 0.8))
    g = render_glyph(char, size, font)
    side = 2 * F + 1
    canvas = np.zeros((side, side), np.uint8)
    off = (side - size) // 2
    src0, dst0 = max(0, -off), max(0, off)
    n = min(size - src0, side - dst0)
    canvas[dst0:dst0 + n, dst0:dst0 + n] = g[src0:src0 + n, src0:src0 + n]
    yy, xx = np.mgrid[-F:F + 1, -F:F + 1]
    canvas[np.hypot(xx, yy) > F] = 0
    return canvas


class TemplateClassifier:
    """So khớp chữ trên quân với tập mẫu (mọi góc xoay)."""

    def __init__(self) -> None:
        self.vectors: list[np.ndarray] = []
        self.labels: list[tuple[Optional[bool], str]] = []  # (là quân đỏ? None = cả hai, loại)
        self._matrix: Optional[np.ndarray] = None
        self.source = ""

    def add(self, mask: np.ndarray, piece_type: str, is_red: Optional[bool] = None) -> None:
        for d in _rotations(mask):
            self.vectors.append(d)
            self.labels.append((is_red, piece_type))
        self._matrix = None

    @classmethod
    def from_fonts(cls, fonts: Optional[list[str]] = None) -> "TemplateClassifier":
        clf = cls()
        fonts = fonts or find_cjk_fonts()[:2]
        for font in fonts:
            for t, chars in GLYPHS.items():
                for ch in chars:
                    for scale in GLYPH_SCALES:
                        clf.add(font_template(ch, font, scale), t)
        clf.source = "font: " + ", ".join(os.path.basename(f) for f in fonts)
        return clf

    def scores(self, mask: np.ndarray, is_red: Optional[bool] = None) -> dict[str, float]:
        out = {t: 0.0 for t in PIECE_TYPES}
        if not self.vectors:
            return out
        R = (mask.shape[0] - 1) / 2
        ds = [d for s, (dx, dy) in _HYPOTHESES
              if (d := descriptor(mask, s, (dx * R, dy * R))) is not None]
        if not ds:
            return out
        if self._matrix is None:
            self._matrix = np.stack(self.vectors)
        sims = (self._matrix @ np.stack(ds).T).max(1)
        for sim, (red, t) in zip(sims, self.labels):
            if red is not None and is_red is not None and red != is_red:
                continue
            if sim > out[t]:
                out[t] = float(sim)
        return out
