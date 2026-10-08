"""Tìm bàn cờ trong ảnh và dựng lưới 9 x 10 giao điểm.

Các bước:
  1. Tìm các tứ giác lớn (mép bàn cờ / khung lưới) làm ứng viên.
  2. Nắn thẳng mỗi ứng viên về hình vuông, tách đường kẻ ngang/dọc bằng phép mở hình thái học.
  3. Khớp một lưới cách đều (10 đường x 9 đường) vào biểu đồ chiếu của đường kẻ.
  4. Xác định chiều bàn cờ nhờ "sông": giữa hàng 4 và 5 không có đường dọc bên trong.
  5. Trả về ma trận đồng nhất (homography) từ toạ độ lưới chuẩn sang ảnh gốc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

WARP = 800          # cạnh ảnh nắn tạm
SPACING = 64        # khoảng cách giao điểm trong ảnh chuẩn
MARGIN = 48         # lề ảnh chuẩn
CANON_W = 8 * SPACING + 2 * MARGIN
CANON_H = 9 * SPACING + 2 * MARGIN


class BoardNotFound(Exception):
    pass


@dataclass
class GridFit:
    # Homography: điểm (cột, hàng) theo đơn vị ô (0..8, 0..9, hàng 0 ở trên ảnh chuẩn) -> pixel ảnh gốc
    H_grid_to_img: np.ndarray
    score: float
    quad: np.ndarray
    river_ratio: float

    def grid_points_img(self) -> np.ndarray:
        cols, rows = np.meshgrid(np.arange(9), np.arange(10))
        pts = np.stack([cols.ravel(), rows.ravel()], 1).astype(np.float32)
        return cv2.perspectiveTransform(pts[None], self.H_grid_to_img)[0].reshape(10, 9, 2)

    def canon_to_img(self) -> np.ndarray:
        """Homography từ pixel ảnh chuẩn sang pixel ảnh gốc."""
        A = np.array([[1 / SPACING, 0, -MARGIN / SPACING], [0, 1 / SPACING, -MARGIN / SPACING], [0, 0, 1]])
        return self.H_grid_to_img @ A

    def warp_canonical(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.canon_to_img(), (CANON_W, CANON_H),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp 4 góc theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái (chiều kim đồng hồ)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # tăng dần góc = chiều kim đồng hồ trong hệ toạ độ ảnh
    start = np.argmin(pts.sum(1))
    return np.roll(pts, -start, axis=0)


def _quad_area(q: np.ndarray) -> float:
    return float(cv2.contourArea(q.astype(np.float32)))


def find_quads(img: np.ndarray, max_candidates: int = 6) -> list[np.ndarray]:
    """Các tứ giác lồi lớn trong ảnh, lớn trước."""
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    scale = min(scale, 1.0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    area_img = gray.shape[0] * gray.shape[1]

    edge_maps = []
    for lo, hi in ((30, 90), (60, 160)):
        e = cv2.Canny(gray, lo, hi)
        edge_maps.append(cv2.dilate(e, np.ones((3, 3), np.uint8)))
    # Phân vùng theo màu: bàn gỗ thường khác màu mặt bàn
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_maps += [otsu, 255 - otsu]

    quads: list[np.ndarray] = []
    for em in edge_maps:
        contours, _ = cv2.findContours(em, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 0.12 * area_img:
                continue
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            for eps in (0.01, 0.02, 0.04):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quads.append(order_corners(approx.reshape(4, 2) / scale))
                    break
    # Dự phòng: bàn cờ chiếm gần hết khung hình
    quads.append(order_corners(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])))

    # Loại trùng lặp
    uniq: list[np.ndarray] = []
    for q in sorted(quads, key=_quad_area, reverse=True):
        if all(np.abs(q - u).max() > 0.02 * max(h, w) for u in uniq):
            uniq.append(q)
    return uniq[:max_candidates]


def _line_masks(warped_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binv = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    k = WARP // 32
    hmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return hmask, vmask


def fit_lattice(profile: np.ndarray, n: int, smin: float, smax: float) -> tuple[float, float, float]:
    """Tìm (vị trí đầu, khoảng cách, điểm) của n đường cách đều khớp nhất với profile."""
    L = len(profile)
    prof = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1), (1, 7), 2).ravel()
    prof = prof / (prof.max() + 1e-6)
    best = (0.0, smin, -1.0)
    k = np.arange(n)
    for s in np.arange(smin, smax, 0.5):
        span = (n - 1) * s
        if span >= L - 1:
            break
        offs = np.arange(0, L - 1 - span, 1.0)
        idx = np.rint(offs[:, None] + k[None, :] * s).astype(int)
        sc = prof[idx].sum(1)
        i = int(np.argmax(sc))
        if sc[i] > best[2]:
            best = (float(offs[i]), float(s), float(sc[i]))
    return best[0], best[1], best[2] / n


def _river_ratio(col_mask: np.ndarray, row_pos: np.ndarray, col_pos: np.ndarray, s_row: float) -> float:
    """Mật độ đường cột trong dải sông so với các dải khác (gần 0 = đúng chiều)."""
    def band_density(r0: int) -> float:
        y0 = int(row_pos[r0] + 0.42 * s_row)
        y1 = int(row_pos[r0] + 0.58 * s_row) + 1
        vals = [col_mask[y0:y1, max(0, int(x) - 3):int(x) + 4].mean() for x in col_pos[1:-1]]
        return float(np.mean(vals)) if vals else 0.0

    river = band_density(4)
    others = np.mean([band_density(r) for r in (0, 1, 2, 3, 5, 6, 7, 8)])
    return float(river / (others + 1e-6))


def fit_grid(img: np.ndarray, quad: np.ndarray) -> Optional[GridFit]:
    quad = order_corners(quad)
    dst = np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]])
    H_q = cv2.getPerspectiveTransform(quad, dst)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    warped = cv2.warpPerspective(gray, H_q, (WARP, WARP))
    hmask, vmask = _line_masks(warped)
    prof_y = hmask.sum(1)  # đường ngang -> đỉnh theo y
    prof_x = vmask.sum(0)  # đường dọc -> đỉnh theo x

    results = []
    # Chiều A: 10 hàng theo trục y, 9 cột theo trục x
    oy, sy, qy = fit_lattice(prof_y, 10, 0.55 * WARP / 9, WARP / 9)
    ox, sx, qx = fit_lattice(prof_x, 9, 0.55 * WARP / 8, WARP / 8)
    rows, cols = oy + np.arange(10) * sy, ox + np.arange(9) * sx
    ratio = _river_ratio(vmask, rows, cols, sy)
    # (cột c, hàng r) -> (x, y) trong ảnh nắn tạm
    A = np.array([[sx, 0, ox], [0, sy, oy], [0, 0, 1]], np.float64)
    results.append(((qx + qy) / 2, ratio, A))

    # Chiều B: 10 hàng theo trục x, 9 cột theo trục y (bàn cờ nằm ngang trong ảnh)
    ox2, sx2, qx2 = fit_lattice(prof_x, 10, 0.55 * WARP / 9, WARP / 9)
    oy2, sy2, qy2 = fit_lattice(prof_y, 9, 0.55 * WARP / 8, WARP / 8)
    rows2, cols2 = ox2 + np.arange(10) * sx2, oy2 + np.arange(9) * sy2
    ratio2 = _river_ratio(hmask.T, rows2, cols2, sx2)
    # cột c -> lên trên (y giảm), hàng r -> sang phải: phép quay, không lật gương
    B = np.array([[0, sx2, ox2], [-sy2, 0, oy2 + 8 * sy2], [0, 0, 1]], np.float64)
    results.append(((qx2 + qy2) / 2, ratio2, B))

    best = max(results, key=lambda t: t[0] * (1 - min(t[1], 1.0)))
    quality, ratio, M = best
    score = quality * (1 - min(ratio, 1.0))
    H = np.linalg.inv(H_q) @ M
    return GridFit(H_grid_to_img=H / H[2, 2], score=float(score), quad=quad, river_ratio=float(ratio))


def refine_grid(img: np.ndarray, fit: GridFit, windows: Sequence[float] = (0.45, 0.3, 0.2)) -> GridFit:
    """Tinh chỉnh lưới: tìm vị trí thật của đường kẻ tại trung điểm mỗi đoạn giữa hai giao điểm
    (chỗ không bị quân che), rồi khớp lại homography bằng RANSAC. Sửa được trường hợp
    tứ giác ban đầu lệch góc hoặc ảnh hơi méo."""
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    S, half = SPACING, int(SPACING * 0.1)
    k = S // 2
    for win in windows:
        canon = fit.warp_canonical(gray_full)
        binv = cv2.adaptiveThreshold(canon, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)
        hmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2, 1)))
        vmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k // 2)))
        w = int(S * win)
        src, dst = [], []
        for r in range(10):          # đường ngang: trung điểm giữa cột c và c+1
            for c in range(8):
                x = int(MARGIN + (c + 0.5) * S)
                y0 = MARGIN + r * S
                lo, hi = max(0, y0 - w), min(CANON_H, y0 + w + 1)
                prof = hmask[lo:hi, x - half:x + half + 1].mean(1)
                if prof.size and prof.max() > 127:
                    src.append((c + 0.5, r))
                    dst.append((x, lo + float(np.argmax(prof))))
        for c in range(9):           # đường dọc: trung điểm giữa hàng r và r+1 (bỏ qua sông)
            for r in range(9):
                if r == 4 and 0 < c < 8:
                    continue
                y = int(MARGIN + (r + 0.5) * S)
                x0 = MARGIN + c * S
                lo, hi = max(0, x0 - w), min(CANON_W, x0 + w + 1)
                prof = vmask[y - half:y + half + 1, lo:hi].mean(0)
                if prof.size and prof.max() > 127:
                    src.append((c, r + 0.5))
                    dst.append((lo + float(np.argmax(prof)), y))
        if len(src) < 20:
            break
        Hc, inl = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 3.0)
        if Hc is None or inl.sum() < 20:
            break
        H = fit.canon_to_img() @ Hc
        fit = GridFit(H_grid_to_img=H / H[2, 2], score=fit.score, quad=fit.quad, river_ratio=fit.river_ratio)
    return fit


def detect_grid(img: np.ndarray, corners: Optional[Sequence[Sequence[float]]] = None) -> GridFit:
    """Tìm lưới bàn cờ. `corners`: 4 góc người dùng chỉ định (nếu tự động sai)."""
    quads = [order_corners(np.float32(corners))] if corners is not None else find_quads(img)
    fits = [refine_grid(img, f) for f in (fit_grid(img, q) for q in quads) if f is not None]
    if not fits:
        raise BoardNotFound("Không tìm thấy bàn cờ trong ảnh")
    best = max(fits, key=lambda f: f.score)
    if best.score < 0.25:
        raise BoardNotFound(
            f"Không chắc đã tìm đúng bàn cờ (điểm {best.score:.2f}). "
            "Hãy chụp thẳng hơn, đủ sáng, thấy rõ cả 4 góc bàn cờ, hoặc chỉ định 4 góc thủ công."
        )
    return best
