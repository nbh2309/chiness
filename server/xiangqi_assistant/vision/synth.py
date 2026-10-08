"""Vẽ ảnh bàn cờ giả (có phối cảnh, nhiễu) để kiểm thử pipeline khi chưa có ảnh thật."""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..board import FILES, RANKS, Position
from .glyphs import BLACK_GLYPH, RED_GLYPH, find_cjk_fonts, load_font

RED_INK = (190, 30, 30)
BLACK_INK = (25, 25, 25)


def draw_board(pos: Position, spacing: int = 80, margin: int = 70, rng: Optional[random.Random] = None,
               font_path: Optional[str] = None, rotate_glyphs: bool = True) -> np.ndarray:
    """Vẽ bàn cờ nhìn thẳng, Đỏ ở dưới. Trả về ảnh BGR."""
    rng = rng or random.Random(0)
    font_path = font_path or find_cjk_fonts()[0]
    w, h = 8 * spacing + 2 * margin, 9 * spacing + 2 * margin
    img = Image.new("RGB", (w, h), (214, 172, 112))
    d = ImageDraw.Draw(img)
    line = (60, 40, 20)
    x0, y0 = margin, margin

    def xy(f: int, row: int) -> tuple[int, int]:  # row 0 = trên cùng
        return x0 + f * spacing, y0 + row * spacing

    d.rectangle([x0 - 12, y0 - 12, x0 + 8 * spacing + 12, y0 + 9 * spacing + 12], outline=line, width=4)
    for row in range(RANKS):
        d.line([xy(0, row), xy(8, row)], fill=line, width=2)
    for f in range(FILES):
        if f in (0, 8):
            d.line([xy(f, 0), xy(f, 9)], fill=line, width=2)
        else:
            d.line([xy(f, 0), xy(f, 4)], fill=line, width=2)
            d.line([xy(f, 5), xy(f, 9)], fill=line, width=2)
    for r0 in (0, 7):
        d.line([xy(3, r0), xy(5, r0 + 2)], fill=line, width=2)
        d.line([xy(5, r0), xy(3, r0 + 2)], fill=line, width=2)

    radius = int(spacing * 0.44)
    font = load_font(font_path, int(radius * 1.15))
    for (f, rank), p in pos.pieces():
        cx, cy = xy(f, 9 - rank)
        cx += rng.randint(-3, 3)
        cy += rng.randint(-3, 3)
        ink = RED_INK if p.isupper() else BLACK_INK
        d.ellipse([cx - radius + 3, cy - radius + 4, cx + radius + 3, cy + radius + 4], fill=(120, 90, 50))
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(238, 214, 168),
                  outline=(150, 110, 60), width=2)
        ir = int(radius * 0.84)
        d.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], outline=ink, width=2)
        glyph = (RED_GLYPH if p.isupper() else BLACK_GLYPH)[p.upper()]
        tile = Image.new("RGBA", (radius * 2, radius * 2), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        l, t, r, b = td.textbbox((0, 0), glyph, font=font)
        td.text((radius - (r - l) / 2 - l, radius - (b - t) / 2 - t), glyph, fill=ink + (255,), font=font)
        angle = rng.uniform(0, 360) if rotate_glyphs else (180 if p.islower() else 0)
        tile = tile.rotate(angle, resample=Image.BICUBIC)
        img.paste(tile, (cx - radius, cy - radius), tile)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_photo(pos: Position, seed: int = 0, size: tuple[int, int] = (1280, 960),
                 view: int = 0, tilt: float = 0.25, font_path: Optional[str] = None,
                 noise: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Ảnh "chụp" bàn cờ: phối cảnh ngẫu nhiên, nền bàn, nhiễu.

    view: 0 = người cầm Đỏ chụp, 1 = xoay 90°, 2 = người cầm Đen chụp, 3 = xoay 270°.
    Trả về (ảnh BGR, 4 góc bàn cờ trong ảnh).
    """
    rng = random.Random(seed)
    board = draw_board(pos, rng=rng, font_path=font_path)
    bh, bw = board.shape[:2]
    W, H = size
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    src = np.roll(src, view, axis=0)  # đổi góc nào nằm ở trên-trái => xoay bàn cờ trong ảnh

    # Hình thang: cạnh xa (trên) hẹp hơn cạnh gần (dưới)
    scale = min(W, H) * rng.uniform(0.75, 0.88)
    cx, cy = W / 2 + rng.uniform(-40, 40), H / 2 + rng.uniform(-30, 30)
    top_half = scale / 2 * (1 - tilt * rng.uniform(0.6, 1.0))
    bot_half = scale / 2
    half_h = scale / 2 * (1 - tilt * 0.4)
    dst = np.float32([
        [cx - top_half, cy - half_h], [cx + top_half, cy - half_h],
        [cx + bot_half, cy + half_h], [cx - bot_half, cy + half_h],
    ])
    dst += np.float32([[rng.uniform(-15, 15), rng.uniform(-15, 15)] for _ in range(4)])
    ang = np.deg2rad(rng.uniform(-8, 8))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = ((dst - [cx, cy]) @ rot.T + [cx, cy]).astype(np.float32)

    bg = np.full((H, W, 3), (70, 95, 120), np.uint8)
    bg = cv2.add(bg, np.random.default_rng(seed).integers(0, 25, (H, W, 3), dtype=np.uint8))
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (W, H))
    mask = cv2.warpPerspective(np.full((bh, bw), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped, bg)

    # Ánh sáng không đều + nhiễu + mờ nhẹ
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 0.85 + 0.25 * (xx / W) * rng.uniform(0.5, 1.0)
    out = np.clip(out.astype(np.float32) * grad[..., None], 0, 255)
    out += np.random.default_rng(seed + 1).normal(0, noise, out.shape)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)
    return out, dst
