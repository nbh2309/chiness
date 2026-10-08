"""Sinh ảnh một quân cờ tổng hợp (ảnh xám CROP x CROP) để huấn luyện bộ phân loại CNN.

Ảnh được cắt đúng như lúc nhận dạng: tâm = tâm quân dò được, cạnh = 2 x bán kính dò được.
Mọi thứ đều ngẫu nhiên hoá để mô hình chịu được khác biệt giữa các bộ cờ thật:
màu mặt quân, màu mực, vòng in trang trí, cỡ/độ đậm chữ, góc xoay, lệch tâm, sai số
bán kính (bộ dò đôi khi bắt vào vòng in), mờ, nhiễu, nén JPEG, ánh sáng.
"""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..board import PIECE_TYPES
from .glyphs import GLYPHS, find_cjk_fonts, load_font

CROP = 48          # kích thước ảnh đưa vào CNN
_HI = 160          # vẽ ở độ phân giải cao rồi thu nhỏ


def _rand_face(rng: random.Random) -> tuple[int, int, int]:
    base = rng.choice([(238, 214, 168), (245, 232, 200), (225, 190, 140), (250, 245, 235), (210, 170, 110)])
    return tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in base)


def _rand_ink(rng: random.Random, red: bool) -> tuple[int, int, int]:
    if red:
        return (rng.randint(150, 230), rng.randint(0, 60), rng.randint(0, 60))
    return rng.choice([(rng.randint(0, 50),) * 3, (0, rng.randint(40, 90), rng.randint(0, 50)),
                       (rng.randint(0, 40), rng.randint(0, 40), rng.randint(60, 110))])


def render_piece(piece_type: str, red: bool, rng: random.Random, fonts: Optional[list[str]] = None) -> np.ndarray:
    fonts = fonts or find_cjk_fonts()
    S = _HI
    Rc = S / 2                                   # bán kính "dò được" = nửa cạnh ảnh cắt
    Rt = Rc / rng.uniform(0.82, 1.12)            # bán kính quân thật
    bg = tuple(int(np.clip(c + rng.randint(-25, 25), 0, 255)) for c in (214, 172, 112))
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    cx = S / 2 + rng.uniform(-0.08, 0.08) * Rc   # sai số tâm dò được
    cy = S / 2 + rng.uniform(-0.08, 0.08) * Rc

    # đường kẻ bàn cờ phía sau quân
    lw = rng.randint(1, 4)
    if rng.random() < 0.9:
        d.line([(0, cy), (S, cy)], fill=(60, 40, 20), width=lw)
    if rng.random() < 0.9:
        d.line([(cx, 0), (cx, S)], fill=(60, 40, 20), width=lw)

    # thân quân (+ bóng / thành quân thấy do góc nhìn)
    sdx, sdy = rng.uniform(-0.12, 0.12) * Rt, rng.uniform(-0.12, 0.12) * Rt
    side = tuple(int(c * rng.uniform(0.45, 0.75)) for c in _rand_face(rng))
    d.ellipse([cx - Rt + sdx, cy - Rt + sdy, cx + Rt + sdx, cy + Rt + sdy], fill=side)
    face = _rand_face(rng)
    edge = tuple(int(c * rng.uniform(0.55, 0.85)) for c in face)
    d.ellipse([cx - Rt, cy - Rt, cx + Rt, cy + Rt], fill=face, outline=edge, width=rng.randint(1, 5))

    ink = _rand_ink(rng, red)
    # vòng in trang trí (0, 1 hoặc 2 vòng)
    n_rings = rng.choices([0, 1, 2], weights=[0.15, 0.65, 0.2])[0]
    rr = Rt * rng.uniform(0.74, 0.9)
    for _ in range(n_rings):
        w = rng.randint(1, 4)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=ink, width=w)
        rr -= rng.uniform(4, 9)

    # chữ
    char = rng.choice(GLYPHS[piece_type])
    font = load_font(rng.choice(fonts), int(Rt * rng.uniform(0.95, 1.35)))
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    stroke = rng.choice([0, 0, 1, 2, 3])
    l, t, r, b = td.textbbox((0, 0), char, font=font, stroke_width=stroke)
    td.text((S / 2 - (r - l) / 2 - l, S / 2 - (b - t) / 2 - t), char, font=font, fill=ink + (255,),
            stroke_width=stroke, stroke_fill=ink + (255,))
    tile = tile.rotate(rng.uniform(0, 360), resample=Image.BICUBIC)
    off = (int(round(cx - S / 2 + rng.uniform(-0.04, 0.04) * Rt)), int(round(cy - S / 2 + rng.uniform(-0.04, 0.04) * Rt)))
    img.paste(tile, off, tile)

    if rng.random() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 2.5)))
    arr = np.array(img).astype(np.float32)
    # ánh sáng: độ sáng/tương phản + dải sáng tối
    arr = arr * rng.uniform(0.6, 1.25) + rng.uniform(-30, 30)
    gy, gx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    arr *= (1 + rng.uniform(-0.25, 0.25) * (gx - 0.5) + rng.uniform(-0.25, 0.25) * (gy - 0.5))[..., None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # hơi méo phối cảnh còn sót sau khi nắn bàn cờ
    k = 0.06
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    dst = src + np.float32([[rng.uniform(-k, k) * S, rng.uniform(-k, k) * S] for _ in range(4)])
    arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (S, S), borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    small_side = rng.randint(18, 64)  # độ phân giải thật của quân trong ảnh chụp
    gray = cv2.resize(gray, (small_side, small_side), interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    gray = np.clip(gray + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 8), gray.shape),
                   0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
        gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return gray


def random_sample(rng: random.Random, fonts: Optional[list[str]] = None) -> tuple[np.ndarray, int]:
    label = rng.randrange(len(PIECE_TYPES))
    return render_piece(PIECE_TYPES[label], rng.random() < 0.5, rng, fonts), label
