"""Chữ Hán trên quân cờ và cách vẽ chúng bằng font hệ thống."""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Mỗi loại quân có nhiều cách viết tuỳ bộ cờ (phồn thể / giản thể / Đỏ-Đen khác chữ).
GLYPHS: dict[str, list[str]] = {
    "K": ["帥", "帅", "將", "将"],
    "A": ["仕", "士"],
    "B": ["相", "象"],
    "N": ["傌", "馬", "马"],
    "R": ["俥", "車", "车"],
    "C": ["炮", "砲", "包"],
    "P": ["兵", "卒"],
}
# Chữ in trên bộ cờ phổ biến nhất (dùng khi vẽ bàn cờ giả)
RED_GLYPH = {"K": "帥", "A": "仕", "B": "相", "N": "傌", "R": "俥", "C": "炮", "P": "兵"}
BLACK_GLYPH = {"K": "將", "A": "士", "B": "象", "N": "馬", "R": "車", "C": "砲", "P": "卒"}

_FONT_PATTERNS = [
    "/usr/share/fonts/**/NotoSerifCJK*",
    "/usr/share/fonts/**/NotoSansCJK*",
    "/usr/share/fonts/**/wqy-zenhei*",
    "/usr/share/fonts/**/wqy-microhei*",
    "/usr/share/fonts/**/*[Kk]ai*",
    "/System/Library/Fonts/PingFang*",
    "/System/Library/Fonts/STHeiti*",
    "C:/Windows/Fonts/msyh*",
    "C:/Windows/Fonts/simsun*",
    "C:/Windows/Fonts/simkai*",
]


def find_cjk_fonts() -> list[str]:
    """Tìm các font có chữ Hán. Có thể chỉ định thêm qua biến XIANGQI_FONTS (ngăn bởi ':')."""
    found: list[str] = []
    env = os.environ.get("XIANGQI_FONTS")
    if env:
        found.extend(p for p in env.split(os.pathsep) if os.path.isfile(p))
    for pat in _FONT_PATTERNS:
        for p in sorted(glob.glob(pat, recursive=True)):
            if p not in found:
                found.append(p)
    return found


@lru_cache(maxsize=64)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_glyph(char: str, size: int = 64, font_path: Optional[str] = None) -> np.ndarray:
    """Vẽ một chữ thành mặt nạ uint8 (255 = nét chữ), kích thước size x size."""
    fonts = [font_path] if font_path else find_cjk_fonts()
    if not fonts:
        raise RuntimeError("Không tìm thấy font chữ Hán. Cài fonts-noto-cjk hoặc đặt XIANGQI_FONTS")
    font = load_font(fonts[0], int(size * 0.8))
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), char, font=font)
    draw.text(((size - (r - l)) / 2 - l, (size - (b - t)) / 2 - t), char, fill=255, font=font)
    return np.array(img)
