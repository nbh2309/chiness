"""Bộ phân loại loại quân bằng CNN (file ONNX, chạy bằng OpenCV DNN, không cần PyTorch).

Huấn luyện: tools/train_cnn.py (ảnh tổng hợp + ảnh quân thật thu từ ảnh hiệu chuẩn).
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from ..board import PIECE_TYPES
from .pieces import Detection
from .piece_render import CROP


def default_model_path() -> str:
    env = os.environ.get("XIANGQI_MODEL")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "models", "piece_cnn.onnx")


def piece_crop(canon_gray: np.ndarray, det: Detection) -> np.ndarray:
    """Cắt quân: tâm = tâm dò được, cạnh = 2 x bán kính, thu về CROP x CROP (giống lúc huấn luyện)."""
    side = int(round(2 * det.radius))
    patch = cv2.getRectSubPix(canon_gray, (side, side), (float(det.center[0]), float(det.center[1])))
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def normalize(batch: np.ndarray) -> np.ndarray:
    """Chuẩn hoá từng ảnh về trung bình 0, độ lệch 1 (bớt phụ thuộc ánh sáng). Dùng chung khi huấn luyện."""
    x = batch.astype(np.float32)
    m = x.mean(axis=(-1, -2), keepdims=True)
    s = x.std(axis=(-1, -2), keepdims=True) + 1e-3
    return (x - m) / s


class CnnClassifier:
    def __init__(self, path: str):
        self.path = path
        self.net = cv2.dnn.readNetFromONNX(path)
        self.source = f"CNN: {os.path.basename(path)}"

    @classmethod
    def load_default(cls) -> Optional["CnnClassifier"]:
        path = default_model_path()
        return cls(path) if os.path.isfile(path) else None

    def predict(self, crops: list[np.ndarray], tta: bool = True) -> np.ndarray:
        """Xác suất (N, 7) theo thứ tự PIECE_TYPES. tta: lấy trung bình trên 4 góc xoay 90°."""
        if not crops:
            return np.zeros((0, len(PIECE_TYPES)), np.float32)
        views = [np.stack(crops)]
        if tta:
            views += [np.stack([np.rot90(c, k).copy() for c in crops]) for k in (1, 2, 3)]
        probs = []
        for v in views:
            blob = normalize(v)[:, None, :, :]
            self.net.setInput(blob)
            logits = self.net.forward()
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs.append(e / e.sum(1, keepdims=True))
        return np.mean(probs, axis=0)

    def scores_for(self, canon: np.ndarray, dets: list[Detection]) -> list[dict[str, float]]:
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY) if canon.ndim == 3 else canon
        probs = self.predict([piece_crop(gray, d) for d in dets])
        return [{t: float(p[i]) for i, t in enumerate(PIECE_TYPES)} for p in probs]
