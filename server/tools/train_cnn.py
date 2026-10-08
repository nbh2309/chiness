"""Huấn luyện CNN phân loại loại quân cờ, xuất ra models/piece_cnn.onnx.

Dữ liệu:
  * Ảnh tổng hợp sinh ngẫu nhiên từ font chữ Hán có trong máy (luôn dùng).
  * Ảnh quân thật (khuyến khích): thư mục có cấu trúc <thư mục>/<loại K|A|B|N|R|C|P>/*.png,
    tạo tự động bằng API /api/calibrate hoặc tools/collect_crops.py từ ảnh thế cờ ban đầu.

Ví dụ:
  python tools/train_cnn.py --epochs 12
  python tools/train_cnn.py --real data/crops --epochs 8 --init models/piece_cnn.pt

Cần: pip install torch  (chỉ để huấn luyện; lúc chạy chỉ cần OpenCV)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xiangqi_assistant.board import PIECE_TYPES  # noqa: E402
from xiangqi_assistant.vision.cnn import normalize  # noqa: E402
from xiangqi_assistant.vision.glyphs import find_cjk_fonts  # noqa: E402
from xiangqi_assistant.vision.piece_render import CROP, random_sample  # noqa: E402


class PieceNet(nn.Module):
    def __init__(self, n_classes: int = len(PIECE_TYPES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 192))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(192 * 3 * 3, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class SynthStream(IterableDataset):
    def __init__(self, n: int, fonts: list[str], seed: int):
        self.n, self.fonts, self.seed = n, fonts, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + wid + random.randrange(1 << 20))
        for _ in range(self.n // nw):
            img, label = random_sample(rng, self.fonts)
            yield normalize(img)[None], label


def augment_real(img: np.ndarray, rng: random.Random) -> np.ndarray:
    S = img.shape[0]
    M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(0, 360), rng.uniform(0.9, 1.1))
    M[:, 2] += [rng.uniform(-0.06, 0.06) * S, rng.uniform(-0.06, 0.06) * S]
    out = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5))
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
    return cv2.resize(out.astype(np.uint8), (CROP, CROP), interpolation=cv2.INTER_AREA)


class RealCrops(Dataset):
    def __init__(self, root: str, repeat: int):
        self.items = []
        for i, t in enumerate(PIECE_TYPES):
            for f in glob.glob(os.path.join(root, t, "*.png")):
                self.items.append((f, i))
        self.repeat = repeat
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        f, label = self.items[idx % len(self.items)]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        return normalize(augment_real(img, self.rng))[None], label


def evaluate(model, fonts, n=3000, seed=12345):
    model.eval()
    rng = random.Random(seed)
    xs, ys = zip(*[random_sample(rng, fonts) for _ in range(n)])
    x = torch.from_numpy(normalize(np.stack(xs))[:, None])
    with torch.no_grad():
        pred = model(x).argmax(1).numpy()
    model.train()
    return float((pred == np.array(ys)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-epoch", type=int, default=40000, help="số ảnh tổng hợp mỗi epoch")
    ap.add_argument("--real", help="thư mục ảnh quân thật <loại>/*.png")
    ap.add_argument("--real-repeat", type=int, default=200, help="mỗi ảnh thật được tăng cường bao nhiêu lần/epoch")
    ap.add_argument("--init", help="tiếp tục từ trọng số .pt có sẵn")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "models", "piece_cnn"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    fonts = find_cjk_fonts()
    if not fonts:
        sys.exit("Không có font chữ Hán. Cài fonts-noto-cjk / fonts-arphic-ukai hoặc đặt XIANGQI_FONTS")
    print("Font dùng để sinh dữ liệu:", *fonts, sep="\n  ")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = PieceNet()
    if args.init and os.path.isfile(args.init):
        model.load_state_dict(torch.load(args.init))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    real = RealCrops(args.real, args.real_repeat) if args.real else None
    if real is not None:
        print(f"Ảnh quân thật: {len(real.items)}")
    steps_per_epoch = args.per_epoch // 256 + (len(real) // 256 if real else 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch + 10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(args.epochs):
        loaders = [DataLoader(SynthStream(args.per_epoch, fonts, epoch), batch_size=256, num_workers=args.workers)]
        if real is not None:
            loaders.append(DataLoader(real, batch_size=256, shuffle=True, num_workers=args.workers))
        total, correct, n = 0.0, 0, 0
        for loader in loaders:
            for x, y in loader:
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                total += loss.item() * len(y)
                correct += int((logits.argmax(1) == y).sum())
                n += len(y)
        acc = evaluate(model, fonts)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / n:.3f}  train acc {correct / n:.3f}  val acc {acc:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out + ".pt")
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 1, CROP, CROP), args.out + ".onnx", input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                      opset_version=13, dynamo=False)
    print("Đã lưu", args.out + ".onnx")


if __name__ == "__main__":
    main()
