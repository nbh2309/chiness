"""Dòng lệnh.

  python -m xiangqi_assistant.cli image anh.jpg --turn w --debug ketqua.jpg
  python -m xiangqi_assistant.cli fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
  python -m xiangqi_assistant.cli calibrate anh_the_co_ban_dau.jpg
  python -m xiangqi_assistant.cli synth --fen "<FEN>" --out gia.jpg   # sinh ảnh giả để thử
"""
from __future__ import annotations

import argparse
import base64
import json
import sys

import cv2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="xiangqi_assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("image", help="phân tích ảnh chụp bàn cờ")
    p.add_argument("path")
    p.add_argument("--turn", default="w", choices=["w", "b", "auto"])
    p.add_argument("--movetime", type=int, default=1500)
    p.add_argument("--corners", help="JSON 4 góc bàn cờ nếu tự động sai")
    p.add_argument("--debug", help="lưu ảnh nhận dạng ra file này")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("fen", help="gợi ý nước đi cho một FEN")
    p.add_argument("fen")
    p.add_argument("--movetime", type=int, default=1500)

    p = sub.add_parser("calibrate", help="thu ảnh quân thật từ ảnh thế cờ ban đầu")
    p.add_argument("path")
    p.add_argument("--corners")

    p = sub.add_parser("synth", help="sinh ảnh bàn cờ giả")
    p.add_argument("--fen", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    from .analysis import Analyzer
    from .board import START_FEN, Position

    if args.cmd == "image":
        img = cv2.imread(args.path)
        if img is None:
            print(f"Không đọc được ảnh {args.path}", file=sys.stderr)
            return 2
        corners = json.loads(args.corners) if args.corners else None
        res = Analyzer().analyze_image(img, turn=args.turn, corners=corners, movetime_ms=args.movetime,
                                       debug=bool(args.debug))
        if args.debug and res.get("debug_jpeg_base64"):
            with open(args.debug, "wb") as fh:
                fh.write(base64.b64decode(res.pop("debug_jpeg_base64")))
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(res["board"])
            print("FEN:", res["fen"])
            for w in res["recognition"]["warnings"] + res["notes"]:
                print("!", w)
            s = res["suggestion"]
            print(s.get("display") if s.get("ok") else "Thế cờ không hợp lệ: " + "; ".join(s["errors"]))
        return 0
    if args.cmd == "fen":
        s = Analyzer().suggest(Position.from_fen(args.fen), args.movetime)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "calibrate":
        from .vision.recognize import collect_crops

        img = cv2.imread(args.path)
        print(collect_crops(img, corners=json.loads(args.corners) if args.corners else None))
        return 0
    if args.cmd == "synth":
        from .vision.synth import render_photo

        img, _ = render_photo(Position.from_fen(args.fen or START_FEN), seed=args.seed, view=args.view)
        cv2.imwrite(args.out, img)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
