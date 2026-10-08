# Trợ lý cờ tướng cho kính AR (Rokid)

Phần mềm này nhận ảnh chụp bàn cờ tướng thật (từ kính AR hoặc điện thoại), nhận dạng thế cờ, rồi gợi ý nước đi tiếp theo bằng ký hiệu tiếng Việt, ví dụ **"Đỏ: Pháo 2 bình 5 | +0.3"**. Gợi ý được hiện lên kính và đọc thành tiếng.

> ⚠️ Chỉ dùng để luyện tập và phân tích. Dùng khi thi đấu hay chơi với người khác là gian lận.

## Kiến trúc

```
[Kính / điện thoại]  -- ảnh JPEG -->  [Server Python]
  android/                              1. Tìm bàn cờ, nắn thẳng, dựng lưới 9x10
  - chụp bằng camera                    2. Phát hiện quân (viền tròn), màu quân (mực đỏ)
  - hiện chữ + đọc giọng nói            3. Phân loại quân bằng CNN (ONNX)
                                        4. Ráp thế cờ theo luật (Sĩ trong cung...), ra FEN
                     <-- gợi ý --       5. Engine Pikafish tính nước tốt nhất
                                        6. Đổi sang ký hiệu "Pháo 2 bình 5"
```

| Thư mục | Nội dung |
|---|---|
| `server/xiangqi_assistant/board.py`, `rules.py` | Bàn cờ, FEN, sinh nước đi hợp lệ (đã kiểm bằng perft) |
| `server/xiangqi_assistant/notation.py` | Ký hiệu tiếng Việt: tấn/thoái/bình, trước/sau |
| `server/xiangqi_assistant/engine/` | Gọi Pikafish qua UCI; engine Python dự phòng (yếu) |
| `server/xiangqi_assistant/vision/` | Nhận dạng ảnh: `geometry.py` (lưới), `pieces.py` (phát hiện quân, màu), `cnn.py` (loại quân), `synth.py` / `piece_render.py` (sinh ảnh giả) |
| `server/xiangqi_assistant/analysis.py` | Ghép mọi bước lại, tự đoán lượt đi khi theo dõi cả ván |
| `server/xiangqi_assistant/api.py` | HTTP API (FastAPI) + trang web thử ở `/` |
| `server/tools/train_cnn.py` | Huấn luyện / tinh chỉnh mô hình nhận dạng quân |
| `server/models/piece_cnn.onnx` | Mô hình đã huấn luyện sẵn (chỉ trên ảnh tổng hợp) |
| `android/` | App Android (Kotlin, CameraX) cho kính / điện thoại |

## Cài đặt và chạy server

```bash
cd server
pip install -r requirements.txt
./scripts/install_pikafish.sh        # tải engine Pikafish vào server/engines/ (cần p7zip-full)
uvicorn xiangqi_assistant.api:app --host 0.0.0.0 --port 8000
```

- Mở `http://<ip-máy-tính>:8000` trên điện thoại để chụp và thử ngay, chưa cần kính.
- Chưa cài Pikafish thì server dùng engine Python dự phòng: chạy được nhưng rất yếu.
- Biến môi trường: `PIKAFISH_PATH` (đường dẫn engine), `XIANGQI_MODEL` (file .onnx), `XIANGQI_DATA` (thư mục dữ liệu), `XIANGQI_FONTS` (font chữ Hán dùng khi huấn luyện).

Dòng lệnh:

```bash
python -m xiangqi_assistant.cli image anh.jpg --turn w --debug ketqua.jpg
python -m xiangqi_assistant.cli fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
python -m xiangqi_assistant.cli synth --out gia.jpg          # sinh ảnh bàn cờ giả để thử
```

### API

| Endpoint | Mô tả |
|---|---|
| `POST /api/analyze` | multipart: `image`, `turn` (`w`/`b`/`auto`), `session_id`, `movetime_ms`, `corners` (JSON 4 góc, tuỳ chọn), `debug` |
| `POST /api/analyze_fen` | JSON `{"fen": "...", "movetime_ms": 1000}` |
| `POST /api/calibrate` | ảnh **thế cờ ban đầu**: lưu 32 ảnh quân thật có nhãn vào `data/crops/` |
| `POST /api/session/{id}/reset` | bắt đầu ván mới |
| `GET /api/health` | engine và mô hình đang dùng |

Với `turn=auto` và cùng một `session_id`, server so sánh với ảnh trước để biết nước vừa đi và đến lượt ai.

## Làm cho nhận dạng chạy tốt với bộ cờ của bạn

Mô hình kèm theo mới chỉ được huấn luyện trên **ảnh tổng hợp** vẽ từ font WenQuanYi. Quân cờ thật thường khắc chữ thư pháp, nên trên ảnh thật độ chính xác sẽ thấp hơn. Cách khắc phục:

1. Xếp bàn cờ ở **thế ban đầu**, chụp 5–10 ảnh ở các góc và ánh sáng khác nhau. Gửi từng ảnh qua nút "Hiệu chuẩn" trên trang web hoặc `python -m xiangqi_assistant.cli calibrate anh.jpg`. Mỗi ảnh cho 32 ảnh quân có nhãn sẵn, vì vị trí ban đầu đã biết trước.
2. Tinh chỉnh mô hình (cần `pip install torch onnx`):
   ```bash
   python tools/train_cnn.py --real data/crops --init models/piece_cnn.pt --epochs 6
   ```
3. Có thể cài thêm font chữ Hán (`fonts-arphic-ukai`, `fonts-arphic-uming`, `fonts-noto-cjk`) rồi huấn luyện lại để mô hình quen nhiều kiểu chữ hơn.

## App Android cho kính

```bash
cd android
gradle wrapper && ./gradlew assembleDebug     # cần Android SDK (Android Studio)
adb install app/build/outputs/apk/debug/app-debug.apk
```

- Lần đầu chạy, nhấn giữ màn hình (hoặc phím Menu) để nhập địa chỉ server, ví dụ `http://192.168.1.10:8000`. Kính và máy tính phải chung mạng Wi-Fi.
- Chạm hoặc OK để chụp. Tăng âm lượng để đổi lượt đi. Lên để bật/tắt tự chụp. Xuống để bắt đầu ván mới.
- Nền đen (trên kính AR màu đen là trong suốt), chữ xanh lá.

## Tình trạng hiện tại

| Phần | Đã kiểm thử |
|---|---|
| Luật cờ, sinh nước đi | ✅ perft khớp số chuẩn (44 / 1 920 / 79 666) |
| Ký hiệu tiếng Việt | ✅ test đơn vị |
| Tìm bàn cờ + lưới (4 hướng chụp, phối cảnh) | ✅ sai số < 4px trên ảnh tổng hợp |
| Phát hiện quân + màu | ✅ đúng 100% trên ảnh tổng hợp |
| Toàn bộ test | ✅ 34 test (`pytest -q`) |
| Phân loại loại quân (CNN) | ✅ ảnh tổng hợp: 39/40 bàn đúng hoàn toàn (sai 1 ô / ~1 000 quân). ❓ ảnh thật: cần dữ liệu của bạn |
| Gọi Pikafish | ✅ với engine giả lập UCI. ❓ chưa chạy Pikafish thật (môi trường build bị chặn mạng) |
| App Android | ❓ **chưa build thử** (môi trường không tải được Android SDK). Có thể cần sửa lỗi biên dịch nhỏ |

Chạy test: `cd server && pip install -r requirements-dev.txt && pytest -q`

## Khó khăn cần cùng giải quyết

1. **Kính Rokid cụ thể chạy app thế nào?** Mình không tìm được thông tin xác thực về mẫu "RV101". Cần bạn xác nhận:
   - Kính có camera không? (Rokid Max/Air không có camera, Rokid Glasses thì có)
   - Kính cài được file APK Android thông thường không, hay phải dùng SDK riêng của Rokid (ví dụ bộ CXR kết nối với điện thoại)?

   Nếu không cài APK được, hướng thay thế là: app chạy trên điện thoại, dùng SDK Rokid để lấy ảnh từ camera kính và đẩy chữ lên màn hình kính. Phần gọi server (`ApiClient.kt`) giữ nguyên.
2. **Ảnh thật**: cần bạn gửi vài ảnh chụp thật từ kính (thế ban đầu và giữa ván) để đo độ chính xác và tinh chỉnh.
3. **Góc nhìn**: camera trên kính nhìn bàn cờ khá xiên. Quân cờ có độ dày nên mặt quân bị lệch khỏi giao điểm. Có thể phải bù độ lệch này theo góc nhìn.
4. **Lượt đi**: một ảnh không cho biết đến lượt ai. Hiện tại dùng cách theo dõi ván (`session_id`), hoặc chọn tay bằng phím.
5. **Độ trễ**: chụp, gửi qua Wi-Fi, nhận dạng (~0.5s), rồi engine (1–2s). Có thể chạy server ngay trên điện thoại sau.
