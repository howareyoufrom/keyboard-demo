# TouchAssist IT3160

Ứng dụng thử nghiệm bàn phím QWERTY tiếng Việt Telex, so sánh bàn phím Baseline với bàn phím tích hợp bộ giải mã Viterbi và mô hình N-gram ký tự.

## Yêu cầu

- Python 3
- Các thư viện: Flask, `datasets`, `underthesea`
- File mô hình `model_wikipedia.pkl` ở thư mục gốc dự án

Cài các thư viện đã khai báo trong `requirements.txt` bằng:

```powershell
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

Từ thư mục gốc `D:\IT3160`, chạy:

```powershell
.\venv\Scripts\python.exe app.py
```

Mở [http://127.0.0.1:5000](http://127.0.0.1:5000) trong trình duyệt. Nếu không dùng môi trường `venv` có sẵn, thay lệnh trên bằng `python app.py` sau khi cài dependencies.

## Sử dụng

- Bàn phím ảo hỗ trợ nhập Telex; nút `VN/EN` bật hoặc tắt xử lý tiếng Việt.
- `AI: Bật/Tắt` điều khiển việc gửi các lần chạm qua API Viterbi.
- `Run tay` thêm một lần chạm lặp trong luồng nhập bàn phím chính.
- Phần **So sánh Baseline và bàn phím tích hợp** nhận chuỗi Telex sạch, sau đó chạy cùng một mẫu lỗi qua hai nhánh.
- Có thể điều chỉnh tỷ lệ nhấn lặp và tỷ lệ dịch điểm chạm sang phím lân cận. Đây là mô phỏng thử nghiệm đơn giản, không phải mô hình y khoa hay dữ liệu người dùng Parkinson.

Ví dụ Telex: `xin chaof` → `xin chào`.

## Thành phần

| Tệp | Vai trò |
|---|---|
| `app.py` | Flask app, trang chính và API `POST /api/decode` |
| `ai_engine.py` | Viterbi, xác suất N-gram ký tự và chấm điểm theo vị trí phím |
| `model_wikipedia.pkl` | Mô hình xác suất N-gram được tạo bởi script huấn luyện |
| `train_wikipedia.py` | Tải một phần Wikipedia tiếng Việt, đếm trigram ký tự và lưu mô hình |
| `templates/index.html` | Giao diện, bộ gõ Telex, mô phỏng lỗi và màn hình so sánh |
| `templates/baseline.html` | Giao diện Baseline độc lập |

### API giải mã

Gửi danh sách lần chạm dạng JSON tới `/api/decode`:

```json
{
  "touches": [
    {"x": 4, "y": 0, "t": 0, "key": "t", "group": 0},
    {"x": 5, "y": 0, "t": 200, "key": "y", "group": 1}
  ]
}
```

Mỗi lần chạm cần có `key`; `x`, `y`, `t`, `group` được dùng khi có sẵn. API trả về:

```json
{"decoded_word": "ty"}
```

## Tạo lại mô hình

Chỉ cần làm bước này nếu `model_wikipedia.pkl` chưa có hoặc muốn huấn luyện lại. Từ thư mục gốc, chạy:

```powershell
.\venv\Scripts\python.exe train_wikipedia.py
```

Script tải tập con Wikipedia tiếng Việt qua Hugging Face Datasets; lần đầu có thể cần Internet và thời gian tải/xử lý. Chạy huấn luyện sẽ ghi đè `model_wikipedia.pkl`.

## Giới hạn hiện tại

- Mô phỏng lệch phím và nhấn lặp là các quy tắc tổng hợp; chưa được hiệu chỉnh từ dữ liệu thực nghiệm và không đại diện đầy đủ cho Parkinson.
- Mô hình N-gram hiện là character-level, không phải mô hình hiểu nghĩa hay từ điển sửa chính tả.
- Tham số tọa độ bàn phím được chuẩn hóa theo bố cục QWERTY mô phỏng; cần hiệu chỉnh và đánh giá nếu thay đổi bố cục.
- Flask đang chạy ở chế độ debug trong cấu hình phát triển; không dùng cấu hình đó để triển khai công khai.

## Hướng cải tiến đề xuất

1. Thêm `requirements.txt` để cài lại môi trường nhất quán.
2. Viết test tự động cho quy tắc Telex, chọn ký tự Viterbi, bỏ lần chạm lặp, chạm lệch và API.
3. Ghi lại bộ dữ liệu kiểm thử có chuỗi dự định gõ và tọa độ/thời gian chạm; đánh giá Baseline và AI trên đúng cùng dữ liệu.
4. Hiệu chỉnh mô phỏng từ dữ liệu người tham gia hoặc nghiên cứu phù hợp; lưu seed để tái lập các lần chạy.
5. Hoàn thiện xử lý đầu vào API, lỗi mô hình và cấu hình triển khai; tắt debug khi chạy ngoài máy phát triển.
