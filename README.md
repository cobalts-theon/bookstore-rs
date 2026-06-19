# BookLog - Simple Python Web Bookstore

BookLog là website bookstore đơn giản dùng:

- **Python + Flask** cho web backend.
- **SQLite** lưu sách, người dùng và đánh giá khi website chạy.
- **HTML + CSS** cho giao diện.
- **Hybrid recommendation** kết hợp content-based và collaborative filtering.

## Cấu trúc chính

```text
app.py                  # Chạy website
templates/index.html    # Một trang HTML duy nhất
static/css/style.css    # Giao diện website
src/booklog/            # Mô hình recommendation
data/demo/              # Dữ liệu demo
```

## Chạy website

```powershell
python -m pip install -r requirements.txt
python app.py
```

Sau đó mở: `http://127.0.0.1:5000`

Website có ba chức năng:

1. Hiển thị danh sách sách.
2. Tìm sách theo tên hoặc tác giả.
3. Chọn reader hoặc sách yêu thích để nhận recommendation.
4. Lưu hoặc cập nhật đánh giá sách của reader vào SQLite.

Ảnh bìa được lấy từ cột `Image-URL-M` trong `Books.csv`, tải theo cơ chế lazy
loading và tự dùng placeholder nếu URL ảnh không còn hoạt động.

Mô hình chỉ được huấn luyện một lần khi website khởi động nên thao tác trên web
không phải huấn luyện lại liên tục. Sau khi lưu rating mới, mô hình sẽ được tải
lại ở request kế tiếp để sử dụng dữ liệu vừa cập nhật.

## SQLite database

Khi chạy lần đầu, website tự tạo `data/booklog.db` và nhập dữ liệu từ các file
CSV đang dùng. Các lần chạy sau sẽ đọc trực tiếp từ SQLite.

Database gồm ba bảng:

- `books`: thông tin sách.
- `users`: thông tin reader.
- `ratings`: đánh giá từ 0 đến 10, liên kết reader với sách.

Có thể đổi vị trí database bằng biến môi trường:

```powershell
$env:BOOKLOG_DATABASE="data/my-booklog.db"
python app.py
```

Muốn tạo lại database từ CSV, xóa file database hiện tại rồi chạy lại website.

## Dùng dữ liệu Kaggle lớn

Không chạy website trực tiếp bằng toàn bộ `data/raw`. Hãy dùng Google Colab để
lọc dữ liệu thành `data/processed` trước. Website sẽ tự dùng `data/processed`
nếu thư mục này tồn tại, nếu không sẽ dùng `data/demo`.

Xem hướng dẫn: `COLAB_GUIDE.md`.

## Đánh giá mô hình

```powershell
python cli.py evaluate --top-k 5
python -m pytest -q
```

Kết quả demo hiện tại:

- RMSE: `1.1060`
- MAE: `0.9360`
- HitRate@5: `0.5000`
- Precision@5: `0.1000`
