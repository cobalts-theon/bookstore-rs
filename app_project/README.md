# BookLog - Frontend + Backend

BookLog là website gợi ý sách dùng Flask, SQLite và hybrid recommendation.
Dự án hiện có đủ các phần chính cho demo website:

- **Backend:** Flask + SQLite + thuật toán recommendation.
- **Frontend:** HTML/CSS hiển thị bookstore, catalog, profile và auth pages.
- **Account flow:** đăng ký, đăng nhập, đăng xuất, lưu rating theo tài khoản.

## Logic thuật toán

Giữ nguyên logic trong `docs/Bookstore recommendation system.docx.md`:

- **Content-based filtering:** TF-IDF trên title, author, publisher và year.
- **Collaborative filtering:** biased matrix factorization.
- **Popularity fallback:** giúp kết quả ổn định hơn.
- **Dynamic hybrid weight:** user mới ưu tiên content-based, user có nhiều rating
  tăng trọng số collaborative filtering.

```text
alpha = user_rating_count / (user_rating_count + 10)
final = 0.9 * (alpha * CF + (1 - alpha) * Content) + 0.1 * Popularity
```

## Cấu trúc

```text
app.py                              # Backend entrypoint
mainproject/booklog/web.py           # Flask routes
mainproject/booklog/database.py      # SQLite
mainproject/booklog/data.py          # Đọc và làm sạch dữ liệu
mainproject/booklog/content.py       # Content-based model
mainproject/booklog/collaborative.py # Collaborative filtering
mainproject/booklog/hybrid.py        # Hybrid recommendation
templates/base.html                  # Layout chung
templates/index.html                 # Trang chủ
templates/catalog.html               # Catalog + tìm kiếm/lọc
templates/login.html                 # Đăng nhập
templates/register.html              # Đăng ký
templates/profile.html               # Hồ sơ + gợi ý cá nhân
templates/about.html                 # Giới thiệu dự án
templates/_book_grid.html            # Component danh sách sách
static/css/style.css                 # Frontend CSS
static/img/book-placeholder.svg      # Ảnh placeholder
mainproject/data/processed/          # Dữ liệu demo đã xử lý
```

## Chức năng website

- Trang chủ giới thiệu BookLog, thống kê dataset và gợi ý nhanh.
- Catalog cho phép tìm sách theo title/author, lọc theo collection và phân trang.
- Người dùng có thể đăng ký/đăng nhập bằng email và mật khẩu.
- Profile hiển thị reader id, số rating đã lưu và danh sách gợi ý cá nhân.
- Khi người dùng đã đăng nhập, rating được lưu vào SQLite và mô hình được train lại
  qua cache refresh.

## Chạy project

```powershell
python -m pip install -r requirements.txt
python app.py
```

Sau đó mở:

```text
http://127.0.0.1:5000
```

Khi chạy lần đầu, backend tự tạo SQLite database tại
`mainproject/data/booklog.db` từ dữ liệu trong `mainproject/data/processed`.
