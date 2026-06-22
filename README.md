# BookLog - Frontend + Backend

BookLog là website gợi ý sách dùng Flask, SQLite và hybrid recommendation.
Dự án hiện chỉ giữ phần cần demo:

- **Backend:** Flask + SQLite + thuật toán recommendation.
- **Frontend:** HTML/CSS hiển thị bookstore và form gợi ý.

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
templates/index.html                 # Frontend HTML
static/css/style.css                 # Frontend CSS
static/img/book-placeholder.svg      # Ảnh placeholder
mainproject/data/processed/          # Dữ liệu demo đã xử lý
```

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
