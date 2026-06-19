# Dùng Google Colab để xử lý dữ liệu lớn

Google Colab phù hợp để lọc dữ liệu lớn một lần. Colab không phù hợp làm backend
ổn định cho website local vì phiên chạy có thể tự ngắt.

## 1. Đưa dữ liệu lên Google Drive

```text
MyDrive/BookLog/raw/Books.csv
MyDrive/BookLog/raw/Ratings.csv
MyDrive/BookLog/raw/Users.csv
MyDrive/BookLog/prepare_data.py
```

File `prepare_data.py` nằm ngay trong dự án này.

## 2. Chạy trong Google Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!python "/content/drive/MyDrive/BookLog/prepare_data.py" \
  --input "/content/drive/MyDrive/BookLog/raw" \
  --output "/content/drive/MyDrive/BookLog/processed" \
  --max-users 500 \
  --max-books 3000 \
  --max-ratings 30000
```

## 3. Chạy website trên máy

Tải thư mục `processed` từ Drive về:

```text
data/processed/Books.csv
data/processed/Ratings.csv
data/processed/Users.csv
```

Sau đó chỉ cần:

```powershell
python app.py
```

Website tự ưu tiên `data/processed`. Nếu máy vẫn chậm, chạy lại Colab với:

```text
--max-users 300 --max-books 1500 --max-ratings 15000
```

