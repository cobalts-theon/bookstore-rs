"""Create a small web-ready dataset from the full Kaggle dataset.

This script is standalone so it can be uploaded and run directly on Colab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="latin-1", errors="replace") as source:
        header = source.readline()
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(
        path,
        sep=separator,
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
    )


def prepare(
    input_dir: Path,
    output_dir: Path,
    max_users: int,
    max_books: int,
    max_ratings: int,
    min_user_ratings: int,
    min_book_ratings: int,
) -> None:
    books = read_csv(input_dir / "Books.csv")
    ratings = read_csv(input_dir / "Ratings.csv")
    users_path = input_dir / "Users.csv"
    users = read_csv(users_path) if users_path.exists() else None

    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip()
    ratings = ratings.dropna(subset=["User-ID", "Book-Rating"])
    ratings = ratings[ratings["Book-Rating"].between(1, 10)].copy()
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"], keep="last")

    books["ISBN"] = books["ISBN"].astype(str).str.strip()
    ratings = ratings[ratings["ISBN"].isin(set(books["ISBN"]))]

    active_users = ratings["User-ID"].value_counts()
    active_users = active_users[active_users >= min_user_ratings].head(max_users).index
    ratings = ratings[ratings["User-ID"].isin(active_users)]

    popular_books = ratings["ISBN"].value_counts()
    popular_books = popular_books[popular_books >= min_book_ratings].head(max_books).index
    ratings = ratings[ratings["ISBN"].isin(popular_books)]

    active_users = ratings["User-ID"].value_counts()
    active_users = active_users[active_users >= min_user_ratings].index
    ratings = ratings[ratings["User-ID"].isin(active_users)]

    if len(ratings) > max_ratings:
        ratings = ratings.sample(max_ratings, random_state=42)

    books = books[books["ISBN"].isin(set(ratings["ISBN"]))].drop_duplicates("ISBN")
    output_dir.mkdir(parents=True, exist_ok=True)
    books.to_csv(output_dir / "Books.csv", index=False, encoding="latin-1")
    ratings.to_csv(output_dir / "Ratings.csv", index=False, encoding="latin-1")

    if users is not None:
        users["User-ID"] = pd.to_numeric(users["User-ID"], errors="coerce")
        users = users[users["User-ID"].isin(set(ratings["User-ID"]))]
        users.to_csv(output_dir / "Users.csv", index=False, encoding="latin-1")

    print(f"Saved processed dataset to: {output_dir}")
    print(f"Books: {len(books):,}")
    print(f"Users: {ratings['User-ID'].nunique():,}")
    print(f"Ratings: {len(ratings):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a smaller BookLog dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--max-books", type=int, default=3_000)
    parser.add_argument("--max-ratings", type=int, default=30_000)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--min-book-ratings", type=int, default=5)
    args = parser.parse_args()
    prepare(
        args.input,
        args.output,
        args.max_users,
        args.max_books,
        args.max_ratings,
        args.min_user_ratings,
        args.min_book_ratings,
    )


if __name__ == "__main__":
    main()
