"""Dataset loading, cleaning, and demo-data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BOOK_COLUMNS = [
    "ISBN",
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
    "Image-URL-S",
    "Image-URL-M",
    "Image-URL-L",
]
RATING_COLUMNS = ["User-ID", "ISBN", "Book-Rating"]
USER_COLUMNS = ["User-ID", "Location", "Age"]


@dataclass(frozen=True)
class BookDataset:
    books: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="latin-1", errors="replace") as source:
        header = source.readline()
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(
        path,
        sep=separator,
        engine="c",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
    )


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result[columns]


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    books = _ensure_columns(books, BOOK_COLUMNS)
    books = books.dropna(subset=["ISBN", "Book-Title"]).copy()
    for column in ["ISBN", "Book-Title", "Book-Author", "Publisher"]:
        books[column] = books[column].fillna("").astype(str).str.strip()
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"], errors="coerce"
    ).fillna(0).astype(int)
    books.loc[
        ~books["Year-Of-Publication"].between(1000, 2100), "Year-Of-Publication"
    ] = 0
    books = books.drop_duplicates(subset="ISBN", keep="first")
    return books.reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame, valid_isbns: set[str]) -> pd.DataFrame:
    ratings = _ensure_columns(ratings, RATING_COLUMNS).copy()
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip()
    ratings = ratings.dropna(subset=["User-ID", "Book-Rating"])
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    ratings = ratings[
        ratings["ISBN"].isin(valid_isbns)
        & ratings["Book-Rating"].between(0, 10)
    ]
    return ratings.drop_duplicates(subset=["User-ID", "ISBN"], keep="last").reset_index(
        drop=True
    )


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = _ensure_columns(users, USER_COLUMNS).copy()
    users["User-ID"] = pd.to_numeric(users["User-ID"], errors="coerce")
    users = users.dropna(subset=["User-ID"])
    users["User-ID"] = users["User-ID"].astype(int)
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    users.loc[~users["Age"].between(5, 100), "Age"] = np.nan
    return users.drop_duplicates(subset="User-ID").reset_index(drop=True)


def load_dataset(data_dir: str | Path) -> BookDataset:
    """Load Kaggle-compatible Books.csv, Ratings.csv, and Users.csv files."""
    data_dir = Path(data_dir)
    required = [data_dir / "Books.csv", data_dir / "Ratings.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files: "
            + ", ".join(missing)
            + ". See README.md for setup instructions."
        )

    books = clean_books(_read_csv(data_dir / "Books.csv"))
    ratings = clean_ratings(_read_csv(data_dir / "Ratings.csv"), set(books["ISBN"]))
    users_path = data_dir / "Users.csv"
    users = clean_users(_read_csv(users_path)) if users_path.exists() else clean_users(
        pd.DataFrame({"User-ID": ratings["User-ID"].unique()})
    )
    return BookDataset(books=books, ratings=ratings, users=users)


def filter_explicit_feedback(
    ratings: pd.DataFrame,
    min_user_ratings: int = 2,
    min_book_ratings: int = 2,
    max_ratings: int | None = None,
) -> pd.DataFrame:
    """Keep useful explicit ratings and iteratively remove sparse entities."""
    result = ratings[ratings["Book-Rating"] > 0].copy()
    for _ in range(2):
        user_counts = result["User-ID"].value_counts()
        result = result[result["User-ID"].isin(user_counts[user_counts >= min_user_ratings].index)]
        book_counts = result["ISBN"].value_counts()
        result = result[result["ISBN"].isin(book_counts[book_counts >= min_book_ratings].index)]
    if max_ratings and len(result) > max_ratings:
        result = result.sample(max_ratings, random_state=42)
    return result.reset_index(drop=True)
