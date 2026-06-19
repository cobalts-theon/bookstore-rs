"""SQLite persistence for BookLog datasets and user ratings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .data import BookDataset


SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    isbn TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    publication_year INTEGER NOT NULL DEFAULT 0,
    publisher TEXT NOT NULL DEFAULT '',
    image_url_s TEXT,
    image_url_m TEXT,
    image_url_l TEXT
);

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    location TEXT NOT NULL DEFAULT '',
    age REAL
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER NOT NULL,
    isbn TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 0 AND 10),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, isbn),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (isbn) REFERENCES books(isbn) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ratings_isbn ON ratings(isbn);
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a configured SQLite connection."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> bool:
    """Create the schema and return whether the database needs initial data."""
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)
        count = connection.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    return count == 0


def _value(value):
    return None if pd.isna(value) else value


def seed_database(database_path: str | Path, dataset: BookDataset) -> None:
    """Import a CSV-backed dataset into an empty SQLite database."""
    books = [
        (
            row["ISBN"],
            row["Book-Title"],
            row["Book-Author"],
            int(row["Year-Of-Publication"]),
            row["Publisher"],
            _value(row["Image-URL-S"]),
            _value(row["Image-URL-M"]),
            _value(row["Image-URL-L"]),
        )
        for _, row in dataset.books.iterrows()
    ]
    users = [
        (
            int(row["User-ID"]),
            _value(row["Location"]) or "",
            _value(row["Age"]),
        )
        for _, row in dataset.users.iterrows()
    ]
    rating_user_ids = {(int(user_id), "", None) for user_id in dataset.ratings["User-ID"]}
    ratings = [
        (int(row["User-ID"]), row["ISBN"], int(row["Book-Rating"]))
        for _, row in dataset.ratings.iterrows()
    ]

    with connect(database_path) as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO books (
                isbn, title, author, publication_year, publisher,
                image_url_s, image_url_m, image_url_l
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            books,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO users (user_id, location, age) VALUES (?, ?, ?)",
            users,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO users (user_id, location, age) VALUES (?, ?, ?)",
            rating_user_ids,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO ratings (user_id, isbn, rating) VALUES (?, ?, ?)",
            ratings,
        )


def load_dataset(database_path: str | Path) -> BookDataset:
    """Load a recommendation dataset from SQLite."""
    with connect(database_path) as connection:
        books = pd.read_sql_query(
            """
            SELECT
                isbn AS "ISBN",
                title AS "Book-Title",
                author AS "Book-Author",
                publication_year AS "Year-Of-Publication",
                publisher AS "Publisher",
                image_url_s AS "Image-URL-S",
                image_url_m AS "Image-URL-M",
                image_url_l AS "Image-URL-L"
            FROM books
            """,
            connection,
        )
        ratings = pd.read_sql_query(
            """
            SELECT
                user_id AS "User-ID",
                isbn AS "ISBN",
                rating AS "Book-Rating"
            FROM ratings
            """,
            connection,
        )
        users = pd.read_sql_query(
            """
            SELECT
                user_id AS "User-ID",
                location AS "Location",
                age AS "Age"
            FROM users
            """,
            connection,
        )
    return BookDataset(books=books, ratings=ratings, users=users)


def save_rating(
    database_path: str | Path, user_id: int, isbn: str, rating: int
) -> None:
    """Create or update one user's rating for a book."""
    if user_id <= 0:
        raise ValueError("Choose an existing reader before rating a book.")
    if not 1 <= rating <= 10:
        raise ValueError("Rating must be between 1 and 10.")

    with connect(database_path) as connection:
        book_exists = connection.execute(
            "SELECT 1 FROM books WHERE isbn = ?", (isbn,)
        ).fetchone()
        if book_exists is None:
            raise ValueError("The selected book does not exist.")
        connection.execute(
            "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
            (user_id,),
        )
        connection.execute(
            """
            INSERT INTO ratings (user_id, isbn, rating)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, isbn) DO UPDATE SET
                rating = excluded.rating,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, isbn, rating),
        )
