from __future__ import annotations

import pandas as pd

from booklog.data import BOOK_COLUMNS, RATING_COLUMNS, USER_COLUMNS, BookDataset
from booklog.database import initialize_database, load_dataset, save_rating, seed_database


def sample_dataset() -> BookDataset:
    return BookDataset(
        books=pd.DataFrame(
            [
                [
                    "isbn-1",
                    "Test Book",
                    "Test Author",
                    2026,
                    "Test Publisher",
                    "",
                    "",
                    "",
                ]
            ],
            columns=BOOK_COLUMNS,
        ),
        ratings=pd.DataFrame([[1, "isbn-1", 8]], columns=RATING_COLUMNS),
        users=pd.DataFrame([[1, "Bangkok", 22]], columns=USER_COLUMNS),
    )


def test_database_is_seeded_and_loaded(tmp_path):
    database_path = tmp_path / "booklog.db"

    assert initialize_database(database_path)
    seed_database(database_path, sample_dataset())
    dataset = load_dataset(database_path)

    assert not initialize_database(database_path)
    assert dataset.books.iloc[0]["Book-Title"] == "Test Book"
    assert dataset.users.iloc[0]["Location"] == "Bangkok"
    assert dataset.ratings.iloc[0]["Book-Rating"] == 8


def test_save_rating_updates_existing_rating(tmp_path):
    database_path = tmp_path / "booklog.db"
    initialize_database(database_path)
    seed_database(database_path, sample_dataset())

    save_rating(database_path, 1, "isbn-1", 10)

    dataset = load_dataset(database_path)
    assert dataset.ratings.iloc[0]["Book-Rating"] == 10
