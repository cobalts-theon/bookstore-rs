"""Simple Flask web interface for the BookLog recommendation system."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from booklog.data import load_dataset as load_csv_dataset  # noqa: E402
from booklog.database import (  # noqa: E402
    initialize_database,
    load_dataset as load_database_dataset,
    save_rating,
    seed_database,
)
from booklog.hybrid import HybridBookRecommender  # noqa: E402


@lru_cache(maxsize=4)
def load_model(data_dir: str, database_path: str):
    """Load the SQLite dataset and train once until ratings change."""
    data_dir_path = Path(data_dir)
    if initialize_database(database_path):
        seed_database(database_path, load_csv_dataset(data_dir_path))
    dataset = load_database_dataset(database_path)
    is_demo = data_dir_path.name == "demo"
    model = HybridBookRecommender().fit(
        dataset,
        min_user_ratings=1 if is_demo else 3,
        min_book_ratings=1 if is_demo else 3,
        max_ratings=None if is_demo else 30_000,
    )
    return dataset, model


def records(frame):
    """Convert books to template-ready records with a lightweight cover URL."""
    result = frame.fillna("").copy()
    if "Image-URL-M" in result:
        result["Cover-URL"] = result["Image-URL-M"].where(
            result["Image-URL-M"].astype(str).str.startswith(("http://", "https://")),
            result.get("Image-URL-L", ""),
        )
        result["Cover-URL"] = result["Cover-URL"].astype(str).str.replace(
            "http://", "https://", n=1
        )
    return result.to_dict(orient="records")


def create_app(
    testing: bool = False,
    data_dir: str | Path | None = None,
    database_path: str | Path | None = None,
):
    app = Flask(__name__)
    processed_dir = ROOT / "data" / "processed"
    default_dir = processed_dir if processed_dir.exists() else ROOT / "data" / "demo"
    app.config.from_mapping(
        TESTING=testing,
        SECRET_KEY=os.environ.get("BOOKLOG_SECRET_KEY", "booklog-development-key"),
        DATA_DIR=str(Path(data_dir or os.environ.get("BOOKLOG_DATA_DIR", default_dir))),
        DATABASE=str(
            Path(
                database_path
                or os.environ.get("BOOKLOG_DATABASE", ROOT / "data" / "booklog.db")
            )
        ),
    )

    @app.get("/")
    def home():
        dataset, model = load_model(app.config["DATA_DIR"], app.config["DATABASE"])
        query = request.args.get("q", "").strip()
        user_id = request.args.get("user_id", type=int)
        favorite = request.args.get("favorite", "").strip()

        books = model.books
        title = "All books"
        message = "Explore popular titles from our curated collection."

        if user_id is not None:
            favorites = [favorite] if favorite else []
            books = model.recommend_for_user(user_id, top_n=12, favorite_titles=favorites)
            title = "Recommended for you"
            weight = round(model.collaborative_weight(user_id) * 100)
            message = f"Personalized with {weight}% collaborative preference."
        elif query:
            mask = (
                books["Book-Title"].str.contains(query, case=False, regex=False)
                | books["Book-Author"].str.contains(query, case=False, regex=False)
            )
            books = books[mask]
            title = f'Search results for "{query}"'
            message = f"{len(books)} matching books found."

        return render_template(
            "index.html",
            books=records(books.head(24)),
            users=sorted(model.collaborative.user_to_index),
            title=title,
            message=message,
            query=query,
            selected_user=user_id,
            selected_favorite=favorite,
            stats={
                "books": len(dataset.books),
                "users": len(model.collaborative.user_to_index),
                "ratings": len(model.explicit_ratings),
            },
        )

    @app.post("/ratings")
    def rate_book():
        user_id = request.form.get("user_id", type=int)
        rating = request.form.get("rating", type=int)
        isbn = request.form.get("isbn", "").strip()
        try:
            if user_id is None or rating is None:
                raise ValueError("Reader and rating are required.")
            load_model(app.config["DATA_DIR"], app.config["DATABASE"])
            save_rating(app.config["DATABASE"], user_id, isbn, rating)
        except ValueError as error:
            flash(str(error), "error")
        else:
            load_model.cache_clear()
            flash("Your rating was saved.", "success")
        return redirect(url_for("home", user_id=user_id))

    return app


app = create_app()


if __name__ == "__main__":
    load_model(app.config["DATA_DIR"], app.config["DATABASE"])
    app.run()
