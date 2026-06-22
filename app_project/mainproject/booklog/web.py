"""Simple Flask web interface for the BookLog recommendation system."""

from __future__ import annotations

import os
from functools import lru_cache, wraps
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "mainproject" / "data"

from mainproject.booklog.data import load_dataset as load_csv_dataset
from mainproject.booklog.database import (
    create_account,
    find_account_by_email,
    find_account_by_id,
    initialize_database,
    load_dataset as load_database_dataset,
    save_rating,
    seed_database,
)
from mainproject.booklog.hybrid import HybridBookRecommender


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


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.account is None:
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login", next=request.full_path))
        return view(**kwargs)

    return wrapped_view


def paginate(frame, page: int, per_page: int = 24):
    total = len(frame)
    page = max(page, 1)
    start = (page - 1) * per_page
    return frame.iloc[start : start + per_page], {
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": max((total + per_page - 1) // per_page, 1),
    }


def create_app(
    testing: bool = False,
    data_dir: str | Path | None = None,
    database_path: str | Path | None = None,
):
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    default_dir = DATA_ROOT / "processed"
    app.config.from_mapping(
        TESTING=testing,
        SECRET_KEY=os.environ.get("BOOKLOG_SECRET_KEY", "booklog-development-key"),
        DATA_DIR=str(Path(data_dir or os.environ.get("BOOKLOG_DATA_DIR", default_dir))),
        DATABASE=str(
            Path(
                database_path
                or os.environ.get(
                    "BOOKLOG_DATABASE", DATA_ROOT / "booklog.db"
                )
            )
        ),
    )

    @app.before_request
    def load_logged_in_account():
        load_model(app.config["DATA_DIR"], app.config["DATABASE"])
        account_id = session.get("account_id")
        g.account = (
            find_account_by_id(app.config["DATABASE"], account_id)
            if account_id is not None
            else None
        )

    @app.context_processor
    def inject_account():
        return {"current_account": g.get("account")}

    def build_books_response(template_name: str = "index.html"):
        dataset, model = load_model(app.config["DATA_DIR"], app.config["DATABASE"])
        query = request.args.get("q", "").strip()
        user_id = request.args.get("user_id", type=int)
        favorite = request.args.get("favorite", "").strip()
        category = request.args.get("category", "").strip()
        page = request.args.get("page", 1, type=int)

        books = model.books
        title = "All books"
        message = "Explore popular titles from our curated collection."

        if g.account and user_id is None and not query and not favorite:
            user_id = g.account["reader_user_id"]

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
        elif category:
            mask = books["Book-Author"].str.contains(category, case=False, regex=False)
            books = books[mask]
            title = f"Books related to {category}"
            message = f"{len(books)} books matched this collection."

        visible_books, pagination = paginate(books.head(120), page)

        return render_template(
            template_name,
            books=records(visible_books),
            users=sorted(model.collaborative.user_to_index),
            title=title,
            message=message,
            query=query,
            category=category,
            pagination=pagination,
            selected_user=user_id,
            selected_favorite=favorite,
            stats={
                "books": len(dataset.books),
                "users": len(model.collaborative.user_to_index),
                "ratings": len(model.explicit_ratings),
            },
        )

    @app.get("/")
    def home():
        return build_books_response()

    @app.get("/catalog")
    def catalog():
        return build_books_response("catalog.html")

    @app.get("/about")
    def about():
        dataset, model = load_model(app.config["DATA_DIR"], app.config["DATABASE"])
        return render_template(
            "about.html",
            stats={
                "books": len(dataset.books),
                "users": len(model.collaborative.user_to_index),
                "ratings": len(model.explicit_ratings),
            },
        )

    @app.route("/register", methods=("GET", "POST"))
    def register():
        if request.method == "POST":
            name = request.form.get("name", "")
            email = request.form.get("email", "")
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            try:
                load_model(app.config["DATA_DIR"], app.config["DATABASE"])
                if len(password) < 6:
                    raise ValueError("Password must contain at least 6 characters.")
                if password != confirm_password:
                    raise ValueError("Passwords do not match.")
                account_id = create_account(
                    app.config["DATABASE"],
                    name,
                    email,
                    generate_password_hash(password),
                )
            except ValueError as error:
                flash(str(error), "error")
            else:
                session.clear()
                session["account_id"] = account_id
                load_model.cache_clear()
                flash("Account created. Welcome to BookLog.", "success")
                return redirect(url_for("profile"))
        return render_template("register.html")

    @app.route("/login", methods=("GET", "POST"))
    def login():
        if request.method == "POST":
            email = request.form.get("email", "")
            password = request.form.get("password", "")
            account = find_account_by_email(app.config["DATABASE"], email)
            if account is None or not check_password_hash(
                account["password_hash"], password
            ):
                flash("Invalid email or password.", "error")
            else:
                session.clear()
                session["account_id"] = account["account_id"]
                flash("Signed in successfully.", "success")
                return redirect(request.args.get("next") or url_for("profile"))
        return render_template("login.html")

    @app.get("/logout")
    def logout():
        session.clear()
        flash("Signed out successfully.", "success")
        return redirect(url_for("home"))

    @app.get("/profile")
    @login_required
    def profile():
        _, model = load_model(app.config["DATA_DIR"], app.config["DATABASE"])
        user_id = g.account["reader_user_id"]
        recommendations = model.recommend_for_user(user_id, top_n=8)
        user_ratings = model.explicit_ratings[model.explicit_ratings["User-ID"] == user_id]
        return render_template(
            "profile.html",
            books=records(recommendations),
            rating_count=len(user_ratings),
            reader_user_id=user_id,
        )

    @app.post("/ratings")
    @login_required
    def rate_book():
        user_id = g.account["reader_user_id"]
        rating = request.form.get("rating", type=int)
        isbn = request.form.get("isbn", "").strip()
        try:
            if rating is None:
                raise ValueError("Rating is required.")
            load_model(app.config["DATA_DIR"], app.config["DATABASE"])
            save_rating(app.config["DATABASE"], user_id, isbn, rating)
        except ValueError as error:
            flash(str(error), "error")
        else:
            load_model.cache_clear()
            flash("Your rating was saved.", "success")
        return redirect(request.form.get("next") or url_for("profile"))

    return app


app = create_app()


if __name__ == "__main__":
    load_model(app.config["DATA_DIR"], app.config["DATABASE"])
    app.run()
