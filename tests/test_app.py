import sqlite3
from pathlib import Path

from app import create_app
from booklog.data import load_dataset


def test_homepage_search_and_recommendations(tmp_path):
    client = create_app(testing=True, database_path=tmp_path / "booklog.db").test_client()

    home = client.get("/")
    recommendations = client.get("/?user_id=-1")

    assert home.status_code == 200
    assert b"Discover books made for your taste" in home.data
    assert b"Image-URL" not in home.data
    assert b"class=\"cover-image\"" in home.data
    assert b"images.amazon.com" in home.data
    assert b"loading=\"lazy\"" in home.data
    assert b"Recommended for you" in recommendations.data
    assert b"% match" in recommendations.data


def test_rating_is_saved_to_sqlite(tmp_path):
    database_path = tmp_path / "booklog.db"
    data_dir = Path(__file__).parents[1] / "data" / "processed"
    dataset = load_dataset(data_dir)
    user_id = int(dataset.ratings.iloc[0]["User-ID"])
    isbn = dataset.books.iloc[0]["ISBN"]
    client = create_app(
        testing=True, data_dir=data_dir, database_path=database_path
    ).test_client()

    response = client.post(
        "/ratings",
        data={"user_id": user_id, "isbn": isbn, "rating": 9},
        follow_redirects=True,
    )

    with sqlite3.connect(database_path) as connection:
        saved_rating = connection.execute(
            "SELECT rating FROM ratings WHERE user_id = ? AND isbn = ?",
            (user_id, isbn),
        ).fetchone()

    assert response.status_code == 200
    assert b"Your rating was saved." in response.data
    assert saved_rating == (9,)
