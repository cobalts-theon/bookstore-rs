from __future__ import annotations

from pathlib import Path

import pytest

from booklog.data import load_dataset
from booklog.evaluation import evaluate
from booklog.hybrid import HybridBookRecommender


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(Path(__file__).parents[1] / "data" / "processed")


@pytest.fixture(scope="module")
def model(dataset):
    return HybridBookRecommender(epochs=5).fit(
        dataset, min_user_ratings=1, min_book_ratings=1
    )


def test_dataset_is_clean(dataset):
    assert len(dataset.books) > 0
    assert dataset.books["ISBN"].is_unique
    assert (dataset.ratings["Book-Rating"] > 0).all()


def test_content_model_resolves_existing_title(model):
    title = model.books.iloc[0]["Book-Title"]
    matched, recommendations = model.recommend_similar(title, top_n=3)
    assert matched == title
    assert title not in set(recommendations["Book-Title"])


def test_personalized_recommendations_exclude_seen_books(model):
    user_id = next(iter(model.collaborative.user_items))
    recommendations = model.recommend_for_user(user_id, top_n=10)
    seen = model.collaborative.user_items[user_id]
    assert set(recommendations["ISBN"]).isdisjoint(seen)
    assert recommendations["Hybrid-Score"].is_monotonic_decreasing


def test_cold_start_uses_and_excludes_favorites(model):
    favorite = model.books.iloc[0]["Book-Title"]
    recommendations = model.recommend_for_user(
        -1, top_n=5, favorite_titles=[favorite]
    )
    assert model.collaborative_weight(-1) == 0
    assert favorite not in set(recommendations["Book-Title"])


def test_collaborative_weight_increases_with_history(model):
    user_id = next(iter(model.collaborative.user_items))
    assert model.collaborative_weight(user_id) > model.collaborative_weight(-1)


def test_vectorized_predictions_match_scalar_predictions(model):
    user_id = next(iter(model.collaborative.user_items))
    isbns = model.books["ISBN"].head(5).tolist()
    vectorized = model.collaborative.predict_many(user_id, isbns)
    scalar = [model.collaborative.predict(user_id, isbn) for isbn in isbns]
    assert vectorized == pytest.approx(scalar)


def test_evaluation_returns_expected_metrics(dataset):
    metrics = evaluate(dataset, top_k=5, max_users=20, factors=8, epochs=3)
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert 0 <= metrics["hit_rate@5"] <= 1
    assert 0 <= metrics["precision@5"] <= 1
