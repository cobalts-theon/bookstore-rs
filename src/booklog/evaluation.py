"""Offline evaluation utilities for the recommendation system."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .data import BookDataset
from .hybrid import HybridBookRecommender


def leave_one_out_split(
    ratings: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    explicit = ratings[ratings["Book-Rating"] > 0].copy()
    eligible = explicit.groupby("User-ID").filter(lambda group: len(group) >= 3)
    test = eligible.groupby("User-ID", group_keys=False).sample(
        n=1, random_state=random_state
    )
    train = explicit.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(
    dataset: BookDataset,
    top_k: int = 10,
    max_users: int = 100,
    factors: int = 16,
    epochs: int = 10,
) -> dict[str, float]:
    train, test = leave_one_out_split(dataset.ratings)
    if test.empty:
        raise ValueError("Evaluation needs at least one user with three explicit ratings.")
    evaluation_dataset = BookDataset(dataset.books, train, dataset.users)
    model = HybridBookRecommender(factors=factors, epochs=epochs).fit(
        evaluation_dataset, min_user_ratings=1, min_book_ratings=1
    )

    predictions = [
        model.collaborative.predict(user_id, isbn)
        for user_id, isbn in test[["User-ID", "ISBN"]].itertuples(index=False, name=None)
    ]
    actual = test["Book-Rating"].to_numpy()

    relevant = test[test["Book-Rating"] >= 7].head(max_users)
    hits = 0
    for user_id, isbn in relevant[["User-ID", "ISBN"]].itertuples(index=False, name=None):
        recommended = model.recommend_for_user(user_id, top_n=top_k)
        hits += int(isbn in set(recommended["ISBN"]))
    users = max(len(relevant), 1)
    return {
        "rmse": float(root_mean_squared_error(actual, predictions)),
        "mae": float(mean_absolute_error(actual, predictions)),
        f"hit_rate@{top_k}": hits / users,
        f"precision@{top_k}": hits / (users * top_k),
        "evaluated_ratings": float(len(test)),
        "evaluated_relevant_users": float(len(relevant)),
    }

