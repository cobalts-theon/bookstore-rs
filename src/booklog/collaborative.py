"""Biased matrix factorization trained with stochastic gradient descent."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BiasedMatrixFactorization:
    def __init__(
        self,
        factors: int = 24,
        epochs: int = 15,
        learning_rate: float = 0.01,
        regularization: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

    def fit(self, ratings: pd.DataFrame) -> "BiasedMatrixFactorization":
        if ratings.empty:
            raise ValueError("Matrix factorization requires at least one explicit rating.")
        self.ratings = ratings[ratings["Book-Rating"] > 0].copy()
        users = self.ratings["User-ID"].unique()
        items = self.ratings["ISBN"].unique()
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        self.index_to_item = np.asarray(items)
        self.global_mean = float(self.ratings["Book-Rating"].mean())
        self.min_rating = float(self.ratings["Book-Rating"].min())
        self.max_rating = float(self.ratings["Book-Rating"].max())
        self.user_counts = self.ratings["User-ID"].value_counts().to_dict()
        self.user_items = self.ratings.groupby("User-ID")["ISBN"].apply(set).to_dict()

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.1, (len(users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(items), self.factors))
        self.user_bias = np.zeros(len(users))
        self.item_bias = np.zeros(len(items))

        samples = [
            (self.user_to_index[user], self.item_to_index[isbn], float(rating))
            for user, isbn, rating in self.ratings[
                ["User-ID", "ISBN", "Book-Rating"]
            ].itertuples(index=False, name=None)
        ]
        for _ in range(self.epochs):
            rng.shuffle(samples)
            for user_index, item_index, rating in samples:
                prediction = self._predict_index(user_index, item_index)
                error = rating - prediction
                user_vector = self.user_factors[user_index].copy()
                self.user_bias[user_index] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user_index]
                )
                self.item_bias[item_index] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item_index]
                )
                self.user_factors[user_index] += self.learning_rate * (
                    error * self.item_factors[item_index]
                    - self.regularization * self.user_factors[user_index]
                )
                self.item_factors[item_index] += self.learning_rate * (
                    error * user_vector
                    - self.regularization * self.item_factors[item_index]
                )
        return self

    def _predict_index(self, user_index: int, item_index: int) -> float:
        return float(
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + self.user_factors[user_index] @ self.item_factors[item_index]
        )

    def predict(self, user_id: int, isbn: str) -> float:
        prediction = self.global_mean
        user_index = self.user_to_index.get(user_id)
        item_index = self.item_to_index.get(isbn)
        if user_index is not None:
            prediction += self.user_bias[user_index]
        if item_index is not None:
            prediction += self.item_bias[item_index]
        if user_index is not None and item_index is not None:
            prediction += self.user_factors[user_index] @ self.item_factors[item_index]
        return float(np.clip(prediction, self.min_rating, self.max_rating))

    def predict_many(self, user_id: int, isbns: list[str]) -> np.ndarray:
        user_index = self.user_to_index.get(user_id)
        predictions = np.full(len(isbns), self.global_mean, dtype=float)
        if user_index is not None:
            predictions += self.user_bias[user_index]

        known_positions: list[int] = []
        known_indices: list[int] = []
        for position, isbn in enumerate(isbns):
            item_index = self.item_to_index.get(isbn)
            if item_index is not None:
                known_positions.append(position)
                known_indices.append(item_index)
        if known_positions:
            positions = np.asarray(known_positions)
            item_indices = np.asarray(known_indices)
            predictions[positions] += self.item_bias[item_indices]
            if user_index is not None:
                predictions[positions] += (
                    self.item_factors[item_indices] @ self.user_factors[user_index]
                )
        return np.clip(predictions, self.min_rating, self.max_rating)
