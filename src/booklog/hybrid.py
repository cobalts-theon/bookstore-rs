"""Dynamic weighted hybrid recommendation model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .collaborative import BiasedMatrixFactorization
from .content import ContentBasedRecommender
from .data import BookDataset, filter_explicit_feedback


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.max() - values.min() if len(values) else 0
    return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)


class HybridBookRecommender:
    def __init__(
        self,
        cold_start_threshold: int = 10,
        factors: int = 24,
        epochs: int = 15,
    ) -> None:
        self.cold_start_threshold = cold_start_threshold
        self.content = ContentBasedRecommender()
        self.collaborative = BiasedMatrixFactorization(factors=factors, epochs=epochs)

    def fit(
        self,
        dataset: BookDataset,
        min_user_ratings: int = 2,
        min_book_ratings: int = 2,
        max_ratings: int | None = None,
    ) -> "HybridBookRecommender":
        self.dataset = dataset
        self.books = dataset.books.drop_duplicates("ISBN").reset_index(drop=True)
        self.content.fit(self.books)
        explicit = filter_explicit_feedback(
            dataset.ratings,
            min_user_ratings=min_user_ratings,
            min_book_ratings=min_book_ratings,
            max_ratings=max_ratings,
        )
        if explicit.empty:
            explicit = dataset.ratings[dataset.ratings["Book-Rating"] > 0].copy()
        self.explicit_ratings = explicit
        self.collaborative.fit(explicit)
        self.ratings_by_user = explicit.groupby("User-ID")
        self._build_popularity()
        return self

    def _build_popularity(self) -> None:
        stats = self.explicit_ratings.groupby("ISBN")["Book-Rating"].agg(["mean", "count"])
        prior = float(self.explicit_ratings["Book-Rating"].mean())
        confidence = 5
        stats["score"] = (
            stats["count"] / (stats["count"] + confidence) * stats["mean"]
            + confidence / (stats["count"] + confidence) * prior
        )
        self.popularity = stats["score"].to_dict()

    def collaborative_weight(self, user_id: int) -> float:
        count = self.collaborative.user_counts.get(user_id, 0)
        return float(count / (count + self.cold_start_threshold))

    def _content_profile(
        self, user_id: int, favorite_titles: list[str] | None
    ) -> np.ndarray:
        isbns: list[str] = []
        values: list[float] = []
        if user_id in self.ratings_by_user.groups:
            history = self.ratings_by_user.get_group(user_id)
            history = history[history["Book-Rating"] >= history["Book-Rating"].mean()]
            isbns.extend(history["ISBN"].tolist())
            values.extend(history["Book-Rating"].astype(float).tolist())
        for title in favorite_titles or []:
            isbn, _ = self.content.resolve_title(title)
            if isbn:
                isbns.append(isbn)
                values.append(10.0)
        return self.content.profile_scores(isbns, values)

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        favorite_titles: list[str] | None = None,
    ) -> pd.DataFrame:
        isbns = self.books["ISBN"].tolist()
        content_scores = _minmax(self._content_profile(user_id, favorite_titles))
        cf_scores = _minmax(self.collaborative.predict_many(user_id, isbns))
        popularity_scores = _minmax(
            np.asarray([self.popularity.get(isbn, 0.0) for isbn in isbns])
        )
        alpha = self.collaborative_weight(user_id)
        # Popularity provides a stable fallback without dominating personalization.
        final_scores = 0.9 * (
            alpha * cf_scores + (1.0 - alpha) * content_scores
        ) + 0.1 * popularity_scores

        seen = set(self.collaborative.user_items.get(user_id, set()))
        for title in favorite_titles or []:
            isbn, matched_title = self.content.resolve_title(title)
            if isbn and matched_title:
                seen.update(
                    self.books.loc[
                        self.books["Book-Title"] == matched_title, "ISBN"
                    ].tolist()
                )
        eligible = [
            index for index, isbn in enumerate(isbns) if isbn not in seen
        ]
        ranked = sorted(eligible, key=lambda index: final_scores[index], reverse=True)[:top_n]
        result = self.books.iloc[ranked].copy()
        result["Content-Score"] = content_scores[ranked]
        result["Collaborative-Score"] = cf_scores[ranked]
        result["Hybrid-Score"] = final_scores[ranked]
        result["Collaborative-Weight"] = alpha
        return result.reset_index(drop=True)

    def recommend_similar(self, title: str, top_n: int = 10) -> tuple[str | None, pd.DataFrame]:
        return self.content.similar_books(title, top_n=top_n)

    def search_titles(self, query: str, limit: int = 20) -> list[str]:
        titles = self.books["Book-Title"].drop_duplicates()
        if not query:
            return titles.head(limit).tolist()
        matches = titles[titles.str.contains(query, case=False, regex=False)]
        return matches.head(limit).tolist()
