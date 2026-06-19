"""TF-IDF content-based recommendation model."""

from __future__ import annotations

from difflib import get_close_matches

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    def __init__(self, max_features: int = 40_000) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            strip_accents="unicode",
        )

    def fit(self, books: pd.DataFrame) -> "ContentBasedRecommender":
        self.books = books.drop_duplicates("ISBN").reset_index(drop=True).copy()
        year = self.books["Year-Of-Publication"].replace(0, "").astype(str)
        metadata = (
            self.books["Book-Title"].fillna("")
            + " "
            + self.books["Book-Author"].fillna("")
            + " "
            + self.books["Publisher"].fillna("")
            + " "
            + year
        )
        self.matrix: csr_matrix = self.vectorizer.fit_transform(metadata)
        self.isbn_to_index = dict(zip(self.books["ISBN"], self.books.index))
        self.title_to_isbn = (
            self.books.drop_duplicates("Book-Title")
            .set_index("Book-Title")["ISBN"]
            .to_dict()
        )
        self.titles = list(self.title_to_isbn)
        return self

    def resolve_title(self, title: str) -> tuple[str | None, str | None]:
        if title in self.title_to_isbn:
            return self.title_to_isbn[title], title
        normalized = {value.casefold(): value for value in self.titles}
        exact = normalized.get(title.strip().casefold())
        if exact:
            return self.title_to_isbn[exact], exact
        matches = get_close_matches(title, self.titles, n=1, cutoff=0.45)
        return (self.title_to_isbn[matches[0]], matches[0]) if matches else (None, None)

    def scores_for_isbn(self, isbn: str) -> np.ndarray:
        index = self.isbn_to_index.get(isbn)
        if index is None:
            return np.zeros(len(self.books))
        return linear_kernel(self.matrix[index], self.matrix).ravel()

    def profile_scores(
        self, rated_isbns: list[str], ratings: list[float] | None = None
    ) -> np.ndarray:
        indices = [self.isbn_to_index[isbn] for isbn in rated_isbns if isbn in self.isbn_to_index]
        if not indices:
            return np.zeros(len(self.books))
        if ratings is None:
            weights = np.ones(len(indices))
        else:
            filtered_ratings = [
                rating
                for isbn, rating in zip(rated_isbns, ratings)
                if isbn in self.isbn_to_index
            ]
            weights = np.asarray(filtered_ratings, dtype=float)
            weights = np.maximum(weights - max(weights.mean() - 1.0, 0.0), 0.25)
        profile = self.matrix[indices].multiply(weights[:, None]).sum(axis=0)
        return linear_kernel(csr_matrix(profile), self.matrix).ravel()

    def similar_books(self, title: str, top_n: int = 10) -> tuple[str | None, pd.DataFrame]:
        isbn, matched_title = self.resolve_title(title)
        if isbn is None:
            return None, pd.DataFrame()
        scores = self.scores_for_isbn(isbn)
        ranked = np.argsort(scores)[::-1]
        ranked = [
            index
            for index in ranked
            if self.books.iloc[index]["Book-Title"] != matched_title and scores[index] > 0
        ][:top_n]
        result = self.books.iloc[ranked].copy()
        result["Content-Score"] = scores[ranked]
        return matched_title, result
