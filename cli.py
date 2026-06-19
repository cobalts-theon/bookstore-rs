"""Command-line interface for BookLog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from booklog.data import load_dataset  # noqa: E402
from booklog.evaluation import evaluate  # noqa: E402
from booklog.hybrid import HybridBookRecommender  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BookLog hybrid recommendation CLI")
    parser.add_argument("--data-dir", default="data/demo", help="Folder containing CSV files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    similar = subparsers.add_parser("similar", help="Recommend similar books")
    similar.add_argument("title")
    similar.add_argument("--top-n", type=int, default=10)

    user = subparsers.add_parser("user", help="Recommend books for a user")
    user.add_argument("user_id", type=int)
    user.add_argument("--favorite", action="append", default=[])
    user.add_argument("--top-n", type=int, default=10)

    evaluation = subparsers.add_parser("evaluate", help="Run leave-one-out evaluation")
    evaluation.add_argument("--top-k", type=int, default=10)
    evaluation.add_argument("--max-users", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset(args.data_dir)
    if args.command == "evaluate":
        print(json.dumps(evaluate(dataset, args.top_k, args.max_users), indent=2))
        return

    model = HybridBookRecommender().fit(dataset, min_user_ratings=1, min_book_ratings=1)
    if args.command == "similar":
        matched, recommendations = model.recommend_similar(args.title, args.top_n)
        if matched is None:
            raise SystemExit(f"No title similar to '{args.title}' was found.")
        print(f"Matched title: {matched}")
    else:
        recommendations = model.recommend_for_user(
            args.user_id, args.top_n, favorite_titles=args.favorite
        )
    columns = [
        column
        for column in [
            "Book-Title",
            "Book-Author",
            "Year-Of-Publication",
            "Content-Score",
            "Collaborative-Score",
            "Hybrid-Score",
        ]
        if column in recommendations
    ]
    print(recommendations[columns].to_string(index=False))


if __name__ == "__main__":
    main()

