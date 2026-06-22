"""Root entrypoint for the BookLog Flask app."""

from __future__ import annotations

from mainproject.booklog.web import app, create_app, load_model  # noqa: E402,F401


if __name__ == "__main__":
    load_model(app.config["DATA_DIR"], app.config["DATABASE"])
    app.run()
