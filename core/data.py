"""Dataset helpers shared across all patterns."""

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of records, one per non-empty line."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_article(path: str | Path, article_id: str) -> str:
    """Return the article text for the given id from a JSONL dataset."""
    for record in load_jsonl(path):
        if record["id"] == article_id:
            return record["article"]
    raise KeyError(f"Article {article_id!r} not found in {path}")


def main() -> None:
    """Smoke test: load one article from the pipeline dataset and print it."""
    path = (
        Path(__file__).parent.parent
        / "patterns"
        / "p01_pipeline"
        / "data"
        / "articles.jsonl"
    )
    print(get_article(path, "grpc-migration"))


if __name__ == "__main__":
    main()
