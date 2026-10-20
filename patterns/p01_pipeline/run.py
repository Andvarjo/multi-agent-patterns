"""Level 1 evaluation: sequential pipeline vs single-agent baseline.

Usage (from the repo root):
    uv run python -m patterns.p01_pipeline.run                     # full eval, 3 runs per case
    uv run python -m patterns.p01_pipeline.run --limit 2 --runs 1  # quick smoke test
    uv run python -m patterns.p01_pipeline.run --from-file results/p01_pipeline_....jsonl
"""

import argparse
from pathlib import Path

from core.data import load_jsonl
from core.harness import print_table, run_eval, summarize
from patterns.p01_pipeline.baseline import run_baseline
from patterns.p01_pipeline.graph import run_pipeline
from patterns.p01_pipeline.judge import judge_summary

DATA_PATH = Path(__file__).parent / "data" / "articles.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pattern 1.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per case and system.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only use the first N cases."
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Summarize an existing results file.",
    )
    args = parser.parse_args()

    if args.from_file:
        records = load_jsonl(args.from_file)
    else:
        dataset = load_jsonl(DATA_PATH)[: args.limit]
        records = run_eval(
            name="p01_pipeline",
            systems={
                "pipeline": lambda case: run_pipeline(case["article"]),
                "baseline": lambda case: run_baseline(case["article"]),
            },
            judge=lambda case, output: judge_summary(case["article"], output),
            dataset=dataset,
            runs=args.runs,
        )

    print()
    print_table(summarize(records))


if __name__ == "__main__":
    main()
