"""Level 2 evaluation: router + specialists vs single-agent baseline.

Usage (from the repo root):
    uv run python -m patterns.p02_router.run                     # full eval, 3 runs per case
    uv run python -m patterns.p02_router.run --limit 3 --runs 1  # quick smoke test
    uv run python -m patterns.p02_router.run --from-file results/p02_router_....jsonl
"""

import argparse
from pathlib import Path

from core.data import load_jsonl
from core.harness import print_table, run_eval, summarize
from core.llm import LLMUsage
from patterns.p02_router.baseline import run_baseline
from patterns.p02_router.graph import run_router
from patterns.p02_router.judge import judge_answer

DATA_PATH = Path(__file__).parent / "data" / "questions.jsonl"


def router_system(case: dict) -> tuple[str, list[LLMUsage], dict]:
    """Adapt run_router to the harness contract, reporting the chosen route as metadata."""
    answer, usage, route = run_router(case["question"])
    return answer, usage, {"route": route}


def baseline_system(case: dict) -> tuple[str, list[LLMUsage], dict]:
    """Adapt run_baseline to the harness contract. The baseline does not route."""
    answer, usage = run_baseline(case["question"])
    return answer, usage, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pattern 2.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per case and system.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N cases.")
    parser.add_argument("--from-file", type=Path, default=None, help="Summarize an existing results file.")
    args = parser.parse_args()

    if args.from_file:
        records = load_jsonl(args.from_file)
    else:
        dataset = load_jsonl(DATA_PATH)[: args.limit]
        records = run_eval(
            name="p02_router",
            systems={"router": router_system, "baseline": baseline_system},
            judge=judge_answer,
            dataset=dataset,
            runs=args.runs,
        )

    print()
    print_table(summarize(records))


if __name__ == "__main__":
    main()