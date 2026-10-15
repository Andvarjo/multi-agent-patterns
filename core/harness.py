"""Generic evaluation harness shared by all patterns.

The harness knows nothing about pipelines or summaries. It only needs:
- a System: takes a dataset case and returns (output, per-agent usage).
- a Judge: takes the case and the output and returns a dict of scores.
"""

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from core.llm import LLMUsage

System = Callable[[dict], tuple[str, list[LLMUsage]]]
Judge = Callable[[dict, str], dict]


def run_one(
    case: dict,
    system_name: str,
    system: System,
    judge: Judge,
    run_index: int,
) -> dict:
    """Run one system on one case once, judge the output, and return a result record."""
    start = time.perf_counter()
    try:
        output, usage = system(case)
        error = None
    except Exception as e:  # a failed run is a result, not a crash
        output, usage, error = "", [], repr(e)
    latency = time.perf_counter() - start

    scores: dict = {}
    judge_error = None
    if error is None:
        try:
            scores = judge(case, output)
        except Exception as e:
            judge_error = repr(e)

    return {
        "case_id": case["id"],
        "system": system_name,
        "run": run_index,
        "output": output,
        "scores": scores,
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "latency_s": round(latency, 3),
        "usage": usage,
        "error": error,
        "judge_error": judge_error,
    }


def run_eval(
    name: str,
    systems: dict[str, System],
    judge: Judge,
    dataset: list[dict],
    runs: int = 3,
    out_dir: Path = Path("results"),
) -> list[dict]:
    """Run every system on every case `runs` times, saving each record to JSONL as it finishes.

    Loop order is case -> run -> system, so systems alternate and share the same API conditions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    total = len(dataset) * runs * len(systems)
    records: list[dict] = []
    print(f"Running {total} executions -> {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for case in dataset:
            for run_index in range(runs):
                for system_name, system in systems.items():
                    record = run_one(case, system_name, system, judge, run_index)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    records.append(record)

                    status = ""
                    if record["error"]:
                        status = f"  SYSTEM ERROR: {record['error']}"
                    elif record["judge_error"]:
                        status = f"  JUDGE ERROR: {record['judge_error']}"
                    print(
                        f"[{len(records)}/{total}] {system_name:<10} {case['id']} "
                        f"run {run_index}: {record['latency_s']}s{status}"
                    )

    print(f"Done. Results saved to {out_path}")
    return records