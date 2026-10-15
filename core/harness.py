"""Generic evaluation harness shared by all patterns.

The harness knows nothing about pipelines or summaries. It only needs:
- a System: takes a dataset case and returns (output, per-agent usage).
- a Judge: takes the case and the output and returns a dict of scores.
"""

import json
import statistics
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


def summarize(records: list[dict]) -> list[dict]:
    """Aggregate records per system: mean scores, score spread across runs, tokens, latency, errors."""
    rows = []
    for system_name in dict.fromkeys(r["system"] for r in records):
        system_records = [r for r in records if r["system"] == system_name]
        ok = [r for r in system_records if r["error"] is None and r["judge_error"] is None]

        criteria = [k for k, v in ok[0]["scores"].items() if isinstance(v, int)] if ok else []
        row: dict = {"system": system_name}
        for criterion in criteria:
            row[criterion] = float(statistics.mean(r["scores"][criterion] for r in ok))

        per_run_avg = [statistics.mean(r["scores"][c] for c in criteria) for r in ok] if criteria else []
        row["avg_score"] = statistics.mean(per_run_avg) if per_run_avg else 0.0
        row["score_std"] = statistics.stdev(per_run_avg) if len(per_run_avg) > 1 else 0.0
        row["llm_calls"] = float(statistics.mean(len(r["usage"]) for r in ok)) if ok else 0.0
        row["input_tokens"] = float(statistics.mean(r["input_tokens"] for r in ok)) if ok else 0.0
        row["output_tokens"] = float(statistics.mean(r["output_tokens"] for r in ok)) if ok else 0.0
        row["latency_s"] = statistics.mean(r["latency_s"] for r in ok) if ok else 0.0
        row["runs"] = len(system_records)
        row["errors"] = len(system_records) - len(ok)
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    """Print the per-system summary as an aligned text table."""
    if not rows:
        print("No results.")
        return
    # Union of columns across systems, in first-seen order (a fully failed system has no scores).
    columns = list(dict.fromkeys(c for row in rows for c in row if c != "system"))
    print("system".ljust(12) + "".join(c.rjust(14) for c in columns))
    for row in rows:
        cells = []
        for c in columns:
            value = row.get(c, "-")
            cells.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        print(row["system"].ljust(12) + "".join(cell.rjust(14) for cell in cells))