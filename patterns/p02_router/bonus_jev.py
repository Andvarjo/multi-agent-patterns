"""Bonus for pattern 2: LLM router vs. Jev, a structured decision model, on the routing step only.

Both routers classify the same 30 questions. No specialists and no judge: the specialists are
identical in both setups, so only the routing decision changes, and routing is checked
deterministically against the dataset.

Usage (from the repo root):
    uv run python -m patterns.p02_router.bonus_jev
    uv run python -m patterns.p02_router.bonus_jev --limit 5
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from typesafe_sdk import Choice, Questions, TypeSafeClient

from core.data import load_jsonl
from patterns.p02_router.agents import router  # importing core.llm also loads .env

DATA_PATH = Path(__file__).parent / "data" / "questions.jsonl"
JEV_MODEL = "jev-1.13.0"  # pinned: confidence values can shift between releases
LOW_CONFIDENCE = 0.6

# Same route descriptions and tie-breaking rule as ROUTER_PROMPT, so the comparison is fair.
JEV_QUESTIONS: Questions = {
    "route": Choice(
        instructions=(
            "Which support route of Ledgerly, an invoicing app for freelancers and small "
            "businesses, should handle the customer's `question`? If the question touches several "
            "routes, choose the one whose policies are needed to answer it."
        ),
        criteria={
            "billing": (
                "Ledgerly plans and prices, subscription charges, payment methods for the "
                "subscription, upgrades and downgrades and their effects, refunds, failed payments, "
                "VAT, discounts, and the fees Ledgerly charges when clients pay invoices."
            ),
            "technical": (
                "How the product works: supported devices and browsers, sync, integrations (Stripe, "
                "PayPal, QuickBooks, Xero, Zapier), CSV import and data export, invoice emails and "
                "deliverability, invoice PDFs, and the API."
            ),
            "account": (
                "Sign-in, passwords, two-factor authentication, profile and email changes, team "
                "members and roles, ownership transfer, and account deletion or restoration."
            ),
            "out_of_scope": (
                "Anything not about using Ledgerly, such as general accounting or tax advice, "
                "comparisons with other products, or unrelated requests."
            ),
        },
    ),
}


def route_with_llm(question: str) -> dict:
    """Reuse the pattern's router node, so this is exactly the router that was evaluated."""
    result = router({"question": question})
    usage = result["usage"][0]
    return {
        "route": result["route"],
        "latency_s": usage["latency_s"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def route_with_jev(client: TypeSafeClient, question: str) -> dict:
    start = time.perf_counter()
    response = client.system_one(state={"question": question}, questions=JEV_QUESTIONS)
    latency = time.perf_counter() - start

    answer = response.choices["route"]
    return {
        "route": answer.choice,
        "confidence": round(answer.confidence, 3),
        "probabilities": {k: round(v, 3) for k, v in answer.probabilities.items()},
        "latency_s": round(latency, 3),
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
        "model": response.model,
    }


def summarize(records: list[dict]) -> None:
    print(f"\n{'router':<8}{'exact':>8}{'acceptable':>12}{'p50 latency':>14}{'mean latency':>14}"
          f"{'in tokens':>11}{'out tokens':>12}")
    for name in ("llm", "jev"):
        rows = [r[name] for r in records]
        exact = statistics.mean(r[name]["route"] == r["expected_route"] for r in records)
        acceptable = statistics.mean(r[name]["route"] in r["acceptable_routes"] for r in records)
        latencies = [x["latency_s"] for x in rows]
        print(f"{name:<8}{exact:>8.0%}{acceptable:>12.0%}{statistics.median(latencies):>13.2f}s"
              f"{statistics.mean(latencies):>13.2f}s"
              f"{statistics.mean(x['input_tokens'] for x in rows):>11.0f}"
              f"{statistics.mean(x['output_tokens'] for x in rows):>12.0f}")

    agreement = statistics.mean(r["llm"]["route"] == r["jev"]["route"] for r in records)
    print(f"\nAgreement between routers: {agreement:.0%}")

    correct = [r["jev"]["confidence"] for r in records if r["jev"]["route"] == r["expected_route"]]
    wrong = [r["jev"]["confidence"] for r in records if r["jev"]["route"] != r["expected_route"]]
    print(f"Jev mean confidence when exact: {statistics.mean(correct):.2f}" if correct else "")
    print(f"Jev mean confidence when not exact: {statistics.mean(wrong):.2f}" if wrong else
          "Jev was exact on every question.")

    print("\nDisagreements and low-confidence Jev decisions:")
    for r in records:
        if r["llm"]["route"] != r["jev"]["route"] or r["jev"]["confidence"] < LOW_CONFIDENCE:
            print(f"  {r['case_id']}: expected={r['expected_route']} llm={r['llm']['route']} "
                  f"jev={r['jev']['route']} (confidence {r['jev']['confidence']:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the LLM router with Jev.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N questions.")
    args = parser.parse_args()

    dataset = load_jsonl(DATA_PATH)[: args.limit]
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"p02_bonus_jev_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    records = []
    with TypeSafeClient(model=JEV_MODEL) as client, out_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(dataset, start=1):
            # Alternate which router goes first, so neither always benefits from warm connections.
            if i % 2:
                llm, jev = route_with_llm(case["question"]), route_with_jev(client, case["question"])
            else:
                jev, llm = route_with_jev(client, case["question"]), route_with_llm(case["question"])

            record = {
                "case_id": case["id"],
                "expected_route": case["expected_route"],
                "acceptable_routes": case["acceptable_routes"],
                "llm": llm,
                "jev": jev,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(f"[{i}/{len(dataset)}] {case['id']}: llm={llm['route']} ({llm['latency_s']}s) "
                  f"jev={jev['route']} ({jev['latency_s']}s, confidence {jev['confidence']})")

    summarize(records)
    print(f"\nRaw results: {out_path}")


if __name__ == "__main__":
    main()