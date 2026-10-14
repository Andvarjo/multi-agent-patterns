"""LLM-as-judge for pattern 1: scores a summary against its source article."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm import get_model

JUDGE_PROMPT = """You are a strict evaluator of technical summaries. You receive the original \
article inside <article> tags and a summary inside <summary> tags.

Score the summary on each criterion from 1 (poor) to 5 (excellent):
- fidelity: every fact, number, and claim in the summary is supported by the article. \
Penalize invented details, distorted nuance (for example, "slightly" becoming "significantly"), \
and claims attributed to the wrong source.
- coverage: the summary captures the most important ideas of the article.
- clarity: a senior engineer can understand it on a first read.
- concision: no filler or repetition, and it respects a limit of 120 words.

Write your reasoning first, then the scores. Be critical: a 5 means you would not change anything."""


class SummaryScore(BaseModel):
    reasoning: str = Field(
        description="Critique of the summary against each criterion, written before scoring."
    )
    fidelity: int = Field(
        ge=1,
        le=5,
        description="All facts, numbers and nuances are supported by the article.",
    )
    coverage: int = Field(
        ge=1, le=5, description="The most important ideas of the article are included."
    )
    clarity: int = Field(
        ge=1, le=5, description="A senior engineer can understand it on a first read."
    )
    concision: int = Field(
        ge=1, le=5, description="No filler or repetition, and at most 120 words."
    )


judge = get_model("judge").with_structured_output(SummaryScore, method="json_schema")


def judge_summary(article: str, summary: str) -> dict:
    """Score a summary against its source article. Returns reasoning plus four 1-5 scores."""
    user_message = (
        f"<article>\n{article}\n</article>\n\n<summary>\n{summary}\n</summary>"
    )
    score = judge.invoke(
        [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    return score.model_dump()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from core.data import load_jsonl
    from patterns.p01_pipeline.baseline import run_baseline
    from patterns.p01_pipeline.graph import run_pipeline

    dataset = load_jsonl(Path(__file__).parent / "data" / "articles.jsonl")
    case_id = sys.argv[1] if len(sys.argv) > 1 else dataset[0]["id"]
    case = next((c for c in dataset if c["id"] == case_id), None)
    if case is None:
        available = ", ".join(c["id"] for c in dataset)
        sys.exit(f"Unknown case id '{case_id}'. Available: {available}")

    for name, system in [("pipeline", run_pipeline), ("baseline", run_baseline)]:
        summary, _ = system(case["article"])
        result = judge_summary(case["article"], summary)

        print(f"===== {name} =====")
        print(summary)
        print()
        print(f"Reasoning: {result['reasoning']}")
        scores = {k: v for k, v in result.items() if k != "reasoning"}
        print(f"Scores: {scores}")
        print()
