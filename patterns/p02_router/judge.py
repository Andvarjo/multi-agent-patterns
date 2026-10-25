from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm import get_model

KNOWLEDGE_DIR = Path(__file__).parent / "data" / "knowledge"
KNOWLEDGE_BASE = "\n\n".join(
    (KNOWLEDGE_DIR / f"{area}.md").read_text(encoding="utf-8")
    for area in ("billing", "technical", "account")
)

JUDGE_PROMPT = """You are a strict evaluator of customer support answers for Ledgerly, an \
invoicing app. You receive:
- the full Ledgerly knowledge base inside <knowledge_base> tags, which is the source of truth;
- the customer's question inside <question> tags;
- the requirements for this question inside <requirements> tags: either facts a correct answer \
must include, or the expected behavior for questions Ledgerly support should not answer;
- the answer to evaluate inside <answer> tags.

Score the answer on each criterion from 1 (poor) to 5 (excellent):
- correctness: every claim in the answer is supported by the knowledge base. Penalize \
contradictions and details that do not appear in the knowledge base. A true fact from the \
knowledge base is never a correctness error, even if it was not required. Simple arithmetic \
derived from the knowledge base (for example, a monthly equivalent of an annual price) is \
acceptable.
- completeness: the answer includes every required fact, or follows the expected behavior.
- concision: the answer addresses what was asked without unrequested information, within \
120 words (60 for out-of-scope questions). Penalize true but irrelevant facts here.

Write your reasoning first, then the scores. Be critical: a 5 means you would not change anything."""


class AnswerScore(BaseModel):
    reasoning: str = Field(
        description="Critique of the summary against each criterion, written before scoring."
    )
    completeness: int = Field(
        ge=1, le=5, description="Includes all required facts.(or in out of scope questions, follows the expected behavior.)"
    )
    correctness: int = Field(
        ge=1, le=5, description="All affirmations are consistent with their reference and don't add policies, numbers or features."
    )
    concision: int = Field(
        ge=1, le=5, description="answers the question without unnecessary information and at most x limit of words."
    )

judge = get_model("judge").with_structured_output(AnswerScore, method="json_schema")

def _reference(case: dict) -> str:
    """Return the reference for a case: the facts an answer must include, or the expected behavior."""
    if case.get("required_facts"):
        facts = "\n".join(f"- {fact}" for fact in case["required_facts"])
        return f"The answer must include:\n{facts}"
    return f"Expected behavior: {case['expected_behavior']}"

def judge_answer(case: dict, output: str, metadata: dict) -> dict:
    """Score a support answer against the case's reference, plus routing accuracy when available."""
    user_message = (
        f"<knowledge_base>\n{KNOWLEDGE_BASE}\n</knowledge_base>\n\n"
        f"<question>\n{case['question']}\n</question>\n\n"
        f"<requirements>\n{_reference(case)}\n</requirements>\n\n"
        f"<answer>\n{output}\n</answer>"
    )
    score = judge.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=user_message),
    ])
    result = score.model_dump()

    if "route" in metadata:
        result["route_correct"] = metadata["route"] in case["acceptable_routes"]
        result["route_exact"] = metadata["route"] == case["expected_route"]

    return result

if __name__ == "__main__":
    from core.data import load_jsonl
    from patterns.p02_router.graph import run_router

    dataset = load_jsonl("patterns/p02_router/data/questions.jsonl")
    case = next(c for c in dataset if c["id"] == "ambiguous-delete-refund")

    answer, usage, route = run_router(case["question"])
    result = judge_answer(case, answer, {"route": route})

    print(f"Route: {route}\n\n{answer}\n")
    print(result)