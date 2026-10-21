from pathlib import Path

from core import llm
from core.llm import LLMUsage

model = llm.get_model("worker")

KNOWLEDGE_PATH = Path(__file__).parent / "data" / "knowledge"

BASELINE_PROMPT = """You are Ledgerly's customer support assistant. Ledgerly is an invoicing app \
for freelancers and small businesses. Answer the customer's question, provided inside <question> \
tags, using only the knowledge bases below.

<billing_knowledge_base>
{billing}
</billing_knowledge_base>

<technical_knowledge_base>
{technical}
</technical_knowledge_base>

<account_knowledge_base>
{account}
</account_knowledge_base>

Rules:
- Answer in at most 120 words, in a friendly and direct tone.
- Include the specific facts the customer needs: prices, limits, time frames and where to find \
settings.
- If the knowledge bases do not contain the answer, say so clearly and suggest contacting \
support. Never invent policies, numbers or features.
- If the question is not about using Ledgerly (for example general accounting or tax advice, \
comparisons with other products, or unrelated requests), reply in at most 60 words: politely \
explain that you can only help with questions about using Ledgerly, and do not attempt to answer it.
- Return plain text only, with no XML tags, markdown or headings."""



AREAS = ("billing", "technical", "account")

knowledge = {
    area: (KNOWLEDGE_PATH / f"{area}.md").read_text(encoding="utf-8")
    for area in AREAS
}
SYSTEM_PROMPT = BASELINE_PROMPT.format(**knowledge)


def run_baseline(question: str) -> tuple[str, list[LLMUsage]]:
    user_message = f"<question>\n{question}\n</question>"
    answer, usage = llm.call_llm(model, SYSTEM_PROMPT, user_message, "baseline")
    return answer, [usage]

if __name__ == "__main__":
    answer, usage = run_baseline("If I delete my account, do I get a refund?")

    print(answer)
    print()
    for record in usage:
        print(record)