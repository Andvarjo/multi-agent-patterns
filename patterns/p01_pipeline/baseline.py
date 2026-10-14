from pathlib import Path

from core.llm import LLMUsage, call_llm, get_model

ARTICLES_PATH = Path(__file__).parent / "data" / "articles.jsonl"

BASELINE_PROMPT = """You summarize technical articles. The article is provided inside <article> tags.

First identify the key ideas, decisions, results, and numbers in the article. Then write the summary.

Output format:
- One paragraph of at most 120 words, for a senior engineering audience.
- Return only the paragraph, with no introduction or closing remarks.

How to write it:
- Connect the ideas logically (cause and effect, problem and solution, decision and outcome).
- Keep numbers and units exactly as written.
- If not everything fits within the word limit, prioritize the most important points.
- Be direct: no filler, hedging, or repetition.

Constraints:
- Use only information explicitly stated in the article. Do not add facts, context, or opinions."""


def run_baseline(article: str) -> tuple[str, list[LLMUsage]]:
    """Run the baseline to compare"""
    user_message = f"<article>\n{article}\n</article>"
    text, metrics = call_llm(
        model=get_model("worker"),
        system_prompt=BASELINE_PROMPT,
        user_message=user_message,
        agent_name="baseline",
    )
    return text, [metrics]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from core.data import load_jsonl

    dataset = load_jsonl(Path(__file__).parent / "data" / "articles.jsonl")
    case_id = sys.argv[1] if len(sys.argv) > 1 else dataset[0]["id"]
    case = next(c for c in dataset if c["id"] == case_id)

    summary, usage = run_baseline(case["article"])

    print(f"Case: {case['id']} ({case['length']}, {case['word_count']} words)\n")
    print(summary)
    print()
    for record in usage:
        print(record)
