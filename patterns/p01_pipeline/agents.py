from core import llm
from patterns.p01_pipeline.state import PipelineState

model = llm.get_model()


EXTRACTOR_PROMPT = """You are the first step in a summarization pipeline. Your output is not shown \
to the end user: it is the input for a writer agent that will turn it into a one-paragraph summary.

Extract the key points from the article provided inside <article> tags.

Output format:
- A bulleted list of 4 to 7 points, one line each.
- Return only the list, with no introduction or closing remarks.

What counts as a key point:
- A concrete idea, decision, result, number, or claim stated in the article.
- Prefer points that carry specific facts (figures, comparisons, outcomes) over generic statements.
- Keep numbers and units exactly as written.

Constraints:
- Use only information explicitly stated in the article. Do not infer, interpret, or add context.
- If the article contains fewer than 4 relevant points, return only the ones that exist."""

WRITER_PROMPT = """You are the second step in a summarization pipeline. You receive key points \
extracted from a technical article, inside <key_points> tags, and write a draft summary that an \
editor agent will polish.

Output format:
- One paragraph of at most 120 words, for a senior engineering audience.
- Return only the paragraph, with no introduction or closing remarks.

How to write it:
- Connect the ideas logically (cause and effect, problem and solution, decision and outcome) \
instead of listing them as separate sentences.
- Keep numbers and units exactly as given.
- If not all points fit within the word limit, prioritize the most important ones and drop minor details.

Constraints:
- Use only the information in the key points. Do not add facts, context, or opinions."""


EDITOR_PROMPT = """You are the final step in a summarization pipeline. You receive a draft summary \
inside <draft> tags and return the final version.

Your job is to improve clarity and concision:
- Remove filler, hedging, and repetition.
- Make sentences more direct and the flow between them smoother.
- Keep the result to one paragraph of at most 120 words.

Constraints:
- Do not add, remove, or change any facts, numbers, or claims.
- If the draft is already clear and concise, return it with minimal changes.
- Return only the final paragraph, with no introduction or comments about your edits."""


def extractor(state: PipelineState) -> dict:
    article = state["article"]
    user_message = f"<article>\n{article}\n</article>"
    response, usage = llm.call_llm(model, EXTRACTOR_PROMPT, user_message, "extractor")
    return {"key_points": response, "usage": [usage]}


def writer(state: PipelineState) -> dict:
    key_points = state["key_points"]
    user_message = f"<key_points>\n{key_points}\n</key_points>"
    response, usage = llm.call_llm(model, WRITER_PROMPT, user_message, "writer")
    return {"draft": response, "usage": [usage]}


def editor(state: PipelineState) -> dict:
    draft = state["draft"]
    user_message = f"<draft>\n{draft}\n</draft>"
    response, usage = llm.call_llm(model, EDITOR_PROMPT, user_message, "editor")
    return {"summary": response, "usage": [usage]}


if __name__ == "__main__":
    sample = (
        "We migrated our API from REST to gRPC. p95 latency dropped from 120 ms to 45 ms, "
        "and payload size fell by 60%. The main cost was rewriting three client SDKs, "
        "which took the team six weeks."
    )
    result = extractor({"article": sample})
    print(result["key_points"])
    print(result["usage"])
