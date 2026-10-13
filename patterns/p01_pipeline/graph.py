from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.llm import LLMUsage
from patterns.p01_pipeline.agents import editor, extractor, writer
from patterns.p01_pipeline.state import PipelineState


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("extractor", extractor)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)

    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()


graph = build_graph()


def run_pipeline(article: str) -> tuple[str, list[LLMUsage]]:
    """Run the full pipeline on one article and return the summary plus per-agent usage."""
    final_state = graph.invoke({"article": article})
    return final_state["summary"], final_state["usage"]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from core.data import load_jsonl

    dataset = load_jsonl(Path(__file__).parent / "data" / "articles.jsonl")
    case_id = sys.argv[1] if len(sys.argv) > 1 else dataset[0]["id"]
    case = next(c for c in dataset if c["id"] == case_id)

    summary, usage = run_pipeline(case["article"])

    print(f"Case: {case['id']} ({case['length']}, {case['word_count']} words)\n")
    print(summary)
    print()
    for record in usage:
        print(record)
