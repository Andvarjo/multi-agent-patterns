from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from patterns.p01_pipeline.state import PipelineState
from patterns.p01_pipeline.agents import extractor, writer, editor
from core.llm import LLMUsage



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
    sample = (
        "We migrated our API from REST to gRPC. p95 latency dropped from 120 ms to 45 ms, "
        "and payload size fell by 60%. The main cost was rewriting three client SDKs, "
        "which took the team six weeks."
    )
    summary, usage = run_pipeline(sample)

    print(summary)
    print()
    for record in usage:
        print(record)
    print()
    print(graph.get_graph().draw_mermaid())