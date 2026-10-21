"""Pattern 2 — Router: one classifier sends each question to a single specialist."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.llm import LLMUsage
from patterns.p02_router.agents import account, billing, out_of_scope, router, technical
from patterns.p02_router.state import Route, RouterState

SPECIALISTS = ("billing", "technical", "account", "out_of_scope")


def select_route(state: RouterState) -> Route:
    """Read the router's decision. No LLM call here: the decision was already made."""
    return state["route"]


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(RouterState)

    builder.add_node("router", router)
    builder.add_node("billing", billing)
    builder.add_node("technical", technical)
    builder.add_node("account", account)
    builder.add_node("out_of_scope", out_of_scope)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", select_route, {name: name for name in SPECIALISTS})
    for name in SPECIALISTS:
        builder.add_edge(name, END)

    return builder.compile()


graph = build_graph()


def run_router(question: str) -> tuple[str, list[LLMUsage], Route]:
    """Run the router system on one question and return (answer, per-agent usage, chosen route)."""
    final_state = graph.invoke({"question": question})
    return final_state["answer"], final_state["usage"], final_state["route"]


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    answer, usage, route = run_router("If I delete my account, do I get a refund?")
    print(f"Route: {route}\n")
    print(answer)
    print()
    for record in usage:
        print(record)