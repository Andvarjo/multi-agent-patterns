"""LLM helpers shared across all patterns."""

import os
import time
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class LLMUsage(TypedDict):
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


ModelRole = Literal["worker", "judge", "router"]

_ENV_VARS: dict[ModelRole, str] = {
    "worker": "WORKER_MODEL",
    "judge": "JUDGE_MODEL",
    "router": "ROUTER_MODEL"
}


def get_model(role: ModelRole = "worker") -> ChatAnthropic:
    """Build the model for the given role, reading its name from the environment.

    "worker" is used by the agents under test; "judge" is a stronger model used to evaluate them.
    """
    return ChatAnthropic(model=os.environ[_ENV_VARS[role]])


def _to_text(content: str | list) -> str:
    """Anthropic responses can be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def call_llm(
    model: ChatAnthropic,
    system_prompt: str,
    user_message: str,
    agent_name: str,
) -> tuple[str, LLMUsage]:
    """Invoke the model and return the response text plus usage metrics for this agent."""
    start = time.perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )
    latency = time.perf_counter() - start

    usage = response.usage_metadata or {}
    metrics: LLMUsage = {
        "agent": agent_name,
        "model": model.model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_s": round(latency, 3),
    }
    return _to_text(response.content).strip(), metrics


if __name__ == "__main__":
    text, metrics = call_llm(
        model=get_model("worker"),
        system_prompt="You are a concise technical assistant.",
        user_message="Explain what a multi-agent system is in one sentence.",
        agent_name="smoke_test",
    )
    print(text)
    print(metrics)
