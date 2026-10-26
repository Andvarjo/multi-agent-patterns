"""State for pattern 2: a router sends each question to one specialist agent."""

import operator
from typing import Annotated, Literal, TypedDict

from core.llm import LLMUsage

Route = Literal["billing", "technical", "account", "out_of_scope"]


class RouterState(TypedDict, total=False):
    question: str
    route: Route
    answer: str
    usage: Annotated[list[LLMUsage], operator.add]
