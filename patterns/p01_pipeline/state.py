import operator
from typing import Annotated, TypedDict

from core.llm import LLMUsage

"""
Main Idea behind this multi agent pipeline is to extract key points from a given article, 
generate a draft based on those key points, and then produce a final summary.
"""


class PipelineState(
    TypedDict, total=False
):  # total=False as first step will not have all fields only the article
    article: str
    key_points: str
    draft: str
    summary: str
    usage: Annotated[list[LLMUsage], operator.add]
