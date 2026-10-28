"""State and schemas for the review analyzer."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from core.llm import LLMUsage

Sentiment = Literal["positive", "negative", "mixed", "neutral"]
Polarity = Literal["positive", "negative"]
Aspect = Literal[
    "battery", "sound", "comfort", "connectivity", "price", "durability", "shipping"
]


class AspectMention(BaseModel):
    aspect: Aspect
    polarity: Polarity


class ReviewExtraction(BaseModel):
    """What the model extracts from one review. The id is attached in code."""

    sentiment: Sentiment
    aspects: list[AspectMention] = Field(
        description="Each aspect the review evaluates, once per aspect."
    )


class IdentifiedReviewExtraction(ReviewExtraction):
    """Extraction with its review id, needed when one call sees many reviews."""

    review_id: str


class BatchExtraction(BaseModel):
    """Baseline output: one extraction per review, in a single call."""

    extractions: list[IdentifiedReviewExtraction]


class ReviewInput(TypedDict):
    """Send payload for one fan-out branch."""

    product: str
    review_id: str
    text: str


class FanoutState(TypedDict, total=False):
    product: str
    reviews: list[dict]  # {"id": str, "text": str}
    extractions: Annotated[list[dict], operator.add]  # IdentifiedReviewExtraction dumps
    aspect_counts: dict[Aspect, dict[Polarity, int]]
    sentiment_counts: dict[Sentiment, int]
    summary: str
    usage: Annotated[list[LLMUsage], operator.add]