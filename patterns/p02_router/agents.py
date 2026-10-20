"""Agents for pattern 2: a router plus one specialist per route."""

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm
from core.llm import LLMUsage
from patterns.p02_router.state import Route, RouterState

model = llm.get_model("worker")

KNOWLEDGE_PATH = Path(__file__).parent / "data" / "knowledge"


class RouteDecision(BaseModel):
    reasoning: str = Field(
        description="Brief reasoning about which route fits, written before choosing."
    )
    route: Route = Field(description="The route that should handle the question.")


classifier = model.with_structured_output(
    RouteDecision, include_raw=True, method="json_schema"
)


ROUTER_PROMPT = """You are the router of Ledgerly's customer support system. Ledgerly is an \
invoicing app for freelancers and small businesses. Classify the customer's question, provided \
inside <question> tags, into exactly one route:

- billing: Ledgerly plans and prices, subscription charges, payment methods for the subscription, \
upgrades and downgrades and their effects, refunds, failed payments, VAT, discounts, and the fees \
Ledgerly charges when clients pay invoices.
- technical: how the product works: supported devices and browsers, sync, integrations \
(Stripe, PayPal, QuickBooks, Xero, Zapier), CSV import and data export, invoice emails and \
deliverability, invoice PDFs, and the API.
- account: sign-in, passwords, two-factor authentication, profile and email changes, team members \
and roles, ownership transfer, and account deletion or restoration.
- out_of_scope: anything not about using Ledgerly, such as general accounting or tax advice, \
comparisons with other products, or unrelated requests.

If a question touches several routes, choose the one whose policies are needed to answer it. \
Think briefly about the choice before giving the route."""

SPECIALIST_PROMPT = """You are Ledgerly's {area} support specialist. Ledgerly is an invoicing app \
for freelancers and small businesses. Answer the customer's question, provided inside <question> \
tags, using only the knowledge base below.

<knowledge_base>
{knowledge}
</knowledge_base>

Rules:
- Answer in at most 120 words, in a friendly and direct tone.
- Include the specific facts the customer needs: prices, limits, time frames and where to find \
settings.
- If the knowledge base does not contain the answer, say so clearly and suggest contacting \
support. Never invent policies, numbers or features.
- Return plain text only, with no XML tags, markdown or headings."""


OUT_OF_SCOPE_PROMPT = """You are Ledgerly's customer support assistant. Ledgerly is an invoicing \
app for freelancers and small businesses. The customer's question, provided inside <question> \
tags, is outside what Ledgerly support can help with.

Reply in at most 60 words: politely explain that you can only help with questions about using \
Ledgerly, and do not attempt to answer the question itself. Return plain text only."""


def router(state: RouterState) -> dict:
    start = time.perf_counter()
    result = classifier.invoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"<question>\n{state['question']}\n</question>"),
        ]
    )
    latency = time.perf_counter() - start

    if result["parsed"] is None:
        raise ValueError(
            f"Router returned an invalid decision: {result['parsing_error']}"
        )

    usage_metadata = result["raw"].usage_metadata or {}
    usage: LLMUsage = {
        "agent": "router",
        "model": model.model,
        "input_tokens": usage_metadata.get("input_tokens", 0),
        "output_tokens": usage_metadata.get("output_tokens", 0),
        "latency_s": round(latency, 3),
    }
    return {"route": result["parsed"].route, "usage": [usage]}


def make_specialist(area: str):
    """Build a specialist node that answers questions using only the knowledge base for `area`."""
    knowledge = (KNOWLEDGE_PATH / f"{area}.md").read_text(encoding="utf-8")
    system_prompt = SPECIALIST_PROMPT.format(area=area, knowledge=knowledge)
    agent_name = f"{area}_specialist"

    def specialist(state: RouterState) -> dict:
        user_message = f"<question>\n{state['question']}\n</question>"
        answer, usage = llm.call_llm(model, system_prompt, user_message, agent_name)
        return {"answer": answer, "usage": [usage]}

    return specialist


def out_of_scope(state: RouterState) -> dict:
    user_message = f"<question>\n{state['question']}\n</question>"
    answer, usage = llm.call_llm(
        model, OUT_OF_SCOPE_PROMPT, user_message, "out_of_scope"
    )
    return {"answer": answer, "usage": [usage]}


billing = make_specialist("billing")
technical = make_specialist("technical")
account = make_specialist("account")

if __name__ == "__main__":
    for question in [
        "How much is the Pro plan if I pay yearly?",
        "If I delete my account, do I get a refund for the rest of my annual subscription?",
        "Is Ledgerly better than FreshBooks?",
    ]:
        print(question, "->", router({"question": question}))

    question = "How much is the Pro plan if I pay yearly?"
    print(billing({"question": question}))
