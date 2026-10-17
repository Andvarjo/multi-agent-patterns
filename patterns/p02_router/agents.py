"""Agents for pattern 2: a router plus one specialist per route."""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm
from core.llm import LLMUsage
from patterns.p02_router.state import Route, RouterState

model = llm.get_model("worker")

class RouteDecision(BaseModel):
    reasoning: str = Field(description="Brief reasoning about which route fits, written before choosing.")
    route: Route = Field(description="The route that should handle the question.")


classifier = model.with_structured_output(RouteDecision, include_raw=True, method="json_schema")


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

def router(state: RouterState) -> dict:
    start = time.perf_counter()
    result = classifier.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"<question>\n{state['question']}\n</question>"),
    ])
    latency = time.perf_counter() - start

    if result["parsed"] is None:
        raise ValueError(f"Router returned an invalid decision: {result['parsing_error']}")

    usage_metadata = result["raw"].usage_metadata or {}
    usage: LLMUsage = {
        "agent": "router",
        "model": model.model,
        "input_tokens": usage_metadata.get("input_tokens", 0),
        "output_tokens": usage_metadata.get("output_tokens", 0),
        "latency_s": round(latency, 3),
    }
    return {"route": result["parsed"].route, "usage": [usage]}



if __name__ == "__main__":
    for question in [
        "How much is the Pro plan if I pay yearly?",
        "If I delete my account, do I get a refund for the rest of my annual subscription?",
        "Is Ledgerly better than FreshBooks?",
    ]:
        print(question, "->", router({"question": question}))