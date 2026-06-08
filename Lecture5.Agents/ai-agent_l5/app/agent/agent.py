import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from .semantic_scraper import extract_offers
from .tools_ext import compute_stats, filter_offers, normalize_offers_currency

_SESSION_STATE = {"offers": []}

SYSTEM = (
    """You are a multi-purpose housing price monitoring agent.
    If the user provides a URL, first extract listings (title, price, currency, url).
    Then, based on the question, choose the right actions: compute statistics, filter by price/keywords,
    Always call tools with named arguments strictly according to their JSON schema.
    If 'offers' is not specified, the tool will use the last extracted listings.
    Normalize currency (to RUB, USD, EUR), compare multiple links, compile a summary report.
    Answer briefly. When needed, show 3-5 example links."""
)


@observe(name="extract_offers")
def _extract_offers_tool(url: str, limit: int = 50) -> list[dict]:
    res = extract_offers(url, limit=limit)
    out = [o.dict() for o in res.offers]
    _SESSION_STATE["offers"] = out
    return out


@observe(name="filter_offers")
def _filter_offers_tool(offers: Optional[List[Dict[str, Any]]] = None, min_price: Optional[int] = None,
                        max_price: Optional[int] = None,
                        text_contains: str = "") -> List[dict]:
    if offers is None:
        offers = _SESSION_STATE.get("offers", [])
    return filter_offers(offers, min_price, max_price, text_contains)


@observe(name="normalize_currency")
def _normalize_offers_currency_tool(offers: Optional[List[Dict[str, Any]]] = None,
                                    target_currency: str = "RUB") -> List[dict]:
    if offers is None:
        offers = _SESSION_STATE.get("offers", [])
    return normalize_offers_currency(offers, target_currency)


@observe(name="compute_stats")
def _compute_stats_tool(prices: Optional[List[int]] = None,
                        offers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if prices is None and offers is None:
        offers = _SESSION_STATE.get("offers", [])
    if prices is None and offers is not None:
        prices = [int(o.get("price", 0)) for o in offers]
    return compute_stats(prices or [])


class ExtractOffersArgs(BaseModel):
    url: str = Field(..., description="Page with listings")
    limit: int = Field(50, description="Maximum number of listings")


class FilterOffersArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of listings {title, price, currency, url}. If not provided, use the last extracted ones.")
    min_price: Optional[int] = Field(None, description="Minimum price")
    max_price: Optional[int] = Field(None, description="Maximum price")
    text_contains: Optional[str] = Field(
        "", description="Substring in the title, e.g. 'studio'")


class NormalizeCurrencyArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of listings. If not provided, use the last extracted ones.")
    target_currency: str = Field(
        "RUB", description="Target currency: RUB|USD|EUR")


class ComputeStatsArgs(BaseModel):
    prices: Optional[List[int]] = Field(None, description="List of prices")
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Alternatively pass listings instead of a price list")


tools = [
    StructuredTool.from_function(
        func=_extract_offers_tool,
        name="extract_offers",
        args_schema=ExtractOffersArgs,
        description="Extract listings (title, price, currency, url) from the given page."
    ),
    StructuredTool.from_function(
        func=_compute_stats_tool,
        name="compute_stats",
        args_schema=ComputeStatsArgs,
        description="Compute statistics for a list of prices or listings: {min,max,avg,median}."
    ),
    StructuredTool.from_function(
        func=_filter_offers_tool,
        name="filter_offers",
        args_schema=FilterOffersArgs,
        description="Filter listings by min_price/max_price and/or by a substring in the title."
    ),
    StructuredTool.from_function(
        func=_normalize_offers_currency_tool,
        name="normalize_offers_currency",
        args_schema=NormalizeCurrencyArgs,
        description="Convert listing prices to a target currency (RUB|USD|EUR)."
    ),
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "User task: {question}\n"
     "Links (may be empty): {urls}\n"
     "Action plan:"
     "1) If there are links, extract listings from each page (extract_offers)."
     "2) If asked to filter, use filter_offers."
     "3) If asked for a different currency, use normalize_offers_currency."
     "4) If statistics are needed, use compute_stats."
     "5) Compile a brief report and include 3-5 example links."
     "If there is no data, say so explicitly."),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOpenAI(model=os.getenv(
    "OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=False,
    return_intermediate_steps=True,
)


@observe(name="run_question")
def run_question(question: str, urls: list[str], max_items: int = 50,
                 session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    langfuse_client = get_client()

    handler = CallbackHandler()

    config = {
        "callbacks": [handler],
        "metadata": {},
        "tags": []
    }

    # Add session_id and user_id to metadata for LangSmith
    if session_id:
        config["metadata"]["session_id"] = session_id
        config["tags"].append(f"session:{session_id}")
    if user_id:
        config["metadata"]["user_id"] = user_id
        config["tags"].append(f"user:{user_id}")

    # Start the trace/observation
    with langfuse_client.start_as_current_observation(
        as_type="span", name="agent_execution"
    ):
        # Propagate session_id and user_id (if provided)
        attrs: Dict[str, str] = {}
        if session_id:
            attrs["session_id"] = session_id
        if user_id:
            attrs["user_id"] = user_id

        if attrs:
            with propagate_attributes(**attrs):
                result = executor.invoke(
                    {"question": question, "urls": urls, "max_items": max_items},
                    config=config
                )
        else:
            result = executor.invoke(
                {"question": question, "urls": urls, "max_items": max_items},
                config=config
            )
            # At this point result contains both the output and intermediate steps
    tools_used: List[str] = []
    intermediate_steps = result.get("intermediate_steps") or []
    for step in intermediate_steps:
        # Each step has the format (AgentAction, tool_output)
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name:
                tools_used.append(tool_name)
        except Exception:
            continue

    return {
        "output": result.get("output"),
        "tools_used": tools_used,
    }
