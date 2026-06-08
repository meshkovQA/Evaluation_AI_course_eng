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
from contextvars import ContextVar
from threading import Lock

# Import for web search via OpenAI Responses API
from openai import OpenAI

_openai_client = OpenAI()

# Per-session state storage
_SESSION_STATES: Dict[str, Dict[str, Any]] = {}
_SESSION_LOCK = Lock()

# Context variable for passing session_id to tools
_current_session_id: ContextVar[Optional[str]] = ContextVar('current_session_id', default=None)


def _get_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get the state for the current session"""
    sid = session_id or _current_session_id.get()
    if not sid:
        sid = "__default__"

    with _SESSION_LOCK:
        if sid not in _SESSION_STATES:
            _SESSION_STATES[sid] = {"offers": []}
        return _SESSION_STATES[sid]


def _set_session_offers(offers: List[Dict[str, Any]], session_id: Optional[str] = None):
    """Save offers for the current session"""
    state = _get_session_state(session_id)
    state["offers"] = offers

SYSTEM = (
    """You are a multi-purpose housing price monitoring agent.

IMPORTANT: You have memory of previously extracted listings within the session.
If the user asks a follow-up question (e.g. "which of them...", "show more...", "filter...")
and does NOT provide a URL, use previously extracted data by calling filter_offers or compute_stats WITHOUT the offers parameter.

Rules:
1) If the user provides a URL, extract listings via extract_offers
2) If the user asks about "these apartments", "among them", "previously found" —
   just call filter_offers/compute_stats without offers, they will use data from memory
3) If asked to filter — call filter_offers with the appropriate parameters
4) If asked to convert currency — normalize_offers_currency
5) If statistics are needed — compute_stats
6) If the user asks about apartment locations, neighborhoods, infrastructure,
   transportation, proximity to the city center, neighborhood safety, or other info
   that is not in the listings — use web_search to look it up online.

IMPORTANT for web_search:
- You MUST include CONTEXT from the current conversation in the search query!
- If the user asks about found apartments, include in the query:
  * City/country from the URL or listings (e.g. "New York", "London")
  * Specific addresses/streets from the found apartments
  * The essence of the user's question
- Examples:
  * Apartments in NYC, question "which are closer to the center?" -> query: "East 53rd street New York Manhattan distance to center"
  * Apartments in London, question "where is the infrastructure better?" -> query: "Baker Street London infrastructure metro shops"
- NEVER make queries without context! Query "apartments closer to the center" is BAD — no link to specific addresses

Always call tools with named arguments strictly according to their JSON schema.
Answer briefly and in the same language as the user's question. When needed, show 3-5 example links."""
)


@observe(name="extract_offers")
def _extract_offers_tool(url: str, limit: int = 50) -> list[dict]:
    res = extract_offers(url, limit=limit)
    out = [o.dict() for o in res.offers]
    # Save to the current session state
    _set_session_offers(out)
    return out


@observe(name="filter_offers")
def _filter_offers_tool(offers: Optional[List[Dict[str, Any]]] = None, min_price: Optional[int] = None,
                        max_price: Optional[int] = None,
                        text_contains: str = "") -> List[dict]:
    if offers is None:
        # Get from the current session state
        offers = _get_session_state().get("offers", [])
    return filter_offers(offers, min_price, max_price, text_contains)


@observe(name="normalize_currency")
def _normalize_offers_currency_tool(offers: Optional[List[Dict[str, Any]]] = None,
                                    target_currency: str = "RUB") -> List[dict]:
    if offers is None:
        # Get from the current session state
        offers = _get_session_state().get("offers", [])
    return normalize_offers_currency(offers, target_currency)


@observe(name="compute_stats")
def _compute_stats_tool(prices: Optional[List[int]] = None,
                        offers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if prices is None and offers is None:
        # Get from the current session state
        offers = _get_session_state().get("offers", [])
    if prices is None and offers is not None:
        prices = [int(o.get("price", 0)) for o in offers]
    return compute_stats(prices or [])


@observe(name="web_search")
def _web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for information online via the OpenAI Responses API with web_search.
    Used to get additional info about neighborhoods,
    infrastructure, transportation, etc.
    """
    try:
        response = _openai_client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=query
        )

        # Extract the text answer and sources
        result = {
            "answer": "",
            "sources": []
        }

        for item in response.output:
            # Text answer
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        result["answer"] = content.text
                        # Extract annotations (sources)
                        if hasattr(content, 'annotations') and content.annotations:
                            for ann in content.annotations:
                                if hasattr(ann, 'url'):
                                    result["sources"].append({
                                        "title": getattr(ann, 'title', ''),
                                        "url": ann.url
                                    })

        return result if result["answer"] else {"info": "No results found"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


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


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Max results (1-10)")


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
    StructuredTool.from_function(
        func=_web_search_tool,
        name="web_search",
        args_schema=WebSearchArgs,
        description="Search for information online. Use for questions about neighborhoods, "
                    "infrastructure, transportation, location, neighborhood safety, "
                    "proximity to the city center, and other info not found in the listings."
    ),
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "User task: {question}\n"
     "New links (may be empty): {urls}\n\n"
     "SESSION CONTEXT:\n{session_context}\n\n"
     "REMEMBER: If no links are given but the user asks about previously found apartments, "
     "use filter_offers or compute_stats WITHOUT the offers parameter — data is already in memory.\n"
     "When using web_search, you MUST take the session context into account!"),
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


def _build_session_context(session_id: Optional[str]) -> str:
    """Build a session context string for the prompt"""
    state = _get_session_state(session_id)
    offers = state.get("offers", [])

    if not offers:
        return "No data yet"

    # Build a short summary of the found apartments
    context_parts = []
    context_parts.append(f"Found {len(offers)} listings")

    # Extract addresses/titles (first 5 as examples)
    titles = [o.get("title", "") for o in offers[:5]]
    if titles:
        context_parts.append(f"Example addresses: {', '.join(titles)}")

    # Extract the city from the URL if available
    urls_in_offers = [o.get("url", "") for o in offers[:1]]
    if urls_in_offers and urls_in_offers[0]:
        context_parts.append(f"Source: {urls_in_offers[0]}")

    return "\n".join(context_parts)


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

    # Set session_id in the context variable for tools
    token = _current_session_id.set(session_id)

    # Build session context
    session_context = _build_session_context(session_id)

    try:
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
                        {"question": question, "urls": urls, "max_items": max_items, "session_context": session_context},
                        config=config
                    )
            else:
                result = executor.invoke(
                    {"question": question, "urls": urls, "max_items": max_items, "session_context": session_context},
                    config=config
                )
    finally:
        # Reset the context variable
        _current_session_id.reset(token)

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
