import os
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langsmith import traceable
from .models import Offer, ExtractionResult
from .http_tools import http_get, strip_boilerplate, absolutize

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0.0)
parser = PydanticOutputParser(pydantic_object=ExtractionResult)

SYSTEM = (
    "You are an AI parser for real estate listings. You receive an HTML page as input. "
    "Find listing cards and extract for each: title, price (integer), currency, and link. "
    "Return strictly JSON according to the given schema. If a link is relative, return it as-is."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Base URL: {base_url}\n"
     "HTML (truncated):\n{html}\n"
     "Required JSON schema:\n{schema}\n"
     "Constraints:\n"
     " - Price as a number only, no spaces.\n"
     " - Currency as in the original text (₽, RUB, $, USD, €, EUR).\n"
     " - No more than {limit} listings.\n"
     "Respond with JSON ONLY, no comments.")
])


def extract_offers_from_html(base_url: str, html: str, limit: int = 50) -> ExtractionResult:
    msgs = PROMPT.format_messages(
        base_url=base_url, html=html, schema=parser.get_format_instructions(), limit=limit
    )
    resp = llm.invoke(msgs)
    text = resp.content
    try:
        data = parser.parse(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        data = parser.parse(m.group(0))
    # absolutize urls
    data.offers = [Offer(title=o.title, price=o.price, currency=o.currency,
                         url=absolutize(base_url, o.url)) for o in data.offers]
    return data


@traceable(name="semantic_scrape")
def extract_offers(url: str, limit: int = 50) -> ExtractionResult:
    raw = http_get(url)
    cleaned = strip_boilerplate(raw)
    return extract_offers_from_html(url, cleaned, limit=limit)
