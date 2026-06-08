# 🏠 Semantic Price Scraper Agent

AI agent for monitoring real estate prices.  
Built on **LangChain** + **OpenAI LLM**, integrated with **LangSmith** and **Langfuse** for full tracing and analytics.

---

## 📖 Features

- 🔎 Extract listings from websites (title, price, currency, url)
- 🧹 Filter listings by price and keywords
- 💱 Convert prices to different currencies (USD, RUB, EUR)
- 📊 Compute price statistics (min, max, avg, median)
- 🌐 **Web search** — get information about neighborhoods, infrastructure, and transit
- 💾 Remember the last result for repeated analysis
- 📈 **Full tracing with Langfuse** — track all LLM and tool calls
- 🔄 **Session management** — group related requests
- 👥 **User ID support** — per-user analytics

---

## 🛠 Available Tools

The agent uses 5 built-in tools, each wrapped with Langfuse `@observe` for full tracing:

### 1. `extract_offers`
Extracts listings from the given page.  
**Input:** URL, limit  
**Output:** array of listings `{title, price, currency, url}`

### 2. `filter_offers`
Filters listings by price and keywords.  
**Input:** list of listings, min/max price, text filter  
**Output:** filtered list

### 3. `normalize_offers_currency`
Converts listing prices to the selected currency (RUB / USD / EUR).  
**Input:** listings + target currency  
**Output:** updated listings

### 4. `compute_stats`
Computes price statistics.  
**Input:** either a list of prices or a list of listings  
**Output:** `{min, max, avg, median}`

### 5. `web_search`
Searches the internet via OpenAI Responses API.  
Used to get additional information about neighborhoods, infrastructure, transit, safety, and other data not present in listings.  
**Input:** query (search query), max_results (1-10)  
**Output:** `{answer, sources[], raw_annotations[]}`

The agent automatically incorporates session context when formulating search queries (city, addresses from found listings).

The agent automatically saves the last list of listings to the session and passes it to tools if the user does not provide `offers` manually.

---

## 🔄 How the Agent Works

1. **Receiving the question**  
   User sends a request with a question and (optionally) URLs.

2. **Creating a trace in Langfuse and LangSmith**  
   Inside `run_question()`, an `agent_execution` observation is started, which automatically captures:
   - LLM calls
   - tool calls
   - intermediate agent steps (Chain-of-Thought hidden)
   - session_id and user_id (if provided)

3. **Building an action plan**  
   The LLM uses the system prompt and selects which tools to call.

4. **Calling tools**  
   Depending on the question:
   - listing extraction
   - filtering
   - currency conversion
   - statistics
   - web search (for neighborhood, infrastructure, etc.)
   All steps are recorded in `intermediate_steps`.

5. **Forming the response**  
   The agent assembles the final result + `tools_used` array in call order.

6. **Returning the result to the client**  
   `/ask` returns:
   ```json
   {
     "output": "agent response",
     "tools_used": ["extract_offers", "filter_offers"],
     "_session_id": "...",
     "_user_id": "..."
   }
   ```

This information can be used for testing, analytics, and verifying agent behavior.

---

## 📡 API Endpoints

### 1. `/ask` — ask the agent a question with tracing

**POST**
```json
{
  "question": "Find apartments cheaper than 2000 dollars",
  "urls": ["https://example.com"],
  "session_id": "uuid-session",
  "user_id": "user123"
}
```

**Using HTTP headers (recommended):**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: 123e4567-e89b-12d3-a456-426614174000" \
  -H "X-User-Id: user123" \
  -d '{
    "question": "Find apartments cheaper than 2000 dollars",
    "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"]
  }'
```

### 2. `/session/new` — create a new session

**GET**
```bash
curl http://localhost:8000/session/new
```

Response:
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Use this ID in the X-Session-Id header for grouping requests"
}
```

### 3. `/monitor` — direct listing extraction from a site

**POST**
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: your-session-id" \
  -d '{
    "url": "https://www.rentalads.com/apartments-for-rent/ny/new-york/",
    "max_items": 50
  }'
```

---

## 🎯 Working with Sessions

### Example of a full workflow in a single session:
```python
import requests
import json

# 1. Create a new session
session_resp = requests.get("http://localhost:8000/session/new")
session_id = session_resp.json()["session_id"]
print(f"Session created: {session_id}")

headers = {
    "Content-Type": "application/json",
    "X-Session-Id": session_id,
    "X-User-Id": "user123"
}

# 2. Extract listings
resp1 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Extract apartments from the site",
        "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"]
    }
)

# 3. Filter by price (same session)
resp2 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Show only apartments cheaper than 2000 dollars"
    }
)

# 4. Convert to rubles (same session)
resp3 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Convert prices to rubles"
    }
)

# 5. Get statistics (same session)
resp4 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Show price statistics"
    }
)

# 6. Search for neighborhood info (web_search using session context)
resp5 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Which neighborhood is best for families with children?"
    }
)
# Agent automatically adds context to the search query (New York, found addresses)
```

---

## 🐳 Docker Compose

### Cloud Langfuse (recommended):
```bash
docker compose up -d --build
```

---

## ⚠️ Limitations

- Agent only works with publicly accessible pages (no authentication)
- Website HTML structure can change, breaking parsing
- A valid OPENAI_API_KEY is required
- Langfuse tracing adds a small latency (~50-100ms)

## 📚 Useful Links

- [Langfuse Documentation](https://docs.langfuse.com)
- [LangChain Integration Guide](https://docs.langfuse.com/integrations/langchain)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
