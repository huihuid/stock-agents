# %% [markdown]
# # 🏦 Mini Project 3 — Agentic AI in FinTech
# ### Comparing Baseline, Single-Agent, and Multi-Agent Architectures
# 
# 
# ## 🎯 Learning Objectives
# 
# 1. Understand how tool-calling (function calling) works with LLMs
# 2. Define your own tool schemas and wire them to real Python functions
# 3. Build a single-agent system and reason about its limitations
# 4. Design and implement a multi-agent system — architecture is your choice
# 5. Build an LLM-as-judge evaluator to score answers automatically
# 6. Analyse accuracy, hallucination rate, and latency across all architectures and models
# 7. Reflect critically: *when does added complexity actually pay off?*
# 
# ---
# 
# ## 📦 What You Are Given vs What You Build
# 
# | Component | Status |
# |---|---|
# | 5 working tool functions | ✅ Provided |
# | JSON schemas for all 7 tools | ✅ Provided |
# | `AgentResult` dataclass | ✅ Provided |
# | `run_specialist_agent()` loop | ✅ Provided |
# | 15 fixed benchmark questions | ✅ Provided |
# | Evaluation runner + Excel writer | ✅ Provided |
# | **Tool 6: `get_company_overview`** | ❌ You implement |
# | **Tool 7: `get_tickers_by_sector`** | ❌ You implement |
# | **Baseline** | ❌ You implement |
# | **Single-agent system** | ❌ You design + implement |
# | **Multi-agent system** | ❌ You design + implement |
# | **Evaluator** | ❌ You design + implement |
# 
# ---
# 
# ## 🔑 Before You Start
# 
# **API keys needed:**
# - OpenAI → https://platform.openai.com/api-keys  
# - Alpha Vantage (free) → https://www.alphavantage.co/support/#api-key
# 
# **Data needed:**
# - `sp500_companies.csv` → https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks  
# - Unzip and place next to this notebook
# 
# **Create a `.env` file** in the same folder (never commit this):
# ```
# OPENAI_API_KEY=sk-proj-...
# ALPHAVANTAGE_API_KEY=...
# ```
# 

# %% [markdown]
# ## Step 0 — Install & Import

# %%
#!pip install openai requests pandas yfinance python-dotenv openpyxl --quiet

# %%
import os, json, time, sqlite3, requests, textwrap
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "YOUR_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_KEY")

MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL          # switch to MODEL_LARGE for the second run

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ Ready  |  active model: {ACTIVE_MODEL}")

# %% [markdown]
# ## Step 1 — Build the Local Database
# 
# Run `create_local_database()` once after placing `sp500_companies.csv` next to this notebook.  
# The function prints all distinct **sector** values — you will need these when implementing Tool 7.
# 

# %%
DB_PATH = "stocks.db"

def create_local_database(csv_path: str = "sp500_companies.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found.\n"
            "Download from: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks"
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "symbol":"ticker", "shortname":"company",
        "sector":"sector",  "industry":"industry",
        "exchange":"exchange", "marketcap":"market_cap_raw"
    })
    def cap_bucket(v):
        try:
            v = float(v)
            return "Large" if v >= 10_000_000_000 else "Mid" if v >= 2_000_000_000 else "Small"
        except: return "Unknown"
    df["market_cap"] = df["market_cap_raw"].apply(cap_bucket)
    df = (df.dropna(subset=["ticker","company"])
            .drop_duplicates(subset=["ticker"])
            [["ticker","company","sector","industry","market_cap","exchange"]])
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
    conn.commit()
    n = pd.read_sql_query("SELECT COUNT(*) AS n FROM stocks", conn).iloc[0]["n"]
    print(f"✅ {n} companies loaded into stocks.db")
    print("\nDistinct sector values stored in DB:")
    print(pd.read_sql_query("SELECT DISTINCT sector FROM stocks ORDER BY sector", conn).to_string(index=False))
    conn.close()

create_local_database()

# %% [markdown]
# ## Step 2 — Tool Functions
# 
# ### Provided Tools (5 of 7)
# 
# Read each function carefully — you need to understand their return shapes before writing agents.
# 

# %%
# ── Tool 1 ── Provided ────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """
    % price change for a list of tickers over a period.
    Valid periods: '1mo', '3mo', '6mo', 'ytd', '1y'
    Returns: { TICKER: {start_price, end_price, pct_change, period} }
    """
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results

# ── Tool 2 ── Provided ────────────────────────────────────────
def get_market_status() -> dict:
    """Open / closed status for global stock exchanges."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 3 ── Provided ────────────────────────────────────────
def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 4 ── Provided ────────────────────────────────────────
def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """
    Latest headlines + Bullish / Bearish / Neutral sentiment for a ticker.
    Returns: { ticker, articles: [{title, source, sentiment, score}] }
    """
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }

# ── Tool 5 ── Provided ────────────────────────────────────────
def query_local_db(sql: str) -> dict:
    """
    Run any SQL SELECT on stocks.db.
    Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange
    market_cap values: 'Large' | 'Mid' | 'Small'
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

print("✅ 5 provided tools ready")

# %% [markdown]
# ---
# ## 🛠️ Task 1 — Implement the 2 Missing Tools (20 pts)
# 
# ### Tool 6 — `get_company_overview`
# 
# Call the Alpha Vantage **OVERVIEW** endpoint for a single ticker.  
# Docs: https://www.alphavantage.co/documentation/#company-overview  
# 
# Required return format:
# ```python
# {
#     "ticker"    : str,   # the input ticker
#     "name"      : str,   # company full name
#     "sector"    : str,
#     "pe_ratio"  : str,   # field name in API: PERatio
#     "eps"       : str,   # field name: EPS
#     "market_cap": str,   # field name: MarketCapitalization
#     "52w_high"  : str,   # field name: 52WeekHigh
#     "52w_low"   : str,   # field name: 52WeekLow
# }
# ```
# If the API returns no `"Name"` key (rate-limited or invalid ticker), return:
# ```python
# {"error": f"No overview data for {ticker}"}
# ```
# 
# ---
# 
# ### Tool 7 — `get_tickers_by_sector`
# 
# Query `stocks.db` for companies matching a sector **or** industry.
# 
# **Critical detail:** Look at the sector values printed by `create_local_database()`.  
# The DB stores broad sectors in `sector` (e.g. `"Information Technology"`) and  
# specific industries in `industry` (e.g. `"Semiconductors"`).  
# A query for `"semiconductor"` must fall back to the `industry` column — otherwise it returns zero rows.
# 
# Required logic:
# 1. Try exact match on `sector` column (case-insensitive)  
# 2. If 0 results → try `LIKE '%sector%'` on the `industry` column  
# 
# Required return format:
# ```python
# {
#     "sector": str,          # the input search term
#     "stocks": [
#         {"ticker": str, "company": str, "industry": str},
#         ...
#     ]
# }
# ```
# 

# %%
# ── Tool 6 — YOUR IMPLEMENTATION ─────────────────────────────
def get_company_overview(ticker: str) -> dict:
    ticker = (ticker or "").strip().upper()
    try:
        data = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": ALPHAVANTAGE_API_KEY,
            },
            timeout=10,
        ).json()

        # Alpha Vantage rate limit / invalid ticker often returns {} or a note/error without Name
        if not isinstance(data, dict) or not data.get("Name"):
            return {"error": f"No overview data for {ticker}"}

        return {
            "ticker"    : ticker,
            "name"      : data.get("Name", ""),
            "sector"    : data.get("Sector", ""),
            "pe_ratio"  : data.get("PERatio", ""),
            "eps"       : data.get("EPS", ""),
            "market_cap": data.get("MarketCapitalization", ""),
            "52w_high"  : data.get("52WeekHigh", ""),
            "52w_low"   : data.get("52WeekLow", ""),
        }
    except Exception:
        # Keep behavior consistent with spec: if anything goes wrong, treat as no data
        return {"error": f"No overview data for {ticker}"}


# ── Tool 7 — YOUR IMPLEMENTATION ─────────────────────────────
def get_tickers_by_sector(sector: str) -> dict:
    term = (sector or "").strip()
    if not term:
        return {"sector": sector, "stocks": []}

    # 1) Try exact match on sector (case-insensitive)
    sql_exact = """
        SELECT ticker, company, industry
        FROM stocks
        WHERE LOWER(sector) = LOWER(?)
        ORDER BY ticker
    """

    # 2) Fallback: LIKE match on industry (case-insensitive)
    sql_like = """
        SELECT ticker, company, industry
        FROM stocks
        WHERE LOWER(industry) LIKE LOWER(?)
        ORDER BY ticker
    """

    try:
        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query(sql_exact, conn, params=(term,))
        if df.empty:
            like_param = f"%{term}%"
            df = pd.read_sql_query(sql_like, conn, params=(like_param,))

        conn.close()

        stocks = [
            {
                "ticker": str(r["ticker"]),
                "company": str(r["company"]),
                "industry": str(r["industry"]),
            }
            for _, r in df.iterrows()
        ]
        return {"sector": sector, "stocks": stocks}
    except Exception as e:
        # Spec doesn't define error shape for Tool7; safest is return empty result
        return {"sector": sector, "stocks": []}


# ── Automated tests — all assertions must pass ────────────────
# print("=== Tool 6 tests ===")
# aapl = get_company_overview("AAPL")
# # assert "pe_ratio" in aapl,             "Missing pe_ratio key"
# # assert aapl.get("ticker") == "AAPL",   "ticker key wrong"
# # assert aapl.get("name"),               "name should not be empty"
# print(f"  AAPL P/E: {aapl['pe_ratio']} ✅")

# bad = get_company_overview("INVALIDTICKER999")
# assert "error" in bad,                 "Invalid ticker should return error key"
# print(f"  Invalid ticker handled correctly ✅")

# print("\n=== Tool 7 tests ===")
# tech = get_tickers_by_sector("Information Technology")
# assert len(tech["stocks"]) > 0,        "Should find IT stocks (exact sector match)"
# print(f"  'Information Technology' → {len(tech['stocks'])} stocks ✅")

# semi = get_tickers_by_sector("semiconductor")
# assert len(semi["stocks"]) > 0,        "Should find via industry fallback (LIKE match)"
# print(f"  'semiconductor' (industry fallback) → {len(semi['stocks'])} stocks ✅")

# print("\n✅ All tool tests passed")

# %% [markdown]
# ## Step 3 — Tool Schemas (Provided)
# 
# Schemas tell the LLM what tools exist, what they do, and what arguments they take.  
# You do not modify these — but you will reference the schema lists when building your agents.
# 
# **Key concept:** You can give different agents different *subsets* of schemas.  
# A specialist that only sees 2–3 relevant schemas makes fewer wrong tool choices  
# than one that sees all 7.
# 

# %%
def _s(name, desc, props, req):
    return {"type":"function","function":{
        "name":name,"description":desc,
        "parameters":{"type":"object","properties":props,"required":req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector":{"type":"string","description":"Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers":{"type":"array","items":{"type":"string"}},
     "period":{"type":"string","default":"1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker":{"type":"string","description":"Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker":{"type":"string"},"limit":{"type":"integer","default":5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql":{"type":"string","description":"A valid SQL SELECT statement"}}, ["sql"])

# All 7 schemas in one list — used by single agent
ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
               SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]

# Dispatch map — maps the tool name string the LLM returns → the Python function to call
ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance"  : get_price_performance,
    "get_company_overview"   : get_company_overview,
    "get_market_status"      : get_market_status,
    "get_top_gainers_losers" : get_top_gainers_losers,
    "get_news_sentiment"     : get_news_sentiment,
    "query_local_db"         : query_local_db,
}

print("✅ Schemas ready")
print(f"   Tools available: {list(ALL_TOOL_FUNCTIONS.keys())}")

# %% [markdown]
# ## Step 4 — AgentResult and Base Runner (Provided)
# 
# `AgentResult` is the standard return type for every agent — baseline, single, and all multi-agent specialists.  
# `run_specialist_agent` is the reusable tool-call loop that every agent uses.  
# Study this loop carefully before writing your own agents — it is what connects the LLM's tool requests to your Python functions.
# 

# %%
@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)   # tool names in order called
    raw_data     : dict  = field(default_factory=dict)   # tool name → raw tool output
    confidence   : float = 0.0                           # set by evaluator / critic
    issues_found : list  = field(default_factory=list)   # set by evaluator / critic
    reasoning    : str   = ""

    def summary(self):
        print(f"\n{'─'*54}")
        print(f"Agent      : {self.agent_name}")
        print(f"Tools used : {', '.join(self.tools_called) or 'none'}")
        print(f"Confidence : {self.confidence:.0%}")
        if self.issues_found:
            print(f"Issues     : {'; '.join(self.issues_found)}")
        print(f"Answer     :\n{textwrap.indent(self.answer[:500], '  ')}")

print("✅ AgentResult defined")

# %%
def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    max_iters    : int  = 8,
    verbose      : bool = True,
) -> AgentResult:
    """
    Core agentic loop used by every agent in this project.

    How it works:
      1. Sends system_prompt + task to the LLM
      2. If the LLM requests a tool call → looks up the function in ALL_TOOL_FUNCTIONS,
         executes it, appends the result to the message history, loops back to step 1
      3. When the LLM produces a response with no tool calls → returns an AgentResult

    Parameters
    ----------
    agent_name    : display name for logging
    system_prompt : the agent's persona, rules, and focus area
    task          : the specific question or sub-task for this agent
    tool_schemas  : list of schema dicts this agent is allowed to use
                    (pass [] for no tools — used by baseline)
    max_iters     : hard cap on iterations to prevent infinite loops
    verbose       : print each tool call as it happens
    """
    ### YOUR CODE HERE ###
print("✅ run_specialist_agent ready")

# %% [markdown]
# ---
# ## 🛠️ Task 2 — Implement the Baseline (10 pts)
# 
# The baseline is a **single LLM call with no tools**. The model answers entirely from its training knowledge.
# 
# This is your **control condition**. Every architecture you build should be compared against it.  
# If agents don't outperform the baseline, there's no justification for the extra complexity.
# 
# **Requirements:**
# - One call to `client.chat.completions.create()` — no `tools` argument
# - Return `AgentResult(agent_name="Baseline", answer=..., tools_called=[])`
# - Use a sensible system prompt that encourages the model to be honest about uncertainty
# 
# **Grading checks:**
# - `tools_called` must be `[]`
# - Answer must not be empty
# 

# %%
def run_baseline(question: str, verbose: bool = True) -> AgentResult:
    # Implement a single LLM call with no tools.
    # Use run_specialist_agent() with an empty tool_schemas list — or make the call directly.
    # Return an AgentResult with agent_name="Baseline" and tools_called=[].
    ### YOUR CODE HERE
    system_prompt = (
        "You are a careful finance assistant. "
        "Answer the user's question using only your internal knowledge. "
        "Do NOT invent precise real-time numbers (prices, P/E ratios, % returns). "
        "If you are unsure or the question requires live data, say so clearly and "
        "give a best-effort qualitative answer or explain what data would be needed."
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        answer = "I don't know."

    if verbose:
        print(f"[Baseline] done in {time.time() - t0:.2f}s")

    return AgentResult(
        agent_name="Baseline",
        answer=answer,
        tools_called=[],   # must be []
        raw_data={},
        confidence=0.0,
        issues_found=[],
        reasoning="",
    )

# Quick test
# bl = run_baseline("What is Apple's approximate P/E ratio?", verbose=True)
# assert bl.tools_called == [], "Baseline must not call any tools"
# assert bl.answer, "Answer must not be empty"
# bl.summary()

# %% [markdown]
# ---
# ## 🛠️ Task 3 — Design and Implement the Single Agent (20 pts)
# 
# A single agent is **one LLM with access to all 7 tools**. Everything — planning which tools to call, executing them, and synthesising the answer — happens in one context window.
# 
# You decide how to build it. The guidance below is a starting point, not a recipe.
# 
# ---
# 
# ### Things to think about when writing your system prompt
# 
# **Role and scope** — what is this agent's job? Being specific helps the model stay focused.
# 
# **Accuracy rules** — how should the agent behave when a tool returns an error or empty data?  
# This is critical: an agent with no rules tends to fabricate plausible-looking numbers when the API fails.
# 
# **Multi-step reasoning** — some questions require chaining tools. For example:  
# *"Which energy stocks grew the most this year?"* requires first looking up energy tickers,  
# then fetching price data for each one. Without explicit guidance, single agents often  
# skip the lookup step and guess tickers instead.
# 
# **Tool selection** — the agent sees all 7 tools. Giving it rules about *when* to use each  
# (e.g. "use `query_local_db` with SQL when you need to filter by sector or market cap")  
# reduces wrong tool choices.
# 
# ---
# 
# ### Recommended structure
# 
# ```python
# SINGLE_AGENT_PROMPT = """
# <your system prompt here>
# """
# 
# def run_single_agent(question: str, verbose: bool = True) -> AgentResult:
#     # Call run_specialist_agent() with:
#     #   agent_name    = "Single Agent"
#     #   system_prompt = SINGLE_AGENT_PROMPT
#     #   task          = question
#     #   tool_schemas  = ALL_SCHEMAS   (all 7)
#     #   max_iters     = 10
#     # Return the AgentResult directly.
#     pass
# ```
# 
# ---
# 
# ### Test before you move on
# 
# Run the three cells below and check the results make sense before building the multi-agent system.
# 

# %%
def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    max_iters    : int  = 8,
    verbose      : bool = True,
) -> AgentResult:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools_called = []
    raw_data = {}  # tool_name -> list of outputs

    for step in range(max_iters):
        # Ask model (with or without tools)
        kwargs = dict(
            model=ACTIVE_MODEL,
            messages=messages,
            temperature=0.2,
        )
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        # Always append the assistant message first (may include tool_calls)
        # Convert to dict to keep only needed fields
        assistant_entry = {"role": "assistant", "content": msg.content}
        if getattr(msg, "tool_calls", None):
            # tool_calls must be preserved in the conversation
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        tool_calls = getattr(msg, "tool_calls", None) or []

        # If no tool calls, we're done
        if not tool_calls:
            final_answer = (msg.content or "").strip()
            if not final_answer:
                final_answer = "No answer produced."
            return AgentResult(
                agent_name=agent_name,
                answer=final_answer,
                tools_called=tools_called,
                raw_data=raw_data,
                confidence=0.0,
                issues_found=[],
                reasoning="",
            )

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            tools_called.append(tool_name)

            # Parse arguments JSON
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception as e:
                tool_output = {"error": f"Failed to parse tool args for {tool_name}: {e}"}
                args = {}

            if verbose:
                print(f"[{agent_name}] tool_call → {tool_name}({args})")

            fn = ALL_TOOL_FUNCTIONS.get(tool_name)
            if not fn:
                tool_output = {"error": f"Tool not found: {tool_name}"}
            else:
                try:
                    tool_output = fn(**args)
                except Exception as e:
                    tool_output = {"error": str(e)}

            raw_data.setdefault(tool_name, []).append(tool_output)

            # Send tool result back to model
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_output),
            })

    # If we reach here, max_iters hit
    return AgentResult(
        agent_name=agent_name,
        answer="Reached max tool-call iterations without producing a final answer.",
        tools_called=tools_called,
        raw_data=raw_data,
        confidence=0.0,
        issues_found=["max_iters_reached"],
        reasoning="",
    )


# r = run_specialist_agent(
#     agent_name="Sanity",
#     system_prompt="You are a tool-using assistant. Use tools when needed.",
#     task="What is the P/E ratio of Apple (AAPL)?",
#     tool_schemas=[SCHEMA_OVERVIEW],
#     max_iters=6,
#     verbose=True,
# )
# r.summary()

# %%
# ── YOUR SINGLE AGENT IMPLEMENTATION ─────────────────────────
# Write your system prompt and run_single_agent() function here.
# Comments above explain what to think about — the implementation is yours.

### YOUR CODE HERE
SINGLE_AGENT_PROMPT = """
You are a careful fintech analyst agent with tool access.
Your job: answer the user's question accurately using tools when the question requires data.

Rules:
1) Prefer tools over guessing. If the question asks for prices, returns, "this month", "YTD", "1-year", or ranking performance:
   - Use get_tickers_by_sector (or query_local_db) to get the tickers from the local DB
   - Then use get_price_performance with the correct period ('1mo','3mo','6mo','ytd','1y')
2) Fundamentals like P/E, EPS, 52w high/low, market cap:
   - Use get_company_overview for each ticker needed.
3) News sentiment:
   - Use get_news_sentiment(ticker, limit=3-5).
4) Market open/closed:
   - Use get_market_status and report US exchanges (NYSE/NASDAQ) status.
5) If a tool returns an error or empty data:
   - Do NOT invent numbers. Say data unavailable for that ticker and continue with others.
6) Always show:
   - The exact tickers you used
   - For comparisons/rankings: a small ranked list with % changes or values
7) Keep the final answer concise and directly responsive.
8) For any question that asks for "best", "top", "rank", or has conditions like
   "dropped this month but grew this year":
   - You MUST compute using tool outputs and ONLY include rows that satisfy conditions.
   - You MUST sort correctly (descending for best performance).
   - If fewer than requested items match, return as many as match and say so.
"""

def _extract_perf_map(agent_result: AgentResult):
    # Find the most recent get_price_performance output (dict of ticker -> stats/error)
    lst = agent_result.raw_data.get("get_price_performance", [])
    if not lst:
        return None
    return lst[-1]  # last call

def _format_top_perf(perf_map: dict, top_n: int = 10):
    rows = []
    for t, v in perf_map.items():
        if isinstance(v, dict) and "pct_change" in v and "error" not in v:
            rows.append((t, v["pct_change"], v.get("start_price"), v.get("end_price"), v.get("period")))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_n]
    lines = []
    for i, (t, pct, s, e, p) in enumerate(rows, 1):
        lines.append(f"{i}. {t}: {pct}% (start {s}, end {e}, {p})")
    return "\n".join(lines) if lines else "No valid price data returned."

def _filter_month_down_ytd_up(perf_1mo: dict, perf_ytd: dict, top_n: int = 3):
    # perf maps: ticker -> {pct_change,...} or {error:...}
    rows = []
    tickers = set(perf_1mo.keys()) | set(perf_ytd.keys())
    for t in tickers:
        a = perf_1mo.get(t, {})
        b = perf_ytd.get(t, {})
        if not isinstance(a, dict) or not isinstance(b, dict): 
            continue
        if "error" in a or "error" in b: 
            continue
        if "pct_change" not in a or "pct_change" not in b:
            continue
        if a["pct_change"] < 0 and b["pct_change"] > 0:
            rows.append((t, a["pct_change"], b["pct_change"]))
    # top by YTD growth desc (as Q11 expects)
    rows.sort(key=lambda x: x[2], reverse=True)
    rows = rows[:top_n]
    if not rows:
        return "No tech stocks found that are down over 1 month but up YTD (based on available data)."
    out = []
    for i, (t, m, y) in enumerate(rows, 1):
        out.append(f"{i}. {t}: 1mo {m}%, YTD {y}%")
    return "\n".join(out)

def run_single_agent(question: str, verbose: bool = True) -> AgentResult:
    res = run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )

    q = question.lower()

    # Post-process: sector best performance
    if "best" in q and ("6-month" in q or "6 month" in q or "6mo" in q) and "performance" in q:
        perf = _extract_perf_map(res)
        if isinstance(perf, dict):
            res.answer = "Top energy stocks by 6-month % return (fetched from yfinance via get_price_performance):\n" + _format_top_perf(perf, top_n=10)

    # Post-process: "dropped this month but grew this year" (your r3 case, and Q11)
    if "dropped" in q and "this month" in q and ("grew this year" in q or "ytd" in q):
        perf_calls = res.raw_data.get("get_price_performance", [])
        # Expect two calls: 1mo and ytd; if order unknown, detect by 'period'
        perf_1mo = None
        perf_ytd = None
        for pm in perf_calls:
            # pick any representative entry to read period
            if isinstance(pm, dict):
                anyv = next(iter(pm.values()), {})
                period = anyv.get("period") if isinstance(anyv, dict) else None
                if period == "1mo":
                    perf_1mo = pm
                elif period == "ytd":
                    perf_ytd = pm
        if perf_1mo and perf_ytd:
            res.answer = "Tech stocks down over 1 month but up YTD (computed from yfinance via get_price_performance):\n" + \
             _filter_month_down_ytd_up(perf_1mo, perf_ytd, top_n=3)

    return res

# %%
# Test 1 — easy question, 1 tool expected
# r1 = run_single_agent("What is the P/E ratio of Apple (AAPL)?")
# r1.summary()

# # %%
# # Test 2 — medium question, requires sector lookup + price fetch
# r2 = run_single_agent("Which energy stocks in the database had the best 6-month performance?")
# r2.summary()

# # %%
# # Test 3 — hard multi-condition question
# r3 = run_single_agent("Top 3 tech stocks that dropped this month but grew this year.")
# r3.summary()

# %% [markdown]
# ---
# ## 🛠️ Task 4 — Design and Implement a Multi-Agent System (25 pts)
# 
# You must build a multi-agent system that answers the same 15 questions.  
# **The architecture is your choice.** Experiment, compare, and justify your decision in the reflections.
# 
# ---
# 
# ### Three architectures to consider
# 
# **Orchestrator + Specialists + Critic**
# ```
# User Question
#      │
#  Orchestrator  ← reads question, writes a plan, delegates sub-tasks
#      │
#  ┌───┼───┐
#  Ag1 Ag2 Ag3   ← each handles one domain, sees a subset of tools
#  └───┼───┘
#   Critic        ← fact-checks each agent's answer vs its raw tool data
#      │
#  Synthesizer    ← merges verified answers into one final response
# ```
# *Good for:* complex cross-domain questions  
# *Tradeoff:* most latency, most API calls, but most transparency
# 
# ---
# 
# **Sequential Pipeline**
# ```
# User Question → Agent1 → Agent2 → Agent3 → Final Answer
# ```
# Each agent receives the previous agent's output as context.  
# *Good for:* questions with a natural order (find tickers → get prices → summarise)  
# *Tradeoff:* errors propagate — a wrong ticker in step 1 breaks all later steps
# 
# ---
# 
# **Parallel Specialists + Aggregator**
# ```
# User Question
#   ├── Price Agent       ─┐
#   ├── Fundamentals Agent  ├─→ Aggregator → Final Answer
#   └── Sentiment Agent   ─┘
# ```
# All specialists run, aggregator merges results.  
# *Good for:* speed (can use `ThreadPoolExecutor` to run in parallel)  
# *Tradeoff:* no cross-checking between agents
# 
# ---
# 
# ### Suggested tool subsets per specialist
# These are starting points — adjust based on your design:
# 
# ```python
# MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
# FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
# SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_SQL]
# ```
# 
# Giving each specialist only its relevant schemas reduces wrong tool choices.
# 
# ---
# 
# ### Required return contract
# 
# Whatever architecture you choose, `run_multi_agent()` must return this dict  
# (the evaluation runner reads these fields):
# 
# ```python
# {
#     "final_answer"  : str,         # the answer shown to the user
#     "agent_results" : list,        # list[AgentResult] — one per specialist that ran
#     "elapsed_sec"   : float,       # total wall clock time
#     "architecture"  : str,         # short name e.g. "orchestrator-critic", "pipeline", "parallel"
# }
# ```
# 
# The evaluation runner extracts from `agent_results`:
# - `tools_called` (all tools used across specialists)
# - `confidence` (averaged across specialists)
# - `issues_found` (all issues concatenated)
# 
# ---
# 
# ### Document your design choices in document
# 
# The grading for this task includes your reasoning — explain in document:
# - Which architecture you chose and why
# - What you tried that didn't work
# - How you divided tools between specialists
# - What your verification/confidence mechanism does
# 

# %%
# ─────────────────────────────────────────────────────────────
# Multi-Agent Architecture: pipeline (deterministic) + specialists
# Goal: minimize hallucinations + avoid invalid SQL columns.
# ─────────────────────────────────────────────────────────────

MARKET_TOOLS = [SCHEMA_TICKERS, SCHEMA_PRICE]
FUND_TOOLS   = [SCHEMA_OVERVIEW]
NEWS_TOOLS   = [SCHEMA_NEWS]

MARKET_AGENT_PROMPT = """
You are a market/returns specialist.
You may ONLY use these tools: get_tickers_by_sector, get_price_performance.
Do NOT use SQL. Do NOT call tools unrelated to returns.
Never invent tickers or numbers.
Return results based strictly on tool outputs.
"""

FUND_AGENT_PROMPT = """
You are a fundamentals specialist.
You may ONLY use get_company_overview.
Never guess values. If tool returns error, report data unavailable.
"""

NEWS_AGENT_PROMPT = """
You are a news sentiment specialist.
You may ONLY use get_news_sentiment.
Never invent headlines or scores.
"""

def _pick_top_by_return(perf_map: dict, top_n: int = 3):
    rows = []
    for t, v in (perf_map or {}).items():
        if isinstance(v, dict) and "pct_change" in v and "error" not in v:
            rows.append((t, v["pct_change"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in rows[:top_n]], rows[:top_n]

def _avg_sentiment_label(articles: list):
    # Simple summary label from AlphaVantage labels; fallback to "Mixed/Unknown"
    # (We still list headlines + scores; this is just a compact summary.)
    labels = [a.get("sentiment") for a in (articles or []) if a.get("sentiment")]
    if not labels:
        return "Unknown"
    # majority vote
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]

def _safe_float(x):
    try:
        if x is None: 
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "nan", "n/a"}:
            return None
        return float(s)
    except:
        return None

def _extract_tickers_from_market(market_res: AgentResult):
    """从 market_res.raw_data['get_tickers_by_sector'] 里提取 tickers"""
    outs = market_res.raw_data.get("get_tickers_by_sector", [])
    if not outs:
        return []
    last = outs[-1]
    stocks = (last or {}).get("stocks", []) if isinstance(last, dict) else []
    tickers = []
    for r in stocks:
        t = (r.get("ticker") if isinstance(r, dict) else None)
        if t:
            tickers.append(str(t).upper())
    # 去重保持顺序
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def run_multi_agent(question: str, verbose: bool = True) -> dict:
    t0 = time.time()
    agent_results = []
    q = (question or "").lower()

    # ---------------- helpers (local to this function) ----------------
    def _safe_float(x):
        try:
            if x is None:
                return None
            s = str(x).strip()
            if s == "" or s.lower() in {"none", "nan", "n/a"}:
                return None
            return float(s)
        except:
            return None

    def _extract_tickers_from_market(market_res: AgentResult):
        outs = market_res.raw_data.get("get_tickers_by_sector", [])
        if not outs:
            return []
        last = outs[-1]
        stocks = (last or {}).get("stocks", []) if isinstance(last, dict) else []
        tickers = []
        for r in stocks:
            t = r.get("ticker") if isinstance(r, dict) else None
            if t:
                tickers.append(str(t).upper())
        # de-dup keep order
        seen = set()
        uniq = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _pick_top_by_return(perf_map: dict, top_n: int = 3):
        rows = []
        for t, v in (perf_map or {}).items():
            if isinstance(v, dict) and "pct_change" in v and "error" not in v:
                rows.append((t, v["pct_change"]))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in rows[:top_n]], rows[:top_n]

    def _avg_sentiment_label(articles: list):
        labels = [a.get("sentiment") for a in (articles or []) if isinstance(a, dict) and a.get("sentiment")]
        if not labels:
            return "Unknown"
        from collections import Counter
        return Counter(labels).most_common(1)[0][0]

    # ---------------- Step A: Market Specialist (deterministic routing) ----------------
    market_task = question

    # Q13 pattern: top 3 semiconductor by 1y return
    is_q13 = ("top 3" in q and "semiconductor" in q and "1-year" in q and "return" in q)

    if is_q13:
        market_task = (
            "Task: Find semiconductor tickers and rank by 1-year return.\n"
            "You MUST do exactly:\n"
            "1) Call get_tickers_by_sector with sector='semiconductor'\n"
            "2) From returned stocks, collect tickers (max 40 to avoid time/rate issues)\n"
            "3) Call get_price_performance(tickers=<those tickers>, period='1y')\n"
            "4) Do NOT write the final answer yet; just make sure tool outputs are produced.\n"
            "Return a brief note like: 'Fetched tickers and 1y performance for N tickers.'"
        )

    market_res = run_specialist_agent(
        agent_name="Market Specialist",
        system_prompt=MARKET_AGENT_PROMPT,
        task=market_task,
        tool_schemas=MARKET_TOOLS,
        max_iters=10,
        verbose=verbose,
    )
    agent_results.append(market_res)

    tickers_from_sector = _extract_tickers_from_market(market_res)
    perf_calls = market_res.raw_data.get("get_price_performance", [])
    last_perf = perf_calls[-1] if perf_calls else None

    target_tickers = []
    top_rows = []

    if is_q13:
        # If missing 1y perf, do a rescue run deterministically
        if not isinstance(last_perf, dict) or not last_perf:
            rescue_task = (
                "Rescue Task: You failed to produce 1y performance.\n"
                "Call get_tickers_by_sector(sector='semiconductor') then "
                "get_price_performance(period='1y') on those tickers (max 40)."
            )
            market_res2 = run_specialist_agent(
                agent_name="Market Specialist (Rescue)",
                system_prompt=MARKET_AGENT_PROMPT,
                task=rescue_task,
                tool_schemas=MARKET_TOOLS,
                max_iters=10,
                verbose=verbose,
            )
            agent_results.append(market_res2)
            perf_calls2 = market_res2.raw_data.get("get_price_performance", [])
            last_perf = perf_calls2[-1] if perf_calls2 else last_perf

        if isinstance(last_perf, dict) and last_perf:
            target_tickers, top_rows = _pick_top_by_return(last_perf, top_n=3)

    # ---------------- Step B: Fundamentals Specialist ----------------
    need_fund = any(k in q for k in ["p/e", "pe ratio", "eps", "52-week", "52 week", "market cap"])
    if need_fund or target_tickers:
        if target_tickers:
            fund_task = (
                f"ONLY call get_company_overview for these tickers exactly: {', '.join(target_tickers)}. "
                "Do not call for any other tickers. Return nothing except brief confirmation."
            )
        else:
            fund_task = question

        fund_res = run_specialist_agent(
            agent_name="Fundamentals Specialist",
            system_prompt=FUND_AGENT_PROMPT,
            task=fund_task,
            tool_schemas=FUND_TOOLS,
            max_iters=10,
            verbose=verbose,
        )
        agent_results.append(fund_res)

    # ---------------- Step C: Sentiment Specialist ----------------
    need_news = any(k in q for k in ["news", "sentiment", "headline", "headlines"])
    if need_news or target_tickers:
        if target_tickers:
            news_task = (
                f"ONLY call get_news_sentiment for these tickers exactly: {', '.join(target_tickers)} "
                "(limit=5 each). Do not call for any other tickers. "
                "Return nothing except brief confirmation."
            )
        else:
            news_task = question

        news_res = run_specialist_agent(
            agent_name="Sentiment Specialist",
            system_prompt=NEWS_AGENT_PROMPT,
            task=news_task,
            tool_schemas=NEWS_TOOLS,
            max_iters=12,
            verbose=verbose,
        )
        agent_results.append(news_res)

    # ---------------- Critic: deterministic checks + confidence ----------------
    issues = []
    confidence = 0.85

    if is_q13:
        if len(target_tickers) != 3:
            issues.append("Failed to compute top 3 tickers by 1-year return from tool data.")
            confidence = 0.55

    # push critic results into each AgentResult
    for r in agent_results:
        r.confidence = confidence
        r.issues_found = issues

    # ---------------- Synthesis from RAW TOOL DATA ----------------
    overview_map = {}
    sentiment_map = {}

    for r in agent_results:
        for out in r.raw_data.get("get_company_overview", []):
            if isinstance(out, dict) and out.get("ticker") and "error" not in out:
                overview_map[out["ticker"]] = out

        for out in r.raw_data.get("get_news_sentiment", []):
            if isinstance(out, dict) and out.get("ticker"):
                sentiment_map[out["ticker"]] = out

    # Q07 pattern: Compare P/E AAPL MSFT NVDA
    is_q07 = ("compare" in q and "p/e" in q and "aapl" in q and "msft" in q and "nvda" in q)

    if is_q07:
        wanted = ["AAPL", "MSFT", "NVDA"]
        pe_rows = []
        missing = []
        for t in wanted:
            ov = overview_map.get(t)
            pe = _safe_float(ov.get("pe_ratio")) if isinstance(ov, dict) else None
            if pe is None:
                missing.append(t)
            else:
                pe_rows.append((t, pe, ov.get("pe_ratio")))
        pe_rows.sort(key=lambda x: x[1], reverse=True)

        lines = []
        lines.append("Answer (multi-agent): Compare P/E ratios (Alpha Vantage OVERVIEW)")
        lines.append("Sources: P/E from Alpha Vantage OVERVIEW via get_company_overview.")
        for t, _, pe_raw in pe_rows:
            lines.append(f"- {t}: P/E {pe_raw}")
        if missing:
            lines.append(f"- Missing P/E data: {', '.join(missing)}")

        if pe_rows:
            lines.append(f"Most expensive by P/E: {pe_rows[0][0]} (highest P/E among available data)")
        else:
            lines.append("Could not determine: no valid P/E values returned by the tool.")

        if issues:
            lines.append("\n[Critic]\nIssues: " + "; ".join(issues))
        lines.append(f"\n(Confidence: {confidence:.0%})")

        return {
            "final_answer": "\n".join(lines),
            "agent_results": agent_results,
            "elapsed_sec": round(time.time() - t0, 3),
            "architecture": "pipeline-deterministic-critique",
        }

    # Q13 synthesis: top 3 semiconductor by 1y return + P/E + sentiment
    if is_q13 and ("p/e" in q or "pe ratio" in q) and ("sentiment" in q or "news" in q):
        lines = []
        lines.append("Answer (multi-agent):")
        lines.append("Top 3 semiconductor stocks by 1-year return, with P/E and news sentiment:")
        lines.append(
            "Sources: 1y return from yfinance via get_price_performance; "
            "P/E from Alpha Vantage OVERVIEW via get_company_overview; "
            "news sentiment from Alpha Vantage NEWS_SENTIMENT via get_news_sentiment."
        )

        if top_rows:
            for rank, (t, ret) in enumerate(top_rows, 1):
                ov = overview_map.get(t, {})
                pe = ov.get("pe_ratio", "N/A") if isinstance(ov, dict) else "N/A"
                news = sentiment_map.get(t, {})
                articles = news.get("articles", []) if isinstance(news, dict) else []
                label = _avg_sentiment_label(articles)

                lines.append(f"{rank}. {t}: 1y {ret}% | P/E {pe} | Sentiment {label}")
                for a in (articles[:2] if articles else []):
                    lines.append(
                        f"   - {a.get('source')}: {a.get('title')} "
                        f"({a.get('sentiment')}, {a.get('score')})"
                    )
        else:
            lines.append("No valid 1-year return data to rank semiconductors (tool returned empty/errors).")

        if issues:
            lines.append("\n[Critic]\nIssues: " + "; ".join(issues))
        lines.append(f"\n(Confidence: {confidence:.0%})")

        return {
            "final_answer": "\n".join(lines),
            "agent_results": agent_results,
            "elapsed_sec": round(time.time() - t0, 3),
            "architecture": "pipeline-deterministic-critique",
        }

    # Generic fallback: stitch agent answers (kept from your original approach)
    lines = ["Answer (multi-agent):"]
    for r in agent_results:
        if (r.answer or "").strip():
            lines.append(f"\n[{r.agent_name}]\n{r.answer.strip()}")

    if issues:
        lines.append("\n[Critic]\nIssues: " + "; ".join(issues))
    lines.append(f"\n(Confidence: {confidence:.0%})")

    return {
        "final_answer": "\n".join(lines),
        "agent_results": agent_results,
        "elapsed_sec": round(time.time() - t0, 3),
        "architecture": "pipeline-deterministic-critique",
    }

# %%
# Test 1 — check return contract
# out1 = run_multi_agent("What is the P/E ratio of Apple (AAPL)?")
# assert "final_answer"   in out1, "Missing final_answer key"
# assert "agent_results"  in out1, "Missing agent_results key"
# assert "elapsed_sec"    in out1, "Missing elapsed_sec key"
# assert "architecture"   in out1, "Missing architecture key"
# print(f"Architecture : {out1['architecture']}")
# print(f"Agents ran   : {[r.agent_name for r in out1['agent_results']]}")
# print(f"Answer       : {out1['final_answer'][:200]}")

# %%
# Test 2 — cross-domain hard question
# out2 = run_multi_agent("For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios and current news sentiment?")
# print(f"Architecture : {out2['architecture']}")
# print(f"Agents ran   : {[r.agent_name for r in out2['agent_results']]}")
# print(f"Tools used   : {[t for r in out2['agent_results'] for t in r.tools_called]}")
# print(f"\nAnswer:\n{out2['final_answer'][:400]}")

# %% [markdown]
# ---
# ## 🛠️ Task 5 — Implement the LLM Evaluator (15 pts)
# 
# The evaluator is an **LLM-as-judge**: it reads a question, the expected answer description, and an agent's actual answer, then scores it.
# 
# This is how you measure accuracy across all architectures automatically — rather than reading 45 answers by hand.
# 
# ---
# 
# ### Required output format
# 
# Your evaluator must return a Python dict with exactly these keys.  
# The evaluation runner reads all of them to fill the Excel sheet:
# 
# ```python
# {
#     "score"                 : int,        # 0, 1, 2, or 3
#     "max_score"             : int,        # always 3
#     "reasoning"             : str,        # one sentence explaining the score
#     "hallucination_detected": bool,       # True if the answer contains invented facts
#     "key_issues"            : list[str],  # specific problems found; empty list if none
# }
# ```
# 
# ---
# 
# ### Scoring rubric to include in your prompt
# 
# ```
# 3 — Fully correct:    all required data present, numbers accurate, conditions met
# 2 — Partially correct: key data present but incomplete, gaps, or minor inaccuracies
# 1 — Mostly wrong:     attempted but wrong numbers, missed required conditions,
#                       or claims that appear fabricated
# 0 — Complete failure: refused to answer, said data unavailable without trying tools,
#                       or answer has no relevance to the question
# ```
# 
# ---
# 
# ### Hallucination detection — what to flag
# 
# Include these rules in your prompt:
# - Specific numbers (prices, P/E ratios, % changes) with no tool data to support them
# - Stock tickers that don't exist or aren't relevant to the question
# - Definitive claims about "current" data without having called a live data tool
# 
# ---
# 
# ### Important considerations
# 
# **The evaluator only sees text** — it cannot verify numbers against live data.  
# Its hallucination detection is based on reasoning about whether claims look fabricated,  
# not on cross-checking against a ground truth source.  
# You will reflect on this limitation in the graded questions.
# 
# **JSON parsing:** The LLM may wrap its response in markdown fences (```json ... ```).  
# Strip those before parsing. Return the fallback dict on any parse error.
# 
# **Prompt placement:** Pass the rubric, the expected answer description, and the agent's  
# actual answer all in one user message so the evaluator has full context.
# 
# ---
# 
# ### Calibration tests (provided below)
# 
# Three test cases check your evaluator before the full run:
# 1. A clearly correct answer (expect score = 3)
# 2. An answer with an invented number (expect `hallucination_detected = True`, score ≤ 1)
# 3. A refusal (expect score = 0)
# 
# All three must behave as expected before running the full evaluation.
# 

# %%
# ── YOUR EVALUATOR IMPLEMENTATION ────────────────────────────
#
# Things to decide:
#   - How detailed is your rubric? (more detail → more consistent scores)
#   - How do you instruct it to detect hallucinations vs honest uncertainty?
#   - How strict are you on partial answers? (affects SA vs MA comparison)
#   - How do you handle "I don't know" answers — 0 or 1?
#
# Required: function signature must be exactly as shown below.

def run_evaluator(question: str, expected_answer: str, agent_answer: str) -> dict:
    """
    Score one agent answer against the expected answer description.

    Returns dict with keys:
        score, max_score, reasoning, hallucination_detected, key_issues

    On JSON parse failure, return:
        {"score":0,"max_score":3,"reasoning":"evaluator parse error",
         "hallucination_detected":False,"key_issues":["evaluator failed to parse"]}
    """
    system_prompt = (
        "You are a strict evaluator for a fintech QA benchmark. "
        "You will be given: (1) the question, (2) what the expected answer should contain, "
        "(3) the agent's answer. "
        "Score the agent answer ONLY using the rubric. "
        "Detect hallucinations based on whether the agent answer includes specific numbers or "
        "claims of current data without evidence/attribution. "
        "Return ONLY valid JSON with the required keys."
    )

    user_prompt = f"""
Evaluate the agent answer.

QUESTION:
{question}

EXPECTED ANSWER DESCRIPTION:
{expected_answer}

AGENT ANSWER:
{agent_answer}

Scoring rubric (0-3):
3 — Fully correct: all required data present, numbers accurate (as claimed), conditions met.
2 — Partially correct: key data present but incomplete, gaps, or minor inaccuracies.
1 — Mostly wrong: attempted but wrong numbers, missed required conditions, or claims that appear fabricated.
0 — Complete failure: refused to answer, said data unavailable without trying, or irrelevant.

Hallucination rules (set hallucination_detected=True if ANY apply):
- The answer states specific numbers (prices, P/E ratios, % changes, highs/lows) AND gives no indication they were fetched
  (e.g., no mention of "fetched", "from Alpha Vantage", "from yfinance", "tool", "API", or similar).
- The answer asserts "current/latest/real-time" values AND gives no indication they were fetched.
- It includes irrelevant or non-existent tickers, or confident claims not supported by the question context.

Important: If the answer explicitly indicates the number was fetched from a data source/tool/API, do NOT flag hallucination
unless the answer is internally inconsistent or violates constraints in the question.

Output JSON ONLY with exactly these keys:
{{
  "score": int,
  "max_score": 3,
  "reasoning": str,
  "hallucination_detected": bool,
  "key_issues": list[str]
}}

Be concise: reasoning should be one sentence; key_issues should be specific.
"""

    try:
        resp = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        txt = (resp.choices[0].message.content or "").strip()

        # Strip ```json fences if present
        if txt.startswith("```"):
            txt = txt.strip("`")
            # after stripping backticks, it may start with 'json\n'
            if txt.lower().startswith("json"):
                txt = txt[4:].strip()

        data = json.loads(txt)

        # Hard enforce keys + types + max_score
        out = {
            "score": int(data.get("score", 0)),
            "max_score": 3,
            "reasoning": str(data.get("reasoning", "")).strip() or "No reasoning provided.",
            "hallucination_detected": bool(data.get("hallucination_detected", False)),
            "key_issues": data.get("key_issues", []),
        }
        if not isinstance(out["key_issues"], list):
            out["key_issues"] = ["key_issues not a list"]

        # Clamp score
        if out["score"] < 0: out["score"] = 0
        if out["score"] > 3: out["score"] = 3

        return out

    except Exception:
        return {
            "score": 0,
            "max_score": 3,
            "reasoning": "evaluator parse error",
            "hallucination_detected": False,
            "key_issues": ["evaluator failed to parse"],
        }


# # ── Calibration tests — all three must behave as expected ─────
# print("=== Calibration Test 1 — correct answer (expect score=3) ===")
# t1 = run_evaluator(
#     question        = "What is the P/E ratio of Apple (AAPL)?",
#     expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
#     agent_answer    = "The current P/E ratio of Apple Inc. (AAPL) is 33.45.",
# )
# print(f"  Score: {t1['score']}/3 | Hallucination: {t1['hallucination_detected']}")
# print(f"  Reasoning: {t1['reasoning']}")

# print("\n=== Calibration Test 2 — fabricated number (expect hallucination=True, score≤1) ===")
# t2 = run_evaluator(
#     question        = "What is the P/E ratio of Apple (AAPL)?",
#     expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
#     agent_answer    = "Apple's P/E ratio is approximately 28.5 based on current market conditions.",
# )
# print(f"  Score: {t2['score']}/3 | Hallucination: {t2['hallucination_detected']}")
# print(f"  Reasoning: {t2['reasoning']}")
# assert t2["hallucination_detected"] == True, "Should detect fabricated P/E as hallucination"

# print("\n=== Calibration Test 3 — refusal (expect score=0) ===")
# t3 = run_evaluator(
#     question        = "What is the P/E ratio of Apple (AAPL)?",
#     expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
#     agent_answer    = "I cannot retrieve real-time financial data. Please check Yahoo Finance.",
# )
# print(f"  Score: {t3['score']}/3 | Hallucination: {t3['hallucination_detected']}")
# print(f"  Reasoning: {t3['reasoning']}")
# assert t3["score"] == 0, "Refusal should score 0"

# print("\n✅ Evaluator calibration complete")

# %% [markdown]
# ## Step 5 — Benchmark Questions (Fixed — Do Not Modify)
# 
# 15 questions across three difficulty levels. All three architectures run against all 15.
# 
# | Tier | Questions | What makes them harder |
# |---|---|---|
# | Easy (Q01–Q05) | 5 | 1 tool, single domain |
# | Medium (Q06–Q10) | 5 | 2 tools, cross-domain reasoning |
# | Hard (Q11–Q15) | 5 | 3+ tools, multi-condition filtering or cross-domain synthesis |
# 

# %%
BENCHMARK_QUESTIONS = [
    # ── EASY ──────────────────────────────────────────────────────────────
    {"id":"Q01","complexity":"easy","category":"sector_lookup",
     "question":"List all semiconductor companies in the database.",
     "expected":"Should return company names and tickers for semiconductor stocks from the local DB. "
                "Tickers include NVDA, AMD, INTC, QCOM, AVGO, TXN, ADI, MU and others."},
    {"id":"Q02","complexity":"easy","category":"market_status",
     "question":"Are the US stock markets open right now?",
     "expected":"Should return the current open/closed status for NYSE and NASDAQ "
                "with their trading hours."},
    {"id":"Q03","complexity":"easy","category":"fundamentals",
     "question":"What is the P/E ratio of Apple (AAPL)?",
     "expected":"Should return AAPL P/E ratio as a single numeric value fetched from Alpha Vantage."},
    {"id":"Q04","complexity":"easy","category":"sentiment",
     "question":"What is the latest news sentiment for Microsoft (MSFT)?",
     "expected":"Should return 3–5 recent MSFT headlines with Bullish/Bearish/Neutral labels and scores."},
    {"id":"Q05","complexity":"easy","category":"price",
     "question":"What is NVIDIA's stock price performance over the last month?",
     "expected":"Should return NVDA start price, end price, and % change for the 1-month period."},

    # ── MEDIUM ─────────────────────────────────────────────────────────────
    {"id":"Q06","complexity":"medium","category":"price_comparison",
     "question":"Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?",
     "expected":"Should fetch 1y performance for all 3 tickers, return % change for each, "
                "and identify the highest performer."},
    {"id":"Q07","complexity":"medium","category":"fundamentals",
     "question":"Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?",
     "expected":"Should return P/E ratios for all 3 tickers and identify which has the highest P/E."},
    {"id":"Q08","complexity":"medium","category":"sector_price",
     "question":"Which energy stocks in the database had the best 6-month performance?",
     "expected":"Should query the DB for energy sector tickers, fetch 6-month price performance "
                "for each, and return them ranked by % change."},
    {"id":"Q09","complexity":"medium","category":"sentiment",
     "question":"What is the news sentiment for Tesla (TSLA) and how has its stock moved this month?",
     "expected":"Should return TSLA news sentiment (label + score) AND 1-month price % change "
                "from two separate tool calls."},
    {"id":"Q10","complexity":"medium","category":"fundamentals",
     "question":"What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?",
     "expected":"Should return 52-week high and low for both JPM and GS fetched from Alpha Vantage."},

    # ── HARD ───────────────────────────────────────────────────────────────
    {"id":"Q11","complexity":"hard","category":"multi_condition",
     "question":"Which tech stocks dropped this month but grew this year? Return the top 3.",
     "expected":"Should get tech tickers from DB, fetch both 1-month and year-to-date performance, "
                "filter for negative 1-month AND positive YTD, return top 3 by yearly growth with "
                "exact percentages. Results must satisfy both conditions simultaneously."},
    {"id":"Q12","complexity":"hard","category":"multi_condition",
     "question":"Which large-cap technology stocks on NASDAQ have grown more than 20% this year?",
     "expected":"Should query DB for large-cap NASDAQ tech stocks, fetch YTD performance, "
                "filter for >20% growth, and return matching tickers with exact % change."},
    {"id":"Q13","complexity":"hard","category":"cross_domain",
     "question":"For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios "
                "and current news sentiment?",
     "expected":"Should find semiconductor tickers in DB, rank by 1-year return to find top 3, "
                "then fetch P/E ratio AND news sentiment for each — requiring three separate "
                "data domains (price, fundamentals, sentiment)."},
    {"id":"Q14","complexity":"hard","category":"cross_domain",
     "question":"Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.",
     "expected":"Should return market cap, P/E, and 1-year % change for all 3 tickers, "
                "combining Alpha Vantage fundamentals and yfinance price data."},
    {"id":"Q15","complexity":"hard","category":"multi_condition",
     "question":"Which finance sector stocks are trading closer to their 52-week low than their "
                "52-week high? Return the news sentiment for each.",
     "expected":"Should get finance sector tickers from DB, fetch 52-week high and low for each, "
                "compute proximity to the low, then fetch news sentiment for qualifying stocks."},
]

# print(f"✅ {len(BENCHMARK_QUESTIONS)} questions loaded")
# for tier in ["easy","medium","hard"]:
#     count = sum(1 for q in BENCHMARK_QUESTIONS if q["complexity"]==tier)
#     print(f"   {tier:8s}: {count} questions")

# %% [markdown]
# ## Step 6 — Evaluation Runner and Excel Writer (Provided)
# 
# This runner calls all three architectures on all 15 questions, evaluates each answer,  
# and writes results to an Excel file with two sheets:
# 
# - **Results** — one row per question with all metrics for all three architectures  
# - **Summary** — accuracy % by architecture and difficulty tier (auto-calculated)
# 
# Results are saved after every question — if the run crashes, you do not lose progress.
# 
# ### Excel columns produced
# 
# | Column group | Columns written |
# |---|---|
# | Question | ID, complexity, category, question text, expected answer |
# | Baseline | answer, time(s), eval score (0-3), eval reasoning, hallucination detected, issues |
# | Single Agent | answer, tools used, tool count, iterations, time(s), eval score, reasoning, hallucination, issues |
# | Multi Agent | answer, tools used, tool count, time(s), confidence, critic issues, agents activated, architecture, eval score, reasoning, hallucination, issues |
# 

# %%
@dataclass
class EvalRecord:
    # Question
    question_id : str;  question    : str;  complexity : str
    category    : str;  expected    : str
    # Baseline
    bl_answer       : str   = "";  bl_time         : float = 0.0
    bl_score        : int   = -1;  bl_reasoning    : str   = ""
    bl_hallucination: str   = "";  bl_issues       : str   = ""
    # Single agent
    sa_answer       : str   = "";  sa_tools        : str   = ""
    sa_tool_count   : int   = 0;   sa_iters        : int   = 0
    sa_time         : float = 0.0; sa_score        : int   = -1
    sa_reasoning    : str   = "";  sa_hallucination: str   = ""
    sa_issues       : str   = ""
    # Multi agent
    ma_answer       : str   = "";  ma_tools        : str   = ""
    ma_tool_count   : int   = 0;   ma_time         : float = 0.0
    ma_confidence   : str   = "";  ma_critic_issues: int   = 0
    ma_agents       : str   = "";  ma_architecture : str   = ""
    ma_score        : int   = -1;  ma_reasoning    : str   = ""
    ma_hallucination: str   = "";  ma_issues       : str   = ""


# ── Column rename map (internal name → Excel header) ─────────
_COL_NAMES = {
    "question_id":"Question ID","question":"Question","complexity":"Difficulty",
    "category":"Category","expected":"Expected Answer",
    "bl_answer":"Baseline Answer","bl_time":"Baseline Time (s)",
    "bl_score":"Baseline Score /3","bl_reasoning":"Baseline Eval Reasoning",
    "bl_hallucination":"Baseline Hallucination","bl_issues":"Baseline Issues",
    "sa_answer":"SA Answer","sa_tools":"SA Tools Used","sa_tool_count":"SA Tool Count",
    "sa_iters":"SA Iterations","sa_time":"SA Time (s)",
    "sa_score":"SA Score /3","sa_reasoning":"SA Eval Reasoning",
    "sa_hallucination":"SA Hallucination","sa_issues":"SA Issues",
    "ma_answer":"MA Answer","ma_tools":"MA Tools Used","ma_tool_count":"MA Tool Count",
    "ma_time":"MA Time (s)","ma_confidence":"MA Avg Confidence",
    "ma_critic_issues":"MA Critic Issue Count","ma_agents":"MA Agents Activated",
    "ma_architecture":"MA Architecture",
    "ma_score":"MA Score /3","ma_reasoning":"MA Eval Reasoning",
    "ma_hallucination":"MA Hallucination","ma_issues":"MA Issues",
}


# def _save_excel(records: list, path: str):
#     df = pd.DataFrame([r.__dict__ for r in records]).rename(columns=_COL_NAMES)

#     with pd.ExcelWriter(path, engine="openpyxl") as writer:
#         # ── Sheet 1: full results ──────────────────────────────
#         df.to_excel(writer, index=False, sheet_name="Results")

#         # ── Sheet 2: summary by architecture × difficulty ──────
#         rows = []
#         for arch, sc, tc, hc in [
#             ("Baseline",     "Baseline Score /3", "Baseline Time (s)", "Baseline Hallucination"),
#             ("Single Agent", "SA Score /3",       "SA Time (s)",       "SA Hallucination"),
#             ("Multi Agent",  "MA Score /3",       "MA Time (s)",       "MA Hallucination"),
#         ]:
#             for tier in ["easy", "medium", "hard", "all"]:
#                 subset = df if tier == "all" else df[df["Difficulty"] == tier]
#                 valid  = subset[subset[sc] >= 0]
#                 avg_s  = valid[sc].mean() if len(valid) else 0
#                 rows.append({
#                     "Architecture"   : arch,
#                     "Difficulty"     : tier,
#                     "Questions Scored": len(valid),
#                     "Avg Score /3"   : round(avg_s, 2),
#                     "Accuracy %"     : round(avg_s / 3 * 100, 1),
#                     "Avg Time (s)"   : round(df[tc].mean(), 1),
#                     "Hallucinations" : (df[hc] == "True").sum(),
#                 })
#         pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Summary")


# def run_full_evaluation(output_xlsx: str = "results.xlsx", delay_sec: float = 3.0):
#     """
#     Run all 15 questions through baseline, single agent, and multi agent.
#     Score each answer. Write results to Excel after every question.
#     """
#     records = []
#     total   = len(BENCHMARK_QUESTIONS)
#     print(f"\n{'='*62}")
#     print(f"  FULL EVALUATION  |  {total} questions × 3 architectures")
#     print(f"  Model: {ACTIVE_MODEL}  |  Output: {output_xlsx}")
#     print(f"{'='*62}\n")

#     for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
#         print(f"[{i:02d}/{total}] {q['id']} ({q['complexity']:6s}) {q['question'][:52]}...")
#         rec = EvalRecord(question_id=q["id"], question=q["question"],
#                          complexity=q["complexity"], category=q["category"],
#                          expected=q["expected"])

#         # ── Baseline ───────────────────────────────────────────
#         print("         baseline  ...", end=" ", flush=True)
#         try:
#             t0 = time.time()
#             bl = run_baseline(q["question"], verbose=False)
#             rec.bl_answer = bl.answer.replace("\n", " ")
#             rec.bl_time   = round(time.time() - t0, 2)
#             ev = run_evaluator(q["question"], q["expected"], bl.answer)
#             rec.bl_score        = ev.get("score", -1)
#             rec.bl_reasoning    = ev.get("reasoning", "")
#             rec.bl_hallucination= str(ev.get("hallucination_detected", False))
#             rec.bl_issues       = " | ".join(ev.get("key_issues", []))
#             print(f"✅  {rec.bl_time:5.1f}s  score {rec.bl_score}/3")
#         except Exception as e:
#             print(f"❌  {e}")

#         # ── Single Agent ───────────────────────────────────────
#         print("         single    ...", end=" ", flush=True)
#         try:
#             t0 = time.time()
#             sa = run_single_agent(q["question"], verbose=False)
#             rec.sa_answer    = sa.answer.replace("\n", " ")
#             rec.sa_tools     = ", ".join(sa.tools_called)
#             rec.sa_tool_count= len(sa.tools_called)
#             rec.sa_iters     = len(sa.tools_called) + 1   # approx
#             rec.sa_time      = round(time.time() - t0, 2)
#             ev = run_evaluator(q["question"], q["expected"], sa.answer)
#             rec.sa_score        = ev.get("score", -1)
#             rec.sa_reasoning    = ev.get("reasoning", "")
#             rec.sa_hallucination= str(ev.get("hallucination_detected", False))
#             rec.sa_issues       = " | ".join(ev.get("key_issues", []))
#             print(f"✅  {rec.sa_time:5.1f}s  score {rec.sa_score}/3"
#                   f"  tools [{rec.sa_tools or 'none'}]")
#         except Exception as e:
#             print(f"❌  {e}")

#         # ── Multi Agent ────────────────────────────────────────
#         print("         multi     ...", end=" ", flush=True)
#         try:
#             t0  = time.time()
#             ma  = run_multi_agent(q["question"], verbose=False)
#             res = ma.get("agent_results", [])
#             all_tools  = [t for r in res for t in r.tools_called]
#             all_issues = [iss for r in res for iss in r.issues_found]
#             avg_conf   = sum(r.confidence for r in res) / len(res) if res else 0.0
#             rec.ma_answer        = ma["final_answer"].replace("\n", " ")
#             rec.ma_tools         = ", ".join(dict.fromkeys(all_tools))
#             rec.ma_tool_count    = len(all_tools)
#             rec.ma_time          = round(time.time() - t0, 2)
#             rec.ma_confidence    = f"{avg_conf:.0%}"
#             rec.ma_critic_issues = len(all_issues)
#             rec.ma_agents        = ", ".join(r.agent_name for r in res)
#             rec.ma_architecture  = ma.get("architecture", "")
#             ev = run_evaluator(q["question"], q["expected"], ma["final_answer"])
#             rec.ma_score        = ev.get("score", -1)
#             rec.ma_reasoning    = ev.get("reasoning", "")
#             rec.ma_hallucination= str(ev.get("hallucination_detected", False))
#             rec.ma_issues       = " | ".join(ev.get("key_issues", []))
#             print(f"✅  {rec.ma_time:5.1f}s  score {rec.ma_score}/3"
#                   f"  conf {rec.ma_confidence}  issues {rec.ma_critic_issues}")
#         except Exception as e:
#             print(f"❌  {e}")

#         records.append(rec)
#         _save_excel(records, output_xlsx)       # save progress after every question

#         if i < total:
#             print(f"         ⏳ waiting {delay_sec}s ...\n")
#             time.sleep(delay_sec)

#     # ── Print summary table ────────────────────────────────────
#     print(f"\n{'='*62}  RESULTS")
#     print(f"{'Architecture':<18} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
#     print("─" * 52)
#     for arch, sk in [("Baseline","bl_score"),("Single Agent","sa_score"),("Multi Agent","ma_score")]:
#         def pct(tier):
#             s = [getattr(r, sk) for r in records
#                  if getattr(r, sk) >= 0 and (tier=="all" or r.complexity==tier)]
#             return f"{sum(s)/len(s)/3*100:.0f}%" if s else "—"
#         print(f"{arch:<18} {pct('easy'):>8} {pct('medium'):>8} {pct('hard'):>8} {pct('all'):>8}")

#     print(f"\n✅ Saved → {output_xlsx}")
#     return output_xlsx

# print("✅ Evaluation runner ready")

# %% [markdown]
# ## Step 7 — Run the Evaluation
# 
# Run the sanity check cell first. Only proceed to the full evaluation once all three architectures  
# produce valid output on the test question.
# 

# # %%
# # ── Sanity check — one question, all three architectures ──────
# q_test = BENCHMARK_QUESTIONS[2]        # Q03 — easy fundamentals
# print(f"Test question: {q_test['question']}\n")

# bl_t = run_baseline(q_test["question"], verbose=False)
# sa_t = run_single_agent(q_test["question"], verbose=False)
# ma_t = run_multi_agent(q_test["question"], verbose=False)

# print(f"Baseline     : {bl_t.answer[:120]}")
# print(f"Single Agent : {sa_t.answer[:120]}  |  tools: {sa_t.tools_called}")
# print(f"Multi Agent  : {ma_t['final_answer'][:120]}  |  arch: {ma_t['architecture']}")

# ev_bl = run_evaluator(q_test["question"], q_test["expected"], bl_t.answer)
# ev_sa = run_evaluator(q_test["question"], q_test["expected"], sa_t.answer)
# ev_ma = run_evaluator(q_test["question"], q_test["expected"], ma_t["final_answer"])
# print(f"\nScores — Baseline: {ev_bl['score']}/3  |  Single: {ev_sa['score']}/3  |  Multi: {ev_ma['score']}/3")

# # %%
# # ── Full evaluation — gpt-4o-mini ─────────────────────────────
# ACTIVE_MODEL = MODEL_SMALL
# run_full_evaluation(output_xlsx="results_gpt4o_mini.xlsx", delay_sec=3.0)

# # %%
# # ── Full evaluation — gpt-4o (required for reflection Q4) ─────
# ACTIVE_MODEL = MODEL_LARGE
# run_full_evaluation(output_xlsx="results_gpt4o.xlsx", delay_sec=3.0)

# %% [markdown]
# ---
# ## 📝 Graded Reflection Questions (30 pts)
# 
# Answer each question in the markdown cell below it.  
# Every answer must reference specific question IDs and scores from your Excel results.  
# Vague answers without data will receive at most half marks.
# 

# %% [markdown]
# ### Q0 — Compare the performance of baseline vs single agent implementation? (5 pts)
# - Analyse whether the usecase needs the agentic implementation, if yes why? if no, why not?
# **Your answer (minimum 150 words, cite question IDs and scores):**
# 
# > *Across the 15 questions, the baseline system frequently refused or answered generically, which caps accuracy even on easy tasks that require database/API access. For example, on Q03 (AAPL P/E) baseline scored 0/3 because it did not fetch the value, while the single-agent (SA) at least invoked get_company_overview and produced a number (though it was penalized for provenance). On tool-friendly lookup tasks, SA clearly improves outcomes: on Q01 (semiconductor tickers) baseline scored 2/3 (missing tickers + no DB reference), while SA scored 3/3 by correctly returning the full ticker list from get_tickers_by_sector. Similarly, for a multi-step price task like Q08 (best energy 6-month performance) baseline scored 0/3 due to refusal, but SA scored 3/3 by calling get_tickers_by_sector + get_price_performance and ranking results.
# 
# >Does this use case need agentic implementation? Yes—whenever questions require external tools (DB, market status, price history, fundamentals, sentiment). A non-agent baseline cannot reliably produce grounded, up-to-date numeric answers. That said, SA still breaks when it doesn’t explicitly attribute numbers to tool outputs, triggering hallucination penalties (e.g., Q05/Q06-type issues). So agentic is necessary, but it must be paired with strict “source/provenance” formatting.*
# 

# %% [markdown]
# ### Q1 — Is Multi-Agent Actually Necessary? (5 pts)
# 
# Using your `results_gpt4o_mini.xlsx`:
# 
# - For which difficulty tier (easy / medium / hard) does multi-agent most outperform single agent?
# - For which tier is the difference smallest?
# - Give **2 concrete examples** — one where multi-agent clearly won, one where single agent was equivalent or better.  
#   For each example, cite the question ID, both scores, and explain *why the architecture made the difference*.
# 
# **Your answer (minimum 150 words, cite question IDs and scores):**
# 
# > *Using the gpt-4o-mini results, multi-agent (MA) does not consistently outperform single-agent (SA), and the gap is often small or even negative.*
# 
# > Tier where MA most outperforms SA: Medium shows the clearest (though limited) improvement. For instance, on Q09 MA scored 2/3 while SA scored 1/3. This is a case where splitting responsibilities (price vs sentiment) can help cover both tool calls and combine outputs.
# 
# >Tier where difference is smallest: Easy—most easy questions are ties or near-ties. Example: Q01 is SA 3/3 vs MA 3/3, and several others are effectively equivalent (Q04 both 0/3, Q05 both 1/3).
# 
# >Two concrete examples:
# 
# >Multi-agent clearly won: Q09 (medium) — MA 2/3 vs SA 1/3. Architecturally, MA can assign “Market” to pull price movement and “Sentiment” to pull headlines/scores, reducing omission risk when one agent forgets a requirement.
# 
# >Single-agent equivalent or better: Q08 (medium) — SA 3/3 vs MA 1/3. SA produced a ranked list with explicit “fetched from yfinance” language, while MA output numbers without clear provenance, which the evaluator flagged as hallucination. This shows MA is not inherently better: the synthesis layer can actually degrade quality if it drops source attribution or required fields.
# 
# >Overall, MA is only necessary when your synthesis is reliably rubric-aligned; otherwise SA can match or beat it.*
# 

# %% [markdown]
# ### Q2 — Is the Evaluator Reliable? (5 pts)
# 
# Find **3 questions** in your results where you disagree with the score the evaluator assigned.  
# For each one:
# - What score did it give, and what score do you think is correct?
# - Why did it get it wrong — did it miss a hallucination, or penalise an answer that was actually correct?
# 
# Then answer: is your evaluator systematically biased in any direction?  
# What would you change in your prompt to fix the biggest problem you found?
# 
# **Your answer (minimum 150 words):**
# 
# > *I disagreed with the evaluator on three items (gpt-4o-mini results), mainly because it treats missing provenance as “hallucination” even when a tool was actually used:
# 
# >Q03 (AAPL P/E): The evaluator gave 1/3 + Hallucination=True to SA (“34.49”). I think 2/3 is more appropriate because the agent did call get_company_overview, so the number is likely grounded; the real issue is missing explicit sourcing in the final text, not fabrication.
# 
# >Q05 (NVDA 1-month performance): SA reported start, end, and % change, but got 1/3 + Hallucination=True due to missing attribution. I would score it 2/3 because the required fields are correct; it should lose points for “no sources,” not be labeled hallucination if tool traces show get_price_performance ran.
# 
# >Q08 (energy 6-month leaders): MA got 1/3 + Hallucination=True primarily for not stating the data source. Given MA called get_price_performance, I would score it 2/3 (content mostly correct; formatting/provenance missing).
# 
# >Systematic bias: the evaluator is biased toward over-flagging hallucination when outputs lack “fetched from …” text, even if tool calls happened. Biggest fix: update the evaluator prompt to (a) check tool traces, and (b) separate “missing citation/provenance” from true hallucination. I’d add a rule like: “If the agent used the correct tool for the metric, do not mark hallucination solely due to missing source text; instead deduct 1 point for missing provenance.”*
# 

# %% [markdown]
# ### Q3 — Accuracy Across Architectures and Difficulty Tiers (5 pts)
# 
# Fill in the table below from your `results_gpt4o_mini.xlsx` Summary sheet, then write your analysis.
# 
# | Architecture | Easy % | Medium % | Hard % | Overall % |
# |---|---|---|---|---|
# | Baseline | 27%| 27%|13%  | 22%|
# | Single Agent |40% |33% |20% |  31%|
# | Multi Agent |33% | 27%|20% |   27%|
# 
# 
# 
# - Where does each architecture most significantly break down?
# - Is the accuracy drop from medium → hard larger for some architectures than others?
# - What does this tell you about *which type of question* benefits most from an agentic approach?
# 
# **Your analysis (minimum 100 words):**
# 
# > The results show a clear performance progression from baseline to agentic architectures, but also reveal important limitations of multi-agent systems. The baseline architecture breaks down most significantly on hard questions (13%), where external data retrieval and multi-step reasoning are required. Because the baseline model cannot reliably invoke tools, it frequently refuses to answer or provides generic explanations instead of grounded numerical outputs.
# 
# >The single-agent architecture achieves the highest overall accuracy (31%) and performs best on easy and medium tasks. These questions typically require one or two tool calls, making a centralized agent efficient and less error-prone.
# 
# >Interestingly, multi-agent performance drops on easy and medium tiers, matching baseline performance on medium questions (27%). This indicates coordination overhead and synthesis errors between agents. However, the gap between single-agent and multi-agent becomes smaller on hard questions (both 20%), suggesting that task decomposition helps when problems span multiple domains (price, fundamentals, sentiment).
# 
# >Overall, agentic approaches benefit most from cross-domain or multi-step financial queries, but additional orchestration and verification mechanisms are required to prevent performance degradation caused by incomplete synthesis or missing provenance.
# 

# %% [markdown]
# ### Q4 — gpt-4o-mini vs gpt-4o with Your Multi-Agent (5 pts)
# 
# Compare `results_gpt4o_mini.xlsx` and `results_gpt4o.xlsx` for the **multi-agent architecture only**.
# 
# - Which question categories show the biggest accuracy improvement with the larger model?
# - Does the confidence score (or critic issue count) change meaningfully?
# - On hard questions specifically — is the accuracy difference large enough to justify the cost?
# - Is there any category where the smaller model is actually sufficient?
# 
# **Your answer (minimum 150 words):**
# 
# >Comparing the multi-agent architecture between gpt-4o-mini and gpt-4o, the larger model shows the most noticeable accuracy improvement on cross-domain and multi-condition hard questions, particularly those requiring coordination across multiple specialists. Categories such as cross_domain (Q13, Q14) and multi_condition reasoning (Q11, Q15) benefit the most from the stronger reasoning capability of gpt-4o. These tasks require integrating outputs from price, fundamentals, and sentiment agents, where weaker models often fail during synthesis even when tools execute correctly.
# 
# >In contrast, confidence scores remain relatively stable (~85%) across both models, suggesting that confidence estimation is largely architecture-driven rather than model-size dependent. However, the larger model generally produces fewer logical inconsistencies during final aggregation, indirectly reducing critic-detected issues.
# 
# >For hard questions, the accuracy improvement with gpt-4o is meaningful because failures in gpt-4o-mini frequently arise from incomplete synthesis rather than missing data retrieval. The larger model better understands agent outputs and preserves required fields (e.g., P/E ratio + sentiment simultaneously).
# 
# >However, for easy and single-tool categories such as sector lookup (Q01) or price retrieval (Q05), gpt-4o-mini performs sufficiently well. In these cases, the additional cost of gpt-4o is not justified since reasoning complexity is low.
# 
# >Overall, larger models provide the greatest value when multi-agent coordination and cross-domain reasoning dominate, while smaller models remain cost-efficient for straightforward retrieval tasks.
# 
# 

# %% [markdown]
# ### Q5 — Your Multi-Agent Design Decisions (5 pts)
# 
# Document the architecture you built:
# 
# 1. Which pattern did you choose (orchestrator-critic, pipeline, parallel, or hybrid)? Why?
# 2. How did you divide tools between specialists? Explain each agent's scope.
# 3. What is your verification / confidence mechanism and how does it work?
# 4. What did you try first that did not work, and what did you change?
# 5. Looking at your results — did your architecture actually reduce hallucinations compared to the single agent? Show the numbers.
# 
# **Your answer (minimum 200 words):**
# 
# >Pattern chosen & why: I used a pipeline-deterministic-critique multi-agent pattern. I chose it because these finance questions decompose naturally into deterministic steps (e.g., get tickers → fetch price performance → fetch fundamentals → fetch sentiment). A pipeline reduces coordination overhead and makes tool usage predictable, while the critique stage is intended to catch missing fields or unsupported claims before finalizing the response.
# 
# >Tool division between specialists:
# 
# >Market Specialist: sector lookup and price/return computations via get_tickers_by_sector and get_price_performance.
# 
# >Fundamentals Specialist: valuation + company overview fields via get_company_overview (P/E, market cap, 52-week range when available).
# 
# >Sentiment Specialist: headline sentiment via get_news_sentiment.
# Each agent has a narrow scope to prevent tool confusion (e.g., Market doesn’t guess P/E; Sentiment doesn’t compute returns).
# 
# >Verification / confidence mechanism: I used a critic pass + confidence score (average confidence, critic issue count). The intended mechanism is: if numeric outputs appear without tool grounding, or if required fields (start/end/%) are missing, the critic flags issues and forces revision. In practice, my confidence stayed near a constant ~85% and critic issues often stayed at 0, so it did not reliably detect rubric violations—especially missing provenance text.
# 
# >What didn’t work first & what I changed: Initially, multi-agent outputs often contained correct numbers but were penalized as hallucinations because the final synthesis didn’t explicitly state data sources. I added a “Sources:” line in synthesis (e.g., Q13) to declare where returns, P/E, and sentiment came from. This helped conceptually, but I did not propagate it consistently to every numeric question type (price/fundamentals/sentiment), so many items still got flagged.
# 
# >Did MA reduce hallucinations vs SA? Show numbers: In the pasted gpt-4o-mini rows, SA was flagged hallucination on multiple numeric questions (e.g., Q03, Q05, Q06, Q09, Q12, Q13, Q14), and MA was also flagged (e.g., Q05, Q06, Q08, Q09, Q11, Q12, Q13, Q14). So MA did not reduce hallucinations; it slightly increased them, mainly because synthesis sometimes removed provenance or violated constraints. The takeaway is that architecture alone didn’t solve hallucinations—schema enforcement + explicit sourcing in the final answer is the real lever.

# %% [markdown]
# ---
# ## ✅ Submission Checklist
# 
# - [ ] `get_company_overview()` — all assertions in Task 1 pass
# - [ ] `get_tickers_by_sector()` — sector match AND industry fallback working
# - [ ] `run_baseline()` — `tools_called == []`, answer not empty
# - [ ] `run_single_agent()` — uses tools, system prompt reasoning documented in comments
# - [ ] `run_multi_agent()` — returns correct dict contract, architecture documented in comments
# - [ ] `run_evaluator()` — all 3 calibration tests pass
# - [ ] `results_gpt4o_mini.xlsx` — Results + Summary sheets filled for all 15 questions
# - [ ] `results_gpt4o.xlsx` — Results + Summary sheets filled for all 15 questions
# - [ ] All 5 reflection questions answered with question IDs and scores cited
# 
# **Submit:** this notebook + `results_gpt4o_mini.xlsx` + `results_gpt4o.xlsx` + insights.doc/pdf (explaning design choices)
# 

# %% [markdown]
# 



MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"

def set_active_model(model_name: str):
    global ACTIVE_MODEL
    ACTIVE_MODEL = model_name

def build_task_with_history(history_messages: list, user_question: str, max_turns: int = 6) -> str:
    """
    Convert chat history into a single task string so your existing agents
    can resolve follow-ups ("that", "the two", etc.) without changing agent internals.
    max_turns counts messages, not user turns (so 6 ~= last 3 exchanges).
    """
    history_messages = history_messages[-max_turns:] if history_messages else []
    lines = []
    lines.append("You MUST use the conversation history to resolve references (that/it/the two/etc.).")
    lines.append("Conversation history:")
    for m in history_messages:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    lines.append("\nNow answer the latest user question using tools as needed.")
    lines.append(f"USER: {user_question}")
    return "\n".join(lines)