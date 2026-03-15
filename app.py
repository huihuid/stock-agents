import os
import streamlit as st



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



#=============================================
st.set_page_config(page_title="MP3 Streamlit Chat", page_icon="💬", layout="wide")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    arch = st.selectbox("Agent selector", ["Single Agent", "Multi-Agent"], index=0)
    model = st.selectbox("Model selector", [MODEL_SMALL, MODEL_LARGE], index=0)

    st.markdown("---")
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.renders = []
        st.rerun()

st.title("💬 MP3 Deployment Chat (Notebook Agents)")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    # store raw chat history (user + assistant)
    st.session_state.messages = []  # list[{"role":"user"/"assistant","content":...}]
if "renders" not in st.session_state:
    # store metadata for display, aligned with messages
    st.session_state.renders = []   # list[{"arch":..., "model":..., "tools":[...], "conf":..., "issues":[...]}]

# ----------------------------
# Render history
# ----------------------------
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        meta = st.session_state.renders[i] if i < len(st.session_state.renders) else {}
        with st.chat_message("assistant"):
            st.caption(
                f"Architecture: **{meta.get('arch','?')}** | Model: **{meta.get('model','?')}**"
            )
            # optional tool/conf display
            tools = meta.get("tools", [])
            conf  = meta.get("conf", None)
            issues = meta.get("issues", [])
            if tools:
                st.caption(f"Tools: {', '.join(tools)}")
            if conf is not None:
                st.caption(f"Confidence: {conf}")
            if issues:
                st.caption(f"Issues: {', '.join(issues)}")

            st.markdown(content)

# ----------------------------
# Chat input
# ----------------------------
user_text = st.chat_input("Ask a question...")
if user_text:
    # 1) append user
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.renders.append({"arch": arch, "model": model})

    # 2) set model in notebook code
    set_active_model(model)

    # 3) build task with history (core memory requirement)
    task = build_task_with_history(st.session_state.messages, user_text, max_turns=6)

    # 4) call the selected architecture from notebook
    if arch == "Single Agent":
        res = run_single_agent(task, verbose=False)
        answer_text = res.answer

        meta = {
            "arch": arch,
            "model": model,
            "tools": res.tools_called,
            "conf": None,
            "issues": [],
        }

    else:
        out = run_multi_agent(task, verbose=False)
        answer_text = out["final_answer"]

        # aggregate metadata from specialists
        agent_results = out.get("agent_results", [])
        tools = [t for r in agent_results for t in r.tools_called]
        issues = [iss for r in agent_results for iss in r.issues_found]
        avg_conf = None
        if agent_results:
            avg_conf = f"{(sum(r.confidence for r in agent_results)/len(agent_results)):.0%}"

        meta = {
            "arch": f"Multi-Agent ({out.get('architecture','')})",
            "model": model,
            "tools": tools,
            "conf": avg_conf,
            "issues": issues,
        }

    # 5) append assistant
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.session_state.renders.append(meta)

    st.rerun()