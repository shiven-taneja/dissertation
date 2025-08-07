from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

import os
import json
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import psycopg2
import psycopg2.extras
from drl_utrans.utils.cfg import load_cfg

from drl_utrans.utils.indicators import make_feature_matrix, FEATURE_COLS

from feature_generation.headline_features import main as enrich_headlines

DATA_ROOT = Path("data")
CACHE_ROOT = DATA_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "GPT OSS 120B 128k")  # per user request
REQUEST_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "40"))

PG_DSN = os.getenv("PG_DSN") 

HEADLINE_TABLE = os.getenv("NEWS_TABLE", "fnspid_news")
HEADLINE_TEXT_COL = os.getenv("NEWS_TEXT_COL", "headline")
HEADLINE_TICKER_COL = os.getenv("NEWS_TICKER_COL", "stock_symbol")
HEADLINE_DATE_COL = os.getenv("NEWS_DATE_COL", "date")


def _ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix == ".csv":
                return pd.read_csv(path)
        except Exception:
            return None
    return None

def _save_cache(df: pd.DataFrame, path: Path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported cache format")
    
# -------------------- Groq LLM helpers --------------------

SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"})

HEADLINE_SYSTEM = "You are a finance assistant. Return only valid JSON."
HEADLINE_TMPL = """You will be given a single financial news headline about a public company or ETF.
Task: Score the directional sentiment for the named instrument over the next few trading days.
Return ONLY valid JSON: {{"score": <float in [-1, 1]>}}

Headline: "{headline}"
"""

TECH_SYSTEM = (
    "You are a quantitative trading assistant. Return ONLY valid JSON with keys 'score' and 'rationale'."
)
TECH_TMPL = """We summarize the current technical state of a security.
Provide a directional sentiment score in [-1, 1] for the next 1–5 trading days.
Use the summary; do not invent facts.
Return ONLY valid JSON: {{"score": <float in [-1, 1]>}}

Ticker: {ticker}
Date: {date}
Summary: {summary}
"""

def groq_chat_json(system: str, user_content: str, response_keys: Tuple[str, ...] = ("score",), retries: int = 3) -> Dict:
    url = f"{GROQ_BASE_URL}/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries):
        try:
            r = SESSION.post(url, data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            obj = json.loads(content)
            # sanity
            for k in response_keys:
                if k not in obj:
                    raise ValueError(f"Missing key '{k}' in response: {obj}")
            return obj
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_headlines(ticker: str) -> pd.DataFrame:
    """Pull raw headlines for ticker from Postgres. Expects date column and optional time."""
    with psycopg2.connect(PG_DSN) as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT {HEADLINE_TICKER_COL} AS ticker,
                       {HEADLINE_TEXT_COL}    AS headline,
                       {HEADLINE_DATE_COL}    AS date,
                FROM {HEADLINE_TABLE}
                WHERE {HEADLINE_TICKER_COL} = %s
                ORDER BY {HEADLINE_DATE_COL} ASC
                """,
                (ticker,),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["headline", "ts"])
    return df

def score_headlines_groq(ticker: str, df_headlines: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE_ROOT / "groq_headlines" / f"{ticker}.parquet"
    _ensure_dirs(cache_path.parent)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    scores = []
    for _, row in df_headlines.iterrows():
        h = str(row["headline"])[:500]
        user = HEADLINE_TMPL.format(headline=h)
        obj = groq_chat_json(HEADLINE_SYSTEM, user, response_keys=("score",))
        s = float(obj["score"])
        s = max(-1.0, min(1.0, s))
        scores.append(s)
    out = df_headlines.copy()
    out["score"] = scores
    _save_cache(out, cache_path)
    return out

def aggregate_headline_daily(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-day sentiment features."""
    df_scores["date"] = pd.to_datetime(df_scores["ts"]).dt.date
    g = df_scores.groupby("date")["score"]
    agg = pd.DataFrame({
        "news_sent_mean_1d": g.mean(),
        "news_sent_std_1d": g.std().fillna(0.0),
        "news_sent_maxabs_1d": g.apply(lambda x: np.abs(x).max() if len(x)>0 else 0.0),
        "news_count_1d": g.size(),
    }).reset_index()
    agg["date"] = pd.to_datetime(agg["date"])
    return agg

def tech_summary_row(row: pd.Series) -> str:
    parts = []
    parts.append(f"close_change_5d: {row['ret_5d']*100:+.2f}%")
    parts.append(f"rsi14: {row['rsi14']:.1f} ({'overbought' if row['rsi14']>70 else 'oversold' if row['rsi14']<30 else 'neutral'})")
    parts.append(f"macd_vs_signal: {'bullish' if row['macd']>row['macd_signal'] else 'bearish'}")
    parts.append(f"sma20_vs_50: {'above' if row['sma20']>row['sma50'] else 'below'}")
    return "; ".join(parts)

def score_tech_sentiment_groq(ticker: str, df_ind: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE_ROOT / "groq_tech" / f"{ticker}.parquet"
    _ensure_dirs(cache_path.parent)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    rows = []
    for idx, r in df_ind.iterrows():
        if pd.isna(r["ret_5d"]) or pd.isna(r["sma50"]):
            continue  # skip warmup
        summary = tech_summary_row(r)
        user = TECH_TMPL.format(ticker=ticker, date=str(idx.date()), summary=summary)
        obj = groq_chat_json(TECH_SYSTEM, user, response_keys=("score",))
        s = float(obj["score"])
        s = max(-1.0, min(1.0, s))
        rows.append({"date": idx, "tech_sent_score": s})
    out = pd.DataFrame(rows)
    _save_cache(out, cache_path)
    return out

# -------------------- Prices & alignment --------------------

def load_prices_csv(ticker: str) -> pd.DataFrame:
    """
    Replace this with your existing price loader. We expect columns:
    ['date','open','high','low','close','volume'] with date as datetime.
    """
    # Placeholder path: data/prices/{ticker}.csv
    p = DATA_ROOT / "prices" / f"{ticker}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing price file: {p}")
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    return df

def build_datasets_with_llm(ticker: str, train_ratio: float = 0.7) -> Tuple[Path, Path]:
    """
    Build train/test CSVs under data/{TICKER}/ with added LLM features.
    The date range is clipped to the min/max dates where headlines exist for the ticker.
    """
    _ensure_dirs(DATA_ROOT)
    raw_prices = load_prices_csv(ticker)

    # Headlines → min/max date for this ticker
    df_h = fetch_headlines(ticker)
    if df_h.empty:
        raise ValueError(f"No headlines for {ticker}")
    min_d = pd.to_datetime(df_h["ts"].min()).normalize()
    max_d = pd.to_datetime(df_h["ts"].max()).normalize()

    # Clip prices to headlines window
    prices = raw_prices.loc[(raw_prices.index >= min_d) & (raw_prices.index <= max_d)].copy()
    if len(prices) < 200:
        raise ValueError(f"Not enough price data in headline window for {ticker}")

    # Headline scores → aggregate daily
    df_scored = score_headlines_groq(ticker, df_h)
    daily_news = aggregate_headline_daily(df_scored)

    # Indicators + tech sentiment
    ind = make_feature_matrix(prices)
    tech_sent = score_tech_sentiment_groq(ticker, ind)

    # Merge all features on date
    feat = prices.copy()
    feat["date"] = feat.index
    feat = feat.merge(daily_news, on="date", how="left")
    feat = feat.merge(tech_sent, on="date", how="left")

    # Fill missing features conservatively
    feat["news_sent_mean_1d"] = feat["news_sent_mean_1d"].fillna(0.0)
    feat["news_sent_std_1d"] = feat["news_sent_std_1d"].fillna(0.0)
    feat["news_sent_maxabs_1d"] = feat["news_sent_maxabs_1d"].fillna(0.0)
    feat["news_sent_count_1d"] = feat["news_count_1d"].fillna(0.0)
    feat["tech_sent_score"] = feat["tech_sent_score"].fillna(0.0)

    # Create base DRL-UTrans feature (open-close difference)
    feat["open_close_diff"] = feat["open"] - feat["close"]

    # Save under data/{TICKER}/
    out_dir = DATA_ROOT / ticker.upper()
    _ensure_dirs(out_dir)

    # Train/test split
    n = len(feat)
    cut = int(n * train_ratio)
    train_df = feat.iloc[:cut].reset_index(drop=True)
    test_df = feat.iloc[cut:].reset_index(drop=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Also save a manifest with column descriptions
    manifest = {
        "ticker": ticker.upper(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "columns": list(feat.columns),
        "date_min": str(feat["date"].min().date()),
        "date_max": str(feat["date"].max().date()),
        "groq_model": GROQ_MODEL,
        "groq_base_url": GROQ_BASE_URL,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return train_path, test_path
