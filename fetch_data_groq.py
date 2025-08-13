from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import psycopg2
import psycopg2.extras
import random
from groq import Groq, InternalServerError, APIStatusError
from drl_utrans.utils.cfg import load_cfg

from drl_utrans.utils.indicators import make_feature_matrix, FEATURE_COLS

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path("data")
CACHE_ROOT = DATA_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
GROQ_MODEL = os.getenv("GROQ_MODEL")  # per user request
REQUEST_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "40"))

PG_DSN = os.getenv("PG_DSN") 

HEADLINE_TABLE = os.getenv("NEWS_TABLE", "fnspid_news")
HEADLINE_TEXT_COL = os.getenv("NEWS_TEXT_COL", "headline")
HEADLINE_TICKER_COL = os.getenv("NEWS_TICKER_COL", "stock_symbol")
HEADLINE_DATE_COL = os.getenv("NEWS_DATE_COL", "date")

PROGRESS_DIR = DATA_ROOT / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

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
        except Exception as e:
            print(f"Warning: Could not load cache from {path}: {e}")
            return None
    return None

def _save_cache(df: pd.DataFrame, path: Path):
    try:
        _ensure_dirs(path.parent)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        elif path.suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported cache format")
    except Exception as e:
        print(f"Warning: Could not save cache to {path}: {e}")

# ==================== Progress Tracking ====================
class ProgressTracker:
    def __init__(self, task_name: str, ticker: str):
        self.task_name = task_name
        self.ticker = ticker
        self.progress_file = PROGRESS_DIR / f"{ticker}_{task_name}.json"
        self.state = self.load_state()
    
    def load_state(self) -> dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed": [], "partial": {}, "last_update": None}
    
    def save_state(self):
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_completed(self, item_id: str) -> bool:
        return item_id in self.state["completed"]
    
    def mark_completed(self, item_id: str):
        if item_id not in self.state["completed"]:
            self.state["completed"].append(item_id)
        self.save_state()
    
    def save_partial(self, key: str, data: any):
        self.state["partial"][key] = data
        self.save_state()
    
    def get_partial(self, key: str) -> any:
        return self.state["partial"].get(key)
    
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
    last_err = None
    for attempt in range(retries):
        try: 
            client = Groq(
                api_key=GROQ_API_KEY,
            )
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature= 0.0,
                top_p= 1.0,
                messages= [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format= {"type": "json_object"},
                reasoning_effort="medium",
                stream= False,
                stop= None,
            )

            result = json.loads(completion.choices[0].message.content)

            if "score" not in result:
                result["score"] = 0.0 
            elif not isinstance(result["score"], (int, float)):
                result["score"] = 0.0
            return result
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}, retrying...")
            last_err = e
        except (InternalServerError, APIStatusError) as e:
            print(f"Groq API error: {e}, retrying ({attempt+1}/{retries})...")
            last_err = e
            # exponential backoff + jitter
            sleep_s = min(20, 0.75 * (2 ** attempt)) + random.random() * 0.5
            time.sleep(sleep_s)
        except Exception as e:
            print(f"Unexpected error: {e}")
            last_err = e
            time.sleep(1)
    
    print(f"ERROR: Groq API failed after retries: {last_err}")
    return {key: (0.0 if key == "score" else "") for key in response_keys}
    
def fetch_headlines(ticker: str) -> pd.DataFrame:
    """Pull raw headlines for ticker from Postgres with error handling."""
    if not PG_DSN:
        print("ERROR: PG_DSN not set")
        return pd.DataFrame()
    
    try:
        with psycopg2.connect(host='localhost', user="postgres", password=PG_DSN, dbname="fnspid") as con:
            with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = f"""
                SELECT {HEADLINE_TICKER_COL} AS ticker,
                       {HEADLINE_TEXT_COL}    AS headline,
                       {HEADLINE_DATE_COL}    AS date
                FROM {HEADLINE_TABLE}
                WHERE {HEADLINE_TICKER_COL} = %s
                ORDER BY {HEADLINE_DATE_COL} ASC
                """
                cur.execute(query, (ticker,))
                rows = cur.fetchall()
    except Exception as e:
        print(f"ERROR fetching headlines: {e}")
        return pd.DataFrame()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["headline", "ts"])
    return df

def score_headlines_groq_incremental(ticker: str, df_headlines: pd.DataFrame) -> pd.DataFrame:
    """Score headlines with incremental saving and recovery."""
    cache_path = CACHE_ROOT / "groq_headlines" / f"{ticker}.parquet"
    _ensure_dirs(cache_path.parent)
    
    # Try to load completed cache first
    cached = _load_cache(cache_path)
    if cached is not None and len(cached) == len(df_headlines):
        print(f"Using cached headline scores for {ticker}")
        return cached
    
    # Progress tracking
    tracker = ProgressTracker("headline_scoring", ticker)
    
    # Load partial results if available
    partial_cache = CACHE_ROOT / "groq_headlines" / f"{ticker}_partial.parquet"
    if partial_cache.exists():
        partial_df = _load_cache(partial_cache)
        if partial_df is not None:
            print(f"Resuming from {len(partial_df)} previously scored headlines")
            start_idx = len(partial_df)
        else:
            partial_df = pd.DataFrame()
            start_idx = 0
    else:
        partial_df = pd.DataFrame()
        start_idx = 0
    
    scores = list(partial_df.get('score', []))
    
    # Score remaining headlines
    for idx in range(start_idx, len(df_headlines)):
        row = df_headlines.iloc[idx]
        h = str(row["headline"])[:500]
        
        # Check if already scored
        headline_id = f"{ticker}_{idx}_{hash(h)}"
        if tracker.is_completed(headline_id):
            # Retrieve saved score
            saved_score = tracker.get_partial(f"score_{idx}")
            if saved_score is not None:
                scores.append(saved_score)
                continue
        
        # Score the headline
        user = HEADLINE_TMPL.format(headline=h)
        obj = groq_chat_json(HEADLINE_SYSTEM, user, response_keys=("score",))
        print(f"[{idx+1}/{len(df_headlines)}] Scored headline: {h[:50]}... → {obj.get('score', 0)}")
        
        s = float(obj.get("score", 0.0))
        s = max(-1.0, min(1.0, s))
        scores.append(s)
        
        # Save progress
        tracker.save_partial(f"score_{idx}", s)
        tracker.mark_completed(headline_id)
        
        # Save partial results every 10 headlines
        if (idx + 1) % 10 == 0:
            partial_out = df_headlines.iloc[:idx+1].copy()
            partial_out["score"] = scores
            _save_cache(partial_out, partial_cache)
            print(f"Saved partial results: {idx+1}/{len(df_headlines)}")
    
    # Final output
    out = df_headlines.copy()
    out["score"] = scores
    _save_cache(out, cache_path)
    
    # Clean up partial cache
    if partial_cache.exists():
        partial_cache.unlink()
    
    return out

def aggregate_headline_daily(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-day sentiment features with error handling."""
    if df_scores.empty:
        return pd.DataFrame()
    
    df_scores["date"] = pd.to_datetime(df_scores["ts"]).dt.date
    g = df_scores.groupby("date")["score"]
    
    agg = pd.DataFrame({
        "news_sent_mean_1d": g.mean(),
        "news_sent_std_1d": g.std().fillna(0.0),
        "news_sent_maxabs_1d": g.apply(lambda x: np.abs(x).max() if len(x) > 0 else 0.0),
        "news_count_1d": g.size(),  # Fixed: was news_count_1d not news_sent_count_1d
    }).reset_index()
    
    agg["date"] = pd.to_datetime(agg["date"])
    return agg

def tech_summary_row(row: pd.Series) -> str:
    """Generate technical summary with error handling."""
    parts = []
    if 'ret_5d' in row and pd.notna(row['ret_5d']):
        parts.append(f"close_change_5d: {row['ret_5d']*100:+.2f}%")

    if 'rsi_14' in row and pd.notna(row['rsi_14']):
        rsi = row['rsi_14']
        rsi_status = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
        parts.append(f"rsi_14: {rsi:.1f} ({rsi_status})")

    if 'macd' in row and 'macd_signal' in row and pd.notna(row['macd']) and pd.notna(row['macd_signal']):
        macd_status = 'bullish' if row['macd'] > row['macd_signal'] else 'bearish'
        parts.append(f"macd_vs_signal: {macd_status}")

    if 'sma20' in row and 'sma50' in row and pd.notna(row['sma20']) and pd.notna(row['sma50']):
        sma_status = 'above' if row['sma20'] > row['sma50'] else 'below'
        parts.append(f"sma20_vs_50: {sma_status}")

    if 'percent_b' in row and pd.notna(row['percent_b']):
        pb = float(row['percent_b'])
        if pb <= 0: zone = "below_lower"
        elif pb < 0.2: zone = "near_lower"
        elif pb <= 0.8: zone = "middle"
        elif pb < 1: zone = "near_upper"
        else: zone = "above_upper"
        parts.append(f"percent_b_zone: {zone}")

    return "; ".join(parts) if parts else "No technical data available"


def score_tech_sentiment_groq_incremental(ticker: str, df_ind: pd.DataFrame) -> pd.DataFrame:
    """Score technical sentiment with incremental saving."""
    cache_path = CACHE_ROOT / "groq_tech" / f"{ticker}.parquet"
    _ensure_dirs(cache_path.parent)

    cached = _load_cache(cache_path)
    if cached is not None:
        print(f"Using cached technical sentiment for {ticker}")
        return cached

    tracker = ProgressTracker("tech_scoring", ticker)

    partial_cache = CACHE_ROOT / "groq_tech" / f"{ticker}_partial.parquet"
    if partial_cache.exists():
        partial_df = _load_cache(partial_cache)
        processed_dates = set(pd.to_datetime(partial_df['date'])) if partial_df is not None else set()
        rows = partial_df.to_dict('records') if partial_df is not None else []
    else:
        processed_dates, rows = set(), []

    def usable(r: pd.Series) -> bool:
        keys = ["ret_5d","rsi_14","macd","macd_signal","sma20","sma50","percent_b"]
        return any(pd.notna(r.get(k, np.nan)) for k in keys)

    total_rows = sum(1 for _, r in df_ind.iterrows() if usable(r))
    current = len(rows)

    for idx, r in df_ind.iterrows():
        if not usable(r):
            continue
        if pd.to_datetime(idx) in processed_dates:
            continue

        current += 1
        summary = tech_summary_row(r)
        user = TECH_TMPL.format(ticker=ticker, date=str(pd.to_datetime(idx).date()), summary=summary)
        obj = groq_chat_json(TECH_SYSTEM, user, response_keys=("score",))

        s = float(obj.get("score", 0.0))
        s = max(-1.0, min(1.0, s))
        print(f"[{current}/{total_rows}] Tech sentiment for {pd.to_datetime(idx).date()}: {s:+.3f}")

        rows.append({"date": pd.to_datetime(idx).normalize(), "tech_sent_score": s})

        if current % 5 == 0:
            partial_out = pd.DataFrame(rows)
            _save_cache(partial_out, partial_cache)

    out = pd.DataFrame(rows).sort_values("date")
    _save_cache(out, cache_path)
    if partial_cache.exists():
        partial_cache.unlink()
    return out


def load_prices_csv(ticker, min_date, max_date):
    cache_file = CACHE_ROOT / f"{ticker}_{min_date.date()}_{max_date.date()}.csv"
    
    if cache_file.exists():
        print(f"Loading cached prices for {ticker}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df
    
    try:
        print(f"Downloading {ticker} [{min_date.date()} → {max_date.date()}]...")
        df = yf.download(ticker, start=min_date, end=max_date, auto_adjust=True, progress=False)
        
        if df.empty:
            raise ValueError(f"No price data retrieved for {ticker}")
        
        # Handle multi-index columns (common with single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Remove timezone to ensure consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Ensure index has consistent name
        df.index.name = 'Date'
        
        # Select and convert columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Save cache
        df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        raise

def get_llm_df(ind, price_sr, prices):
    llm_df = pd.DataFrame(index=ind.index)
    llm_df["rsi_14"] = ind["rsi_14"]
    llm_df["macd"] = ind["macd"]
    llm_df["macd_signal"] = ind["macd_sig"]        # alias for LLM
    # SMAs from original Close series, then reindex to the aligned dates
    sma20 = prices["Close"].rolling(20).mean()
    sma50 = prices["Close"].rolling(50).mean()
    llm_df["sma20"] = sma20.reindex(ind.index)
    llm_df["sma50"] = sma50.reindex(ind.index)
    # 5-day return and %B using your existing bollinger outputs
    llm_df["ret_5d"] = price_sr.pct_change(5)
    llm_df["percent_b"] = (ind["close"] - ind["boll_low"]) / (ind["boll_up"] - ind["boll_low"])
    llm_df.replace([np.inf,-np.inf], np.nan, inplace=True)
    llm_df.dropna(inplace=True)
    return llm_df

def build_datasets_with_llm(ticker: str, train_ratio: float = 0.7, validate_only: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Build train/test CSVs with LLM features.
    
    Args:
        ticker: Stock ticker symbol
        train_ratio: Ratio of data for training
        validate_only: If True, only run validation without processing
    
    Returns:
        Tuple of (train_path, test_path) or (None, None) if validation fails
    """
    print("=" * 60)
    print(f"Starting pipeline for {ticker}")
    print("=" * 60)
    
    _ensure_dirs(DATA_ROOT)
    
    # Check for existing completed dataset
    out_dir = DATA_ROOT / ticker.upper()
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    manifest_path = out_dir / "manifest.json"
    
    if train_path.exists() and test_path.exists() and manifest_path.exists():
        print(f"\n✅ Complete dataset already exists for {ticker}")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")
        response = input("Regenerate? (y/N): ").strip().lower()
        if response != 'y':
            return train_path, test_path
    
    try:
        # Fetch headlines
        print("\n📰 Fetching headlines...")
        df_h = fetch_headlines(ticker)
        print(f"   Found {len(df_h)} headlines")
        
        if df_h.empty:
            raise ValueError(f"No headlines found for {ticker}")
        
        min_d = pd.to_datetime(df_h["ts"].min()).normalize()
        max_d = pd.to_datetime(df_h["ts"].max()).normalize()
        print(f"   Date range: {min_d.date()} to {max_d.date()}")
        
        # Load prices
        print("\n📈 Loading price data...")
        prices= load_prices_csv(ticker, min_date=min_d, max_date=max_d)
        print(f"   Loaded {len(prices)} price records")
        
        if len(prices) < 200:
            raise ValueError(f"Insufficient price data ({len(prices)} < 200 required)")
        
        # Score headlines
        print("\n🤖 Scoring headlines with LLM...")
        df_scored = score_headlines_groq_incremental(ticker, df_h)
        daily_news = aggregate_headline_daily(df_scored)
        print(f"   Aggregated to {len(daily_news)} daily sentiment records")
        
        print(prices.head())
        # Technical indicators
        print("\n📊 Generating technical indicators...")
        features, closes = make_feature_matrix(prices)
        aligned_dates = prices.index[-len(features) :]

        feat_df  = pd.DataFrame(features, columns=FEATURE_COLS, index=aligned_dates)
        price_sr = pd.Series(closes, index=aligned_dates, name="close")

        ind = pd.concat([feat_df, price_sr], axis=1)

        print(f"   Generated indicators for {len(ind)} days")

        llm_df = get_llm_df(ind, price_sr, prices)
        # Score technical sentiment
        print("\n🤖 Scoring technical sentiment with LLM...")
        tech_sent = score_tech_sentiment_groq_incremental(ticker, llm_df)
        print(f"   Scored {len(tech_sent)} technical records")
        
        # Merge all features
        print("\n🔄 Merging all features...")
        feat = ind.copy()
        feat['date'] = feat.index
        feat = feat.reset_index(drop=True)
        
        # Merge with proper date alignment
        feat = feat.merge(daily_news, on="date", how="left")
        feat = feat.merge(tech_sent, on="date", how="left")
        
        # Fill missing features
        feat["news_sent_mean_1d"] = feat["news_sent_mean_1d"].fillna(0.0)
        feat["news_sent_std_1d"] = feat["news_sent_std_1d"].fillna(0.0)
        feat["news_sent_maxabs_1d"] = feat["news_sent_maxabs_1d"].fillna(0.0)
        feat["news_count_1d"] = feat["news_count_1d"].fillna(0)  # Fixed column name
        feat["tech_sent_score"] = feat["tech_sent_score"].fillna(0.0)
        
        # Create base feature
        
        # Split train/test
        print("\n✂️  Splitting train/test...")
        n = len(feat)
        cut = int(n * train_ratio)
        
        train_tmp = feat.iloc[: cut]
        mu   = train_tmp[FEATURE_COLS].mean()
        sigma= train_tmp[FEATURE_COLS].std(ddof=0)
        feat[FEATURE_COLS] = (feat[FEATURE_COLS] - mu) / sigma

        train_df = feat.iloc[:cut].reset_index(drop=True)
        test_df = feat.iloc[cut:].reset_index(drop=True)
        
        print(f"   Train: {len(train_df)} rows")
        print(f"   Test: {len(test_df)} rows")
        
        
        # Save datasets
        _ensure_dirs(out_dir)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save manifest
        manifest = {
            "ticker": ticker.upper(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "columns": list(feat.columns),
            "date_min": str(feat["date"].min().date()),
            "date_max": str(feat["date"].max().date()),
            "groq_model": GROQ_MODEL,
            "groq_base_url": GROQ_BASE_URL,
            "generated_at": datetime.now().isoformat(),
            "train_ratio": train_ratio
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ Successfully created datasets for {ticker}")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")
        print(f"   Manifest: {manifest_path}")
        print("=" * 60)
        
        return train_path, test_path
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return None, None
    