# add_news_sentiment.py (PostgreSQL‑backed FNSPID)
"""Augment train/test CSVs with FinBERT sentiment **pulled from a local
PostgreSQL copy of the FNSPID news dataset**.

Why move to Postgres?
---------------------
* **One‑time download / ingest** – avoid repeated HuggingFace streaming.
* **FAST point queries** – `SELECT headline FROM fnspid_news
  WHERE stock_symbol='AAPL' AND date='2010‑02‑01';` returns in milliseconds
  once indexed.
* Fits naturally into pgAdmin workflows.

Prerequisites
-------------
1. A running PostgreSQL ≥12 and superuser access (pgAdmin UI or psql).
2. A table `fnspid_news(stock_symbol TEXT, date DATE, headline TEXT)` filled
   with the entire dataset (see README in the chat message that accompanied
   this file for setup steps).
3. Environment variable `PG_CONN` holding your libpq connection string, e.g.
   ```bash
   export PG_CONN="host=localhost dbname=fnspid user=postgres password=secret"
   ```
4. `pip install psycopg2-binary transformers pandas` (add `torch` for GPU).

Usage
-----
```bash
python add_news_sentiment.py   # defaults to AAPL + data/AAPL_*.csv
```

If you prefer explicit arguments:
```python
from pathlib import Path
from add_news_sentiment import main
main(
    ticker="MSFT",
    train=Path("data/MSFT_train.csv"),
    test=Path("data/MSFT_test.csv"),
    pg_conn="host=localhost dbname=fnspid user=... password=..."
)
```
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from sqlalchemy import create_engine
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# --------------------------------------------------------------------------- #
MODEL_NAME = "ProsusAI/finbert"
TABLE_NAME = "fnspid_news"       # change if you used a different table name

# --------------------------------------------------------------------------- #
# ---------------------  FinBERT pipeline builder  --------------------------- #

def build_finbert() -> pipeline:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        truncation=True,
        batch_size=32,
        device=-1,  # set 0 for GPU
    )

# --------------------------------------------------------------------------- #
# --------------------  PostgreSQL headline retrieval  ----------------------- #



def db_headlines(ticker: str) -> List[str]:
    engine = create_engine(os.getenv("PG_CONN"))
    hds = pd.read_sql(f"SELECT stock_symbol, date, headline FROM fnspid_news WHERE stock_symbol=%s;",
                 con=engine, params=(ticker.upper(),))
    return hds
# --------------------------------------------------------------------------- #
# --------------------  Sentiment scoring & enrichment  ---------------------- #

def sentiment_score(headlines: List[str], clf: pipeline) -> float:
    if not headlines:
        return 0.0
    diffs = []
    for i in range(0, len(headlines), 32):
        for result in clf(headlines[i : i + 32]):
            scores = {d["label"].lower(): d["score"] for d in result}
            diffs.append(scores.get("positive", 0.0) - scores.get("negative", 0.0))
    return float(sum(diffs) / len(diffs))


def enrich(csv_path: Path, ticker: str, clf: pipeline) -> None:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "news_sentiment" in df.columns:
        print(f"{csv_path.name}: news_sentiment already exists – skipping")
        return

    headlines = db_headlines(ticker).groupby("date")["headline"].apply(list).to_dict()
    print(headlines)
    return
    scores = {}
    for d in df["Date"].dt.date.unique():
        scores[d] = sentiment_score(db_headlines(ticker, d), clf)

    df["news_sentiment"] = df["Date"].dt.date.map(scores).fillna(0.0)

    backup = csv_path.with_suffix(f"{csv_path.suffix}.bak_{int(time.time())}")
    csv_path.rename(backup)
    df.to_csv(csv_path, index=False)
    print(f"{csv_path.name}: added news_sentiment (backup → {backup.name})")

# --------------------------------------------------------------------------- #
#                                    CLI                                       #
# --------------------------------------------------------------------------- #

def main(
    ticker: str,
    df
) -> None:
    clf = build_finbert()
    df = enrich(df, ticker, clf)
    return df

if __name__ == "__main__":
    main("AAPL", Path("data/AAPL_train.csv"), Path("data/AAPL_test.csv"))
