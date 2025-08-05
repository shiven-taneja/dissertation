#!/usr/bin/env python
"""
Download OHLCV with yfinance, build 14-dim feature matrix, and
time-split into train / test CSVs.

Usage:
    python scripts/fetch_data.py --ticker AAPL --start 2010-01-01 \
                                 --end 2025-01-01 --split 2018-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml

from drl_utrans.utils.cfg import load_cfg

from drl_utrans.utils.indicators import make_feature_matrix, FEATURE_COLS

from feature_generation.headline_features import main as enrich_headlines
import numpy as np


# ─────────────────────────────── helpers ──────────────────────────────
def build_feature(df: pd.DataFrame) -> pd.Series:
    delta = (df["Open"] - df["Close"]).astype(np.float32)
    delta = delta.rename(columns = {"AAPL": "delta_oc"})
    return delta



# ───────────────────────────────  main  ───────────────────────────────
def main(ticker: str, start: str, end: str, split_date: str):
    print(f"Downloading {ticker}  [{start} → {end}] …", flush=True)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Open", "Close"]]             # we only need O & C

    # 1-dim feature matrix -------------------------------------------------
    feat = build_feature(df)               # (T,) Series, index = dates
    closes = df["Close"].astype(np.float32)
    closes = closes.rename(columns = {"AAPL": "close"})

    merged = pd.concat([feat, closes], axis=1).dropna()
    print("Final matrix shape:", merged.shape)   # (T, 2)

    # -------------------- train / test split -----------------------------
    train = merged.loc[: split_date]
    test  = merged.loc[split_date :]

    mu = train["delta_oc"].mean()
    sigma = train["delta_oc"].std(ddof=0)
    merged["delta_oc"] = (merged["delta_oc"] - mu) / sigma

    outdir = Path("data"); outdir.mkdir(exist_ok=True)
    train.to_csv(outdir / f"{ticker}_train.csv")
    test .to_csv(outdir / f"{ticker}_test.csv")
    print("Saved", outdir / f"{ticker}_train.csv", "and test.csv")

    return outdir / f"{ticker}_train.csv", outdir / f"{ticker}_test.csv"

# def main(ticker, start, end, split_date):
#     print(f"Downloading {ticker} [{start} → {end}] …")
#     df = yf.download(ticker, start=start, end=end, auto_adjust=True)
#     df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

#     features, closes = make_feature_matrix(df)
#     print("Feature matrix", features.shape)

#     # Align dates after dropna
#     aligned_dates = df.index[-len(features) :]
    
#     closes = closes.ravel()                # <- 1-D, shape (T,)

#     feat_df  = pd.DataFrame(features, columns=FEATURE_COLS, index=aligned_dates)
#     price_sr = pd.Series(closes, index=aligned_dates, name="close")

#     merged = pd.concat([feat_df, price_sr], axis=1)
#     train_tmp = merged.loc[: split_date]
#     mu   = train_tmp[FEATURE_COLS].mean()
#     sigma= train_tmp[FEATURE_COLS].std(ddof=0)
#     merged[FEATURE_COLS] = (merged[FEATURE_COLS] - mu) / sigma

#     # merged = enrich_headlines(ticker, merged)

#     outdir = Path("data")
#     outdir.mkdir(parents=True, exist_ok=True)

#     train = merged.loc[: split_date]
#     test = merged.loc[split_date :]

#     train.to_csv(outdir / f"{ticker}_train.csv", index=True)
#     test.to_csv(outdir / f"{ticker}_test.csv", index=True)
#     print("Saved", outdir / f"{ticker}_train.csv", "and test.csv")

#     return outdir / f"{ticker}_train.csv", outdir / f"{ticker}_test.csv"


if __name__ == "__main__":
    main()
