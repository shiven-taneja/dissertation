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

from drl_utrans.utils.indicators import make_feature_matrix, FEATURE_COLS


def parse_cfg(cfg_path: Path):
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="override ticker in cfg")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--split", default=None, help="train/test split date")
    parser.add_argument(
        "--cfg",
        default=Path(__file__).resolve().parents[1] / "dissertation/drl_utrans/configs/defaults.yaml",
        type=Path,
    )
    parser.add_argument("--outdir", default="data", type=Path)
    return parser.parse_args()


def main():
    args = cli()
    cfg = parse_cfg(args.cfg)

    ticker = args.ticker or cfg["data"]["ticker"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    split_date = args.split or cfg["data"]["split_date"]

    print(f"Downloading {ticker} [{start} → {end}] …")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    features, closes = make_feature_matrix(df)
    print("Feature matrix", features.shape)

    # Align dates after dropna
    aligned_dates = df.index[-len(features) :]
    
    closes = closes.ravel()                # <- 1-D, shape (T,)

    feat_df  = pd.DataFrame(features, columns=FEATURE_COLS, index=aligned_dates)
    price_sr = pd.Series(closes, index=aligned_dates, name="close")

    merged = pd.concat([feat_df, price_sr], axis=1)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train = merged.loc[: split_date]
    test = merged.loc[split_date :]

    train.to_csv(outdir / f"{ticker}_train.csv", index=True)
    test.to_csv(outdir / f"{ticker}_test.csv", index=True)
    print("Saved", outdir / f"{ticker}_train.csv", "and test.csv")


if __name__ == "__main__":
    main()
