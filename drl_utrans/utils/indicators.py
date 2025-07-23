"""
Pure-pandas / numpy technical-indicator utilities for DRL-UTrans.
Returns *np.ndarray* so downstream code never sees DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ───────────────────────────── basic helpers ──────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _safe_diff(series: pd.Series) -> pd.Series:
    """diff() but first value = 0 instead of NaN."""
    d = series.diff()
    d.iloc[0] = 0.0
    return d


# ───────────────────────────── indicators (paper set) ─────────────────────
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    line = fast_ema - slow_ema
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll)


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + mult * std
    lower = ma - mult * std
    return upper, lower


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    rsv = 100 * (close - ll) / (hh - ll)
    k = rsv.ewm(alpha=1.0 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


# ───────────────────────────── feature builder ────────────────────────────
FEATURE_COLS = [
    "macd",
    "macd_sig",
    "macd_hist",
    "cci",
    "wr_14",
    "boll_up",
    "boll_low",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "ema20",
    "close_delta",
    "open_close_diff",
    "rsi_14",
]


def make_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Convert an OHLCV DataFrame (with `Open High Low Close Volume` columns)
    into a `(T, 14)` float32 NumPy matrix in the exact order required
    by UTrans.
    """
    df = df.copy()

    # MACD trio
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["Close"])

    # CCI
    df["cci"] = cci(df["High"], df["Low"], df["Close"])

    # Williams %R
    df["wr_14"] = williams_r(df["High"], df["Low"], df["Close"])

    # Bollinger bands (upper / lower)
    df["boll_up"], df["boll_low"] = bollinger(df["Close"])

    # KDJ
    df["kdj_k"], df["kdj_d"], df["kdj_j"] = kdj(df["High"], df["Low"], df["Close"])

    # EMA-20
    df["ema20"] = _ema(df["Close"], 20)

    # Close delta & open-close diff
    df["close_delta"] = _safe_diff(df["Close"])
    df["open_close_diff"] = df["Open"] - df["Close"]

    # RSI-14
    df["rsi_14"] = rsi(df["Close"])

    # Clean infinities / NaNs produced by divisions
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    return features, df["Close"].iloc[-len(features) :].to_numpy(dtype=np.float32)
