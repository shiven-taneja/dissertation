from __future__ import annotations
import numpy as np
import pandas as pd


def cagr(equity: np.ndarray, dt: pd.DatetimeIndex | None = None, freq: str = "D") -> float:
    """
    Compound Annual Growth Rate from equity curve.
    If `dt` is None we assume equally–spaced points with given `freq`.
    """
    if dt is None:
        years = len(equity) / _freq_to_years(freq)
    else:
        years = (dt[-1] - dt[0]).days / 365.25
    return (equity[-1] / equity[0]) ** (1 / years) - 1


def sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of daily returns (rf in same units)."""
    if returns.std(ddof=1) == 0:
        return 0.0
    daily_excess = returns - rf / 252
    return np.sqrt(252) * daily_excess.mean() / daily_excess.std(ddof=1)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()


def _freq_to_years(freq: str) -> float:
    if freq.upper().startswith("D"):
        return 252.0
    if freq.upper().startswith("H"):
        return 252.0 * 6.5 * 60  # trading minutes
    raise ValueError(freq)
