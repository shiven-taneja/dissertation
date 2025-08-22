# ===============================================================
# file: metrics.py (Corrected)
# ===============================================================
from __future__ import annotations
import numpy as np
import pandas as pd

def cagr(
    equity: np.ndarray, 
    dt: pd.DatetimeIndex | None = None, 
    freq: str = "D"
) -> float:
    """
    Calculates the Compound Annual Growth Rate from an equity curve.
    """
    if dt is None:
        years = len(equity) / _freq_to_years(freq)
    else:
        # Ensure we have at least two data points to calculate time delta
        if len(dt) < 2:
            return 0.0
        years = (dt[-1] - dt[0]).days / 365.25
    
    if years == 0:
        return 0.0
        
    return (equity[-1] / equity[0]) ** (1 / years) - 1

def sharpe(
    returns: np.ndarray, 
    rf: float = 0.02,  # Updated default Risk-Free Rate to 2.0%
    downside_only: bool = False
) -> float:
    """
    Calculates the annualised Sharpe ratio (or Sortino ratio if downside_only=True).
    
    Args:
        returns (np.ndarray): Daily returns of the strategy.
        rf (float): Annual risk-free rate.
        downside_only (bool): If True, calculates the Sortino ratio by using only
                              downside deviation. Defaults to False.

    Returns:
        float: The calculated ratio.
    """
    if returns.size < 2:
        return 0.0

    daily_rfr = rf / 252
    daily_excess = returns - daily_rfr
    
    if downside_only:
        # Calculate downside deviation for Sortino ratio
        downside_returns = daily_excess[daily_excess < 0]
        if downside_returns.size < 2:
            return np.nan # Not enough data to calculate downside deviation
        denominator = downside_returns.std(ddof=1)
    else:
        # Calculate standard deviation for Sharpe ratio
        denominator = daily_excess.std(ddof=1)

    if denominator == 0:
        # If there's no deviation, the ratio is technically infinite if mean > 0.
        # Returning 0.0 is a practical choice to avoid errors.
        return 0.0
        
    return np.sqrt(252) * daily_excess.mean() / denominator


def max_drawdown(equity: np.ndarray) -> float:
    """
    Calculates the maximum drawdown from an equity curve.
    """
    if equity.size < 2:
        return 0.0
        
    peak = np.maximum.accumulate(equity)
    # Ensure peak is never zero to avoid division by zero
    peak[peak == 0] = 1.0 
    drawdown = (equity - peak) / peak
    return drawdown.min()


def _freq_to_years(freq: str) -> float:
    """Converts frequency string to a fraction of a year."""
    freq = freq.upper()
    if freq.startswith("D"):
        return 252.0
    if freq.startswith("H"):
        return 252.0 * 6.5  # Standard trading hours in a day
    if freq.startswith("MIN"):
        return 252.0 * 6.5 * 60 # Trading minutes in a day
    raise ValueError(f"Unsupported frequency: {freq}")