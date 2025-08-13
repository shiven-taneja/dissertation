# ============================
# file: evaluate.py
# ============================
"""
Evaluation utility. Date handling:
- Reads CSV, moves `date` to DatetimeIndex (ISO or epoch ms/sec supported).
- Uses this index for pricing periods and metrics.

It can optionally save plots (only enabled for *best* runs by the ablation runner).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent

HEADLINE_COLS = [
    "news_sent_mean_1d",
    "news_sent_std_1d",
    "news_sent_maxabs_1d",
    "news_count_1d"
    ] 
TECH_SENT_COLs = ["tech_sent_score"]

FEATURE_COLS = [
    "macd",
    "kdj_k",
    "open_close_diff",
    "rsi_14",
]


def run_episode(env: PaperSingleStockEnv, agent: DrlUTransAgent, debug: bool = False):
    state_nd = env.reset()
    state = torch.from_numpy(state_nd).float()

    equity_curve = [env.portfolio_value()]
    actions_taken = []
    weights_used = []

    step_count = 0
    total_buys = total_sells = 0

    while True:
        act, w = agent.select_action(state, eval_mode=True)
        nxt, rew, done, info = env.step((act, w))

        actual_action = info["action"]
        if actual_action == 0:
            total_buys += 1
        elif actual_action == 1:
            total_sells += 1
        actions_taken.append(actual_action)
        weights_used.append(w)

        if debug and step_count % 50 == 0:
            print(
                f"Step {step_count:3d}: Action={act}→{actual_action}, "
                f"Weight={w:.2f}, Shares={info['shares']:4d}, "
                f"Cash=${info['cash']:.2f}, Value=${info['portfolio_value']:.2f}, "
                f"Reward={rew:.2f}"
            )

        state = torch.from_numpy(nxt).float()
        equity_curve.append(env.portfolio_value())
        step_count += 1
        if done:
            break

    return np.asarray(equity_curve), actions_taken, weights_used


def evaluate_one(
    *,
    ticker: str,
    run_type: str,
    test_csv: str | Path,
    ckpt_path: str | Path,
    commission_rate: float = 0.001,
    investment_capacity: int = 500,
    window_size: int = 12,
    results_root: str | Path = "results",
    debug: bool = False,
    save_plots: bool = False,
    train_csv: str | Path = None,
) -> Dict:
    """Evaluate a single checkpoint; optionally save plots & metrics."""
    test_csv = Path(test_csv)
    df_test = pd.read_csv(test_csv)
    s = df_test["date"]
    idx = pd.to_datetime(s, errors="coerce")
    df_test = df_test.drop(columns=["date"])  # keep date only as index
    df_test.index = idx


    if run_type == "baseline":
        feature_cols = FEATURE_COLS
    elif run_type == "headline":
        feature_cols = FEATURE_COLS + HEADLINE_COLS
    elif run_type == "techsent":
        feature_cols = FEATURE_COLS + TECH_SENT_COLs
    elif run_type == "all":
        feature_cols = FEATURE_COLS + HEADLINE_COLS + TECH_SENT_COLs
    feats = df_test[feature_cols].to_numpy(dtype=np.float32)
    prices = df_test["close"].to_numpy(dtype=np.float32)

    env = PaperSingleStockEnv(
        feats,
        prices,
        window_size=window_size,
        ic_shares=investment_capacity,
        commission=commission_rate,
        train_mode=False,
    )

    agent = DrlUTransAgent(state_dim=(window_size, feats.shape[1] + 3))

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("policy", checkpoint)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.epsilon = 0.0

    equity, actions, _ = run_episode(env, agent, debug=debug)
    dates = df_test.index[: len(equity)]
    rets = np.diff(equity) / equity[:-1]

    # Buy & Hold
    bh_equity = env.ic0 * prices[: len(equity)] / prices[0] * env.cash0 / env.ic0
    bh_rets = np.diff(bh_equity) / bh_equity[:-1]

    metrics = {
        "final_return_%": float((equity[-1] / equity[0] - 1) * 100),
        "CAGR_%": float(cagr(equity, dates) * 100),
        "Sharpe": float(sharpe(rets)),
        "MaxDD_%": float(max_drawdown(equity) * 100),
        "total_trades": int(sum(1 for a in actions if a != 2)),
        "bh_final_return_%": float((bh_equity[-1] / bh_equity[0] - 1) * 100),
        "bh_CAGR_%": float(cagr(bh_equity, dates) * 100),
        "bh_Sharpe": float(sharpe(bh_rets)),
        "bh_MaxDD_%": float(max_drawdown(bh_equity) * 100),
    }

    # save core artifacts
    results_root = Path(results_root)
    plots_dir = results_root / "plots" / ticker
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_root / f"{ticker}_{run_type}_equity.csv"
    pd.DataFrame({
        "date": dates,
        "equity": equity,
        "bh_equity": bh_equity,
    }).to_csv(out_csv, index=False)

    eq_png = act_png = None
    if save_plots:
        # equity vs B&H (best runs only)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label=f"DRL-UTrans ({run_type})", linewidth=2)
        plt.plot(dates, bh_equity, label="Buy&Hold", alpha=0.8)
        plt.ylabel("Portfolio Value ($)")
        plt.title(f"Equity Curve – {ticker} ({run_type})")
        plt.legend(); plt.grid(True, alpha=0.3)
        eq_png = plots_dir / f"equity_{ticker}_{run_type}.png"
        plt.tight_layout(); plt.savefig(eq_png, dpi=150); plt.close()

        # optional actions plot for best only
        buy_idx = [i for i, a in enumerate(actions) if a == 0]
        sell_idx = [i for i, a in enumerate(actions) if a == 1]
        plt.figure(figsize=(12, 6))
        plt.plot(prices[: len(actions)], label="Price", alpha=0.7)
        if buy_idx:
            plt.scatter(buy_idx, [prices[i] for i in buy_idx], marker='^', s=30, label='Buy')
        if sell_idx:
            plt.scatter(sell_idx, [prices[i] for i in sell_idx], marker='v', s=30, label='Sell')
        plt.title(f"Trading Actions – {ticker} ({run_type})")
        plt.xlabel("Time Step"); plt.ylabel("Price ($)")
        plt.legend(); plt.grid(True, alpha=0.3)
        act_png = plots_dir / f"actions_{ticker}_{run_type}.png"
        plt.tight_layout(); plt.savefig(act_png, dpi=150); plt.close()

        if train_csv is not None and Path(train_csv).exists():
            df_train = pd.read_csv(train_csv)
            s = df_train["date"]
            idx = pd.to_datetime(s, errors="coerce")
            df_train = df_train.drop(columns=["date"])  # keep date only as index
            df_train.index = idx
            full_prices = pd.concat([df_train["close"], df_test["close"]])
            full_dates = full_prices.index
            # scale B&H to start at same cash as model equity
            cash0 = float(equity[0])
            full_bh_equity = cash0 * (full_prices / full_prices.iloc[0]).to_numpy()
            test_start, test_end = df_test.index[0], df_test.index[-1]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(full_dates, full_bh_equity, label="Buy&Hold (full)", linewidth=1.3)
            ax.plot(dates, equity, label=f"DRL-UTrans ({run_type}) – test", linewidth=2)
            ax.axvspan(test_start, test_end, alpha=0.12, label="Test period")
            ax.set_ylabel("Portfolio Value ($)")
            ax.set_title(f"{ticker} ({run_type})")
            ax.legend(); ax.grid(True, alpha=0.3)
            ctx_png = plots_dir / f"context_equity_{ticker}_{run_type}.png"
            fig.tight_layout(); fig.savefig(ctx_png, dpi=150); plt.close(fig)

    # metrics json
    metrics_path = results_root / f"{ticker}_{run_type}_metrics.json"
    with metrics_path.open("w") as f:
        import json; json.dump(metrics, f, indent=2)

    return {
        "metrics": metrics,
        "equity": equity,
        "bh_equity": bh_equity,
        "dates": dates,
        "plots": {"equity_png": str(eq_png) if eq_png else None, "actions_png": str(act_png) if act_png else None},
        "metrics_path": str(metrics_path),
        "equity_csv": str(out_csv),
    }
