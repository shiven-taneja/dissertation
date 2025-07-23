#!/usr/bin/env python
"""
Load a trained checkpoint and evaluate on the held-out test slice.
Produces:
    • plots/curve_<ticker>.png
    • results/<ticker>_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from drl_utrans.utils.cfg import load_cfg, merge_cli
from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent


# ───────────────────────────── CLI ──────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, default="drl_utrans/configs/defaults.yaml")
    p.add_argument("--ckpt", type=Path, required=True, help="checkpoint .pt from train.py")
    p.add_argument("--ticker", help="override ticker in cfg")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


# ─────────────────────────── evaluation ─────────────────────────
def run_episode(env: PaperSingleStockEnv, agent: DrlUTransAgent, already_reset: bool = False):
    state_nd = env._state() if already_reset else env.reset()
    state = torch.from_numpy(state_nd).float()
    equity_curve = [env.portfolio_value()]
    while True:
        act, w = agent.select_action(state)
        nxt, rew, done, _ = env.step((act, w))
        state = torch.from_numpy(nxt).float()
        equity_curve.append(env.portfolio_value())
        if done:
            break
    return np.asarray(equity_curve)


def main():
    args = cli()
    cfg = merge_cli(load_cfg(args.cfg), args)
    ticker = cfg["data"]["ticker"]

    # ------------- load test CSV -----------------------------
    test_csv = Path(cfg["data"]["test_csv"].format(ticker=ticker))
    df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    feats = df[df.columns[:-1]].to_numpy(dtype=np.float32)
    prices = df["close"].to_numpy(dtype=np.float32)

    # ------------- build deterministic test env --------------
    env = PaperSingleStockEnv(
        feats,
        prices,
        window_size=cfg["model"]["window_size"],
        ic_shares=cfg["data"]["investment_capacity"],
        commission=cfg["data"]["commission_rate"],
        train_mode=False,
    )

    # ------------- restore agent -----------------------------
    agent = DrlUTransAgent(
        state_dim=(cfg["model"]["window_size"], cfg["model"]["feature_dim"])
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")["policy"]
    agent.policy_net.load_state_dict(state_dict)
    agent.epsilon = 0.0  # fully greedy

    # ------------- run episode -------------------------------
    equity = run_episode(env, agent, already_reset=True)
    dates = df.index[: len(equity)]
    rets = np.diff(equity) / equity[:-1]

    # baseline buy-and-hold on same slice
    bh_equity = env.ic0 * prices[: len(equity)]
    bh_rets = np.diff(bh_equity) / bh_equity[:-1]

    # ------------- metrics -----------------------------------
    metrics = {
        "final_return_%": (equity[-1] / equity[0] - 1) * 100,
        "CAGR_%": cagr(equity, dates) * 100,
        "Sharpe": sharpe(rets),
        "MaxDD_%": max_drawdown(equity) * 100,
        "bh_final_return_%": (bh_equity[-1] / bh_equity[0] - 1) * 100,
        "bh_CAGR_%": cagr(bh_equity, dates) * 100,
        "bh_Sharpe": sharpe(bh_rets),
        "bh_MaxDD_%": max_drawdown(bh_equity) * 100,
    }
    for k, v in metrics.items():
        print(f"{k:>15}: {v:8.2f}")

    # ------------- save outputs ------------------------------
    outdir = args.outdir
    outdir.mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    # equity curve plot
    plt.figure(figsize=(8, 4))
    plt.plot(dates, equity, label="DRL-UTrans")
    plt.plot(dates, bh_equity, label="Buy&Hold", alpha=0.5)
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Equity curve – {ticker}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"curve_{ticker}.png", dpi=150)

    # metrics json
    with (outdir / f"{ticker}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
