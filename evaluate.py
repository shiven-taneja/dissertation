# scripts/evaluate_ppo.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.ppo_utrans import PPOUTransAgent, PPOConfig, weight_center_from_bin
from drl_utrans.models.utrans_ac import UTransActorCritic
from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown


def run_episode(env: PaperSingleStockEnv, agent: PPOUTransAgent, debug=True):
    state_nd = env.reset()
    state = torch.from_numpy(state_nd).float()

    equity_curve = [env.portfolio_value()]
    actions_taken = []
    weights_used = []

    step_count = 0
    total_buys = total_sells = 0

    while True:
        a, w, _, _, _, _ = agent.select_action(state, eval_mode=True)
        nxt, rew, done, info = env.step((a, w))

        actual_action = info['action']
        if actual_action == 0:
            total_buys += 1
        elif actual_action == 1:
            total_sells += 1

        actions_taken.append(actual_action)
        weights_used.append(w)

        if debug and step_count % 50 == 0:
            print(f"Step {step_count:3d}: Action={a}→{actual_action}, Weight={w:.2f}, "
                  f"Shares={info['shares']:4d}, Cash=${info['cash']:.2f}, "
                  f"Value=${info['portfolio_value']:.2f}, Reward={rew:.2f}")

        state = torch.from_numpy(nxt).float()
        equity_curve.append(env.portfolio_value())

        step_count += 1
        if done:
            break

    print(f"\nEpisode Summary: steps={step_count} buys={total_buys} sells={total_sells} holds={step_count - total_buys - total_sells}")
    return np.asarray(equity_curve), actions_taken, weights_used


def main(ticker: str, test_csv: str, ckpt_path: str, commission_rate=0.001, investment_capacity=500, window_size=12, feature_dim=14):
    df_test = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    feats = df_test[df_test.columns[:-1]].to_numpy(dtype=np.float32)
    prices = df_test["close"].to_numpy(dtype=np.float32)

    env = PaperSingleStockEnv(
        feats,
        prices,
        window_size=window_size,
        ic_shares=investment_capacity,
        commission=commission_rate,
        train_mode=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = PPOConfig(state_dim=(window_size, feature_dim))
    agent = PPOUTransAgent(cfg)
    agent.model.load_state_dict(checkpoint["model"])
    agent.model.eval()

    print(f"Loaded model from: {ckpt_path}")

    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)
    equity, actions, weights = run_episode(env, agent, debug=True)

    dates = df_test.index[: len(equity)]
    rets = np.diff(equity) / equity[:-1]

    # Buy & Hold baseline
    bh_equity = env.ic0 * prices[: len(equity)] / prices[0] * env.cash0 / env.ic0
    bh_rets = np.diff(bh_equity) / bh_equity[:-1]

    metrics = {
        "final_return_%": float((equity[-1] / equity[0] - 1) * 100),
        "CAGR_%": float(cagr(equity, dates) * 100),
        "Sharpe": float(sharpe(rets)),
        "MaxDD_%": float(max_drawdown(equity) * 100),
        "total_trades": sum(1 for a in actions if a != 2),
        "bh_final_return_%": float((bh_equity[-1] / bh_equity[0] - 1) * 100),
        "bh_CAGR_%": float(cagr(bh_equity, dates) * 100),
        "bh_Sharpe": float(sharpe(bh_rets)),
        "bh_MaxDD_%": float(max_drawdown(bh_equity) * 100),
    }

    print("\n" + "="*50)
    print("Final Metrics")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:>20}: {v:8.2f}")

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    # Plot equity vs B&H
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity, label="PPO-UTrans", linewidth=2)
    plt.plot(dates, bh_equity, label="Buy&Hold", alpha=0.7)
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Equity Curve - {ticker}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"ppo_eval_{ticker}.png", dpi=150)

    with (outdir / f"{ticker}_ppo_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    main(
        ticker="AAPL",
        test_csv="data/AAPL_test.csv",
        ckpt_path="checkpoints/ppo_utrans_AAPL.pt",
    )