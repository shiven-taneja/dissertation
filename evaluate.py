#!/usr/bin/env python
"""
Fixed evaluation script with proper debugging and feature normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from drl_utrans.utils.cfg import load_cfg
from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent


def run_episode(env: PaperSingleStockEnv, agent: DrlUTransAgent, debug=True):
    """Run evaluation episode with debugging."""
    state_nd = env.reset()
    state = torch.from_numpy(state_nd).float()
    
    equity_curve = [env.portfolio_value()]
    actions_taken = []
    weights_used = []
    
    step_count = 0
    total_buys = 0
    total_sells = 0
    
    while True:
        # Get action from agent
        act, w = agent.select_action(state, eval_mode=True)
        
        # Execute action
        nxt, rew, done, info = env.step((act, w))
        
        # Track what actually happened
        actual_action = info['action']
        if actual_action == 0:
            total_buys += 1
        elif actual_action == 1:
            total_sells += 1
            
        actions_taken.append(actual_action)
        weights_used.append(w)
        
        # Debug output
        if debug and step_count % 50 == 0:
            print(f"Step {step_count:3d}: "
                  f"Action={act}→{actual_action}, "
                  f"Weight={w:.2f}, "
                  f"Shares={info['shares']:4d}, "
                  f"Cash=${info['cash']:.2f}, "
                  f"Value=${info['portfolio_value']:.2f}, "
                  f"Reward={rew:.2f}")
        
        # Update state
        state = torch.from_numpy(nxt).float()
        equity_curve.append(env.portfolio_value())
        
        step_count += 1
        if done:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {step_count}")
    print(f"  Total buys: {total_buys}")
    print(f"  Total sells: {total_sells}")
    print(f"  Total holds: {step_count - total_buys - total_sells}")
    print(f"  Final portfolio value: ${equity_curve[-1]:.2f}")
    print(f"  Return: {(equity_curve[-1]/equity_curve[0] - 1)*100:.2f}%")
    
    return np.asarray(equity_curve), actions_taken, weights_used


def main(ticker, test_csv, ckpt_path, commission_rate=0.001, investment_capacity=500, window_size=12, feature_dim=14):
    
    # Load test data
    test_csv = Path(test_csv)
    df_test = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    
    # Extract features and prices
    feats = df_test[df_test.columns[:-1]].to_numpy(dtype=np.float32)
    prices = df_test["close"].to_numpy(dtype=np.float32)
    
    print(f"\nTest data shape: {feats.shape}")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Build environment
    env = PaperSingleStockEnv(
        feats,
        prices,
        window_size=window_size,
        ic_shares=investment_capacity,
        commission=commission_rate,
        train_mode=False,
    )
    
    # Load agent
    agent = DrlUTransAgent(
        state_dim=(window_size, feature_dim)
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "policy" in checkpoint:
        state_dict = checkpoint["policy"]
    else:
        state_dict = checkpoint
    
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.epsilon = 0.0  # No exploration during evaluation
    
    print(f"\nLoaded model from: {ckpt_path}")
    
    # Run evaluation
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)
    
    equity, actions, weights = run_episode(env, agent, debug=True)
    dates = df_test.index[: len(equity)]
    rets = np.diff(equity) / equity[:-1]
    
    # Buy & Hold baseline
    bh_equity = env.ic0 * prices[: len(equity)] / prices[0] * env.cash0 / env.ic0
    bh_rets = np.diff(bh_equity) / bh_equity[:-1]
    
    # Calculate metrics
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
    
    # Save outputs
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Equity curves
    plt.subplot(2, 1, 1)
    plt.plot(dates, equity, label="DRL-UTrans", linewidth=2)
    plt.plot(dates, bh_equity, label="Buy&Hold", alpha=0.7)
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Equity Curve - {ticker}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Actions plot
    plt.subplot(2, 1, 2)
    buy_points = [i for i, a in enumerate(actions) if a == 0]
    sell_points = [i for i, a in enumerate(actions) if a == 1]
    
    plt.plot(prices[:len(actions)], color='gray', alpha=0.5, label='Price')
    if buy_points:
        plt.scatter(buy_points, [prices[i] for i in buy_points], 
                   color='green', marker='^', s=50, label='Buy', zorder=5)
    if sell_points:
        plt.scatter(sell_points, [prices[i] for i in sell_points], 
                   color='red', marker='v', s=50, label='Sell', zorder=5)
    
    plt.ylabel("Price ($)")
    plt.xlabel("Time Step")
    plt.title("Trading Actions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"evaluation_{ticker}.png", dpi=150)
    plt.show()
    
    # Save metrics
    with (outdir / f"{ticker}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    main(
        ticker="AAPL",
        test_csv="data/AAPL_test.csv",
        train_csv="data/AAPL_train.csv",  # Pass training CSV for normalization
        ckpt_path="checkpoints/utrans_AAPL_final.pt",
        commission_rate=0.001,
        investment_capacity=500
    )