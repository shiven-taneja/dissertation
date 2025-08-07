import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch

from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown  # adapt if needed

RESULTS_ROOT = Path("results")

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_metrics(ticker: str, metrics: Dict, variant: str, out_tag: str):
    out_dir = RESULTS_ROOT / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"metrics_{variant}_{out_tag}.json"
    import json
    with open(f, "w") as fp:
        json.dump(metrics, fp, indent=2)

def run_episode(env, agent, already_reset: bool = False):
    state_nd = env._state() if already_reset else env.reset()
    state = torch.from_numpy(state_nd).float()
    equity_curve = [env.portfolio_value()]
    actions = []
    while True:
        with torch.no_grad():
            a, w = agent.select_action(state, eval_mode=True)
        nxt_state, reward, done, info = env.step((a, w))
        actions.append((a, w))
        equity_curve.append(env.portfolio_value())
        if done:
            break
        state = torch.from_numpy(nxt_state).float()
    equity_curve = np.array(equity_curve, dtype=np.float32)
    return equity_curve, actions

def run_and_plot(ticker: str, env, agent, variant: str, out_tag: str, eq_train: np.ndarray = None) -> Dict:
    plots_dir = RESULTS_ROOT / "plots" / ticker.upper()
    plots_dir.mkdir(parents=True, exist_ok=True)

    eq_test, actions = run_episode(env, agent, already_reset=False)
    bh = env.buy_and_hold_curve()

    # Save raw arrays for combined plots
    np.save(plots_dir / f"eq_test_{variant}_{out_tag}.npy", eq_test)
    np.save(plots_dir / f"bh_test_{out_tag}.npy", bh)
    if eq_train is not None:
        np.save(plots_dir / f"eq_train_{variant}_{out_tag}.npy", eq_train)

    metrics = {
        "final_return_%": float((eq_test[-1] / eq_test[0] - 1) * 100),
        "CAGR_%": float(cagr(eq_test) * 100),
        "Sharpe": float(sharpe(eq_test)),
        "MaxDD_%": float(max_drawdown(eq_test) * 100),
        "bh_final_return_%": float((bh[-1] / bh[0] - 1) * 100),
        "bh_CAGR_%": float(cagr(bh) * 100),
        "bh_Sharpe": float(sharpe(bh)),
        "bh_MaxDD_%": float(max_drawdown(bh) * 100),
    }
    save_metrics(ticker, metrics, variant=variant, out_tag=out_tag)

    # 1) Equity curve (test)
    fig1 = plt.figure()
    plt.plot(eq_test, label=f"{variant}")
    plt.plot(bh, label="Buy & Hold", linestyle="--")
    plt.title(f"{ticker} – Equity Curve (Test) – {variant}")
    plt.legend()
    p1 = plots_dir / f"equity_{variant}_{out_tag}.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    # 2) Actions (test)
    buys = [i for i,(a,w) in enumerate(actions) if a==0]
    sells= [i for i,(a,w) in enumerate(actions) if a==1]
    fig2 = plt.figure()
    plt.plot(eq_test, label="Equity")
    if len(buys)>0:
        plt.scatter(np.array(buys), eq_test[np.array(buys)], marker="^", label="Buy", s=15)
    if len(sells)>0:
        plt.scatter(np.array(sells), eq_test[np.array(sells)], marker="v", label="Sell", s=15)
    plt.title(f"{ticker} – Actions (Test) – {variant}")
    plt.legend()
    p2 = plots_dir / f"actions_{variant}_{out_tag}.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    return metrics

def combined_test_plot(ticker: str, out_tag: str, variants: List[str]):
    plots_dir = RESULTS_ROOT / "plots" / ticker.upper()
    eqs = []
    labels = []
    for v in variants:
        f = plots_dir / f"eq_test_{v}_{out_tag}.npy"
        if f.exists():
            eqs.append(np.load(f))
            labels.append(v)
    bh = np.load(plots_dir / f"bh_test_{out_tag}.npy")
    fig = plt.figure()
    for e, l in zip(eqs, labels):
        plt.plot(e, label=l)
    plt.plot(bh, label="Buy & Hold", linestyle="--")
    plt.title(f"{ticker} – Combined Equity Curves (Test)")
    plt.legend()
    fig.savefig(plots_dir / f"equity_combined_{out_tag}.png", bbox_inches="tight")
    plt.close(fig)

def train_then_test_plot(ticker: str, out_tag: str, variants: List[str]):
    plots_dir = RESULTS_ROOT / "plots" / ticker.upper()
    # For each variant, try to load both train and test arrays
    curves_train = []
    curves_test = []
    labels = []
    for v in variants:
        ftr = plots_dir / f"eq_train_{v}_{out_tag}.npy"
        fte = plots_dir / f"eq_test_{v}_{out_tag}.npy"
        if ftr.exists() and fte.exists():
            curves_train.append(np.load(ftr))
            curves_test.append(np.load(fte))
            labels.append(v)
    # BH for test already saved; for train we don't have BH from env, so omit BH on train side.
    bh_test = np.load(plots_dir / f"bh_test_{out_tag}.npy")

    # Build a single timeline: concatenate train (normalized to start at 1) and test (start where train leaves off)
    fig = plt.figure()
    offset = 0
    for ct, ce, l in zip(curves_train, curves_test, labels):
        # Normalize each to 1 at start for fair visual
        ct_norm = ct / ct[0]
        ce_norm = ce / ce[0] * ct_norm[-1]
        x_train = np.arange(len(ct_norm))
        x_test = np.arange(len(ct_norm), len(ct_norm)+len(ce_norm))
        plt.plot(x_train, ct_norm, alpha=0.8, label=f"{l} (train)")
        plt.plot(x_test, ce_norm, alpha=0.9, label=f"{l} (test)")
    # Add BH on test only
    bhn = bh_test / bh_test[0]
    x_bh = np.arange(len(curves_train[0]), len(curves_train[0]) + len(bhn)) if curves_train else np.arange(len(bhn))
    plt.plot(x_bh, bhn, linestyle="--", label="Buy & Hold (test)")
    plt.title(f"{ticker} – Train (left) then Test (right): all 4 variants + BH")
    plt.legend()
    fig.savefig(plots_dir / f"train_then_test_{out_tag}.png", bbox_inches="tight")
    plt.close(fig)