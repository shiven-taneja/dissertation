from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent

# -------------------------- feature helpers ---------------------------
HEADLINE_COLS = [
    "news_sent_mean_1d",
    "news_sent_std_1d",
    "news_sent_maxabs_1d",
    "news_count_1d"
    ] 
TECH_SENT_COLs = ["tech_sent_score"]

FEATURE_COLS = [
    # "macd",
    # "kdj_k",
    "open_close_diff",
    # "rsi_14",
]

# ------------------------------ train ---------------------------------

def train_one(
    *,
    ticker: str,
    run_type: str,  # 'baseline' | 'headline' | 'techsent' | 'all'
    train_csv: str | Path,
    commission_rate: float = 0.001,
    investment_capacity: int = 500,
    epochs: int = 50,
    seed: int | None = 26,
    window_size: int = 12,
    lr: float = 1e-3,
    batch_size: int = 20,
    gamma: float = 0.99,
    replay_memory_size: int = 10_000,
    target_update_freq: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.99,
    weight_loss_coef: float = 0.0,
    rand_weights: bool = True,
    output: bool = True,
    runs_dir: str | Path = "runs",
    checkpoints_dir: str | Path = "checkpoints",
) -> Dict:
    """Train once and return metadata (including checkpoint path)."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_csv = Path(train_csv)

    # Read and normalize index
    df = pd.read_csv(train_csv)

    s = df["date"]
    idx = pd.to_datetime(s, errors="coerce")
    df = df.drop(columns=["date"])  # keep date only as index
    df.index = idx


    if run_type == "baseline":
        feature_cols = FEATURE_COLS
    elif run_type == "headline":
        feature_cols = FEATURE_COLS + HEADLINE_COLS
    elif run_type == "techsent":
        feature_cols = FEATURE_COLS + TECH_SENT_COLs
    elif run_type == "all":
        feature_cols = FEATURE_COLS + HEADLINE_COLS + TECH_SENT_COLs

    features = df[feature_cols].to_numpy(dtype=np.float32)
    prices = df["close"].to_numpy(dtype=np.float32)

    env = PaperSingleStockEnv(
        features,
        prices,
        window_size=window_size,
        ic_shares=investment_capacity,
        commission=commission_rate,
        train_mode=True,
        seed=seed,
    )

    agent = DrlUTransAgent(
        state_dim=(window_size, features.shape[1] + 3),
        lr=lr,
        batch_size=batch_size,
        gamma=gamma,
        memory_size=replay_memory_size,
        target_update_freq=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        weight_loss_coef=weight_loss_coef,
        rand_weights=rand_weights,
    )

    logdir = Path(runs_dir) / f"{ticker}_{run_type}_seed{seed or 0}"
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir.as_posix())

    ckptdir = Path(checkpoints_dir)
    ckptdir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_wall = perf_counter()
    last_info = None

    for ep in range(epochs):
        if output:
            print(f"Starting epoch {ep + 1}/{epochs} ...", flush=True)
        state_nd = env.reset()
        state = torch.from_numpy(state_nd).float()

        ep_reward = 0.0
        losses: List[float] = []
        actions_log: List[int] = []
        weights_log: List[float] = []

        while True:
            action, w = agent.select_action(state)
            next_state, reward, done, info = env.step((action, w))
            last_info = info

            next_state_t = torch.from_numpy(next_state).float()
            actions_log.append(info["action"])  # executed action
            weights_log.append(abs(info["weight"]))

            agent.store_transition(state, action, w, reward, next_state_t, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state_t
            ep_reward += reward
            global_step += 1

            if done:
                break

        writer.add_scalar("reward/episode", ep_reward, ep)
        if losses:
            writer.add_scalar("loss/mean", mean(losses), ep)
        writer.add_scalar("epsilon", agent.epsilon, ep)
        if last_info is not None:
            writer.add_scalar("portfolio/value", last_info.get("portfolio_value", 0.0), ep)

        elapsed = perf_counter() - start_wall
        buys = sum(1 for a in actions_log if a == 0)
        sells = sum(1 for a in actions_log if a == 1)
        avg_w = float(np.mean(weights_log)) if weights_log else 0.0
        if output:
            print(
                f"Ep {ep+1:03d}/{epochs}  "
                f"reward {ep_reward:8.1f}  ε {agent.epsilon:4.2f}  "
                f"steps {global_step:5d}  time {elapsed/60:4.1f}m  "
                f"buys {buys:4d}  sells {sells:4d}  avg_w {avg_w:6.2f}",
                flush=True,
            )
        agent.decay_epsilon()

    # save checkpoint (temp name; runner will promote best)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"{run_type}_ee{epsilon_end}_wl{weight_loss_coef}_rw{int(rand_weights)}_{stamp}"
    ckpt_path = Path(checkpoints_dir) / f"utrans_{ticker}_{suffix}.pt"
    torch.save({"policy": agent.policy_net.state_dict()}, ckpt_path)

    meta = {
        "ckpt_path": str(ckpt_path),
        "feature_cols": feature_cols,
        "feature_dim": int(features.shape[1]),
        "ticker": ticker,
        "run_type": run_type,
        "seed": seed,
        "hparams": {
            "commission_rate": commission_rate,
            "investment_capacity": investment_capacity,
            "epochs": epochs,
            "window_size": window_size,
            "lr": lr,
            "batch_size": batch_size,
            "gamma": gamma,
            "replay_memory_size": replay_memory_size,
            "target_update_freq": target_update_freq,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "weight_loss_coef": weight_loss_coef,
            "rand_weights": rand_weights,
        },
        "logdir": str(logdir),
    }
    return meta
