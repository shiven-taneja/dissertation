#!/usr/bin/env python
"""
End-to-end training script for DRL-UTrans.

Usage example:
    python scripts/train.py --cfg drl_utrans/configs/default.yaml \
                            --ticker AAPL --epochs 50 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from drl_utrans.utils.cfg import load_cfg, merge_cli
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent


# ──────────────────────────────── CLI ────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfg",
        type=Path,
        default=Path("drl_utrans/configs/defaults.yaml"),
        help="YAML config",
    )
    p.add_argument("--ticker", help="override ticker in cfg")
    p.add_argument("--epochs", type=int, help="override epoch count")
    p.add_argument("--seed", type=int, help="global random seed")
    p.add_argument("--logdir", type=Path, default=Path("runs"))
    p.add_argument("--ckptdir", type=Path, default=Path("checkpoints"))
    return p.parse_args()


# ─────────────────────────────── main ────────────────────────────────
def main():
    args = cli()
    cfg = merge_cli(load_cfg(args.cfg), args)
    ticker = cfg["data"]["ticker"]
    epochs = cfg["agent"]["epochs"]
    seed = cfg["agent"].get("seed", None)

    # ------------- deterministic behaviour (optional) --------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------- load training CSV -------------------------------
    train_csv = Path(cfg["data"]["train_csv"].format(ticker=ticker))
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run fetch_data.py first."
        )
    df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
    features = df[df.columns[:-1]].to_numpy(dtype=np.float32)  # 14 features
    prices = df["close"].to_numpy(dtype=np.float32)

    # ------------- build environment -------------------------------
    env = PaperSingleStockEnv(
        features,
        prices,
        window_size=cfg["model"]["window_size"],
        ic_shares=cfg["data"]["investment_capacity"],
        commission=cfg["data"]["commission_rate"],
        train_mode=True,
        seed=seed,
    )

    # ------------- build agent -------------------------------------
    agent = DrlUTransAgent(
        state_dim=(cfg["model"]["window_size"], cfg["model"]["feature_dim"]),
        lr=cfg["agent"]["learning_rate"],
        batch_size=cfg["agent"]["batch_size"],
        gamma=cfg["agent"]["gamma"],
        memory_size=cfg["agent"]["replay_memory_size"],
        target_update_freq=cfg["agent"]["target_update_freq"],
        epsilon_start=cfg["agent"]["epsilon_start"],
        epsilon_end=cfg["agent"]["epsilon_end"],
        epsilon_decay=cfg["agent"]["epsilon_decay"],
    )

    # ------------- logging -----------------------------------------
    logdir = args.logdir / f"{ticker}_seed{seed or 0}"
    writer = SummaryWriter(logdir.as_posix())
    ckptdir = args.ckptdir
    ckptdir.mkdir(exist_ok=True)

    # ------------- training loop -----------------------------------
    global_step = 0
    start_wall = perf_counter()
    for ep in range(epochs):
        print(f"Starting epoch {ep + 1}/{epochs} ...", flush=True)
        state_nd = env.reset()
        state = torch.from_numpy(state_nd).float()
        ep_reward = 0.0
        losses = []
        actions_log = []
        weights_log = []


        while True:
            action, w = agent.select_action(state)
            next_state, reward, done, info = env.step((action, w))
            next_state_t = torch.from_numpy(next_state).float()
            actions_log.append(info["action"])
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

        # ---- end episode metrics ----
        writer.add_scalar("reward/episode", ep_reward, ep)
        if losses:
            writer.add_scalar("loss/mean", mean(losses), ep)
        writer.add_scalar("epsilon", agent.epsilon, ep)
        writer.add_scalar("portfolio/value", info["portfolio_value"], ep)

        elapsed = perf_counter() - start_wall
        buys  = sum(1 for a in actions_log if a == 0)
        sells = sum(1 for a in actions_log if a == 1)
        avg_w = np.mean(weights_log) if weights_log else 0.0
        print(
            f"Ep {ep+1:03d}/{epochs}  "
            f"reward {ep_reward:8.1f}  "
            f"ε {agent.epsilon:4.2f}  "
            f"steps {global_step:5d}  "
            f"time {elapsed/60:4.1f}m"
            f"  buys {buys:4d}  sells {sells:4d} "
            f"avg_w {avg_w:6.2f}",
            flush=True,
        )
        agent.decay_epsilon()

    # ------------- save checkpoint ---------------------------------
    ckpt_path = ckptdir / f"utrans_{ticker}_final.pt"
    torch.save({"policy": agent.policy_net.state_dict()}, ckpt_path)
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
