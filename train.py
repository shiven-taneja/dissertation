from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from drl_utrans.utils.cfg import load_cfg
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent


# ─────────────────────────────── main ────────────────────────────────
def main(ticker, train_csv, commission_rate = 0.001, investment_capacity = 500, epochs = 50, seed = 26, window_size = 12, feature_dim = 14, lr = 0.001, batch_size = 20, gamma = 0.9, replay_memory_size = 10000, target_update_freq = 500, epsilon_start = 1.0, epsilon_end = 0.4, epsilon_decay = 0.99, output = True):
    seed = seed
    # ------------- deterministic behaviour (optional) --------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------- load training CSV -------------------------------
    train_csv = Path(train_csv)
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
        window_size=window_size,
        ic_shares=investment_capacity,
        commission=commission_rate,
        train_mode=True,
        seed=seed,
    )

    # ------------- build agent -------------------------------------
    agent = DrlUTransAgent(
        state_dim=(window_size, feature_dim),
        lr=lr,
        batch_size=batch_size,
        gamma=gamma,
        memory_size=replay_memory_size,
        target_update_freq=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    # ------------- logging -----------------------------------------
    logdir = default=Path("runs") / f"{ticker}_seed{seed or 0}"
    writer = SummaryWriter(logdir.as_posix())
    ckptdir = Path("checkpoints")
    ckptdir.mkdir(exist_ok=True)

    # ------------- training loop -----------------------------------
    global_step = 0
    start_wall = perf_counter()
    for ep in range(epochs):
        if output:
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
        if output:
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
    return ckpt_path


if __name__ == "__main__":
    main(
        ticker="AAPL",
        start="2010-01-01",
        end="2025-01-01",
        train_csv="data/AAPL_train.csv",
        split_date="2018-01-01",
        commission_rate=0.001,
        investment_capacity=500,
        epochs=50,
        seed=42,
    )
