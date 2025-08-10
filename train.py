# scripts/train_ppo.py
from __future__ import annotations
from pathlib import Path
from statistics import mean
from time import perf_counter
import numpy as np
import torch

from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.ppo_utrans import PPOUTransAgent, PPOConfig, weight_center_from_bin


def main(
    ticker: str,
    train_csv: str,
    commission_rate: float = 0.001,
    investment_capacity: int = 500,
    epochs: int = 50,
    seed: int | None = 26,
    window_size: int = 12,
    feature_dim: int = 14,
    lr: float = 3e-4,
    rollout_steps: int = 4096,
    minibatch_size: int = 256,
    update_epochs: int = 10,
    n_weight_bins: int = 11,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ---------- load data ----------
    import pandas as pd
    df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
    features = df[df.columns[:-1]].to_numpy(dtype=np.float32)
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

    cfg = PPOConfig(
        state_dim=(window_size, feature_dim),
        lr=lr,
        rollout_steps=rollout_steps,
        minibatch_size=minibatch_size,
        update_epochs=update_epochs,
        n_weight_bins=n_weight_bins,
    )
    agent = PPOUTransAgent(cfg)

    ckptdir = Path("checkpoints")
    ckptdir.mkdir(exist_ok=True)

    start = perf_counter()
    for ep in range(epochs):
        state_nd = env.reset()
        state = torch.from_numpy(state_nd).float()
        ep_reward = 0.0
        steps_collected = 0

        while steps_collected < cfg.rollout_steps:
            a, w, v, _, wb, _ = agent.select_action(state) 
            next_state, reward, done, info = env.step((a, w))

            
            a_exec = info["action"]                                  # 0/1/2 actually taken
            nonhold_exec = (a_exec != 2)
            wb_exec = wb if nonhold_exec else 0

            # recompute joint log-prob for the EXECUTED action under current policy
            logp_exec = agent.exec_logprob(state, a_exec, wb_exec)

            agent.store(state, a_exec, wb_exec, nonhold_exec,
                        float(reward), bool(done), float(v), float(logp_exec))

            state = torch.from_numpy(next_state).float()
            ep_reward += reward
            steps_collected += 1

            if done:
                # restart episode within same rollout collection
                state = torch.from_numpy(env.reset()).float()

        agent.update()
        elapsed = (perf_counter() - start) / 60
        print(f"Ep {ep+1:03d}/{epochs}  reward {ep_reward:10.2f}  steps {steps_collected:5d}  time {elapsed:5.2f}m")

    ckpt_path = ckptdir / f"ppo_utrans_{ticker}.pt"
    torch.save({"model": agent.model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
    print("Saved checkpoint to", ckpt_path)
    return ckpt_path

if __name__ == "__main__":
    # Example
    main(
        ticker="AAPL",
        train_csv="data/AAPL_train.csv",
        epochs=10,
    )