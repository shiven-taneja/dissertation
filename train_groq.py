import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

HEADLINE_COLS = ["news_sent_mean_1d", "news_sent_std_1d", "news_sent_maxabs_1d", "news_sent_count_1d"]
TECH_COLS = ["tech_sent_score"]

def load_variant_features(df: pd.DataFrame, feature_variant: str) -> np.ndarray:
    base = ["macd", "macd_sig", "macd_hist", "cci", "wr_14", "boll_up", "boll_low", "kdj_k", "kdj_d", "kdj_j", "ema20", "close_delta", "open_close_diff", "rsi_14"]
    if feature_variant == "baseline":
        cols = base
    elif feature_variant == "headline":
        cols = base + HEADLINE_COLS
    elif feature_variant == "tech":
        cols = base + TECH_COLS
    elif feature_variant == "both":
        cols = base + HEADLINE_COLS + TECH_COLS
    else:
        raise ValueError(f"Unknown feature_variant: {feature_variant}")
    X = df[cols].values.astype(np.float32)
    return X

def prepare_env_arrays(train_csv: Path, test_csv: Path, feature_variant: str, window: int = 12):
    train = pd.read_csv(train_csv, parse_dates=["date"])
    test  = pd.read_csv(test_csv, parse_dates=["date"])

    X_train = load_variant_features(train, feature_variant)
    X_test  = load_variant_features(test, feature_variant)

    prices_train = train["close"].values.astype(np.float32)
    prices_test  = test["close"].values.astype(np.float32)

    def to_windows(X, L=12):
        out = []
        for i in range(L, len(X)):
            out.append(X[i-L:i])
        return np.stack(out, axis=0)

    feats_train = to_windows(X_train, window)
    feats_test  = to_windows(X_test, window)

    prices_train_w = prices_train[window:]
    prices_test_w  = prices_test[window:]

    return feats_train, prices_train_w, feats_test, prices_test_w

def evaluate_equity(env, agent):
    import numpy as np, torch
    state_nd = env.reset()
    state = torch.from_numpy(state_nd).float()
    eq = [env.portfolio_value()]
    while True:
        with torch.no_grad():
            a, w = agent.select_action(state, eval_mode=True)
        ns, r, done, info = env.step((a, w))
        eq.append(env.portfolio_value())
        if done:
            break
        state = torch.from_numpy(ns).float()
    return np.array(eq, dtype=np.float32)

def train_one_variant(
    ticker: str,
    train_csv: Path,
    test_csv: Path,
    feature_variant: str,
    out_tag: str,
    epsilon_end: float,
    weight_loss_coef: float,
    rand_weights: bool,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    from drl_utrans.envs.single_stock import PaperSingleStockEnv    
    from drl_utrans.agent.drl_utrans import DrlUTransAgent
    from evaluate_groq import run_and_plot, save_metrics

    feats_tr, prices_tr, feats_te, prices_te = prepare_env_arrays(train_csv, test_csv, feature_variant)
    env_train = PaperSingleStockEnv(features=feats_tr, prices=prices_tr, window_size=feats_tr.shape[1],
                                    ic_shares=500, commission=0.0, train_mode=True)
    env_test  = PaperSingleStockEnv(features=feats_te, prices=prices_te, window_size=feats_te.shape[1],
                                    ic_shares=500, commission=0.0, train_mode=False)

    agent = DrlUTransAgent(
        state_dim=(feats_tr.shape[1], feats_tr.shape[2]),
        gamma=0.99,
        target_update_freq=200,
        epsilon_end=epsilon_end,
        rand_weights=rand_weights,
        weight_loss_coef=weight_loss_coef,
        device=device,
    )

    agent.train(env_train, epochs=50)

    # Save train equity for the "train-then-test" plot
    eq_train = evaluate_equity(env_train, agent)

    res = run_and_plot(
        ticker=ticker,
        env=env_test,
        agent=agent,
        variant=feature_variant,
        out_tag=out_tag,
        eq_train=eq_train,
    )
    return res