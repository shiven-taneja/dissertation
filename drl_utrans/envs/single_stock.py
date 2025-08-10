from __future__ import annotations

import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch


class PaperSingleStockEnv:
    """
    Fixed single-stock trading environment with proper state handling
    and action execution.
    """

    def __init__(
        self,
        features: np.ndarray | pd.DataFrame,
        prices: np.ndarray | pd.Series,
        window_size: int = 12,
        ic_shares: int = 500,
        commission: float = 0.001,
        train_mode: bool = True,
        seed: Optional[int] = None,
    ):
        # Store market data
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        elif not isinstance(features, np.ndarray):
            features = np.asarray(features)
        self.X = features.astype(np.float32)

        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        self.P = prices.astype(np.float32)

        assert len(self.X) == len(self.P), "features and price length mismatch!"

        # Static hyper-params
        self.L = window_size
        self.ic0 = ic_shares
        self.commission = commission
        self.train_mode = train_mode
        self.rng = random.Random(seed)
        self._prev_e = 0.0
        # Pointer bounds - fixed to prevent looking ahead
        self.max_ptr = len(self.X) - 1  # last valid index

        self.reward_scale = 1e-4
        self.lookahead = 1
        
        # Episode bookkeeping
        self.reset()

    def _state(self) -> np.ndarray:
        """Return current state window (historical data only)."""
        # FIXED: Return window ending at current ptr (not including future)
        start_idx = max(0, self.ptr - self.L + 1)
        end_idx = self.ptr + 1
        state = self.X[start_idx:end_idx]
        
        # Pad with zeros if at the beginning
        if state.shape[0] < self.L:
            padding = np.zeros((self.L - state.shape[0], state.shape[1]), dtype=np.float32)
            state = np.concatenate([padding, state], axis=0)
        
        return state

    def _done(self) -> bool:
        """Episode ends when we reach the end of data."""
        return self.ptr >= self.max_ptr - 1

    def _cost_basis(self) -> float:
        return self.I / self.H if self.H > 0 else 0.0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # if self.train_mode:
        #     # Random start for training (ensure we have enough data)
        #     min_start = self.L - 1
        #     max_start = max(min_start, self.max_ptr - 100)  # Ensure at least 100 steps
        #     self.ptr = self.rng.randint(min_start, max_start)
        # else:
        #     # Fixed start for evaluation
        #     self.ptr = self.L - 1

        self.ptr = self.L - 1
        
        # Portfolio variables
        self.H = 0      # shares held
        self.I = 0.0    # invested amount
        self.IC = self.ic0  # investment capacity
        
        # Initialize cash
        P0 = self.P[self.ptr]
        self.cash0 = P0 * self.ic0
        self.cash = self.cash0
        
        return self._state()

    def step(self, act_weight: Tuple[int, float]):
        action, w = act_weight
        w = np.clip(w, 0.0, 1.0)

        # Prices and wealth BEFORE trade (paper uses b_t + p_t h_t)
        P_t = self.P[self.ptr]
        V_t_pre = self.cash + self.H * P_t

        # === execute trade at price P_t ===
        action_taken = 2
        if action == 0:  # BUY
            max_can_buy = int(self.cash / (P_t * (1.0 + self.commission)))
            max_allowed = self.IC - self.H
            max_shares  = min(max_can_buy, max_allowed)
            if max_shares >= 100:
                B = int(round(w * max_shares / 100)) * 100
                B = max(100, min(B, max_shares))
                trade_val = P_t * B
                fee = trade_val * self.commission
                self.cash -= (trade_val + fee)
                self.H    += B
                action_taken = 0

        elif action == 1:  # SELL
            if self.H >= 100:
                S = int(round(w * self.H / 100)) * 100
                S = max(100, min(S, self.H))
                trade_val = P_t * S
                fee = trade_val * self.commission
                self.cash += (trade_val - fee)
                self.H    -= S
                action_taken = 1

        # === advance time ===
        self.ptr += 1
        done = self._done()

        # lookahead (default 1 step)
        next_idx = min(self.ptr + self.lookahead - 1, len(self.P) - 1)
        P_next = self.P[next_idx]

        # Wealth AFTER move with post-trade holdings: b_{t+1} + p_{t+1} h_{t+1}
        V_next = self.cash + self.H * P_next

        # Paper reward (no commission term): Δ portfolio value
        reward = (V_next - V_t_pre) * self.reward_scale

        # Next state (for the immediate next step)
        next_state = self._state() if not done else np.zeros_like(self._state())

        info = {
            "portfolio_value": self.portfolio_value(),
            "action": action_taken,
            "weight": float(w),
            "shares": int(self.H),
            "cash": float(self.cash),
            "price": float(P_t),
        }
        return next_state.astype(np.float32), float(reward), done, info

    def portfolio_value(self, P_t: Optional[float] = None) -> float:
        """Calculate total portfolio value."""
        if P_t is None:
            P_t = self.P[min(self.ptr, len(self.P) - 1)]
        return self.cash + self.H * P_t