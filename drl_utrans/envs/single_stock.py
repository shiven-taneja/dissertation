from __future__ import annotations

import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch


class PaperSingleStockEnv:
    """
    Single-stock trading environment that *exactly* follows
    Algorithm 1 (reward) from Yang et al. 2023, with these extras:

    • `train_mode=True`  → each reset() picks a random start index
      so episodes cover diverse sub-segments of the training set.
    • `train_mode=False` → deterministic test episode that runs once
      from the first index to the very end.

    • Commission fee (`commission`) deducted on every trade
      (buy + sell) just like FinRL.

    State returned by reset/step is a **NumPy (window, feature) float32**.
    Helper `.to_tensor()` gives you a ready PyTorch float tensor.
    """

    # ------------------------------------------------------------------ #
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
        # -------- store market data ----------------------------------- #
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        elif not isinstance(features, np.ndarray):
            features = np.asarray(features)
        self.X = features.astype(np.float32)

        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        self.P = prices.astype(np.float32)

        assert len(self.X) == len(
            self.P
        ), "features and price length mismatch!"

        # -------- static hyper-params --------------------------------- #
        self.L = window_size
        self.ic0 = ic_shares
        self.cash0 = None 
        self.commission = commission
        self.train_mode = train_mode
        self.rng = random.Random(seed)

        # pointer bounds
        self.max_ptr = len(self.X) - 2  # last valid index  (P_t exists)

        # episode bookkeeping
        self.ptr: int
        self.H: int
        self.I: float
        self.IC: int
        self.reset()

    # ================================================================== #
    # internal helpers
    # ================================================================== #
    def _state(self) -> np.ndarray:
        """Return current state window (view, not copy)."""
        return self.X[self.ptr - self.L + 1: self.ptr +1]
        # return self.X[self.ptr - self.L: self.ptr] 

    def _done(self) -> bool:
        """Episode ends when we can’t compute P_{t+1}."""
        return self.ptr >= self.max_ptr

    def _cost_basis(self) -> float:
        return self.I / self.H if self.H > 0 else 0.0

    # ================================================================== #
    # gym-like API
    # ================================================================== #
    def reset(self) -> np.ndarray:
        """
        Returns first state window.
        In train_mode we pick a random start index so that
        window & at least one step fit inside data.
        """
        # if self.train_mode:
        #     self.ptr = self.rng.randint(self.L, self.max_ptr - 1)
        # else:  # test mode
        #     if hasattr(self, "_used_once"):
        #         raise RuntimeError("Test env can be run only once.")
        #     self._used_once = True
        #     self.ptr = self.L  # deterministic start

        self.ptr = self.L
        
        # portfolio vars
        self.H = 0  # shares held
        self.I = 0.0  # invested amount $
        self.IC = self.ic0  # remaining capacity (shares)

        P0 = self.P[self.ptr]
        self.cash0 = P0 * self.ic0
        self.cash  = self.cash0

        return self._state()

    # ------------------------------------------------------------------ #
    def step(self, act_weight: Tuple[int, float]):
        """
        Parameters
        ----------
        act_weight : (action_id, weight)
            action_id : 0 buy, 1 sell, 2 hold
            weight    : ∈[0,1] ignored for hold
        Returns
        -------
        next_state : np.ndarray  (window, feat) float32
        reward     : float
        done       : bool
        info       : dict   (current portfolio value)
        """
        action, w = act_weight
        P_t = self.P[self.ptr]
        cost_basis = self._cost_basis()
        reward = 0.0

        # -------- BUY -------------------------------------------------- #
        if action == 0:
            if self.IC - self.H <= 100:
                action = 2
            else:
                B = max(100, round(((self.IC - self.H) * w) / 100) * 100)
                trade_val = P_t * B
                fee = trade_val * self.commission
                self.cash -= trade_val + fee

                self.H += B
                self.I += trade_val  
                cost_basis = self._cost_basis()
                reward = (cost_basis - P_t) * B 
                self.IC -= B

        # -------- SELL ------------------------------------------------- #
        elif action == 1:
            if self.H <= 0: # no shares to sell
                action = 2
            else:
                cost_basis = self._cost_basis()
                S = max(100, round((self.H * w) / 100) * 100)
                S = min(S, self.H)
                trade_val = P_t * S
                fee = trade_val * self.commission
                self.cash += trade_val - fee

                self.H -= S
                self.I -= cost_basis * S  # remove cost basis of sold shares
                reward = (P_t - cost_basis) * S 
                # self.IC  += int(round(reward / P_t))
                self.IC += S
            

        # -------- HOLD ------------------------------------------------- #
        elif self.H > 0:
            if self.I > 0:
                reward = (P_t - cost_basis) * self.H
            else: 
                reward = 0

        # ------ advance time ------------------------------------------ #
        self.ptr += 1
        done = self._done()
        next_state = (
            self._state() if not done else np.zeros_like(self._state())
        )
        info = {"portfolio_value": self.portfolio_value(),
        "action": action, "weight": w, "shares": self.H}
        return next_state.astype(np.float32), reward, done, info

    # ================================================================== #
    # convenience
    # ================================================================== #
    def portfolio_value(self, P_t: Optional[float] = None) -> float:
        """cash value of held shares + cost basis remainder (IC not modelled)."""
        if P_t is None:
            P_t = self.P[self.ptr]
        return self.cash + self.H * P_t

    # quick torch helper
    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state).float()
