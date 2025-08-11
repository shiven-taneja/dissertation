from __future__ import annotations
import random
from typing import Tuple, Optional
import numpy as np
import pandas as pd

class PaperSingleStockEnv:
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
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        elif not isinstance(features, np.ndarray):
            features = np.asarray(features)
        self.X = features.astype(np.float32)

        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        self.P = prices.astype(np.float32)

        assert len(self.X) == len(self.P), "features and price length mismatch!"

        self.L = window_size
        self.ic0 = ic_shares
        self.commission = commission
        self.train_mode = train_mode
        self.rng = random.Random(seed)

        self.max_ptr = len(self.X) - 1
        self.reset()

    def _portfolio_features(self, P_t: float) -> np.ndarray:
        equity = self.cash + self.H * P_t
        position_frac = (self.H / max(self.IC, 1e-8))
        cash_frac = (self.cash / max(equity, 1e-8))
        remaining_capacity = (self.IC - self.H) / max(self.IC, 1e-8)
        pf = np.array([position_frac, cash_frac, remaining_capacity], dtype=np.float32)
        return np.clip(pf, -2.0, 2.0)

    def _state(self) -> np.ndarray:
        start_idx = max(0, self.ptr - self.L + 1)
        end_idx = self.ptr + 1
        base = self.X[start_idx:end_idx]
        if base.shape[0] < self.L:
            padding = np.zeros((self.L - base.shape[0], base.shape[1]), dtype=np.float32)
            base = np.concatenate([padding, base], axis=0)
        P_t = self.P[self.ptr]
        pf = self._portfolio_features(P_t)
        pf_block = np.tile(pf[None, :], (self.L, 1))
        state = np.concatenate([base, pf_block], axis=1)
        return state.astype(np.float32)

    def _done(self) -> bool:
        return self.ptr >= self.max_ptr - 1

    def _cost_basis(self) -> float:
        return self.I / self.H if self.H > 0 else 0.0

    def reset(self) -> np.ndarray:
        self.ptr = self.L - 1
        self.H = 0
        self.I = 0.0
        self.IC = self.ic0
        P0 = self.P[self.ptr]
        self.cash0 = P0 * self.ic0
        self.cash = self.cash0
        return self._state()

    def portfolio_value(self, P_t: Optional[float] = None) -> float:
        if P_t is None:
            P_t = self.P[min(self.ptr, len(self.P) - 1)]
        return self.cash + self.H * P_t

    def step(self, act_weight: Tuple[int, float]):
        """
        Execute action and return next state.
        
        Actions:
        - 0: Buy
        - 1: Sell  
        - 2: Hold
        """
        action, w = act_weight
        
        # Ensure weight is valid
        w = np.clip(w, 0.0, 1.0)
        
        # Get current price
        P_t = self.P[self.ptr]
        reward = 0.0
        
        # Debug info
        old_H = self.H
        old_cash = self.cash
        
        # === BUY ACTION ===
        if action == 0:
            max_can_buy = int(self.cash / (P_t * (1 + self.commission)))
            max_allowed = self.IC - self.H
            max_shares = min(max_can_buy, max_allowed)
            
            if max_shares >= 100:
                # Calculate shares to buy based on weight
                B = int(round(w * max_shares / 100)) * 100
                B = max(100, min(B, max_shares))
                
                # Execute buy
                trade_val = P_t * B
                fee = trade_val * self.commission
                total_cost = trade_val + fee
                
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.H += B
                    self.I += trade_val
                    
                    # Calculate reward (paper formula)
                    cost_basis = self._cost_basis()
                    reward = (cost_basis - P_t) * B
                    action_taken = 0
                else:
                    action_taken = 2  # Can't afford, hold instead
            else:
                action_taken = 2  # Not enough capacity, hold
                
        # === SELL ACTION ===
        elif action == 1:
            if self.H >= 100:
                # Calculate shares to sell based on weight
                S = int(round(w * self.H / 100)) * 100
                S = max(100, min(S, self.H))
                
                # Execute sell
                cost_basis = self._cost_basis()
                trade_val = P_t * S
                fee = trade_val * self.commission
                
                self.cash += trade_val - fee
                self.H -= S
                self.I = max(0, self.I - cost_basis * S)
                
                # Calculate reward (paper formula)
                reward = (P_t - cost_basis) * S
                
                # Update IC based on profit
                # if reward > 0:
                #     self.IC += int(reward / P_t)
                # self.IC += int(reward / P_t)
                action_taken = 1
            else:
                action_taken = 2  # No shares to sell, hold
                
        # === HOLD ACTION ===
        else:
            action_taken = 2
            if self.H > 0:
                cost_basis = self._cost_basis()
                reward = (P_t - cost_basis) * self.H

        # Advance time
        self.ptr += 1
        done = self._done()
        
        # Get next state
        if not done:
            next_state = self._state()
        else:
            next_state = np.zeros_like(self._state())
        
        # Prepare info dict
        info = {
            "portfolio_value": self.portfolio_value(),
            "action": action_taken,
            "weight": w,
            "shares": self.H,
            "cash": self.cash,
            "price": P_t
        }
        
        return next_state.astype(np.float32), reward, done, info
