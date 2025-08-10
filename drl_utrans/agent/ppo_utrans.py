# drl_utrans/agent/ppo_utrans.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from drl_utrans.models.utrans_ac import UTransActorCritic

# ------------------------------- utils ---------------------------------

def weight_center_from_bin(bin_idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Map integer bin indices ➜ weight in [0,1] at bin centers.
    For n_bins=11, centers are [0.0, 0.1, ..., 1.0].
    """
    if n_bins == 1:
        return torch.zeros_like(bin_idx, dtype=torch.float32)
    return (bin_idx.to(torch.float32) / (n_bins - 1)).clamp(0.0, 1.0)

@dataclass
class PPOConfig:
    state_dim: Tuple[int, int] = (12, 14)
    n_actions: int = 3
    n_weight_bins: int = 11
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    rollout_steps: int = 2048
    update_epochs: int = 50
    minibatch_size: int = 256
    n_transformer_heads: int = 8
    n_transformer_layers: int = 1

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []          # (B,) long in {0,1,2}
        self.wbins = []            # (B,) long in {0..n_bins-1}
        self.nonhold = []          # (B,) bool mask where action != Hold
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []         # joint log-prob (masked for Hold)

    def clear(self):
        self.__init__()

class PPOUTransAgent:
    """PPO agent with factorized policy over (action_type, weight_bin).

    - If action == Hold (2), the weight is forced to 0.0 and its log-prob/entropy
      are masked from the loss.
    - Otherwise, the chosen weight is the bin center in [0,1].
    """
    def __init__(self, cfg: PPOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        L, F = cfg.state_dim
        self.model = UTransActorCritic(
            input_dim=F,
            n_actions=cfg.n_actions,
            n_weight_bins=cfg.n_weight_bins,
            n_transformer_heads=cfg.n_transformer_heads,
            n_transformer_layers=cfg.n_transformer_layers,
        ).to(self.device)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer()

    # ------------------------- interaction ------------------------- #
    def _dist_and_value(self, state_t: torch.Tensor):
        action_logits, weight_logits, value = self.model(state_t)
        adist = Categorical(logits=action_logits)
        wdist = Categorical(logits=weight_logits)
        return adist, wdist, value

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, eval_mode: bool = False):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        state = state.float().to(self.device)

        adist, wdist, value = self._dist_and_value(state)
        if eval_mode:
            a = torch.argmax(adist.probs, dim=-1)
            wb = torch.argmax(wdist.probs, dim=-1)
        else:
            a = adist.sample()
            wb = wdist.sample()

        # mask weight if Hold
        nonhold = (a != 2)
        w = weight_center_from_bin(wb, self.cfg.n_weight_bins)
        w = torch.where(nonhold, w, torch.zeros_like(w))

        # joint logprob with mask
        logprob = adist.log_prob(a)
        logprob = torch.where(nonhold, logprob + wdist.log_prob(wb), logprob)

        return (
            int(a.item()),
            float(w.item()),
            float(value.item()),
            float(logprob.item()),
            int(wb.item()),
            bool(nonhold.item()),
        )

    def store(self, state: torch.Tensor, action: int, wbin: int, nonhold: bool, reward: float, done: bool, value: float, logprob: float):
        self.buffer.states.append(state.cpu().float())
        self.buffer.actions.append(action)
        self.buffer.wbins.append(wbin)
        self.buffer.nonhold.append(nonhold)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)
        self.buffer.logprobs.append(logprob)

    # --------------------------- update ---------------------------- #
    def _compute_gae(self, next_value: float, gamma: float, lam: float):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        dones   = torch.tensor(self.buffer.dones, dtype=torch.float32)
        values  = torch.tensor(self.buffer.values, dtype=torch.float32)

        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + values
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, returns

    def update(self):
        cfg = self.cfg
        device = self.device

        states  = torch.stack(self.buffer.states).to(device)              # (T, L, F)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=device)
        wbins   = torch.tensor(self.buffer.wbins, dtype=torch.long, device=device)
        nonhold = torch.tensor(self.buffer.nonhold, dtype=torch.bool, device=device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=device)
        values  = torch.tensor(self.buffer.values, dtype=torch.float32, device=device)

        # Bootstrap with last state
        with torch.no_grad():
            _, _, next_value = self._dist_and_value(states[-1:].to(device))
        adv, returns = self._compute_gae(float(next_value.item()), cfg.gamma, cfg.gae_lambda)
        adv, returns = adv.to(device), returns.to(device)

        T = states.size(0)
        idx = np.arange(T)
        mb = cfg.minibatch_size

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, mb):
                sel = idx[start:start + mb]
                sb_states  = states[sel]
                sb_actions = actions[sel]
                sb_wbins   = wbins[sel]
                sb_nonhold = nonhold[sel]
                sb_oldlogp = old_logprobs[sel]
                sb_adv     = adv[sel]
                sb_returns = returns[sel]

                adist, wdist, value = self._dist_and_value(sb_states)

                a_logp = adist.log_prob(sb_actions)
                w_logp = wdist.log_prob(sb_wbins)
                joint_logp = torch.where(sb_nonhold, a_logp + w_logp, a_logp)

                ratio = (joint_logp - sb_oldlogp).exp()
                surr1 = ratio * sb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * sb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value, sb_returns)

                # Entropy (mask weight entropy for hold)
                ent = adist.entropy() + (wdist.entropy() * sb_nonhold.float())
                entropy = ent.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    @torch.no_grad()
    def exec_logprob(self, state: torch.Tensor, action: int, wbin: int):
        if state.dim()==2: state = state.unsqueeze(0)
        state = state.float().to(self.device)
        adist, wdist, _ = self._dist_and_value(state)
        a_lp = adist.log_prob(torch.tensor([action], device=self.device))
        if action == 2:   # Hold → ignore weight head
            return float(a_lp.item())
        w_lp = wdist.log_prob(torch.tensor([wbin], device=self.device))
        return float((a_lp + w_lp).item())