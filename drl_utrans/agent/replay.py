from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import torch


class ReplayMemory:
    """
    Simple FIFO experience replay buffer.

    Each item stored is a 6-tuple:
        (state, action, weight, reward, next_state, done)
    where *state* and *next_state* are **CPU tensors** to avoid GPU bloat.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    # ------------------------------------------------------------------ #
    def push(
        self,
        state: torch.Tensor,
        action: int,
        weight: float,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        # store CPU clones to keep replay on host memory
        self.buffer.append(
            (
                state.cpu().detach(),
                action,
                weight,
                reward,
                next_state.cpu().detach(),
                done,
            )
        )

    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, weights, rewards, next_states, dones = zip(*batch)

        states_t = torch.stack(states)                 # (B, L, F)
        actions_t = torch.tensor(actions, dtype=torch.long)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.stack(next_states)
        dones_t = torch.tensor(dones, dtype=torch.bool)
        return states_t, actions_t, weights_t, rewards_t, next_states_t, dones_t

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.buffer)
