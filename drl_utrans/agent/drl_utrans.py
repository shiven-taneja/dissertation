import random
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from drl_utrans.models.utrans import UTransNet
from drl_utrans.agent.replay import ReplayMemory

class DrlUTransAgent:
    def __init__(
        self,
        state_dim: Tuple[int, int] = (12, 1),
        lr: float = 3e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        memory_size: int = 100000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,   # unused now (kept for compat)
        weight_loss_coef: float = 0.5,
        rand_weights: bool = True,
        device: Optional[str] = None,
    ):
        self.window_size, self.feature_dim = state_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.weight_loss_coef = weight_loss_coef
        self.rand_weights = rand_weights

        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = UTransNet(
            input_dim=self.feature_dim,
            n_actions=3,
            n_transformer_heads=8,
            n_transformer_layers=1,
        ).to(self.device)

        self.target_net = UTransNet(
            input_dim=self.feature_dim,
            n_actions=3,
            n_transformer_heads=8,
            n_transformer_layers=1,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RAdam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        # exploration (step-based)
        self.global_steps = 0
        self.eps_start, self.eps_end = epsilon_start, epsilon_end
        self.eps_steps = 200_000    # decay horizon
        self.warmup_steps = 5_000   # no updates until buffer has this many samples
        self.epsilon = self.eps_start

        self.train_steps = 0

    def _update_epsilon(self):
        frac = min(1.0, self.global_steps / self.eps_steps)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: torch.Tensor, eval_mode=False) -> Tuple[int, float]:
        if state.dim() == 2:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        if not eval_mode:
            self._update_epsilon()

        if (not eval_mode) and (random.random() < self.epsilon):
            action = random.randrange(3)
            weight = random.uniform(0.1, 1.0) if (self.rand_weights and action != 2) else (0.1 if action != 2 else 0.0)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                action_logits, action_weight = self.policy_net(state.float())
            if not eval_mode:
                self.policy_net.train()
            action = int(torch.argmax(action_logits, dim=1).item())
            weight = float(torch.clamp(action_weight, 0.0, 1.0).item())
            weight = weight if action != 2 else 0.0

        if not eval_mode:
            self.global_steps += 1
        return action, weight

    def store_transition(self, state, action: int, weight: float, reward: float, next_state, done: bool):
        self.memory.push(state, action, weight, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        # warmup
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, weights, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device).float()
        next_states = next_states.to(self.device).float()
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device)
        weights = weights.to(self.device).float()

        q_values, pred_weights = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online, _ = self.policy_net(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target, _ = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            max_next_q = max_next_q * (~dones)
            targets = rewards + self.gamma * max_next_q

        q_loss = nn.functional.smooth_l1_loss(state_action_values, targets)

        # Auxiliary weight regression only for buy/sell
        mask = (actions != 2)
        if mask.any():
            pw = pred_weights.squeeze(-1)[mask]
            tw = weights[mask].clamp(0.0, 1.0)
            weight_loss = nn.functional.mse_loss(pw, tw)
            total_loss = q_loss + self.weight_loss_coef * weight_loss
        else:
            total_loss = q_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(total_loss.item())
