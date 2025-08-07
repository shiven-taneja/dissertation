import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# from drl_utrans.models.utrans import UTransModel
from drl_utrans.models.utrans import UTransNet
from drl_utrans.agent.replay import ReplayMemory

class DrlUTransAgent:
    """
    Deep Q-learning agent utilizing the UTrans model (policy network and target network).
    Implements experience replay, epsilon-greedy exploration, target network updates,
    and uses Huber loss for training.
    """
    def __init__(
        self,
        state_dim: Tuple[int, int] = (12, 1),
        lr: float = 1e-3,
        batch_size: int = 20,
        gamma: float = 0.99,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.95,
        device: Optional[str] = None,
    ):
        self.window_size, self.feature_dim = state_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.eps_delta = (epsilon_start - epsilon_end) / 50
        self.weight_loss_coef = 0.5 
        # Set device for computations
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_net = UTransNet(
            input_dim=self.feature_dim,
            n_actions=3,  # Buy / Sell / Hold
            n_transformer_heads=8,
            n_transformer_layers=1,
        ).to(self.device)
        self.target_net = UTransNet(
            input_dim=self.feature_dim,
            n_actions=3,  # Buy / Sell / Hold
            n_transformer_heads=8,
            n_transformer_layers=1,
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizer: RAdam as specified in the paper
        self.optimizer = optim.RAdam(self.policy_net.parameters(), lr=lr)
        # Experience replay memory
        self.memory = ReplayMemory(memory_size)
        # Counter for training steps (for target network updates)
        self.train_steps = 0
        # Compile model for performance (PyTorch 2.x)
        if hasattr(torch, 'compile'):
            self.policy_net = torch.compile(self.policy_net)
            self.target_net = torch.compile(self.target_net)

    def select_action(self, state: torch.Tensor, eval_mode=False) -> Tuple[int, float]:
        """Choose an action and weight using epsilon-greedy."""
        if state.dim() == 2:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        
        # During training, use epsilon-greedy
        if not eval_mode and random.random() < self.epsilon:
            action = random.randrange(3)
            # Use random weights during exploration for better learning
            # weight = random.uniform(0.1, 1.0) if action != 2 else 0.0
            weight = 0.1 if action != 2 else 0.0
        else:
            # Exploit: use network predictions
            self.policy_net.eval()
            with torch.no_grad():
                action_logits, action_weight = self.policy_net(state.float())
            if not eval_mode:
                self.policy_net.train()
            
            action = int(torch.argmax(action_logits).item())
            # Use the network's weight prediction directly
            weight = float(torch.clamp(action_weight, 0.0, 1.0).item())
            weight = weight if action != 2 else 0.0
        
        return action, weight

    def store_transition(self, state: torch.Tensor, action: int, weight: float,
                         reward: float, next_state: torch.Tensor, done: bool):
        """Store a new transition in replay memory."""
        self.memory.push(state, action, weight, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Sample batch and train with both Q-loss and weight loss."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, weights, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device).float()
        next_states = next_states.to(self.device).float()
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Get Q-values and weight predictions
        q_values, pred_weights = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q, _ = self.target_net(next_states)
            max_next_q = next_q.max(dim=1).values
            max_next_q[dones] = 0.0
            targets = rewards + self.gamma * max_next_q
        
        # Q-value loss
        q_loss = nn.functional.smooth_l1_loss(state_action_values, targets)
        
        # Weight prediction loss (only for buy/sell actions)
        action_mask = (actions != 2)  # Not hold
        if action_mask.any():
            pred_weights_masked = pred_weights.squeeze(-1)[action_mask]
            target_weights = weights[action_mask]
            weight_loss = nn.functional.mse_loss(pred_weights_masked, target_weights)
            total_loss = q_loss + self.weight_loss_coef * weight_loss
        else:
            total_loss = q_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return float(total_loss.item())

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # self.epsilon = max(self.epsilon_end,
        #            self.epsilon - (1-self.epsilon_end))
        self.epsilon = max(self.epsilon_end, self.epsilon - self.eps_delta)

    def train(self, env, num_episodes: int = 50, max_steps: Optional[int] = None):
        """
        Train the agent on the given environment for a certain number of episodes.
        The environment should implement reset() and step(action, weight).
        """
        for episode in range(num_episodes):
            state = env.reset()  # state is a numpy array
            for t in range(max_steps or int(1e6)):
                # Convert numpy state to tensor for the policy network
                state_tensor = torch.from_numpy(state).float()
                action, weight = self.select_action(state_tensor)
                
                # Perform action in environment, passing action and weight separately
                next_state, reward, done, _ = env.step(action, weight)
                
                # Convert numpy next_state to tensor for storage
                next_state_tensor = torch.from_numpy(next_state).float()
                reward = float(reward)

                # Store transition and perform a training step
                self.store_transition(state_tensor, action, weight, reward, next_state_tensor, done)
                self.train_step()
                
                # Update state for the next iteration
                state = next_state
                
                if done:
                    break
            # Decay epsilon at the end of each episode
            self.decay_epsilon()