import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
from agent.drl_utrans import DrlUTransAgent

def test_train_step_updates_weights():
    # Initialize agent with smaller batch size for testing
    agent = DrlUTransAgent(state_dim=(12, 14), lr=0.001, batch_size=10, memory_size=50)
    # Fill replay memory with dummy transitions
    for i in range(10):
        state = torch.randn(12, 14)
        next_state = torch.randn(12, 14)
        action = i % 3  # cycle actions 0,1,2
        weight = 0.0 if action == 2 else random.random()
        reward = random.uniform(-1.0, 1.0)
        done = (i == 9)  # last transition ends episode
        agent.store_transition(state, action, weight, reward, next_state, done)
    # Save parameters before training
    params_before = [p.clone() for p in agent.policy_net.parameters()]
    # Perform one training step
    loss = agent.train_step()
    # Check loss is returned and is a float
    assert loss is not None, "train_step() did not return a loss"
    assert isinstance(loss, float), "Loss is not a float"
    assert loss >= 0.0, "Loss is negative"
    # Check that at least one parameter has changed after training
    params_after = list(agent.policy_net.parameters())
    updated = False
    for p_before, p_after in zip(params_before, params_after):
        if not torch.allclose(p_before, p_after, atol=1e-6):
            updated = True
            break
    assert updated, "Policy network parameters did not update after one training step"


if __name__ == "__main__":
    test_train_step_updates_weights()
    print("Test passed: train_step updates weights and returns valid loss.")