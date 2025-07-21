import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utrans import UTransModel

def test_forward_output_shape():
    model = UTransModel(window_size=12, feature_dim=14)
    model.eval()
    batch_size = 4
    dummy_input = torch.randn(batch_size, 12, 14)
    with torch.no_grad():
        action_logits, action_weight = model(dummy_input)
    # Verify output shapes
    assert action_logits.shape == (batch_size, 3), \
        f"Unexpected action_logits shape: {action_logits.shape}"
    assert action_weight.shape == (batch_size, 1), \
        f"Unexpected action_weight shape: {action_weight.shape}"
    # Verify action_weight values are within [0, 1]
    assert (0.0 <= action_weight).all() and (action_weight <= 1.0).all(), \
        "Action weight output not in [0, 1]"

if __name__ == "__main__":
    test_forward_output_shape()
    print("Test passed: forward method produces expected output shapes and values.")