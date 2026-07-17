# Modern PyTorch backend

Stage 2 replaces the historical PyTorch 0.2 implementation with a supported,
versioned policy-value backend. The legacy source remains under
`alphazero_gomoku/legacy/` for reference.

## Architecture

`PolicyValueModel` contains a convolutional stem, a configurable residual trunk,
and separate policy and value heads. `NetworkConfig` is serialized into every
checkpoint so architecture reconstruction does not depend on command-line flags.

The default network uses:

- 4 input state planes;
- 64 trunk channels;
- 4 residual blocks;
- a 2-channel policy head;
- a 1-channel value head with a 128-unit hidden layer.

Small configurations can be used for tests and low-resource experiments:

```python
from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet

config = NetworkConfig(
    board_width=6,
    board_height=6,
    channels=32,
    residual_blocks=2,
)
network = PolicyValueNet(6, 6, config=config, device="auto")
```

## Training API

`train_batch` applies the AlphaZero objective:

```text
loss = policy_cross_entropy + value_mean_squared_error
```

AdamW supplies weight decay. Gradients are clipped before each update. CUDA
training enables automatic mixed precision by default; CPU execution stays in
float32. `train_step` preserves the two-value return contract expected by the
existing training pipeline.

## Checkpoints

Checkpoints contain:

- format version and model type;
- complete `NetworkConfig`;
- model state dictionary;
- optimizer state dictionary when requested;
- training step and user metadata.

```python
network.save_model("models/current_policy.pt", metadata={"run": "baseline"})
restored = PolicyValueNet.from_checkpoint("models/current_policy.pt")
```

Loading uses `weights_only=True` and an explicit device mapping. A checkpoint
with an unsupported format version or mismatched architecture is rejected with a
clear error instead of being partially loaded.

## Installation and tests

```bash
python -m pip install -e ".[dev,train]"
python -m pytest tests/test_policy_value_net_pytorch.py
```

The backend tests cover output normalization, legal-action masking, a real
optimizer update, checkpoint round trips, terminal boards, and MCTS integration.
