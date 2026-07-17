"""Modern PyTorch policy-value network for AlphaZero Gomoku."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Architecture settings stored alongside every checkpoint."""

    board_width: int
    board_height: int
    input_channels: int = 4
    channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden_size: int = 128

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def action_size(self) -> int:
        return self.board_width * self.board_height


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Scalar values produced by one optimizer update."""

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    gradient_norm: float


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by the shared trunk."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual, inplace=True)


class PolicyValueModel(nn.Module):
    """Residual trunk with separate policy and value heads."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(config.channels) for _ in range(config.residual_blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, config.policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.policy_channels * config.action_size, config.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(config.channels, config.value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.value_channels * config.action_size, config.value_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(self.stem(states))
        policy_logits = self.policy_head(features)
        log_policy = F.log_softmax(policy_logits, dim=1)
        value = self.value_head(features)
        return log_policy, value


class PolicyValueNet:
    """Training and inference facade compatible with the existing MCTS code."""

    def __init__(
        self,
        board_width: int,
        board_height: int,
        model_file: str | Path | None = None,
        use_gpu: bool | None = None,
        *,
        device: str | torch.device = "auto",
        config: NetworkConfig | None = None,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 5.0,
        amp: bool = True,
        compile_model: bool = False,
        compile_backend: str | None = None,
    ) -> None:
        if config is None:
            config = NetworkConfig(board_width=board_width, board_height=board_height)
        if (config.board_width, config.board_height) != (board_width, board_height):
            raise ValueError("network config dimensions must match the requested board")

        self.config = config
        self.board_width = board_width
        self.board_height = board_height
        self.device = self._resolve_device(device, use_gpu)
        self.use_gpu = self.device.type == "cuda"
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = bool(amp and self.device.type == "cuda")
        self.compile_enabled = compile_model
        self.compile_backend = compile_backend

        self.model = PolicyValueModel(config).to(self.device)
        self.policy_value_net = self.model
        if compile_model and compile_backend is not None:
            self._forward_model = torch.compile(self.model, backend=compile_backend)
        elif compile_model:
            self._forward_model = torch.compile(self.model)
        else:
            self._forward_model = self.model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)
        self.training_step = 0
        self.last_metrics: TrainingMetrics | None = None

        if model_file is not None:
            self.load_model(model_file)

    @staticmethod
    def _resolve_device(
        device: str | torch.device,
        use_gpu: bool | None,
    ) -> torch.device:
        if use_gpu is not None:
            device = "cuda" if use_gpu else "cpu"
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return resolved

    def _states_to_tensor(self, states: Sequence[np.ndarray] | np.ndarray) -> Tensor:
        array = np.ascontiguousarray(np.asarray(states, dtype=np.float32))
        expected = (self.config.input_channels, self.board_height, self.board_width)
        if array.ndim != 4 or tuple(array.shape[1:]) != expected:
            raise ValueError(f"states must have shape (batch, {expected}), got {array.shape}")
        return torch.from_numpy(array).to(self.device)

    def _forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                return self._forward_model(states)
        except Exception as error:
            if not self.compile_enabled or self._forward_model is self.model:
                raise
            warnings.warn(
                f"torch.compile failed ({type(error).__name__}); falling back to eager mode",
                RuntimeWarning,
                stacklevel=2,
            )
            self._forward_model = self.model
            self.compile_enabled = False
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                return self.model(states)

    def policy_value(
        self,
        state_batch: Sequence[np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return action probabilities and values for a batch of states."""
        self.model.eval()
        states = self._states_to_tensor(state_batch)
        with torch.inference_mode():
            log_policy, values = self._forward(states)
        probabilities = log_policy.exp().float().cpu().numpy()
        return probabilities, values.float().cpu().numpy()

    def policy_value_fn(self, board: Any) -> tuple[Iterable[tuple[int, float]], float]:
        """Evaluate one board using the callback contract expected by MCTS."""
        if (board.width, board.height) != (self.board_width, self.board_height):
            raise ValueError("board dimensions do not match the network checkpoint")
        legal_positions = np.asarray(board.availables, dtype=np.int64)
        state = np.ascontiguousarray(board.current_state()[None, ...], dtype=np.float32)
        probabilities, values = self.policy_value(state)
        if len(legal_positions) == 0:
            return iter(()), float(values[0, 0])
        legal_probabilities = probabilities[0, legal_positions]
        probability_sum = float(legal_probabilities.sum())
        if probability_sum <= 0.0 or not np.isfinite(probability_sum):
            legal_probabilities = np.full(
                len(legal_positions),
                1.0 / len(legal_positions),
                dtype=np.float32,
            )
        else:
            legal_probabilities = legal_probabilities / probability_sum
        action_priors = zip(legal_positions.tolist(), legal_probabilities.tolist(), strict=True)
        return action_priors, float(values[0, 0])

    def train_batch(
        self,
        state_batch: Sequence[np.ndarray] | np.ndarray,
        mcts_probs: Sequence[np.ndarray] | np.ndarray,
        winner_batch: Sequence[float] | np.ndarray,
        learning_rate: float | None = None,
    ) -> TrainingMetrics:
        """Perform one AlphaZero policy-value optimizer update."""
        self.model.train()
        states = self._states_to_tensor(state_batch)
        target_policy = torch.as_tensor(
            np.ascontiguousarray(mcts_probs, dtype=np.float32),
            device=self.device,
        )
        target_values = torch.as_tensor(
            np.ascontiguousarray(winner_batch, dtype=np.float32),
            device=self.device,
        ).reshape(-1)

        expected_policy_shape = (states.shape[0], self.config.action_size)
        if tuple(target_policy.shape) != expected_policy_shape:
            raise ValueError(
                f"mcts_probs must have shape {expected_policy_shape}, got {target_policy.shape}"
            )
        if target_values.shape[0] != states.shape[0]:
            raise ValueError("winner_batch length must match state_batch")
        if learning_rate is not None:
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate

        self.optimizer.zero_grad(set_to_none=True)
        log_policy, values = self._forward(states)
        # Compute losses in FP32 even when the forward pass uses CUDA AMP.
        # This keeps reductions stable and avoids mixed-dtype MSE backward errors.
        policy_loss = -(target_policy * log_policy.float()).sum(dim=1).mean()
        value_loss = F.mse_loss(values.float().reshape(-1), target_values)
        loss = policy_loss + value_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            entropy = -(log_policy.exp() * log_policy).sum(dim=1).mean()

        self.training_step += 1
        metrics = TrainingMetrics(
            loss=float(loss.detach().cpu()),
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
        )
        self.last_metrics = metrics
        return metrics

    def train_step(
        self,
        state_batch: Sequence[np.ndarray] | np.ndarray,
        mcts_probs: Sequence[np.ndarray] | np.ndarray,
        winner_batch: Sequence[float] | np.ndarray,
        lr: float,
    ) -> tuple[float, float]:
        """Backward-compatible training API used by the existing pipeline."""
        metrics = self.train_batch(state_batch, mcts_probs, winner_batch, lr)
        return metrics.loss, metrics.entropy

    def get_policy_param(self) -> Mapping[str, Tensor]:
        """Return the uncompiled model state for compatibility."""
        return self.model.state_dict()

    def save_model(
        self,
        model_file: str | Path,
        *,
        include_optimizer: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save a portable, versioned checkpoint."""
        path = Path(model_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_type": "residual_policy_value_net",
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "training_step": self.training_step,
            "metadata": dict(metadata or {}),
        }
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_model(self, model_file: str | Path, *, load_optimizer: bool = False) -> None:
        """Load a modern checkpoint or a compatible raw state dictionary."""
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            format_version = int(checkpoint.get("format_version", 0))
            if format_version != CHECKPOINT_FORMAT_VERSION:
                raise ValueError(f"unsupported checkpoint format version: {format_version}")
            saved_config = NetworkConfig(**checkpoint["config"])
            if saved_config != self.config:
                raise ValueError(
                    "checkpoint architecture does not match this network; "
                    "use PolicyValueNet.from_checkpoint instead"
                )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.training_step = int(checkpoint.get("training_step", 0))
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return
        if not isinstance(checkpoint, Mapping):
            raise ValueError("unsupported PyTorch checkpoint format")
        self.model.load_state_dict(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        model_file: str | Path,
        *,
        device: str | torch.device = "auto",
        load_optimizer: bool = False,
        amp: bool = True,
        compile_model: bool = False,
        compile_backend: str | None = None,
    ) -> PolicyValueNet:
        """Construct the matching architecture directly from a checkpoint."""
        map_location = cls._resolve_device(device, use_gpu=None)
        checkpoint = torch.load(model_file, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
            raise ValueError("checkpoint does not contain modern architecture metadata")
        config = NetworkConfig(**checkpoint["config"])
        network = cls(
            config.board_width,
            config.board_height,
            device=map_location,
            config=config,
            amp=amp,
            compile_model=compile_model,
            compile_backend=compile_backend,
        )
        network.load_model(model_file, load_optimizer=load_optimizer)
        return network


class Net(PolicyValueModel):
    """Compatibility constructor for the historical ``Net(width, height)`` API."""

    def __init__(self, board_width: int, board_height: int) -> None:
        super().__init__(NetworkConfig(board_width=board_width, board_height=board_height))
