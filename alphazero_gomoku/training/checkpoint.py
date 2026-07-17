"""Complete experiment snapshots for interruption-safe training."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet

from .config import ExperimentConfig
from .replay_buffer import ReplayBuffer
from .reproducibility import capture_random_state, restore_random_state

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingState:
    iteration: int = 0
    total_games: int = 0
    total_positions: int = 0
    best_score: float = 0.0
    learning_rate_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class ResumeBundle:
    config: ExperimentConfig
    state: TrainingState
    network: PolicyValueNet
    replay: ReplayBuffer
    random_generator: np.random.Generator


class CheckpointManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        network: PolicyValueNet,
        replay: ReplayBuffer,
        state: TrainingState,
        config: ExperimentConfig,
        random_generator: np.random.Generator,
    ) -> Path:
        if len(replay) == 0:
            raise ValueError("cannot checkpoint without replay data")
        destination = self.root / f"step_{state.iteration:06d}"
        temporary = self.root / f".{destination.name}.tmp"
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        network.save_model(temporary / "model.pt", metadata={"iteration": state.iteration})
        replay.save(temporary / "replay.npz")
        manifest = {
            "config": config.to_dict(),
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "training_state": asdict(state),
            "random_state": capture_random_state(random_generator),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.rename(destination)
        (self.root / "latest").write_text(destination.name, encoding="utf-8")
        return destination

    def latest(self) -> Path:
        pointer = self.root / "latest"
        if not pointer.is_file():
            raise FileNotFoundError("no latest checkpoint pointer exists")
        destination = self.root / pointer.read_text(encoding="utf-8").strip()
        if not destination.is_dir():
            raise FileNotFoundError(f"checkpoint directory is missing: {destination}")
        return destination

    def load(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "auto",
    ) -> ResumeBundle:
        source = Path(checkpoint) if checkpoint is not None else self.latest()
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(manifest["config"])
        format_version = int(manifest.get("format_version", 1))
        if format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format version: {format_version}")
        state = TrainingState(**manifest["training_state"])
        network = PolicyValueNet.from_checkpoint(
            source / "model.pt",
            device=device,
            load_optimizer=True,
            amp=config.network.amp,
            compile_model=config.network.compile_model,
            compile_backend=config.network.compile_backend,
        )
        replay = ReplayBuffer.load(source / "replay.npz")
        generator = np.random.default_rng()
        restore_random_state(manifest["random_state"], generator)
        return ResumeBundle(config, state, network, replay, generator)
