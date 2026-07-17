"""Randomness controls shared by self-play and optimization."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RandomContext:
    seed: int
    numpy: np.random.Generator


def seed_everything(seed: int, *, deterministic: bool = False) -> RandomContext:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)

    try:
        import torch
    except ImportError:
        return RandomContext(seed=seed, numpy=generator)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return RandomContext(seed=seed, numpy=generator)


def capture_random_state(generator: np.random.Generator) -> dict[str, Any]:
    """Capture JSON-serializable Python, NumPy, and PyTorch RNG states."""
    legacy = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_generator": generator.bit_generator.state,
        "numpy_legacy": {
            "name": legacy[0],
            "keys": legacy[1].tolist(),
            "position": legacy[2],
            "has_gauss": legacy[3],
            "cached_gaussian": legacy[4],
        },
    }
    try:
        import torch
    except ImportError:
        return state
    state["torch_cpu"] = torch.get_rng_state().tolist()
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.tolist() for value in torch.cuda.get_rng_state_all()]
    return state


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def restore_random_state(state: dict[str, Any], generator: np.random.Generator) -> None:
    """Restore a state produced by :func:`capture_random_state`."""
    random.setstate(_nested_tuple(state["python"]))
    generator.bit_generator.state = state["numpy_generator"]
    legacy = state["numpy_legacy"]
    np.random.set_state(
        (
            legacy["name"],
            np.asarray(legacy["keys"], dtype=np.uint32),
            int(legacy["position"]),
            int(legacy["has_gauss"]),
            float(legacy["cached_gaussian"]),
        )
    )
    try:
        import torch
    except ImportError:
        return
    if "torch_cpu" in state:
        torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8) for value in state["torch_cuda"]]
        )
