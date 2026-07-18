"""Centralized batched neural inference for parallel MCTS actors."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet


@dataclass(frozen=True, slots=True)
class InferenceStats:
    requests: int = 0
    batches: int = 0
    max_batch_size: int = 0
    inference_seconds: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        return self.requests / self.batches if self.batches else 0.0


@dataclass(slots=True)
class _Request:
    state: np.ndarray
    legal_positions: tuple[int, ...]
    completed: threading.Event
    result: tuple[list[tuple[int, float]], float] | None = None
    error: BaseException | None = None


_STOP = object()


class BatchedPolicyEvaluator:
    """Merge synchronous actor requests into bounded neural-network batches."""

    def __init__(
        self,
        network: PolicyValueNet,
        *,
        batch_size: int,
        max_wait_ms: float,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.network = network
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_ms / 1_000.0
        self._queue: queue.Queue[_Request | object] = queue.Queue()
        self._stats = InferenceStats()
        self._stats_lock = threading.Lock()
        self._closed = False
        self._state_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._run,
            name="alphazero-batched-inference",
            daemon=True,
        )
        self._worker.start()

    def __enter__(self) -> BatchedPolicyEvaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def policy_value_fn(self, board: Any) -> tuple[list[tuple[int, float]], float]:
        """Evaluate one board through the centralized batching worker."""
        if (board.width, board.height) != (
            self.network.board_width,
            self.network.board_height,
        ):
            raise ValueError("board dimensions do not match the network")
        request = _Request(
            state=np.ascontiguousarray(board.current_state(), dtype=np.float32),
            legal_positions=tuple(int(move) for move in board.availables),
            completed=threading.Event(),
        )
        with self._state_lock:
            if self._closed:
                raise RuntimeError("batched evaluator is closed")
            self._queue.put(request)
        request.completed.wait()
        if request.error is not None:
            raise RuntimeError("batched inference failed") from request.error
        if request.result is None:
            raise RuntimeError("batched inference returned no result")
        return request.result

    def snapshot(self) -> InferenceStats:
        with self._stats_lock:
            return self._stats

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._worker.join()
        if self._worker.is_alive():
            raise RuntimeError("batched inference worker did not stop")

    def _collect_batch(self, first: _Request) -> tuple[list[_Request], bool]:
        requests = [first]
        should_stop = False
        deadline = perf_counter() + self.max_wait_seconds
        while len(requests) < self.batch_size:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                should_stop = True
                break
            requests.append(item)
        return requests, should_stop

    def _evaluate(self, requests: list[_Request]) -> None:
        started = perf_counter()
        probabilities, values = self.network.policy_value(
            np.stack([request.state for request in requests])
        )
        duration = perf_counter() - started

        for index, request in enumerate(requests):
            legal = np.asarray(request.legal_positions, dtype=np.int64)
            if legal.size == 0:
                priors: list[tuple[int, float]] = []
            else:
                masked = probabilities[index, legal]
                total = float(masked.sum())
                if total <= 0.0 or not np.isfinite(total):
                    masked = np.full(legal.size, 1.0 / legal.size, dtype=np.float32)
                else:
                    masked = masked / total
                priors = list(zip(legal.tolist(), masked.astype(float).tolist(), strict=True))
            request.result = (priors, float(values[index, 0]))

        with self._stats_lock:
            self._stats = InferenceStats(
                requests=self._stats.requests + len(requests),
                batches=self._stats.batches + 1,
                max_batch_size=max(self._stats.max_batch_size, len(requests)),
                inference_seconds=self._stats.inference_seconds + duration,
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            requests, should_stop = self._collect_batch(item)
            try:
                self._evaluate(requests)
            except BaseException as error:
                for request in requests:
                    request.error = error
            finally:
                for request in requests:
                    request.completed.set()
            if should_stop:
                return
