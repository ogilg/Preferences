from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .data import PairwiseActivationData

if TYPE_CHECKING:
    from src.types import BinaryPreferenceMeasurement


@dataclass
class BTResult:
    weights: np.ndarray  # [coef..., intercept] — same format as Ridge probes
    layer: int
    train_accuracy: float
    train_loss: float
    n_epochs: int


def _bt_loss_and_grad(
    w: np.ndarray,
    h_winners: np.ndarray,
    h_losers: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray]:
    diff = h_winners - h_losers
    logits = diff @ w[:-1] + w[-1]

    # Numerically stable sigmoid loss: -log(sigmoid(x)) = log(1 + exp(-x))
    loss = np.mean(np.logaddexp(0, -logits)) + 0.5 * l2_lambda * np.sum(w[:-1] ** 2)

    # Gradient: sigmoid(-x) * (-diff) = (1 - sigmoid(x)) * (-diff)
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    grad_scale = (probs - 1) / len(logits)  # negative because -log(sigmoid)
    grad_w = diff.T @ grad_scale + l2_lambda * w[:-1]
    grad_b = grad_scale.sum()

    return loss, np.append(grad_w, grad_b)


def train_bt(
    data: PairwiseActivationData,
    layer: int,
    lr: float = 0.01,
    l2_lambda: float = 1.0,
    batch_size: int = 64,
    max_epochs: int = 1000,
    patience: int = 10,
    rng: np.random.Generator | None = None,
) -> BTResult:
    if rng is None:
        rng = np.random.default_rng()

    d_model = data.activations[layer].shape[1]
    w = np.zeros(d_model + 1)  # [coef..., intercept]

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        h_winners, h_losers = data.get_batch(layer, batch_size, rng)
        loss, grad = _bt_loss_and_grad(w, h_winners, h_losers, l2_lambda)
        w -= lr * grad

        if loss < best_loss - 1e-6:
            best_loss = loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Compute final accuracy on all pairs
    acts = data.activations[layer]
    all_winners = acts[data.pairs[:, 0]]
    all_losers = acts[data.pairs[:, 1]]
    logits = (all_winners - all_losers) @ w[:-1] + w[-1]
    accuracy = (logits > 0).mean()

    final_loss, _ = _bt_loss_and_grad(w, all_winners, all_losers, l2_lambda)

    return BTResult(
        weights=w,
        layer=layer,
        train_accuracy=float(accuracy),
        train_loss=float(final_loss),
        n_epochs=epoch + 1,
    )


def train_for_comparisons(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    measurements: list[BinaryPreferenceMeasurement],
    **train_kwargs,
) -> tuple[list[BTResult], dict[int, np.ndarray]]:
    data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)

    if len(data.pairs) == 0:
        return [], {}

    results = []
    probes = {}

    for layer in sorted(activations.keys()):
        result = train_bt(data, layer, **train_kwargs)
        results.append(result)
        probes[layer] = result.weights

    return results, probes
