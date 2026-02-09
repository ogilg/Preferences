from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .data import PairwiseActivationData


@dataclass
class BTResult:
    weights: np.ndarray  # [coef..., intercept] — same format as Ridge probes
    layer: int
    train_accuracy: float
    train_loss: float
    n_iterations: int


def _bt_loss_and_grad(
    w: np.ndarray,
    diff: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray]:
    logits = diff @ w[:-1] + w[-1]

    # Numerically stable sigmoid loss: -log(sigmoid(x)) = log(1 + exp(-x))
    loss = np.mean(np.logaddexp(0, -logits)) + 0.5 * l2_lambda * np.sum(w[:-1] ** 2)

    # Gradient: sigmoid(-x) * (-diff) = (1 - sigmoid(x)) * (-diff)
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    grad_scale = (probs - 1) / len(logits)
    grad_w = diff.T @ grad_scale + l2_lambda * w[:-1]
    grad_b = grad_scale.sum()

    return loss, np.append(grad_w, grad_b)


def train_bt(
    data: PairwiseActivationData,
    layer: int,
    l2_lambda: float = 1.0,
) -> BTResult:
    acts = data.activations[layer]
    diff = acts[data.pairs[:, 0]] - acts[data.pairs[:, 1]]

    d_model = acts.shape[1]
    w0 = np.zeros(d_model + 1)

    n_iterations = 0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_iterations
        n_iterations += 1
        return _bt_loss_and_grad(w, diff, l2_lambda)

    result = minimize(objective, w0, method="L-BFGS-B", jac=True)
    w = result.x

    logits = diff @ w[:-1] + w[-1]
    accuracy = (logits > 0).mean()

    return BTResult(
        weights=w,
        layer=layer,
        train_accuracy=float(accuracy),
        train_loss=float(result.fun),
        n_iterations=n_iterations,
    )


def train_for_comparisons(
    data: PairwiseActivationData,
    l2_lambda: float = 1.0,
) -> tuple[list[BTResult], dict[int, np.ndarray]]:
    if len(data.pairs) == 0:
        return [], {}

    results = []
    probes = {}

    for layer in sorted(data.activations.keys()):
        result = train_bt(data, layer, l2_lambda=l2_lambda)
        results.append(result)
        probes[layer] = result.weights

    return results, probes
