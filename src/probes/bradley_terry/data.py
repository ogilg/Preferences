from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.types import BinaryPreferenceMeasurement


@dataclass
class PairwiseActivationData:
    activations: dict[int, np.ndarray]
    pairs: np.ndarray  # (n_pairs, 2) array of (winner_idx, loser_idx)

    @classmethod
    def from_measurements(
        cls,
        measurements: list[BinaryPreferenceMeasurement],
        task_ids: np.ndarray,
        activations: dict[int, np.ndarray],
    ) -> PairwiseActivationData:
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

        pairs = []
        for m in measurements:
            if m.choice == "refusal":
                continue
            a_id, b_id = m.task_a.id, m.task_b.id
            if a_id not in id_to_idx or b_id not in id_to_idx:
                continue
            a_idx, b_idx = id_to_idx[a_id], id_to_idx[b_id]
            if m.choice == "a":
                pairs.append((a_idx, b_idx))
            else:
                pairs.append((b_idx, a_idx))

        return cls(activations=activations, pairs=np.array(pairs) if pairs else np.empty((0, 2), dtype=int))

    def get_batch(
        self,
        layer: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.choice(len(self.pairs), size=min(batch_size, len(self.pairs)), replace=False)
        acts = self.activations[layer]
        return acts[self.pairs[indices, 0]], acts[self.pairs[indices, 1]]
