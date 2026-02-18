"""Compute probe weight cosine similarities across 10 seeds."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import Ridge

from src.probes.core.linear_probe import train_and_evaluate

load_dotenv()

PARAPHRASES_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrases.json")
EXISTING_ACTIVATIONS = Path("activations/gemma_3_27b/activations_prompt_last.npz")
PARA_ACTIVATIONS = Path("experiments/probe_science/paraphrase_augmentation/paraphrase_activations.npz")

LAYER = 31
CV_FOLDS = 5
N_ALPHA_SWEEP = 9
TEST_FRACTION = 0.2
SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 8888, 9999, 7777]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    with open(PARAPHRASES_FILE) as f:
        paraphrases = json.load(f)
    utilities = {p["task_id"]: p["utility"] for p in paraphrases}
    task_ids_100 = list(utilities.keys())

    existing = np.load(EXISTING_ACTIVATIONS, allow_pickle=True)
    existing_task_ids = list(existing["task_ids"])
    existing_acts = existing[f"layer_{LAYER}"]
    existing_id_to_idx = {tid: i for i, tid in enumerate(existing_task_ids)}

    para_data = np.load(PARA_ACTIVATIONS, allow_pickle=True)
    para_task_ids = list(para_data["task_ids"])
    para_acts = para_data[f"layer_{LAYER}"]
    para_id_to_idx = {tid: i for i, tid in enumerate(para_task_ids)}

    available_ids = [tid for tid in task_ids_100 if tid in existing_id_to_idx]
    alphas = np.logspace(0, 7, N_ALPHA_SWEEP)

    sims_base_aug = []
    sims_base_para = []
    sims_aug_para = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sorted_ids = sorted(available_ids, key=lambda tid: utilities[tid])
        quartile_size = len(sorted_ids) // 4
        train_ids = []
        for q in range(4):
            start = q * quartile_size
            end = (q + 1) * quartile_size if q < 3 else len(sorted_ids)
            q_ids = sorted_ids[start:end]
            n_test_q = max(1, int(len(q_ids) * TEST_FRACTION))
            q_test = set(rng.choice(len(q_ids), size=n_test_q, replace=False))
            for j, tid in enumerate(q_ids):
                if j not in q_test:
                    train_ids.append(tid)

        def get_orig(ids):
            indices = [existing_id_to_idx[tid] for tid in ids]
            return existing_acts[indices], np.array([utilities[tid] for tid in ids])

        def get_para(ids):
            indices = [para_id_to_idx[f"{tid}_para"] for tid in ids]
            return para_acts[indices], np.array([utilities[tid] for tid in ids])

        train_orig_acts, train_orig_labels = get_orig(train_ids)
        train_para_acts, train_para_labels = get_para(train_ids)
        aug_acts = np.concatenate([train_orig_acts, train_para_acts])
        aug_labels = np.concatenate([train_orig_labels, train_para_labels])

        probes = {}
        for name, X, y in [
            ("baseline", train_orig_acts, train_orig_labels),
            ("augmented", aug_acts, aug_labels),
            ("paraphrase_only", train_para_acts, train_para_labels),
        ]:
            probe, _, _ = train_and_evaluate(X, y, CV_FOLDS, alphas=alphas)
            w = np.concatenate([probe.coef_.flatten(), [probe.intercept_]])
            probes[name] = w

        sims_base_aug.append(cosine_sim(probes["baseline"], probes["augmented"]))
        sims_base_para.append(cosine_sim(probes["baseline"], probes["paraphrase_only"]))
        sims_aug_para.append(cosine_sim(probes["augmented"], probes["paraphrase_only"]))

    print("Probe weight cosine similarities (10 seeds):")
    print(f"  Baseline vs Augmented:      {np.mean(sims_base_aug):.6f} ± {np.std(sims_base_aug):.6f}")
    print(f"  Baseline vs Paraphrase-only: {np.mean(sims_base_para):.6f} ± {np.std(sims_base_para):.6f}")
    print(f"  Augmented vs Paraphrase-only: {np.mean(sims_aug_para):.6f} ± {np.std(sims_aug_para):.6f}")

    for name, vals in [
        ("base_aug", sims_base_aug),
        ("base_para", sims_base_para),
        ("aug_para", sims_aug_para),
    ]:
        print(f"  {name} per seed: {[f'{v:.6f}' for v in vals]}")


if __name__ == "__main__":
    main()
