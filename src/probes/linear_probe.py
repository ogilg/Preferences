from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


class LinearProbe:
    def __init__(self, max_iter: int = 1000):
        self.model = LogisticRegression(max_iter=max_iter, multi_class="multinomial")

    def fit(self, activations: np.ndarray, labels: np.ndarray) -> LinearProbe:
        self.model.fit(activations, labels)
        return self

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.model.predict(activations)

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(activations)

    def evaluate(self, activations: np.ndarray, labels: np.ndarray) -> dict:
        y_pred = self.predict(activations)
        return classification_report(labels, y_pred, output_dict=True)

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[LinearProbe, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    probe = LinearProbe().fit(X_train, y_train)
    results = probe.evaluate(X_test, y_test)

    cv_scores = cross_val_score(probe.model, activations, labels, cv=cv_folds)
    results["cv_mean"] = float(cv_scores.mean())
    results["cv_std"] = float(cv_scores.std())

    return probe, results
