"""Online logistic decision head for drift-commit gating (v2 protocol)."""
from __future__ import annotations

from typing import Tuple


class DecisionHeadV2:
    """
    Online logistic regression for drift-event commit gating.

    Trained retrospectively when ground truth becomes available at t + 24h (via runner hook).
    """

    def __init__(self, alpha: float = 0.5, l2: float = 1.0):
        from river.linear_model import LogisticRegression
        from river.optim import SGD

        self._model = LogisticRegression(optimizer=SGD(0.01), l2=l2)
        self._alpha = float(alpha)
        self._n_seen = 0
        self._n_committed = 0

    def predict_commit(self, features: dict) -> Tuple[bool, float]:
        """
        Predict whether to commit a drift event given numerical features dict.

        First 5 events always commit (cold-start).
        """
        self._n_seen += 1
        if self._n_seen <= 5:
            self._n_committed += 1
            return True, 1.0
        probs = self._model.predict_proba_one(features)
        p = float(probs.get(True, 0.0))
        commit = p >= self._alpha
        if commit:
            self._n_committed += 1
        return commit, p

    def learn_one(self, features: dict, label: bool) -> None:
        self._model.learn_one(features, bool(label))

    def stats(self) -> dict:
        return {
            "n_seen": self._n_seen,
            "n_committed": self._n_committed,
            "commit_rate": float(self._n_committed / max(self._n_seen, 1)),
        }
