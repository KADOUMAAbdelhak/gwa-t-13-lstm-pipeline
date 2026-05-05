"""Decision head for drift-triggered update commitment."""
from __future__ import annotations

from typing import Tuple


class UpdateDecisionHead:
    """
    Online logistic regression that decides whether a drift-triggered update should be committed.

    ``alpha`` is the locked sensitivity threshold (UC2 alpha narrative): larger alpha
    means fewer commits (more conservative), smaller alpha means more commits.
    """

    def __init__(self, alpha: float = 0.5, l2: float = 1.0):
        from river.linear_model import LogisticRegression
        from river.optim import SGD

        self._model = LogisticRegression(optimizer=SGD(0.01), l2=l2)
        self._alpha = alpha
        self._n_seen = 0
        self._n_committed = 0

    def predict_commit(self, features: dict) -> Tuple[bool, float]:
        self._n_seen += 1
        if self._n_seen <= 3:
            self._n_committed += 1
            return True, 1.0
        p = self._model.predict_proba_one(features).get(True, 0.0)
        commit = p >= self._alpha
        if commit:
            self._n_committed += 1
        return commit, float(p)

    def learn_one(self, features: dict, label: bool) -> None:
        self._model.learn_one(features, bool(label))

    def stats(self) -> dict:
        return {
            "n_seen": self._n_seen,
            "n_committed": self._n_committed,
        }
