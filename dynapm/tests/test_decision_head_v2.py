"""Tests for DecisionHeadV2."""
from __future__ import annotations

from dynapm.core.decision_head_v2 import DecisionHeadV2


def test_seed_phase_commits():
    h = DecisionHeadV2(alpha=0.9999)
    for i in range(5):
        c, p = h.predict_commit({"a": float(i)})
        assert c is True and p == 1.0
    st = h.stats()
    assert st["n_seen"] == 5 and st["n_committed"] == 5


def test_alpha_threshold_basic():
    feats = {"x": 1.0, "y": 0.0}
    loose = DecisionHeadV2(alpha=0.2, l2=0.001)
    strict = DecisionHeadV2(alpha=0.95, l2=0.001)
    # Move past seed
    for _ in range(6):
        loose.predict_commit(feats)
        strict.predict_commit(feats)

    label = True
    for _ in range(60):
        loose.learn_one(feats, label)
        strict.learn_one(feats, label)

    n_loose = sum(1 for _ in range(30) if loose.predict_commit(feats)[0])
    n_strict = sum(1 for _ in range(30) if strict.predict_commit(feats)[0])
    assert n_loose >= n_strict


def test_negative_label_decreases_commit_prob():
    feats = {"f": 1.0}
    h = DecisionHeadV2(alpha=0.05, l2=0.5)
    for _ in range(6):
        h.predict_commit(feats)

    probs = []
    for _ in range(10):
        h.learn_one(feats, True)
        probs.append(h._model.predict_proba_one(feats).get(True, 0.0))
    p_pos = probs[-1]

    h2 = DecisionHeadV2(alpha=0.05, l2=0.5)
    for _ in range(6):
        h2.predict_commit(feats)
    for _ in range(50):
        h2.learn_one(feats, False)
        h2.learn_one(feats, False)
        h2.learn_one(feats, True)
    p_mixed = h2._model.predict_proba_one(feats).get(True, 0.0)

    assert p_mixed <= p_pos
