"""Ring-disjoint cross-validation.

The headline protocol. Folds are formed over *connected components* of the
observed graph, so every application sharing a device, dealer, account or
guarantor with another lands in the same fold. The model therefore can never see
any part of a ring in training and be scored on another part in test -- which is
exactly what a random row-level split permits and what makes ring-detection
benchmarks so often report unusable numbers.

Fold membership is derived only from graph connectivity. Labels are never
consulted, so the split itself is unsupervised and cannot leak.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def ring_disjoint_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Assign an integer fold id per application."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    n_groups = len(uniq)

    if n_groups >= n_splits:
        gkf = GroupKFold(n_splits=n_splits)
        fold = np.zeros(len(groups), dtype=int)
        for i, (_, test) in enumerate(gkf.split(groups, groups=groups)):
            fold[test] = i
        return fold

    # Fewer groups than splits: fall back to hashing groups into folds so the
    # ring-disjoint guarantee still holds.
    rng = np.random.default_rng(seed)
    assign = rng.permutation(n_groups) % n_splits
    return assign[np.searchsorted(uniq, groups)]


def shuffle_label_control(labels: np.ndarray, fold: np.ndarray,
                          seed: int = 0) -> np.ndarray:
    """Permute labels *within* each fold, preserving base rates and fold structure.

    If a model trained on these shuffled labels still beats chance, the apparent
    skill is coming from the features' ability to separate folds or from a leak,
    not from any real relationship to fraud. This is the null hypothesis the
    whole report is measured against.
    """
    rng = np.random.default_rng(seed)
    out = labels.copy()
    for f in np.unique(fold):
        m = fold == f
        out[m] = rng.permutation(out[m])
    return out
