"""Models: the node-only baseline, the graph-regularised learner, and propagation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 4000):
    lr = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
    lr.fit(X, y)
    return lr


#: Defaults chosen by nested ring-disjoint CV (see experiments/nested_cv.py).
#:
#: The earlier values were max_depth=6, min_samples_leaf=20, lr=0.08. The leaf
#: size of 20 was actively harmful: with 131 positives over 5 folds there are
#: only ~26 positives per fold, so a leaf demanding 20 samples can barely split
#: on the positive class at all. That single setting was holding the node-only
#: baseline at chance (AUC-PR 0.038) when it was actually worth 0.237.
GBT_DEFAULTS = dict(max_depth=6, min_samples_leaf=10, max_iter=250,
                    learning_rate=0.05, l2_regularization=1.0)

#: The search space used for inner-fold selection in nested CV.
#:
#: ``min_samples_leaf`` is deliberately capped at 10. This is the same failure
#: mode documented in README section 8: with ~131 positives, a 5-fold outer split
#: and a 3-fold inner split leaves ~17 positives per inner training fold, so any
#: leaf size near 20 cannot split on the positive class and the selected model
#: collapses toward chance. Leaving ``l = 20`` in the grid let inner-fold noise
#: re-select it and dragged the nested node-only score back to 0.05. The grid is
#: also kept small on purpose: with only ~105 positives in each inner OOF vector,
#: a 36-point grid has enough selection variance to pick a bad config by luck.
GBT_GRID = [(d, l, i, r)
            for d in (3, 6) for l in (5, 10)
            for i in (300, 500) for r in (0.05, 0.08)]


def fit_gbt(X: np.ndarray, y: np.ndarray, seed: int = 0, **overrides):
    """Histogram gradient boosting -- the tabular workhorse.

    Chosen over XGBoost/LightGBM only because those are not installed in the
    sandbox; on Colab they are drop-in substitutes (see the Colab notebook).
    """
    n_pos = int(y.sum())
    params = {**GBT_DEFAULTS, **overrides}
    return HistGradientBoostingClassifier(
        random_state=seed,
        class_weight="balanced" if n_pos else None,
        **params,
    ).fit(X, y)


def select_gbt(Xv: np.ndarray, y: np.ndarray, groups: np.ndarray,
               inner_splits: int = 3, seed: int = 0):
    """Pick hyperparameters using a ring-disjoint split of the *training rows only*.

    Returns (best_params, inner_score). Calling this on the rows you intend to
    score would be selection on the test set; the caller is responsible for
    keeping the outer test fold out of `Xv`.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    inner = list(GroupKFold(n_splits=inner_splits).split(Xv, y, groups=groups))
    best = (-1.0, GBT_GRID[0])
    for (d, l, i, r) in GBT_GRID:
        sc = np.zeros(len(y))
        for itr, ite in inner:
            s = StandardScaler().fit(Xv[itr])
            m = HistGradientBoostingClassifier(
                random_state=seed, class_weight="balanced", max_depth=d,
                min_samples_leaf=l, max_iter=i, learning_rate=r,
                l2_regularization=1.0)
            m.fit(s.transform(Xv[itr]), y[itr])
            sc[ite] = m.predict_proba(s.transform(Xv[ite]))[:, 1]
        if y.sum() and y.sum() < len(y):
            a = float(average_precision_score(y, sc))
            if a > best[0]:
                best = (a, (d, l, i, r))
    return best[1], best[0]


@dataclass
class GraphRegularisedLogistic:
    """Logistic regression penalised for disagreeing with graph neighbours.

    Objective:  min_w  CE(y, Xw)  +  lambda/2 * sum_{i~j} w_ij (x_i.w - x_j.w)^2

    The second term is w^T (X^T L X) w, so it is still a quadratic in w and can be
    folded into the ridge penalty -- which means the standard solver needs only a
    modified regulariser, not a custom optimiser. This is the classical
    graph-Laplacian regularised semi-supervised formulation (Zhou et al., NIPS
    2004; Belkin & Niyogi, ICML 2004) applied to a supervised objective.

    Intuition for a first-time reader: a plain classifier scores each application
    from its own columns. This one is additionally told "applications that share a
    device or a guarantor should receive similar scores". A ring -- which is
    structurally coherent but individually ordinary -- then gets pulled into a
    consistently high (or consistently low) band, while a genuinely isolated
    unusual application is left alone.
    """

    lam: float = 0.5
    C: float = 1.0
    max_iter: int = 4000

    def fit(self, X: np.ndarray, y: np.ndarray, A: sp.csr_matrix):
        from scipy.sparse.linalg import LinearOperator

        A = sp.csr_matrix(A, dtype=float)
        deg = np.asarray(A.sum(axis=1)).ravel()
        L = sp.diags(deg) - A                     # graph Laplacian
        # Feature-space Laplacian: M = X^T L X  (d x d)
        M = (X.T @ (L @ X)).toarray() if sp.issparse(X) else X.T @ (L @ X)
        M = 0.5 * (M + M.T)                        # enforce symmetry numerically
        Xa = np.asarray(X)
        n, d = Xa.shape
        base = Xa.T @ Xa
        H = base + (self.lam / max(self.C, 1e-12)) * M + np.eye(d) * 1e-6

        # Newton-IRLS on the penalised logistic objective.
        w = np.zeros(d)
        for _ in range(60):
            eta = Xa @ w
            p = 1.0 / (1.0 + np.exp(-eta))
            # class weights so the minority class is not swamped
            wpos = 0.5 * n / max(int(y.sum()), 1)
            wneg = 0.5 * n / max(int((1 - y).sum()), 1)
            r = np.where(y == 1, wpos, wneg)
            sw = r * p * (1 - p) + 1e-9
            grad = Xa.T @ (r * (p - y)) + (self.lam / max(self.C, 1e-12)) * (M @ w)
            A_ = (Xa * sw[:, None]).T @ Xa + (self.lam / max(self.C, 1e-12)) * M \
                + np.eye(d) * 1e-6
            try:
                step = np.linalg.solve(A_, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A_, grad, rcond=None)[0]
            w_new = w - step
            if not np.all(np.isfinite(w_new)):
                break
            if np.max(np.abs(w_new - w)) < 1e-7:
                w = w_new
                break
            w = w_new
        self.w_ = w
        self._scale = 1.0
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.w_


def personalised_pagerank_score(A: sp.csr_matrix, seed: np.ndarray,
                                alpha: float = 0.85, iters: int = 50) -> np.ndarray:
    """Diffuse a seed suspicion vector over the graph (PPR propagation).

    A pure propagation baseline: no features, no labels. It answers "if the
    currently confirmed bad applications are the seeds, who else does the graph
    implicate?" Useful as a reference point and as the L4 ring scorer's diffusion
    step.
    """
    A = sp.csr_matrix(A, dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    safe = np.where(deg > 0, deg, 1.0)
    s = seed / max(seed.sum(), 1e-12)
    r = s.copy()
    for _ in range(iters):
        r = alpha * (A.T @ (r / safe)) + (1 - alpha) * s
    return np.asarray(r).ravel()
