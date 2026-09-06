from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def full_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    out = {"n": len(y_true), "n_pos": int(y_true.sum()),
           "base_rate": float(y_true.mean())}
    if out["n_pos"] == 0 or out["n_pos"] == out["n"]:
        return out
    out["auc_roc"] = float(roc_auc_score(y_true, score))
    out["auc_pr"] = float(average_precision_score(y_true, score))
    out["lift_pr"] = out["auc_pr"] / out["base_rate"]
    order = np.argsort(-score)
    for frac in (0.01, 0.05, 0.10):
        k = max(int(round(frac * len(y_true))), 1)
        out[f"recall_at_{int(frac*100)}pct"] = float(y_true[order[:k]].sum()) / out["n_pos"]
    return out


def best_single_feature(y_true: np.ndarray, X) -> tuple[str, float]:
    """AUC-PR of the single most predictive column, sign-corrected.

    This is the honest floor for any "our model beats the simple thing" claim.
    A tuned ensemble that cannot beat one column used as a raw score is not
    beating anything, and this project was reporting exactly that: the node-only
    GBT scored at chance (0.038) while `n_guarantors` alone scored 0.2325. The
    baseline was misconfigured, not the data clean.
    """
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    best_name, best_auc = None, -1.0
    cols = getattr(X, "columns", range(np.asarray(X).shape[1]))
    arr = np.asarray(X, dtype=float)
    for j, name in enumerate(cols):
        v = arr[:, j]
        if not np.isfinite(v).all():
            v = np.nan_to_num(v)
        if np.all(v == v[0]):
            continue
        a = float(average_precision_score(y_true, v))
        a = max(a, float(average_precision_score(y_true, -v)))
        if a > best_auc:
            best_auc, best_name = a, str(name)
    return best_name, best_auc
