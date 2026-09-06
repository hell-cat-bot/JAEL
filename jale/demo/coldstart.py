"""Day-one workflow: no trained model, a handful of confirmed cases.

TVS does not store a ``ring_id`` column and on day one has almost no confirmed
ring labels, so a supervised L4 cannot train. This is the honest answer to that:
seed personalised PageRank with the few applications an investigator has already
confirmed and let the graph implicate the rest. No training, no features -- only
the confirmed seeds and the observable graph.

Measured on SMOKE (see ``experiments/improve_propagation.py``): 3 / 5 / 10 seeds
give AUC-PR 0.29 / 0.39 / 0.42 against a 0.038 base rate. That is a usable
triage queue from nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..eval.metrics import full_metrics
from ..models.models import personalised_pagerank_score


def propagate_from_seeds(A: sp.csr_matrix, app_ids: pd.Index,
                         seed_app_ids: list[str], alpha: float = 0.85,
                         iters: int = 50) -> pd.Series:
    """Return a suspicion score per application, diffused from the seeds."""
    pos = pd.Series(np.arange(len(app_ids)), index=app_ids)
    seed = np.zeros(len(app_ids))
    seed[pos.reindex(seed_app_ids).dropna().astype(int).to_numpy()] = 1.0
    r = personalised_pagerank_score(A, seed, alpha=alpha, iters=iters)
    return pd.Series(r, index=app_ids, name="coldstart_score")


def evaluate_coldstart(A: sp.csr_matrix, app_ids: pd.Index, y_true: np.ndarray,
                       seed_counts=(3, 5, 10), n_repeats: int = 20,
                       seed: int = 0) -> dict:
    """Average AUC-PR over ``n_repeats`` random draws of confirmed seeds.

    Seeds are drawn from the true positives (an investigator confirms real
    cases); the seed applications are then excluded from scoring so the metric is
    not inflated by grading the seeds themselves.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    pos = np.flatnonzero(y == 1)
    ids = np.asarray(app_ids)
    rows = []
    for k in seed_counts:
        aucs, recalls = [], []
        for _ in range(n_repeats):
            s_idx = rng.choice(pos, min(k, len(pos)), replace=False)
            score = propagate_from_seeds(A, app_ids, [ids[i] for i in s_idx]).to_numpy()
            keep = np.ones(len(y), bool); keep[s_idx] = False
            m = full_metrics(y[keep], score[keep])
            aucs.append(m.get("auc_pr", np.nan))
            recalls.append(m.get("recall_at_5pct", np.nan))
        rows.append({
            "n_seeds": k,
            "auc_pr_mean": round(float(np.nanmean(aucs)), 4),
            "auc_pr_sd": round(float(np.nanstd(aucs)), 4),
            "recall_at_5pct_mean": round(float(np.nanmean(recalls)), 3),
        })
    return {"base_rate": round(float(y.mean()), 4), "by_seed_count": rows}
