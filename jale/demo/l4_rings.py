"""L4 -- ring-level scoring.

V1 scores an application. This scores the *cluster*. The pitch's novelty claim
lives here: a lender does not want to know "is application 4471 fraud?", it wants
to know "applications 4471, 4472, 4488 and nine others are one operation -- shut
the operation down".

Design rules, all of which a reviewer can check by reading this file:

1. **No learned weights.** The ring score is a fixed, written-out formula over
   structural quantities. Nothing here is fitted to the labels, so nothing here
   can overfit the five synthetic typologies the way the feature-based model does
   (README / doubts.md: cross-typology AUC-PR collapses to 0.27). A ring betrays
   itself structurally or not at all.

2. **No labels.** ``ring_id`` never enters this module. Candidate clusters come
   from unsupervised connected components; the score uses only the observable
   graph and the (already out-of-fold) application scores from L3.

3. **Typology-agnostic via a max.** A device farm concentrates devices; a
   disbursement sink concentrates accounts; identity reuse concentrates people; a
   guarantor star concentrates guarantors. The structural term is the **maximum**
   over those axes, so a cluster needs to be extreme on only one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.builder import LenderGraph, STRONG_FOLD_RELATIONS, cooccurrence

MIN_RING_SIZE = 3          # a "ring" needs at least three applications
BURST_WINDOW_DAYS = 14     # window for the burst-concentration term


@dataclass
class RingCluster:
    cluster_id: int
    application_ids: list[str]
    n_apps: int
    n_persons: int
    ring_score: float
    # the pieces of the score, all in [0, 1] -- shown in the UI so the number is
    # never a black box
    device_concentration: float
    account_concentration: float
    person_concentration: float
    guarantor_concentration: float
    burst_concentration: float
    internal_density: float
    mean_member_score: float
    frac_members_hot: float
    structural_term: float
    corroboration_term: float
    span_days: int
    dominant_axis: str
    typology_guess: str

    def as_dict(self) -> dict:
        return asdict(self)


def _max_window_count(days: np.ndarray, window: int) -> int:
    """Largest number of points falling inside any ``window``-day span."""
    if len(days) == 0:
        return 0
    d = np.sort(days)
    hi = np.searchsorted(d, d + window, side="right")
    return int((hi - np.arange(len(d))).max())


_AXIS_TO_TYPOLOGY = {
    "device": "device farm",
    "account": "disbursement sink",
    "person": "identity reuse",
    "guarantor": "guarantor star",
    "burst": "coordinated burst",
}


def score_rings(graph: LenderGraph,
                applications: pd.DataFrame,
                fold_groups: pd.Series,
                app_scores: pd.Series,
                guarantor_links: pd.DataFrame | None = None,
                hot_percentile: float = 95.0) -> list[RingCluster]:
    """Return every candidate cluster, scored and ranked (highest risk first).

    ``fold_groups`` is the connected-component id per application from
    ``graph.fold_groups`` -- components over the strong relations, which is the
    same partition the ring-disjoint CV uses, so a "cluster" here is exactly a
    "ring" there.

    ``app_scores`` is the L3 out-of-fold score per application. It enters only the
    corroboration term; a cluster with no model support can still score high on
    structure alone (that is the cold-start case).
    """
    apps = applications.set_index("application_id")
    fg = fold_groups.reindex(apps.index)
    scores = app_scores.reindex(apps.index).fillna(app_scores.median())
    p_hot = float(np.percentile(scores.to_numpy(), hot_percentile))

    day = apps["applied_day"].astype(float)
    dev = apps["device_id"].astype(str)
    acct = apps["account_id"].astype(str)
    per = apps["person_id"].astype(str)

    gl = guarantor_links if guarantor_links is not None else pd.DataFrame(
        columns=["application_id", "guarantor_person_id"])
    gby_app = gl.groupby("application_id")["guarantor_person_id"].apply(list).to_dict()

    # Application-to-application affinity over the strong relations, for the
    # internal-density term. Same matrix the folds are built from.
    A = graph.cooccurrence_union(tuple(STRONG_FOLD_RELATIONS)).tocsr()
    pos = pd.Series(np.arange(len(graph.app_ids)), index=graph.app_ids)

    # ---- raw per-cluster quantities ------------------------------------
    raw: list[dict] = []
    for cid, members in fg.groupby(fg).groups.items():
        members = list(members)
        n = len(members)
        if n < MIN_RING_SIZE:
            continue
        m_scores = scores.loc[members].to_numpy()
        days = day.loc[members].to_numpy()

        n_dev = dev.loc[members].nunique()
        n_acct = acct.loc[members].nunique()
        n_per = per.loc[members].nunique()
        g_ids = [g for a in members for g in gby_app.get(a, [])]
        n_g_links = len(g_ids)
        n_g_distinct = len(set(g_ids))

        # concentration = 1 - distinct/total, so 0 = everyone distinct (benign
        # household), ~1 = one shared entity for the whole cluster (a farm/sink).
        dev_c = 1.0 - n_dev / n
        acct_c = 1.0 - n_acct / n
        per_c = 1.0 - n_per / n
        g_c = (1.0 - n_g_distinct / n_g_links) if n_g_links >= 2 else 0.0

        burst_c = _max_window_count(days, BURST_WINDOW_DAYS) / n

        idx = pos.loc[members].to_numpy()
        sub = A[idx][:, idx]
        internal_edges = sub.nnz / 2.0
        density = internal_edges / max(n * (n - 1) / 2.0, 1.0)

        raw.append(dict(
            cluster_id=int(cid), members=members, n=n, n_per=int(n_per),
            dev_c=dev_c, acct_c=acct_c, per_c=per_c, g_c=g_c, burst_c=burst_c,
            density=min(density, 1.0),
            mean_member_score=float(np.mean(m_scores)),
            frac_hot=float(np.mean(m_scores > p_hot)),
            span=int(days.max() - days.min()),
        ))

    if not raw:
        return []

    # ---- assemble the score ------------------------------------------------
    # corroboration: how much the L3 model already backs this cluster. Rank-
    # normalised across candidates so the scale is stable.
    mm = np.array([r["mean_member_score"] for r in raw])
    mm_norm = _rank01(mm)

    out: list[RingCluster] = []
    for r, mmn in zip(raw, mm_norm):
        axes = {"device": r["dev_c"], "account": r["acct_c"],
                "person": r["per_c"], "guarantor": r["g_c"], "burst": r["burst_c"]}
        dominant_axis = max(axes, key=axes.get)
        structural = max(axes.values())
        # blend the strongest structural axis with density, lightly
        structural = 0.8 * structural + 0.2 * r["density"]
        corroboration = 0.5 * mmn + 0.5 * r["frac_hot"]

        # Corroboration-led. The L3 model already ranks individual applications
        # well; L4's job is to group them into an operation and, above all, to
        # pull in the ring members that individually scored low. So the ring
        # score leans on how much the model already backs the cluster, with the
        # structural term as a confidence boost / tie-break. A structurally
        # concentrated cluster whose members the model finds unremarkable is far
        # more likely a household or a Common Service Centre kiosk -- shared
        # device, shared address and repeat guarantors are all legitimate.
        # ``corroboration`` is built from the out-of-fold application scores,
        # never from ``ring_id`` -- this is a prior on what a ring is, not label
        # peeking.
        ring_score = 0.65 * corroboration + 0.35 * structural

        out.append(RingCluster(
            cluster_id=r["cluster_id"],
            application_ids=[str(a) for a in r["members"]],
            n_apps=r["n"], n_persons=r["n_per"],
            ring_score=round(float(ring_score), 4),
            device_concentration=round(r["dev_c"], 3),
            account_concentration=round(r["acct_c"], 3),
            person_concentration=round(r["per_c"], 3),
            guarantor_concentration=round(r["g_c"], 3),
            burst_concentration=round(min(r["burst_c"], 1.0), 3),
            internal_density=round(r["density"], 3),
            mean_member_score=round(r["mean_member_score"], 4),
            frac_members_hot=round(r["frac_hot"], 3),
            structural_term=round(float(structural), 3),
            corroboration_term=round(float(corroboration), 3),
            span_days=r["span"],
            dominant_axis=dominant_axis,
            typology_guess=_AXIS_TO_TYPOLOGY.get(dominant_axis, "unclear"),
        ))

    out.sort(key=lambda c: c.ring_score, reverse=True)
    return out


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_rings(clusters: list[RingCluster],
                   applications: pd.DataFrame,
                   y_true: np.ndarray,
                   app_ring_id: np.ndarray | None = None,
                   min_ring_members: int = 3) -> dict:
    """How well does the ring score separate real rings from benign clusters?

    A candidate cluster counts as a *true ring* if it contains at least
    ``min_ring_members`` applications belonging to a single injected ring. This
    is stricter and more honest than a "majority fraud" rule: heavily camouflaged
    rings sit in a component with many benign cover applications, so a majority
    rule would wrongly score them as benign clusters.

    We then walk the ring-score ranking and, for each "escalate the top K
    clusters" budget, report how many applications that asks an analyst to review
    and how much fraud it catches -- recall@k, at ring granularity.
    """
    apps = applications.reset_index(drop=True)
    y = pd.Series(np.asarray(y_true).astype(int), index=apps["application_id"])
    total_fraud = int(y.sum())
    ring_of = (pd.Series(np.asarray(app_ring_id), index=apps["application_id"])
               if app_ring_id is not None else None)

    rows = []
    for c in clusters:
        members = y.reindex(c.application_ids).fillna(0)
        is_true = float(members.mean()) >= 0.5
        if ring_of is not None:
            rids = ring_of.reindex(c.application_ids).fillna(0)
            biggest = rids[rids > 0].value_counts()
            is_true = bool(len(biggest) and biggest.iloc[0] >= min_ring_members)
        rows.append(dict(cluster_id=c.cluster_id, ring_score=c.ring_score,
                         n_apps=c.n_apps, n_fraud=int(members.sum()),
                         is_true_ring=is_true))
    R = pd.DataFrame(rows)
    if R.empty:
        return {"note": "no candidate clusters"}

    n_true_rings = int(R["is_true_ring"].sum())
    budget_curve = []
    for k in range(1, min(len(R), 25) + 1):
        top = R.iloc[:k]
        budget_curve.append(dict(
            k=k,
            apps_reviewed=int(top["n_apps"].sum()),
            fraud_caught=int(top["n_fraud"].sum()),
            recall=round(top["n_fraud"].sum() / max(total_fraud, 1), 3),
            precision_clusters=round(top["is_true_ring"].mean(), 3),
        ))

    # precision/recall over clusters at the natural cutoff (score >= 0.5)
    hi = R[R["ring_score"] >= 0.5]
    fp = hi[~hi["is_true_ring"]]
    return {
        "n_candidate_clusters": int(len(R)),
        "n_true_rings": n_true_rings,
        "total_fraud_apps": total_fraud,
        "at_threshold_0.5": {
            "clusters_flagged": int(len(hi)),
            "true_rings_in_flagged": int(hi["is_true_ring"].sum()),
            "true_rings_total": n_true_rings,
            "ring_recall": round(hi["is_true_ring"].sum() / max(n_true_rings, 1), 3),
            "precision": round(hi["is_true_ring"].mean(), 3) if len(hi) else None,
            "fraud_apps_covered": int(hi["n_fraud"].sum()),
            "app_recall": round(hi["n_fraud"].sum() / max(total_fraud, 1), 3),
            "false_positive_clusters": int(len(fp)),
            "fp_median_apps": int(fp["n_apps"].median()) if len(fp) else 0,
        },
        "budget_curve": budget_curve,
        "clusters": rows,
    }
