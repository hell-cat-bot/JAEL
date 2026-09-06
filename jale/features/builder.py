"""Feature construction for JALE V1.

Two feature families, kept strictly separate so that the ablation
"does the graph actually help?" is meaningful:

* ``build_node_features``  -- tabular attributes of the application and the
  applicant. Contains **no** relational information whatsoever.
* ``build_graph_features`` -- everything derived from shared entities.

Discipline enforced here
------------------------
1. Neither family reads a label. ``ring_id`` never enters this module.
2. ``ObservationTime.APPLICATION`` suppresses all EMI-derived columns, because at
   application time they do not exist. This is the faithful setting for problem
   (e) ("predict fraud ecosystems *before* fraud occurs") and it removes the
   shortcut whereby a node-only model wins on delinquency alone.
3. Every categorical encoding is *frequency* encoding computed inside
   ``fit_transform``-style helpers that the pipeline calls on TRAIN rows only.
   Target encoding is deliberately avoided: it is the single easiest way to leak
   a label into a feature and it buys almost nothing here.
4. Graph features are computed transductively over the whole observable graph.
   That is legitimate -- the graph is observable data, not an outcome -- and it
   matches how a production system would run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import ObservationTime
from ..graph.builder import RELATIONS, LenderGraph, cooccurrence, neighbour_counts

NUMERIC_NODE = ["loan_amount", "tenure_months", "down_payment", "monthly_income",
                "applicant_age_at_application", "n_guarantors"]
CATEG_NODE = ["product", "employment_type", "application_state"]


# --------------------------------------------------------------------------
# node (tabular) features
# --------------------------------------------------------------------------
def build_node_features(applications: pd.DataFrame,
                        emi: pd.DataFrame | None,
                        observation_time: str = ObservationTime.APPLICATION,
                        ) -> pd.DataFrame:
    """Tabular, relation-free features keyed on application_id."""
    df = applications.set_index("application_id")
    out = pd.DataFrame(index=df.index)

    for c in NUMERIC_NODE:
        if c in df.columns:
            out[c] = df[c].astype(float)

    # Derived ratios that a credit underwriter would compute by hand.
    eps = 1.0
    out["down_payment_ratio"] = out["down_payment"] / (out["loan_amount"] + eps)
    out["income_to_amount"] = out["monthly_income"] / (out["loan_amount"] + eps)
        # NOTE: these are the *scheduled* instalments, computable at application time
    # from the sanctioned amount and tenor. They are deliberately NOT named
    # "emi_*" so they cannot be confused with repayment-history columns, which
    # are gated on POST_DISBURSEMENT.
    out["scheduled_emi_amount"] = out["loan_amount"] / out["tenure_months"].clip(lower=1)
    out["scheduled_emi_to_income"] = out["scheduled_emi_amount"] / (out["monthly_income"] + eps)
    out["amount_per_tenure"] = out["loan_amount"] / out["tenure_months"].clip(lower=1)

    for c in CATEG_NODE:
        if c in df.columns:
            # Frequency encoding, not target encoding: the mapping is a property
            # of the feature distribution, not of the outcome.
            freq = df[c].astype(str).value_counts()
            out[f"{c}_freq"] = df[c].astype(str).map(freq).astype(float)
            out[f"{c}_code"] = df[c].astype("category").cat.codes.astype(float)

    if observation_time == ObservationTime.POST_DISBURSEMENT:
        if emi is None or len(emi) == 0:
            raise ValueError("POST_DISBURSEMENT requires an EMI schedule")
        e = emi.copy()
        agg = e.groupby("application_id").agg(
            emi_n_installments=("installment_no", "size"),
            emi_n_missed=("missed", "sum"),
            emi_max_dpd=("days_past_due", "max"),
            emi_mean_dpd=("days_past_due", "mean"),
            emi_total_dpd=("days_past_due", "sum"),
        )
        agg["emi_miss_rate"] = agg["emi_n_missed"] / agg["emi_n_installments"].clip(lower=1)
        agg["emi_ever_dpd30"] = (agg["emi_max_dpd"] >= 30).astype(float)
        first = e[e.installment_no == 1].set_index("application_id")
        agg["emi_first_missed"] = first["missed"].reindex(agg.index).fillna(0).astype(float)
        out = out.join(agg, how="left").fillna({"emi_n_installments": 0,
                                                "emi_n_missed": 0,
                                                "emi_max_dpd": 0,
                                                "emi_mean_dpd": 0,
                                                "emi_total_dpd": 0,
                                                "emi_miss_rate": 0,
                                                "emi_ever_dpd30": 0,
                                                "emi_first_missed": 0})
    elif observation_time != ObservationTime.APPLICATION:
        raise ValueError(f"unknown observation_time {observation_time!r}")

    out.index.name = "application_id"
    return out


# --------------------------------------------------------------------------
# graph features
# --------------------------------------------------------------------------
def _time_windowed_burst(applications: pd.DataFrame, M: sp.csr_matrix,
                         order_index: pd.Index,
                         windows: tuple[int, ...] = (7, 30)) -> pd.DataFrame:
    """Applications sharing a node within +/- w days of this application.

    Burst timing is the core of ring tradecraft: a device farm files many
    applications within a few days. Counting co-occurrence *without* the time
    dimension would treat a kiosk that has served a village for three years
    identically to a farm that filed twelve applications last week -- which is
    exactly the false positive this project has to avoid.

    For each node, its applications are sorted by day and a binary search gives,
    for each application, how many of its co-applicants fall inside the window on
    either side. O(n log n) per node.
    """
    day_map = pd.Series(applications["applied_day"].to_numpy(dtype=float),
                        index=pd.Index(applications["application_id"]))
    day_map = day_map.reindex(order_index).fillna(0.0)
    out = pd.DataFrame(index=order_index)
    Mc = M.tocsc()
    for w in windows:
        counts = np.zeros(len(order_index))
        for j in range(Mc.shape[1]):
            rows = Mc.indices[Mc.indptr[j]:Mc.indptr[j + 1]]
            if len(rows) < 2:
                continue
            d = day_map.to_numpy()[rows]
            order = np.argsort(d, kind="stable")
            ds = d[order]
            rs = rows[order]
            lo = np.searchsorted(ds, ds - w, side="left")
            hi = np.searchsorted(ds, ds + w, side="right")
            # (hi - lo - 1) = number of *other* applications within the window
            counts[rs] += (hi - lo - 1)
        out[f"burst_{w}d"] = counts
    return out


def build_graph_features(graph: LenderGraph,
                         applications: pd.DataFrame,
                         node_features: pd.DataFrame,
                         ) -> pd.DataFrame:
    """Relational features. Nothing here reads a label."""
    idx = graph.app_ids
    out = pd.DataFrame(index=idx)
    app_pos = pd.Series(np.arange(len(idx)), index=idx)

    # ---- degree / co-occurrence counts per relation ----------------------
    for rel, M in graph.incidence.items():
        if M.shape[1] == 0:
            out[f"deg_{rel}"] = 0.0
            out[f"node_deg_{rel}"] = 0.0
            continue
        out[f"deg_{rel}"] = np.asarray(M.sum(axis=1)).ravel()
        # how heavily used is *this* application's node of this relation?
        node_deg = np.asarray(M.sum(axis=0)).ravel()
        per_app_node_deg = np.asarray(M.multiply(node_deg).sum(axis=1)).ravel()
        deg = np.maximum(np.asarray(M.sum(axis=1)).ravel(), 1.0)
        out[f"node_deg_{rel}"] = per_app_node_deg / deg
        out[f"co_{rel}"] = neighbour_counts(M)

    # ---- burst timing ----------------------------------------------------
    for rel in ("device", "account", "guarantor", "person"):
        M = graph.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        b = _time_windowed_burst(applications, M, idx, windows=(7, 30))
        for c in b.columns:
            out[f"{rel}_{c}"] = b[c].to_numpy()

    # ---- entity diversity within a shared node --------------------------
    # A legitimate kiosk serves many *different* people over a long period; a
    # device farm serves many different PANs in a short period. Entropy of the
    # applicant population on a node separates these without any label.
    for rel in ("device", "dealer", "account"):
        M = graph.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        pid_code = applications.set_index("application_id")["person_id"] \
            .astype("category").cat.codes.reindex(idx).fillna(-1).to_numpy()
        Ml = M.tocsc()
        ent = np.zeros(len(idx))
        for j in range(Ml.shape[1]):
            rows = Ml.indices[Ml.indptr[j]:Ml.indptr[j + 1]]
            if len(rows) < 2:
                continue
            vals, counts = np.unique(pid_code[rows], return_counts=True)
            p = counts / counts.sum()
            h = float(-(p * np.log(p + 1e-12)).sum())
            ent[rows] = h
        out[f"entropy_{rel}"] = ent

    # ---- neighbourhood aggregation of node features ---------------------
    # Mean/max/std of *observable* applicant attributes among applications that
    # share a device or dealer. A ring is unusually homogeneous for its size.
    nf = node_features.reindex(idx)
    for rel in ("device", "dealer", "account", "guarantor"):
        M = graph.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        C = cooccurrence(M)
        deg = np.asarray(C.sum(axis=1)).ravel()
        safe = np.where(deg > 0, deg, 1.0)
        for col in ("loan_amount", "monthly_income", "applicant_age_at_application"):
            if col not in nf.columns:
                continue
            v = nf[col].fillna(nf[col].mean()).to_numpy(dtype=float)
            nbr_mean = np.asarray(C @ v).ravel() / safe
            nbr_sq = np.asarray(C @ (v ** 2)).ravel() / safe
            nbr_var = np.maximum(nbr_sq - nbr_mean ** 2, 0.0)
            # Applications with no neighbour on this relation must read as 0, not
            # as "the neighbour mean equals zero". Without this mask `nbrdiff`
            # degenerates into a copy of the raw value and silently becomes a
            # duplicate of a node feature inside the graph block, which would
            # inflate the graph block's apparent contribution in the ablation.
            has = (deg > 0).astype(float)
            out[f"nbrmean_{rel}_{col}"] = nbr_mean * has
            out[f"nbrstd_{rel}_{col}"] = np.sqrt(nbr_var) * has
            out[f"nbrdiff_{rel}_{col}"] = (v - nbr_mean) * has
            out[f"hasnbr_{rel}_{col}"] = has

    # ---- structural ------------------------------------------------------
    out["component_size"] = _component_sizes(graph)

    # Personalised PageRank over the union graph: a global centrality measure
    # that local degree statistics cannot capture.
    out["ppr"] = _personalised_pagerank(graph, idx)

    out.index.name = "application_id"
    return out.fillna(0.0)


def _union_adjacency(graph: LenderGraph) -> sp.csr_matrix:
    """Application-application adjacency summed across strong relations."""
    n = graph.n_apps()
    A = sp.csr_matrix((n, n), dtype=np.float32)
    for rel in ("device", "account", "guarantor", "person"):
        M = graph.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        C = (M @ M.T).tocsr()
        C.setdiag(0.0)
        A = A + C
    A.data[:] = np.minimum(A.data, 1.0)
    return A.tocsr()


def _component_sizes(graph: LenderGraph) -> np.ndarray:
    A = _union_adjacency(graph)
    n_comp, labels = sp.csgraph.connected_components(A, directed=False)
    sizes = np.bincount(labels)
    return sizes[labels].astype(float)


def _personalised_pagerank(graph: LenderGraph, idx: pd.Index,
                           alpha: float = 0.85, iters: int = 30) -> np.ndarray:
    """Uniform-teleport PageRank centrality on the union graph.

    Implemented as plain power iteration on the sparse adjacency: no extra
    dependency, and it converges in a few dozen iterations at this scale.
    """
    A = _union_adjacency(graph)
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = deg == 0
    P = A.T.tocsr()
    norm = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        leak = alpha * r[dangling].sum()
        r_new = alpha * P.dot(r * norm) + (1 - alpha) / n + leak / n
        s = r_new.sum()
        if s > 0:
            r_new /= s
        if np.abs(r_new - r).sum() < 1e-10:
            r = r_new
            break
        r = r_new
    # Report rank-normalised centrality so the scale is comparable across graphs.
    order = np.argsort(-r)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return 1.0 - ranks / max(n - 1, 1)
