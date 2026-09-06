"""L5 -- explanation.

A score column is not a fraud system. An investigator needs to know *why* an
application was flagged, in terms they can verify against the file: which device,
which account, which guarantor, who else is on it, how fast it moved.

Everything here is read off the observable graph. No labels, no model internals
beyond the application's own feature vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.builder import LenderGraph, RELATIONS

# feature name -> (template, higher_is_worse). {n} is filled with the rounded
# value. Only graph features an investigator can act on are given phrasings.
_PHRASINGS: dict[str, tuple[str, bool]] = {
    "co_device":      ("shares its device fingerprint with {n} other application(s)", True),
    "co_account":     ("disburses to an account used by {n} other application(s)", True),
    "co_person":      ("same applicant identity as {n} other application(s)", True),
    "co_guarantor":   ("shares a guarantor with {n} other application(s)", True),
    "co_dealer":      ("sourced by a dealer also on {n} other application(s) in this cluster", True),
    "node_deg_device":    ("its device fingerprint appears on {n} applications in total", True),
    "node_deg_account":   ("its disbursement account appears on {n} applications in total", True),
    "node_deg_guarantor": ("its guarantor backs {n} applications in total", True),
    "device_burst_7d":    ("{n} applications filed on this device within 7 days", True),
    "device_burst_30d":   ("{n} applications filed on this device within 30 days", True),
    "account_burst_7d":   ("{n} applications to this account within 7 days", True),
    "guarantor_burst_30d": ("{n} applications backed by this guarantor within 30 days", True),
    "component_size":     ("sits inside a connected cluster of {n} applications", True),
    "entropy_device":     ("low identity diversity on the shared device (entropy {n})", False),
    "entropy_account":    ("low identity diversity on the shared account (entropy {n})", False),
}


def _neighbours_on_relation(graph: LenderGraph, rel: str, app_row: int,
                            limit: int = 12) -> list[dict]:
    """Other applications that touch the same node(s) of ``rel`` as ``app_row``."""
    M = graph.incidence.get(rel)
    if M is None or M.shape[1] == 0:
        return []
    cols = M[app_row].indices
    if len(cols) == 0:
        return []
    # rows sharing any of those columns
    sub = M[:, cols]
    shared = np.asarray(sub.sum(axis=1)).ravel()
    others = np.flatnonzero(shared > 0)
    others = others[others != app_row]
    # node id(s) shared -- invert node_index
    inv = {v: k for k, v in graph.node_index.get(rel, {}).items()}
    node_ids = [inv.get(int(c)) for c in cols]
    return [{
        "shared_node": [str(x) for x in node_ids],
        "n_other_apps": int(len(others)),
        "sample_app_rows": [int(x) for x in others[:limit]],
    }]


def explain_application(app_id: str,
                        graph: LenderGraph,
                        applications: pd.DataFrame,
                        graph_features: pd.DataFrame,
                        app_score: float,
                        score_percentile: float,
                        ring_lookup: dict | None = None,
                        top_k_signals: int = 5) -> dict:
    """Case note for one application."""
    app_ids = list(graph.app_ids)
    row = app_ids.index(app_id)
    apps = applications.set_index("application_id")
    rec = apps.loc[app_id]

    # ---- shared entities per relation --------------------------------
    shared: dict[str, list[dict]] = {}
    for rel in ("device", "account", "person", "guarantor", "dealer"):
        info = _neighbours_on_relation(graph, rel, row)
        info = [d for d in info if d["n_other_apps"] > 0]
        if info:
            # map sample rows -> application ids
            for d in info:
                d["sample_app_ids"] = [app_ids[r] for r in d.pop("sample_app_rows")]
            shared[rel] = info

    # ---- most extreme signals for this application -------------------
    gf = graph_features.reindex([app_id])
    med = graph_features.median(numeric_only=True)
    iqr = (graph_features.quantile(0.75) - graph_features.quantile(0.25)).replace(0, np.nan)
    signals = []
    for col in graph_features.columns:
        if col not in _PHRASINGS:
            continue
        val = float(gf[col].iloc[0]) if col in gf else 0.0
        template, higher_worse = _PHRASINGS[col]
        z = (val - float(med.get(col, 0.0))) / float(iqr.get(col, np.nan) or np.nan) \
            if not np.isnan(iqr.get(col, np.nan)) else 0.0
        extreme = z if higher_worse else -z
        if extreme <= 0.5:
            continue
        pct = float((graph_features[col] <= val).mean() * 100)
        signals.append({
            "feature": col,
            "value": round(val, 3),
            "portfolio_percentile": round(pct, 1),
            "text": template.format(n=int(val) if val == int(val) else round(val, 2)),
            "_rank": extreme,
        })
    signals.sort(key=lambda s: s["_rank"], reverse=True)
    for s in signals:
        s.pop("_rank")
    signals = signals[:top_k_signals]

    # ---- ring membership -------------------------------------------
    ring = (ring_lookup or {}).get(app_id)

    return {
        "application_id": app_id,
        "score": round(float(app_score), 4),
        "score_percentile": round(float(score_percentile), 1),
        "applicant": {
            "person_id": str(rec["person_id"]),
            "product": str(rec.get("product", "")),
            "loan_amount": float(rec.get("loan_amount", 0.0)),
            "district": str(rec.get("application_district", "")),
            "state": str(rec.get("application_state", "")),
            "applied_day": int(rec.get("applied_day", 0)),
            "n_guarantors": int(rec.get("n_guarantors", 0)),
        },
        "shared_entities": shared,
        "top_signals": signals,
        "ring": ring,
    }
