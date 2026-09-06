#!/usr/bin/env python3
"""Run the whole pipeline once and emit a single JSON the demo UI reads.

    python -m jale.demo.build_demo_data --profile SMOKE

No model runs in the browser. Everything the demo shows -- the scored queue, the
ring clusters, the case notes, the cold-start example, the evaluation tab -- is
computed here, out of fold, and written to ``demo/demo_data.json``.

Honesty carried over from V1:
* application scores are out-of-fold under ring-disjoint CV (a ring is never
  split across train and test);
* features are built from ``raw/`` only, with the label-segregation assertion;
* ``ObservationTime.APPLICATION`` -- no repayment history reaches a feature;
* the evaluation block reports the ring-disjoint number as the headline and the
  random-split number only to quantify the leak.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..config import SMOKE, FULL, ObservationTime
from ..data.generator import build as build_dataset
from ..eval.metrics import best_single_feature, full_metrics
from ..eval.splits import shuffle_label_control
from ..features.builder import build_graph_features, build_node_features
from ..graph.builder import build_graph, fold_groups
from ..models.models import fit_gbt
from ..resolution.apply import (apply_resolution, merge_diagnostics,
                                resolved_person_map)
from .coldstart import evaluate_coldstart, propagate_from_seeds
from .explain import explain_application
from .l4_rings import evaluate_rings, score_rings

N_SPLITS = 5
SEED = 0
OUT = Path("demo/demo_data.json")


def _oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold GBT scores under ring-disjoint CV (scaler fit on train only)."""
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(Xv, y, groups=groups):
        sc = StandardScaler().fit(Xv[tr])
        m = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=SEED)
        oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return oof


def _typology_generalisation(apps, g, nf, gf, y, groups, lab, rings,
                             node_cols, g_cols) -> list[dict]:
    """Train on four typologies, test on the fifth. Reproduces
    experiments/typology_generalisation.py, compactly."""
    typ = (lab.merge(rings[["ring_id", "typology"]], on="ring_id", how="left")
              .set_index("application_id")["typology"].reindex(g.app_ids))
    X_all = nf.reindex(g.app_ids)[node_cols].join(
        gf.reindex(g.app_ids)[g_cols], how="left")
    Xv = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    clean = np.array([gg for gg in np.unique(groups) if y[groups == gg].sum() == 0])
    rng = np.random.default_rng(0)
    rows = []
    for held in sorted(typ.dropna().unique()):
        held_apps = (typ == held).to_numpy()
        te_g = np.isin(groups, np.unique(groups[held_apps]))
        pool = np.flatnonzero(np.isin(groups, clean) & (y == 0) & ~te_g)
        extra = np.zeros(len(y), bool)
        if len(pool):
            extra[rng.choice(pool, min(max(int(held_apps.sum() * 26), 200),
                                       len(pool)), replace=False)] = True
        te = te_g | extra
        tr = ~te
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(Xv[tr])
        s = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0).predict_proba(
            sc.transform(Xv[te]))[:, 1]
        rows.append({"held_out": held, "n_test": int(te.sum()),
                     "positives": int(y[te].sum()),
                     "auc_pr": round(full_metrics(y[te], s).get("auc_pr", float("nan")), 4)})
    return rows


def _entity_resolution(persons, truth, mapping, linker, auc_raw, auc_resolved,
                       r5_raw, r5_resolved) -> dict:
    """Fellegi-Sunter linkage quality plus its cost when wired into the graph."""
    diag = merge_diagnostics(mapping, truth)
    return {
        "threshold_derived": round(float(getattr(linker, "_last_threshold", 2.04)), 3),
        "pi_estimated": round(float(getattr(linker, "pi_", float("nan"))), 4),
        "clusters_merged": diag["clusters_merged"],
        "correct_merges": diag["correct_merges"],
        "false_merges": diag["false_merges"],
        "records_welded_wrongly": diag["records_welded_wrongly"],
        "auc_pr_perfect_id": round(float(auc_raw), 4),
        "auc_pr_resolved_id": round(float(auc_resolved), 4),
        "delta_auc_pr": round(float(auc_resolved - auc_raw), 4),
        "recall5_perfect_id": round(float(r5_raw), 3),
        "recall5_resolved_id": round(float(r5_resolved), 3),
        "note": ("Every other number on this page uses the generator's perfect "
                 "person_id. This row replaces it with ids from our own "
                 "unsupervised linker. The gap is the cost of not having a "
                 "clean identity column; a PAN/Aadhaar-verified NBFC identity "
                 "spine closes most of it."),
    }


def build(profile_name: str = "SMOKE") -> dict:
    cfg = SMOKE if profile_name.upper() == "SMOKE" else FULL
    t0 = time.time()

    root = build_dataset(cfg, f"data/jale_{profile_name.lower()}")
    tabs = {f.stem: pd.read_parquet(f)
            for f in sorted((root / "raw").glob("*.parquet"))}
    banned = ("ring_id", "human_id", "is_kiosk")
    assert not any(c.startswith("label") or c in banned
                   for df in tabs.values() for c in df.columns), \
        "a ground-truth column leaked into raw/"

    apps = tabs["applications"]
    lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
    rings = pd.read_parquet(root / "labels" / "rings.parquet")
    id_truth = pd.read_parquet(root / "labels" / "person_identity_truth.parquet")
    y = (lab.set_index("application_id")["ring_id"]
            .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
    app_ring_id = (lab.set_index("application_id")["ring_id"]
                   .reindex(apps["application_id"]).fillna(0).astype(int).to_numpy())

    g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
    A_union = g.cooccurrence_union()
    groups = fold_groups(g)
    groups_arr = groups.reindex(g.app_ids).to_numpy()

    nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
    gf = build_graph_features(g, apps, nf)
    node_cols = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
    g_cols = [c for c in gf.columns if c != "ppr"]

    X_node = nf.reindex(g.app_ids)[node_cols]
    X_graph = gf.reindex(g.app_ids)[g_cols]
    X_all = X_node.join(X_graph, how="left")

    print("  scoring (ring-disjoint OOF) ...")
    s_ng = _oof(X_all, y, groups_arr)
    s_n = _oof(X_node, y, groups_arr)
    s_g = _oof(X_graph, y, groups_arr)

    app_score = pd.Series(s_ng, index=g.app_ids)
    pct = pd.Series(app_score.rank(pct=True) * 100, index=g.app_ids)

    # ---- L4 ------------------------------------------------------------
    print("  L4 ring scoring ...")
    clusters = score_rings(g, apps, groups, app_score,
                           guarantor_links=tabs["guarantor_links"])
    ring_eval = evaluate_rings(clusters, apps, y, app_ring_id=app_ring_id)
    true_ring = {r["cluster_id"]: r["is_true_ring"] for r in ring_eval["clusters"]}
    # attach fraud counts for display, and a lookup app_id -> ring summary
    yb = pd.Series(y, index=g.app_ids)
    ring_lookup = {}
    clusters_out = []
    for c in clusters:
        d = c.as_dict()
        d["n_fraud"] = int(yb.reindex(c.application_ids).fillna(0).sum())
        d["is_true_ring"] = bool(true_ring.get(c.cluster_id, False))
        clusters_out.append(d)
        for aid in c.application_ids:
            ring_lookup[aid] = {
                "cluster_id": c.cluster_id, "ring_score": c.ring_score,
                "n_members": c.n_apps, "typology_guess": c.typology_guess,
                "dominant_axis": c.dominant_axis,
            }

    # ---- L5 explanations for a curated subset ------------------------
    print("  L5 explanations ...")
    order = np.argsort(-s_ng)
    curated = set(np.array(g.app_ids)[order[:150]])          # top of the queue
    curated |= set(np.array(g.app_ids)[y == 1])              # every true fraud
    curated |= set(ring_lookup.keys())                       # every ring member
    # a few benign hard negatives: large clusters with no fraud
    for c in clusters:
        if yb.reindex(c.application_ids).fillna(0).sum() == 0 and c.n_apps >= 10:
            curated.update(c.application_ids[:6])
    explanations = {}
    for aid in curated:
        explanations[aid] = explain_application(
            aid, g, apps, X_graph, float(app_score[aid]), float(pct[aid]),
            ring_lookup=ring_lookup)

    # ---- graph edges for clusters of a drawable size ----------------
    print("  graph edges ...")
    pos = pd.Series(np.arange(len(g.app_ids)), index=g.app_ids)
    drawable = {c.cluster_id for c in clusters if 3 <= c.n_apps <= 90}
    member_rows = pos.reindex(
        [a for c in clusters if c.cluster_id in drawable
         for a in c.application_ids]).astype(int).to_numpy()
    row_set = set(member_rows.tolist())
    edges = []
    for rel in ("device", "account", "person", "guarantor"):
        M = g.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        C = sp.triu(M @ M.T, k=1).tocoo()
        for i, j, w in zip(C.row, C.col, C.data):
            if i in row_set and j in row_set:
                edges.append([g.app_ids[i], g.app_ids[j], rel, int(w)])

    # ---- cold-start -------------------------------------------------
    print("  cold-start ...")
    cs_eval = evaluate_coldstart(A_union, g.app_ids, y)
    rng = np.random.default_rng(SEED)
    seed_ids = [g.app_ids[i] for i in rng.choice(np.flatnonzero(y == 1), 3, replace=False)]
    cs_scores = propagate_from_seeds(A_union, g.app_ids, seed_ids)
    cs_rank = cs_scores.drop(index=seed_ids).sort_values(ascending=False)
    cs_example = {
        "seed_application_ids": list(seed_ids),
        "seed_rings": [int(app_ring_id[list(g.app_ids).index(s)]) for s in seed_ids],
        "top_implicated": [
            {"application_id": aid, "coldstart_score": round(float(v), 5),
             "is_fraud": bool(yb[aid]),
             "ring_id": int(app_ring_id[list(g.app_ids).index(aid)])}
            for aid, v in cs_rank.head(25).items()],
    }

    # ---- evaluation block -----------------------------------------
    print("  audits ...")
    fold_id = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(GroupKFold(N_SPLITS).split(X_all, y, groups=groups_arr)):
        fold_id[te] = i
    yl = shuffle_label_control(y, fold_id, seed=1)
    s_ctrl = _oof(X_all, yl, groups_arr)
    ctrl = full_metrics(yl, s_ctrl)

    oof_rand = np.zeros(len(y))
    Xv_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xv_all, y):
        sc = StandardScaler().fit(Xv_all[tr])
        oof_rand[te] = fit_gbt(sc.transform(Xv_all[tr]), y[tr], seed=SEED
                               ).predict_proba(sc.transform(Xv_all[te]))[:, 1]
    rand = full_metrics(y, oof_rand)

    bn, ba = best_single_feature(y, X_node)
    bgn, bga = best_single_feature(y, X_graph)

    typ_gen = _typology_generalisation(apps, g, nf, gf, y, groups_arr, lab, rings,
                                       node_cols, g_cols)

    # entity resolution IN the path: rebuild the graph on linker-resolved ids and
    # rescore, so the cost of not having a perfect identity column is measured.
    print("  entity resolution in the path ...")
    mapping, linker = resolved_person_map(tabs["persons"])
    a2, gl2, p2 = apply_resolution(apps, tabs["guarantor_links"],
                                   tabs["persons"], mapping)
    g2 = build_graph(a2, gl2, p2)
    grp2 = fold_groups(g2).reindex(g2.app_ids).to_numpy()
    nf2 = build_node_features(a2, None, ObservationTime.APPLICATION)
    gf2 = build_graph_features(g2, a2, nf2)
    n2 = [c for c in nf2.columns if not c.endswith(("_freq", "_code"))]
    g2c = [c for c in gf2.columns if c != "ppr"]
    X2 = nf2.reindex(g2.app_ids)[n2].join(gf2.reindex(g2.app_ids)[g2c], how="left")
    y2 = (lab.set_index("application_id")["ring_id"]
             .reindex(g2.app_ids).fillna(0).to_numpy() > 0).astype(int)
    s2 = _oof(X2, y2, grp2)
    m_res = full_metrics(y2, s2)
    er = _entity_resolution(tabs["persons"], id_truth, mapping, linker,
                            full_metrics(y, s_ng)["auc_pr"], m_res["auc_pr"],
                            full_metrics(y, s_ng)["recall_at_5pct"],
                            m_res["recall_at_5pct"])

    m_ng, m_n, m_g = full_metrics(y, s_ng), full_metrics(y, s_n), full_metrics(y, s_g)

    # nested-CV figures are slow (~10 min); if scripts/run_v1.py --nested has been
    # run, fold its numbers in as the "selection-bias-free" headline.
    nested = {}
    rep = Path(f"reports/v1_{profile_name.upper()}.json")
    if rep.exists():
        try:
            rr = json.loads(rep.read_text()).get("results", {})
            for k in ("NESTED node-only", "NESTED graph-only", "NESTED node+graph"):
                if k in rr:
                    nested[k.replace("NESTED ", "") + " GBT (nested CV)"] = rr[k]
        except Exception:
            pass

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": profile_name.upper(),
            "n_applications": int(len(apps)),
            "n_fraud": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "n_rings_injected": int(len(rings)),
            "python": platform.python_version(),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "applications": [
            {
                "id": aid,
                "score": round(float(app_score[aid]), 4),
                "score_pct": round(float(pct[aid]), 1),
                "is_fraud": bool(yb[aid]),
                "ring_id": int(app_ring_id[i]),
                "cluster_id": int(groups_arr[i]),
                "in_ring_cluster": aid in ring_lookup,
                "product": str(apps.iloc[i]["product"]),
                "loan_amount": float(apps.iloc[i]["loan_amount"]),
                "district": str(apps.iloc[i]["application_district"]),
                "state": str(apps.iloc[i]["application_state"]),
                "applied_day": int(apps.iloc[i]["applied_day"]),
                "n_guarantors": int(apps.iloc[i]["n_guarantors"]),
            }
            for i, aid in enumerate(g.app_ids)
        ],
        "rings": clusters_out,
        "ring_eval": ring_eval,
        "explanations": explanations,
        "graph_edges": edges,
        "coldstart": {"evaluation": cs_eval, "worked_example": cs_example},
        "evaluation": {
            "models": {
                "node-only GBT (ring-disjoint OOF)": m_n,
                "graph-only GBT (ring-disjoint OOF)": m_g,
                "node+graph GBT (ring-disjoint OOF)": m_ng,
                **nested,
            },
            "audits": {
                "shuffled_label_control": ctrl,
                "random_split": rand,
                "ring_disjoint": m_ng,
                "leak_gap_auc_pr": round(rand["auc_pr"] - m_ng["auc_pr"], 4),
                "best_single_node_feature": {"name": bn, "auc_pr": round(ba, 4)},
                "best_single_graph_feature": {"name": bgn, "auc_pr": round(bga, 4)},
            },
            "typology_generalisation": typ_gen,
            "entity_resolution": er,
        },
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["SMOKE", "FULL"], default="SMOKE")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"[demo] building demo data ({args.profile}) ...")
    payload = build(args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    kb = out.stat().st_size / 1024
    print(f"[demo] wrote {out}  ({kb:.0f} KB, {payload['meta']['elapsed_sec']}s)")
    ev = payload["evaluation"]["models"]["node+graph GBT (ring-disjoint OOF)"]
    re_ = payload["ring_eval"]
    print(f"       application AUC-PR = {ev['auc_pr']:.3f}  "
          f"recall@5% = {ev['recall_at_5pct']:.3f}")
    print(f"       ring clusters: {re_.get('n_candidate_clusters')} candidates, "
          f"{re_.get('n_true_rings')} true rings")


if __name__ == "__main__":
    main()
