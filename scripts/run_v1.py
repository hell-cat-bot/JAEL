#!/usr/bin/env python3
"""JA-LE V1 end-to-end run.

Pipeline: generate -> resolve identities -> build graph -> build features ->
ring-disjoint CV -> audits. Nothing in here reads a fraud label except the final
scoring comparison; entity resolution, graph construction, feature construction
and fold assignment are all unsupervised.

Usage:  python scripts/run_v1.py [--profile SMOKE|FULL]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jale.config import FULL, SMOKE, ObservationTime
from jale.data.generator import LenderGraphGenerator
from jale.eval.metrics import best_single_feature, full_metrics
from jale.eval.splits import ring_disjoint_folds, shuffle_label_control
from jale.features.builder import build_graph_features, build_node_features
from jale.graph.builder import build_graph, fold_groups
from jale.models.models import (GBT_GRID, GraphRegularisedLogistic, fit_gbt,
                                fit_logistic, select_gbt)

N_SPLITS = 5
SEED = 0


def prep(X: pd.DataFrame, fit_mask: np.ndarray):
    """Scale on the training rows only -- fitting the scaler on all rows leaks."""
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    sc = StandardScaler().fit(Xv[fit_mask])
    return sc.transform(Xv)


def cv_score(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str,
             A: sp.csr_matrix | None = None) -> np.ndarray:
    """Return out-of-fold scores under ring-disjoint (group) CV."""
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(X, y, groups=groups):
        mask = np.zeros(len(y), dtype=bool); mask[tr] = True
        Xtr, Xte = prep(X, mask)[tr], prep(X, mask)[te]
        if kind == "lr":
            m = fit_logistic(Xtr, y[tr]); s = m.decision_function(Xte)
        elif kind == "gbt":
            m = fit_gbt(Xtr, y[tr], seed=SEED); s = m.predict_proba(Xte)[:, 1]
        elif kind == "graph_lr":
            m = GraphRegularisedLogistic(lam=0.5).fit(Xtr, y[tr], A[tr][:, tr])
            s = m.decision_function(Xte)
        else:
            raise ValueError(kind)
        oof[te] = s
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["SMOKE", "FULL"], default="SMOKE")
    ap.add_argument("--nested", action="store_true",
                    help="also run nested CV: hyperparameters are chosen inside each "
                         "outer training split only, so the reported number carries no "
                         "selection bias. Slower (~10 min on SMOKE).")
    args = ap.parse_args()
    cfg = SMOKE if args.profile == "SMOKE" else FULL

    t0 = time.time()
    print(f"[1/6] generating synthetic portfolio ({args.profile}) ...")
    # Write to disk with labels physically segregated, then read the feature
    # tables back from `raw/`. The evaluation code only ever sees `raw/`; the
    # labels are read from `labels/` at the single point where scores are
    # compared to truth. Leakage is therefore impossible by construction rather
    # than by convention -- there is no label column in any frame this pipeline
    # builds features from.
    from jale.data.generator import build as build_dataset
    root = build_dataset(cfg, f"data/jale_{args.profile.lower()}")
    tabs = {f.stem: pd.read_parquet(f)
            for f in sorted((root / "raw").glob("*.parquet"))}
    assert not any(any(c.startswith("label") or c in ("ring_id", "human_id", "is_kiosk")
                       for c in df.columns) for df in tabs.values()), \
        "a ground-truth column leaked into the observable tables"
    lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
    apps = tabs["applications"]
    y = (lab.set_index("application_id")["ring_id"]
            .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
    print(f"      {len(apps)} applications | {int(y.sum())} fraud "
          f"({y.mean()*100:.2f}% base rate)")
    print(f"      loaded from {root}/raw/ (labels held separately in labels/)")

    print("[2/6] building graph ...")
    g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
    A = g.cooccurrence_union()
    groups = fold_groups(g).to_numpy()
    print(f"      relations: " + ", ".join(f"{r}={g.incidence[r].shape[1]}" for r in g.relations()))
    print(f"      ring-disjoint fold groups: {len(np.unique(groups))} "
          f"(largest {pd.Series(groups).value_counts().max()})")

    print("[3/6] building features ...")
    nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
    gf = build_graph_features(g, apps, nf)
    NODE_COLS = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
    G_COLS = [c for c in gf.columns if c != "ppr"]
    print(f"      node features: {len(NODE_COLS)} | graph features: {len(G_COLS)}")

    print("[4/6] cross-validating under ring-disjoint splits ...")
    results = {}
    # nf and gf are both indexed by application_id; align both to the graph's
    # canonical application order so rows and labels stay in lockstep.
    X_node = nf.reindex(g.app_ids)[NODE_COLS]
    X_all = X_node.join(gf.reindex(g.app_ids)[G_COLS], how="left")
    assert X_all.index.equals(pd.Index(g.app_ids))
    assert not X_all.isna().all(axis=0).any(), "an entirely-empty feature column"
    for name, X, kind in [
        ("node-only  logistic", X_node, "lr"),
        ("node-only  GBT",      X_node, "gbt"),
        ("node+graph GBT",      X_all,  "gbt"),
        ("node+graph graph-LR", X_all,  "graph_lr"),
    ]:
        s = cv_score(X, y, groups, kind, A=A)
        results[name] = full_metrics(y, s)
        m = results[name]
        print(f"      {name:22s} AUC-PR={m['auc_pr']:.4f} (lift {m['lift_pr']:.1f}x)  "
              f"AUC-ROC={m['auc_roc']:.4f}  recall@5%={m['recall_at_5pct']:.3f}")

    print("[4b] honest floor: best single feature used as a raw score ...")
    # Guards against a misconfigured baseline flattering the model. A tuned
    # ensemble that cannot beat one column used raw is not beating anything.
    for lbl, X in [("node", X_node), ("graph", gf[G_COLS].reindex(g.app_ids))]:
        nm, auc = best_single_feature(y, X)
        print(f"      best single {lbl:5s} feature: {nm:32s} AUC-PR={auc:.4f} "
              f"(lift {auc/y.mean():.1f}x)")

    if args.nested:
        print("[4c] NESTED CV - hyperparameters chosen without seeing the test fold ...")
        # Selection on the same folds used for reporting inflates the score; on
        # SMOKE the sweep overstated node+graph by +0.037 AUC-PR. Nested CV
        # removes that. See experiments/nested_cv.py.
        for lbl, X in [("node-only", X_node), ("graph-only", gf[G_COLS].reindex(g.app_ids)),
                       ("node+graph", X_all)]:
            Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups=groups):
                params, _ = select_gbt(Xv[tr], y[tr], groups[tr])
                sc = StandardScaler().fit(Xv[tr])
                m = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=SEED,
                            max_depth=params[0], min_samples_leaf=params[1],
                            max_iter=params[2], learning_rate=params[3])
                oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
            r = full_metrics(y, oof)
            results[f"NESTED {lbl}"] = r
            print(f"      {lbl:11s} nested AUC-PR={r['auc_pr']:.4f} "
                  f"lift={r['lift_pr']:.1f}x R@5%={r['recall_at_5pct']:.3f}")

    print("[5/6] AUDIT - shuffled-label control (must sit at chance) ...")
    # The control must permute labels within *cross-validation folds*, not within
    # connected components: most components are single applications, so shuffling
    # inside them is a no-op and the "control" silently keeps the real labels.
    fold_id = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X_all, y, groups=groups)):
        fold_id[te] = i
    yl = shuffle_label_control(y, fold_id, seed=1)
    assert float(np.mean(yl)) == float(np.mean(y)), "control changed the base rate"
    for name, X in [("node-only  GBT", X_node), ("node+graph GBT", X_all)]:
        s = cv_score(X, yl, groups, "gbt", A=A)
        ctrl = full_metrics(yl, s)
        verdict = "PASS (at chance)" if ctrl["auc_pr"] < 1.5 * ctrl["base_rate"] \
            else "FAIL: features predict permuted labels -> investigate leakage"
        print(f"      {name:16s} AUC-PR={ctrl['auc_pr']:.4f} vs base {ctrl['base_rate']:.4f}"
              f" -> {verdict}")
        results[f"CONTROL shuffled {name}"] = ctrl

    print("[6/6] AUDIT - random row split vs ring-disjoint split ...")
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X_all, y):
        mask = np.zeros(len(y), dtype=bool); mask[tr] = True
        Xtr, Xte = prep(X_all, mask)[tr], prep(X_all, mask)[te]
        oof[te] = fit_gbt(Xtr, y[tr], seed=SEED).predict_proba(Xte)[:, 1]
    leaky = full_metrics(y, oof)
    print(f"      random split AUC-PR   = {leaky['auc_pr']:.4f}")
    print(f"      ring-disjoint AUC-PR  = {results['node+graph GBT']['auc_pr']:.4f}")
    gap = leaky['auc_pr'] - results['node+graph GBT']['auc_pr']
    print(f"      gap                   = {gap:+.4f}  "
          f"({'random split is inflated by ring leakage' if gap > 0.01 else 'no material leakage'})")

    out = {"profile": args.profile, "results": results,
           "shuffled_control": ctrl, "random_split": leaky,
           "n_graph_features": len(G_COLS), "elapsed_sec": round(time.time()-t0, 1)}
    os.makedirs("reports", exist_ok=True)
    path = f"reports/v1_{args.profile}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
