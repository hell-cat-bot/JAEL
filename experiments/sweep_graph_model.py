"""Hyperparameter sweep on the graph model.

`min_samples_leaf=20` was crippling the node-only baseline (0.038 vs 0.242 when
lowered). With 131 positives over 5 folds there are only ~26 positives per fold,
so a leaf requiring 20 samples can barely split on the positive class at all.

Question: does the same fix lift the node+graph model, or does it just overfit?
Swept under ring-disjoint CV, so any gain is honest.
"""
import sys; sys.path.insert(0, "/home/user/jale")
from pathlib import Path
import itertools
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics

root = Path(build_ds(SMOKE, "data/jale_smoke"))
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
y = (lab.set_index("application_id")["ring_id"]
        .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
gf = build_graph_features(g, apps, nf)
NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
G = [c for c in gf.columns if c != "ppr"]
X_node = nf.reindex(g.app_ids)[NODE]
X_g = gf.reindex(g.app_ids)[G]
X_all = X_node.join(X_g, how="left")
groups = fold_groups(g).to_numpy()
folds = list(GroupKFold(n_splits=5).split(X_node, y, groups=groups))

def cv(X, **kw):
    oof = np.zeros(len(y))
    for tr, te in folds:
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        sc = StandardScaler().fit(Xv[tr])
        m = HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **kw)
        m.fit(sc.transform(Xv[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return full_metrics(y, oof)

print(f"base rate = {y.mean():.4f}   n_pos = {y.sum()}\n")
for label, X in [("node-only", X_node), ("graph-only", X_g), ("node+graph", X_all)]:
    print(f"=== {label} ===")
    best = (0, None)
    for depth, leaf, iters, lr in itertools.product(
            [2, 3, 6], [5, 10, 20], [250, 500], [0.05, 0.08]):
        r = cv(X, max_depth=depth, min_samples_leaf=leaf, max_iter=iters,
               learning_rate=lr, l2_regularization=1.0)
        if r["auc_pr"] > best[0]:
            best = (r["auc_pr"], (depth, leaf, iters, lr), r)
    d, lf, it, lrr = best[1]
    print(f"  best: depth={d} leaf={lf} iters={it} lr={lrr} -> "
          f"AUC-PR={best[2]['auc_pr']:.4f} lift={best[2]['lift_pr']:.1f}x "
          f"R@5%={best[2]['recall_at_5pct']:.3f} R@1%={best[2]['recall_at_1pct']:.3f}")
    cur = cv(X, max_depth=6, min_samples_leaf=20, max_iter=250,
             learning_rate=0.08, l2_regularization=1.0)
    print(f"  current default                -> AUC-PR={cur['auc_pr']:.4f} "
          f"lift={cur['lift_pr']:.1f}x R@5%={cur['recall_at_5pct']:.3f}")
    print(f"  delta                          -> {best[2]['auc_pr']-cur['auc_pr']:+.4f}\n")
