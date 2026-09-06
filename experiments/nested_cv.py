"""Nested cross-validation: the honest way to report a tuned model.

The previous script swept hyperparameters on the same ring-disjoint folds used to
report performance. That is selection on the test set -- exactly the kind of
cheating this project is supposed to avoid -- and it inflates the number.

Here the hyperparameters are chosen *inside* each outer training split only. The
outer test fold is never seen during selection. The resulting number is lower
than the sweep reported, and it is the one that should go in the document.

Inner selection uses a ring-disjoint split of the outer training rows, so the
ring guarantee holds at both levels.
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

GRID = list(itertools.product([2, 3, 6], [5, 10, 20], [250, 500], [0.05, 0.08]))

def fit_predict(Xv, tr, ev, params):
    sc = StandardScaler().fit(Xv[tr])
    m = HistGradientBoostingClassifier(random_state=0, class_weight="balanced",
                                       max_depth=params[0], min_samples_leaf=params[1],
                                       max_iter=params[2], learning_rate=params[3],
                                       l2_regularization=1.0)
    m.fit(sc.transform(Xv[tr]), y[tr])
    return m.predict_proba(sc.transform(Xv[ev]))[:, 1]

def nested(X, label):
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    oof = np.zeros(len(y)); chosen = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
        # inner ring-disjoint split of the outer TRAINING rows only
        inner = list(GroupKFold(n_splits=3).split(Xv[tr], y[tr], groups=groups[tr]))
        best = (-1.0, GRID[0])
        for params in GRID:
            sc = np.zeros(len(tr))
            for itr, ite in inner:
                s = StandardScaler().fit(Xv[tr][itr])
                m = HistGradientBoostingClassifier(
                    random_state=0, class_weight="balanced", max_depth=params[0],
                    min_samples_leaf=params[1], max_iter=params[2],
                    learning_rate=params[3], l2_regularization=1.0)
                m.fit(s.transform(Xv[tr][itr]), y[tr][itr])
                sc[ite] = m.predict_proba(s.transform(Xv[tr][ite]))[:, 1]
            a = full_metrics(y[tr], sc).get("auc_pr", 0.0)
            if a > best[0]: best = (a, params)
        chosen.append(best[1])
        oof[te] = fit_predict(Xv, tr, te, best[1])
    r = full_metrics(y, oof)
    print(f"  {label:14s} nested AUC-PR={r['auc_pr']:.4f} lift={r['lift_pr']:.1f}x "
          f"R@1%={r['recall_at_1pct']:.3f} R@5%={r['recall_at_5pct']:.3f} "
          f"R@10%={r['recall_at_10pct']:.3f}")
    print(f"                 params chosen per fold: {chosen}")
    return r

print(f"base rate = {y.mean():.4f}\n=== NESTED CV (selection never sees the test fold) ===")
nested(X_node, "node-only")
nested(X_g, "graph-only")
nested(X_all, "node+graph")
