"""The node-only baseline is understated. Fix it.

`n_guarantors` alone, used as a raw score, gives AUC-PR 0.2325 (6.2x lift) while
the node-only GBT gives 0.0376 (chance). A single coarse integer beats a tuned
gradient-boosted model by 6x. That means one of two things:

  (i)  the GBT is misconfigured for 131 positives, or
  (ii) the GBT is correctly refusing to fit 11 mostly-noise features and the
       univariate score is the honest node-only number.

Either way, reporting "node-only = chance, therefore the graph supplies 17.7x"
is wrong. The honest framing is "node-only ~6.2x, graph takes it to 17.7x".

This script establishes the honest node-only ceiling several ways.
"""
import sys; sys.path.insert(0, "/home/user/jale")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
X_all = X_node.join(gf.reindex(g.app_ids)[G], how="left")
groups = fold_groups(g).to_numpy()
folds = list(GroupKFold(n_splits=5).split(X_node, y, groups=groups))

def run(X, make, name):
    oof = np.zeros(len(y))
    for tr, te in folds:
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        sc = StandardScaler().fit(Xv[tr])
        m = make()
        m.fit(sc.transform(Xv[tr]), y[tr])
        oof[te] = (m.decision_function(sc.transform(Xv[te]))
                   if hasattr(m, "decision_function")
                   else m.predict_proba(sc.transform(Xv[te]))[:, 1])
    r = full_metrics(y, oof)
    print(f"  {name:38s} AUC-PR={r['auc_pr']:.4f} lift={r['lift_pr']:5.1f}x "
          f"R@5%={r['recall_at_5pct']:.3f}")
    return r["auc_pr"]

print("=== NODE-ONLY, several model families ===")
run(X_node, lambda: HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=250, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0, class_weight="balanced"),
    "GBT (current default)")
run(X_node, lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=5,
        l2_regularization=1.0, random_state=0, class_weight="balanced"),
    "GBT (shallow, small leaf)")
run(X_node, lambda: HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=300, min_samples_leaf=3,
        random_state=0, class_weight="balanced"),
    "GBT (depth 2)")
run(X_node, lambda: RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=0,
        class_weight="balanced_subsample"), "RandomForest (depth 4)")
run(X_node, lambda: LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"),
    "Logistic (raw features)")
run(X_node.rank(pct=True),
    lambda: LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"),
    "Logistic (rank-transformed)")

print("\n=== best single node feature (honest node-only floor) ===")
best, bestc = 0.0, None
for c in NODE:
    v = X_node[c].to_numpy(float)
    r = full_metrics(y, v)
    a = max(r["auc_pr"], full_metrics(y, -v)["auc_pr"])
    if a > best: best, bestc = a, c
    print(f"  {c:29s} AUC-PR={r['auc_pr']:.4f} lift={r['lift_pr']:5.1f}x")
print(f"  -> best single feature: {bestc} at AUC-PR={best:.4f} ({best/0.0377:.1f}x lift)")

print("\n=== NODE+GRAPH reference (unchanged) ===")
run(X_all, lambda: HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=250, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0, class_weight="balanced"),
    "GBT node+graph")

print("\n=== does adding n_guarantors alone to the graph block matter? ===")
Xg_only = gf.reindex(g.app_ids)[G]
run(Xg_only, lambda: HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=250, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0, class_weight="balanced"),
    "GBT graph-only (no node feats)")
