"""Is the 0.667 -> 0.717 improvement real, or is it noise dressed up as progress?

Three independent checks, because one is not enough:

  1. Per-fold paired comparison. Both configs scored on identical folds, so the
     difference is paired. Shows whether the gain is consistent or driven by one
     lucky fold.

  2. Bootstrap confidence intervals on AUC-PR for each config, and on the
     difference. Resamples applications with replacement. Caveat: resampling
     breaks ring-disjointness, so this measures sampling noise in the metric, not
     leakage. It answers "could this gap arise from 131 positives alone?".

  3. Model-family convergence. If leaf=20 was genuinely broken rather than merely
     suboptimal, then several *different* model families should all land near
     0.24 on node-only features and none should land at 0.038.
"""
import sys; sys.path.insert(0, "/home/user/jale")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.models.models import fit_gbt

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
X_all = nf.reindex(g.app_ids)[NODE].join(gf.reindex(g.app_ids)[G], how="left")
groups = fold_groups(g).to_numpy()
Xv = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
folds = list(GroupKFold(n_splits=5).split(Xv, y, groups=groups))

OLD = dict(max_depth=6, min_samples_leaf=20, max_iter=250, learning_rate=0.08)
NEW = dict(max_depth=6, min_samples_leaf=10, max_iter=250, learning_rate=0.05)

def oof_scores(params):
    o = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(Xv[tr])
        m = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0, **params)
        o[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return o

s_old, s_new = oof_scores(OLD), oof_scores(NEW)

print("=== 1. per-fold paired comparison ===")
print(f"{'fold':>4s} {'n':>6s} {'pos':>5s} {'old':>8s} {'new':>8s} {'delta':>8s}")
d = []
for i, (tr, te) in enumerate(folds):
    a = average_precision_score(y[te], s_old[te])
    b = average_precision_score(y[te], s_new[te])
    d.append(b - a)
    print(f"{i:>4d} {len(te):6d} {y[te].sum():5d} {a:8.4f} {b:8.4f} {b-a:+8.4f}")
d = np.array(d)
print(f"  mean delta {d.mean():+.4f}   sd {d.std(ddof=1):.4f}   "
      f"folds improved: {(d>0).sum()}/5")
print(f"  paired t on 5 folds: t={d.mean()/(d.std(ddof=1)/np.sqrt(5)):.2f} "
      f"(df=4, critical 2.78 at p=0.05 two-sided)")

print("\n=== 2. bootstrap CIs (2000 resamples) ===")
rng = np.random.default_rng(0)
n = len(y)
bo, bn, bd = [], [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if y[idx].sum() == 0 or y[idx].sum() == len(idx):
        continue
    a = average_precision_score(y[idx], s_old[idx])
    b = average_precision_score(y[idx], s_new[idx])
    bo.append(a); bn.append(b); bd.append(b - a)
bo, bn, bd = map(np.array, (bo, bn, bd))
for lbl, v in [("old (leaf=20)", bo), ("new (leaf=10)", bn)]:
    print(f"  {lbl:14s} mean={v.mean():.4f}  95% CI [{np.percentile(v,2.5):.4f}, "
          f"{np.percentile(v,97.5):.4f}]")
print(f"  difference       mean={bd.mean():+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, "
      f"{np.percentile(bd,97.5):+.4f}]")
print(f"  P(new > old) = {(bd > 0).mean():.4f}")

print("\n=== 3. model-family convergence on node-only features ===")
Xn = nf.reindex(g.app_ids)[NODE].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)

def oof_model(make):
    o = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(Xn[tr])
        m = make().fit(sc.transform(Xn[tr]), y[tr])
        o[te] = m.predict_proba(sc.transform(Xn[te]))[:, 1]
    return average_precision_score(y, o)

def gbt(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **kw)

for lbl, mk in [
    ("GBT leaf=20 (OLD)", lambda: gbt(**OLD)),
    ("GBT leaf=10 (NEW)", lambda: gbt(**NEW)),
    ("GBT depth=3 leaf=5", lambda: gbt(max_depth=3, min_samples_leaf=5,
                                       max_iter=250, learning_rate=0.08)),
    ("GBT depth=2 leaf=5", lambda: gbt(max_depth=2, min_samples_leaf=5,
                                       max_iter=300, learning_rate=0.05)),
    ("RandomForest d=4", lambda: RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=0,
        class_weight="balanced_subsample")),
    ("ExtraTrees d=6", lambda: ExtraTreesClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=0,
        class_weight="balanced_subsample")),
]:
    print(f"  {lbl:22s} {oof_model(mk):.4f}")
print("  If leaf=20 were merely suboptimal, other families would scatter around it.")
print("  If it was structurally broken, every other family lands far above it.")
