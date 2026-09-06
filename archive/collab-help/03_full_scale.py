"""Does anything break at 120,000 persons? SMOKE is 6,000."""
import sys, os, time, resource
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from jale.config import FULL, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics, best_single_feature
from jale.models.models import fit_gbt

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

t0 = time.time(); print(f"[gen] start, RSS {rss():.0f} MB")
root = build_ds(FULL, "data/jale_full")
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
print(f"[gen] {len(apps):,} apps in {time.time()-t0:.0f}s, RSS {rss():.0f} MB")

lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
y = (lab.set_index("application_id")["ring_id"]
        .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
print(f"[gen] {y.sum():,} fraud ({y.mean():.2%})")

t = time.time(); g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
A = g.cooccurrence_union(); groups = fold_groups(g).to_numpy()
print(f"[graph] {A.shape[0]:,} nodes {A.nnz:,} edges, {len(np.unique(groups)):,} groups "
      f"(largest {pd.Series(groups).value_counts().max()}) in {time.time()-t:.0f}s, RSS {rss():.0f} MB")

t = time.time()
nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
gf = build_graph_features(g, apps, nf)
NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
G = [c for c in gf.columns if c != "ppr"]
X = nf.reindex(g.app_ids)[NODE].join(gf.reindex(g.app_ids)[G], how="left")
print(f"[features] {X.shape} in {time.time()-t:.0f}s, RSS {rss():.0f} MB")

for lbl, XX in [("node-only", X[NODE]), ("node+graph", X)]:
    Xv = XX.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    t = time.time(); oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xv, y, groups=groups):
        sc = StandardScaler().fit(Xv[tr])
        oof[te] = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0).predict_proba(
            sc.transform(Xv[te]))[:, 1]
    m = full_metrics(y, oof)
    print(f"[cv] {lbl:11s} AUC-PR={m['auc_pr']:.4f} lift={m['lift_pr']:.1f}x "
          f"R@5%={m['recall_at_5pct']:.3f}  {time.time()-t:.0f}s RSS {rss():.0f} MB")
print(f"\nDONE in {time.time()-t0:.0f}s, peak RSS {rss():.0f} MB. Paste back verbatim.")
