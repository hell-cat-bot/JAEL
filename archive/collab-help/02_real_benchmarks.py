"""Our protocol, applied to somebody else's data.

The submission's biggest weakness is that every number so far comes from data we
generated ourselves. This applies JA-LE's ring-disjoint protocol to public
graph-fraud benchmarks. Published numbers on these datasets use random splits,
so ours will be lower -- that is the point.

Start with YelpChi. T-Finance and DGraph-Fin need more memory; uncomment one at
a time. CPU is fine.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from jale.eval.metrics import full_metrics, best_single_feature
from jale.models.models import fit_gbt
from jale.data.public_datasets import (load_yelpchi, load_bwgnn_pt, load_dgraphfin,
                                       subsample, ring_disjoint_folds_from_adjacency)

LOADERS = [("YelpChi", load_yelpchi, None)]
# LOADERS.append(("T-Finance", load_bwgnn_pt, "/content/data/T-Finance.pt"))
# LOADERS.append(("DGraph-Fin", load_dgraphfin, "/content/data/dgraphfin.npz"))

for name, fn, arg in LOADERS:
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    try:
        g = fn(arg) if arg else fn()
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        continue
    print(" ", g.summary())
    g = subsample(g, 0.25) if g.X.shape[0] > 60_000 else g
    print("  after subsample:", g.summary())

    Xraw = np.nan_to_num(g.X, nan=0.0, posinf=0.0, neginf=0.0)
    yr = g.y.astype(int)
    folds = ring_disjoint_folds_from_adjacency(g.A, n_splits=5)
    print("  fold sizes:", np.bincount(folds), " positives per fold:",
          np.bincount(folds[yr == 1]))

    nm, auc = best_single_feature(yr, Xraw)
    print(f"  honest floor: best single feature {nm} AUC-PR={auc:.4f} "
          f"(base {yr.mean():.4f})")

    for split, name2 in [("ring", "ring-disjoint"), ("random", "random split")]:
        oof = np.zeros(len(yr))
        if split == "ring":
            it = GroupKFold(n_splits=5).split(Xr, yr, groups=folds)
        else:
            it = StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, yr)
        for tr, te in it:
            sc = StandardScaler().fit(Xraw[tr])   # training rows only
            oof[te] = fit_gbt(sc.transform(Xraw[tr]), yr[tr], seed=0).predict_proba(
                sc.transform(Xraw[te]))[:, 1]
        m = full_metrics(yr, oof)
        print(f"  {name2:14s} AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f} "
              f"R@5%={m['recall_at_5pct']:.3f}")
print("\nDONE. Paste everything above back verbatim.")
