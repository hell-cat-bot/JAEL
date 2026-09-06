"""Does a GNN beat gradient boosting on the SAME graph, features, folds, metric?

The whole plan assumes it might. Nobody has checked. This answers it, and also
answers the more interesting question: does a GNN generalise across fraud
typologies better than the feature-based model does? (In the sandbox the
feature-based model collapses from 0.754 to 0.273 mean held-out AUC-PR when the
held-out typology was never seen in training.)

Run on a Colab GPU runtime. ~15 min.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_gbt
from jale.models.torch_gnn import to_pyg_data, build_model, train_and_eval

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print("device:", DEVICE)

root = build_ds(SMOKE, "data/jale_colab")
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
A = g.cooccurrence_union()
lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
rings = pd.read_parquet(root / "labels" / "rings.parquet")
y = (lab.set_index("application_id")["ring_id"]
        .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
typ = (lab.merge(rings[["ring_id", "typology"]], on="ring_id", how="left")
          .set_index("application_id")["typology"].reindex(g.app_ids))
nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
gf = build_graph_features(g, apps, nf)
NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
G = [c for c in gf.columns if c != "ppr"]
X_all = nf.reindex(g.app_ids)[NODE].join(gf.reindex(g.app_ids)[G], how="left")
groups = fold_groups(g).to_numpy()

# Scaler is fitted INSIDE each fold, on training rows only. Fitting it once on the
# whole matrix leaks each test fold's mean/variance into training.
Xraw = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
y64 = y.astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(Xraw, y64, groups=groups))

# Message passing uses STRONG relations only. Measured on SMOKE: the full
# cooccurrence_union() has 511,652 edges of which 403,062 (78.8%) cross fold
# boundaries, 98.6% of them from the dealer relation -- a transductive GNN would
# pass messages from train into test and destroy the ring-disjoint guarantee.
# The strong-relations graph has 8,294 edges and ZERO cross-fold edges.
from jale.graph.builder import STRONG_FOLD_RELATIONS
A_msg = g.cooccurrence_union(tuple(STRONG_FOLD_RELATIONS))
print(f"message-passing graph: {A_msg.nnz:,} edges (full union: {A.nnz:,})")

print("\n=== PART 1: GNN vs GBT, identical ring-disjoint folds ===")
print(f"{'model':10s} {'AUC-PR':>8s} {'AUC-ROC':>8s} {'R@5%':>7s}   (per fold)")
results = {}
for kind in ["gbt", "sage", "gat", "gcn", "caregnn", "bwgnn"]:
    t0, per = time.time(), []
    if kind == "gbt":
        oof = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(Xraw[tr])
            oof[te] = fit_gbt(sc.transform(Xraw[tr]), y[tr], seed=0).predict_proba(
                sc.transform(Xraw[te]))[:, 1]
        m = full_metrics(y, oof); per = [m["auc_pr"]] * 5
    else:
        for tr, te in folds:
            sc = StandardScaler().fit(Xraw[tr])
            Xv = sc.transform(Xraw).astype(np.float32)
            va, te2 = te[: len(te) // 2], te[len(te) // 2:]
            trm = np.zeros(len(y), bool); trm[tr] = True
            vam = np.zeros(len(y), bool); vam[va] = True
            tem = np.zeros(len(y), bool); tem[te2] = True
            data = to_pyg_data(A_msg, Xv, y64, trm, vam, tem)
            model = build_model(kind, in_dim=Xv.shape[1], hidden=128)
            _, mm = train_and_eval(model, data, epochs=150, device=DEVICE)
            per.append(mm["auc_pr"])
        m = {"auc_pr": float(np.mean(per)), "auc_roc": float(np.nan),
             "recall_at_5pct": float(np.nan)}
    results[kind] = m
    print(f"{kind:10s} {m['auc_pr']:8.4f} {m.get('auc_roc', float('nan')):8.4f} "
          f"{m.get('recall_at_5pct', float('nan')):7.3f}   "
          f"{[round(v,3) for v in per]}  {time.time()-t0:.0f}s")

print("\n=== PART 2: cross-typology generalisation (the interesting test) ===")
print("Train on four typologies, test on the fifth. Feature-based GBT collapses")
print("to 0.273 mean here; if a GNN holds up better, that is the headline finding.")
clean = np.array([gg for gg in np.unique(groups) if y[groups == gg].sum() == 0])
rng = np.random.default_rng(0)
print(f"\n{'typology':20s} " + " ".join(f"{k:>9s}" for k in ["gbt", "sage", "caregnn", "bwgnn"]))
for held in sorted(typ.dropna().unique()):
    held_apps = (typ == held).to_numpy()
    te_g = np.isin(groups, np.unique(groups[held_apps]))
    pool = np.flatnonzero(np.isin(groups, clean) & (y == 0) & ~te_g)
    extra = np.zeros(len(y), bool)
    if len(pool):
        extra[rng.choice(pool, min(max(int(held_apps.sum() * 26), 200), len(pool)),
                         replace=False)] = True
    te = te_g | extra; tr = ~te
    if y[te].sum() == 0 or y[tr].sum() == 0:
        continue
    out = []
    for kind in ["gbt", "sage", "caregnn", "bwgnn"]:
        sc = StandardScaler().fit(Xraw[tr])
        if kind == "gbt":
            s = fit_gbt(sc.transform(Xraw[tr]), y[tr], seed=0).predict_proba(
                sc.transform(Xraw[te]))[:, 1]
        else:
            Xv = sc.transform(Xraw).astype(np.float32)
            trm = np.zeros(len(y), bool); trm[tr] = True
            vam = np.zeros(len(y), bool); vam[te & (y == 0)] = True
            tem = np.zeros(len(y), bool); tem[te & (y == 1)] = True
            data = to_pyg_data(A_msg, Xv, y64, trm, vam, tem)
            model = build_model(kind, in_dim=Xv.shape[1], hidden=128)
            p, _ = train_and_eval(model, data, epochs=150, device=DEVICE)
            s = p[te]
        out.append(full_metrics(y[te], s).get("auc_pr", float("nan")))
    print(f"{held:20s} " + " ".join(f"{v:9.4f}" for v in out))
print("\nDONE. Paste everything above back verbatim, including any tracebacks.")
