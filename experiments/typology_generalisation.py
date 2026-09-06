"""Does the model learn RINGS, or does it learn THESE rings?

Train on four fraud typologies, test on the fifth. If performance collapses when
the held-out typology is structurally unlike the others, the model has memorised
typology-specific signatures rather than learning what a ring is -- which is the
difference between a system that survives a new fraud pattern and one that does not.

Leakage note: this uses the typology label to build the split, which is fine for a
diagnostic but is NOT the headline protocol. Reported separately for that reason.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_gbt

root = Path(build_ds(SMOKE, "data/jale_smoke"))
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
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
X_node = nf.reindex(g.app_ids)[NODE]
X_all = X_node.join(gf.reindex(g.app_ids)[G], how="left")

groups = fold_groups(g).to_numpy()

print("Test set = every application in a fold group touched by the held-out typology,")
print("plus benign applications from groups that contain no fraud at all. This keeps")
print("the split ring-disjoint AND gives the test set both classes.\n")

clean_groups = np.array([gg for gg in np.unique(groups)
                         if y[groups == gg].sum() == 0])
rng = np.random.default_rng(0)

print(f"{'held-out typology':20s} {'n_test':>7s} {'pos':>4s} {'rate':>7s} {'node-only':>10s} {'node+graph':>11s}")
rows = []
for held in sorted(typ.dropna().unique()):
    held_apps = (typ == held).to_numpy()
    # every group touched by this typology goes to test, whole
    touched = np.unique(groups[held_apps])
    te_group = np.isin(groups, touched)
    # add benign applications from fraud-free groups so the test set has negatives
    n_neg = max(int(held_apps.sum() * (1 - y.mean()) / y.mean()), 200)
    pool = np.flatnonzero(np.isin(groups, clean_groups) & (y == 0) & ~te_group)
    extra = np.zeros(len(y), bool)
    if len(pool):
        extra[rng.choice(pool, min(n_neg, len(pool)), replace=False)] = True
    te = te_group | extra
    tr = ~te
    if y[te].sum() == 0 or y[tr].sum() == 0:
        continue
    out = []
    for X in (X_node, X_all):
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        sc = StandardScaler().fit(Xv[tr])
        m = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0)
        s = m.predict_proba(sc.transform(Xv[te]))[:, 1]
        out.append(full_metrics(y[te], s).get("auc_pr", float("nan")))
    rows.append((held, int(te.sum()), int(y[te].sum()), out[0], out[1]))
    print(f"{held:20s} {te.sum():7d} {y[te].sum():4d} {y[te].mean():7.3f} "
          f"{out[0]:10.4f} {out[1]:11.4f}")

print("\nmean held-out AUC-PR:  node-only %.4f   node+graph %.4f"
      % (np.nanmean([r[3] for r in rows]), np.nanmean([r[4] for r in rows])))
print("in-distribution reference (see reports/v1_SMOKE.json): "
      "node+graph ring-disjoint ~0.73, random split ~0.85")
