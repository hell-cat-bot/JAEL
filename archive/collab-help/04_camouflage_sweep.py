"""Where does the system stop working?

Sweeps the fraction of ring-adjacent camouflage applications. At 0% the rings are
bare; at high values a ring is mostly innocent-looking applications and the
structural signal drowns. Reporting our own breaking point is more persuasive
than not reporting it.

Requires a config knob for camouflage rate. If BenignStructure / RingTypology has
no such field, this script prints the fields it found and exits -- send that back
and the knob can be added.
"""
import sys, os, dataclasses
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from jale.config import SMOKE, ObservationTime
from jale.data.generator import LenderGraphGenerator, save_dataset
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_gbt

knobs = [f.name for f in dataclasses.fields(SMOKE.benign)] if hasattr(SMOKE, "benign") else []
ring_fields = []
if hasattr(SMOKE, "ring_typologies"):
    for t in SMOKE.ring_typologies:
        ring_fields = [f.name for f in dataclasses.fields(t)]
        break
print("BenignStructure fields:", knobs)
print("RingTypology fields   :", ring_fields)
CAND = [k for k in knobs + ring_fields if "camou" in k.lower()]
if not CAND:
    print("\nNo camouflage knob found. Paste the two field lists above back and the")
    print("knob will be added to jale/config.py.")
    sys.exit(0)
print("camouflage knob(s):", CAND)

for rate in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    cfg = SMOKE
    setattr(cfg.benign, CAND[0], rate) if CAND[0] in knobs else None
    gen = LenderGraphGenerator(cfg)
    tabs = gen.generate()
    apps = tabs["applications"]
    y = (apps["ring_id"] > 0).astype(int).to_numpy()
    g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
    groups = fold_groups(g).to_numpy()
    nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
    gf = build_graph_features(g, apps, nf)
    NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
    G = [c for c in gf.columns if c != "ppr"]
    X = nf.reindex(g.app_ids)[NODE].join(gf.reindex(g.app_ids)[G], how="left")
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xv, y, groups=groups):
        sc = StandardScaler().fit(Xv[tr])
        oof[te] = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0).predict_proba(
            sc.transform(Xv[te]))[:, 1]
    m = full_metrics(y, oof)
    print(f"camouflage x{rate:4.2f}  n={len(y):5d} pos={y.sum():4d} ({y.mean():.2%})  "
          f"AUC-PR={m['auc_pr']:.4f} lift={m['lift_pr']:.1f}x R@5%={m['recall_at_5pct']:.3f}")
print("\nDONE. Paste back verbatim.")
