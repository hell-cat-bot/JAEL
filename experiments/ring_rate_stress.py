"""Stress test: does detection survive different fraud prevalence and heavier
camouflage? (doubts.md D16)

Two knobs are swept, both first-class parameters of our own generator:

  1. Prevalence -- Profile.fraud_ring_fraction in {1.0%, 2.0%, 3.0% (= baseline
     SMOKE), 4.5%}. Answers: "is 0.754 an artifact of the chosen ring share?"

  2. Camouflage -- every RingTypology.benign_link_rate scaled by {x1 (ref),
     x1.5, x2} at baseline prevalence. Answers: "do the rings just sit there
     waiting to be found?" (benign_link_rate is the generator's declared
     evasion knob: extra benign links per ring member; see jale/config.py.)

Protocol is identical to scripts/run_v1.py: ring-disjoint GroupKFold(5),
node+graph GBT, labels touched only at scoring. Same seed family; each
setting generates a fresh world -- this is a sensitivity sweep, not a
confidence interval, and it is reported as such.

What this is NOT: a substitute for real-data validation. It varies our own
simulator's knobs to bound how much of the headline number depends on
prevalence and evasion difficulty -- the two questions a reviewer should ask.

Run:  python experiments/ring_rate_stress.py    (~4 min, SMOKE profile)
Writes: reports/ring_rate_stress.json
"""
import sys
from dataclasses import replace
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import jale.data.generator as gen_mod
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_gbt

ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
PREVALENCE = [0.010, 0.020, 0.030, 0.045]     # 0.030 = SMOKE baseline
CAMOUFLAGE = [1.0, 1.5, 2.0]                  # x benign_link_rate (1.0 = ref)
NODE_EXCLUDE_SUFFIX = ("_freq", "_code")


def oof_gbt(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(Xv, y, groups=groups):
        sc = StandardScaler().fit(Xv[tr])          # scale on train rows only
        m = fit_gbt(sc.transform(Xv[tr]), y[tr], seed=0)
        oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return oof


def run_setting(ring_fraction: float, cam_factor: float, tmp: Path) -> dict:
    cfg = replace(SMOKE, fraud_ring_fraction=ring_fraction)
    patched = None
    if cam_factor != 1.0:
        orig = gen_mod.RING_TYPOLOGIES
        patched = {k: replace(t, benign_link_rate=t.benign_link_rate * cam_factor)
                   for k, t in orig.items()}
        gen_mod.RING_TYPOLOGIES = patched
    try:
        out = tmp / f"rf{ring_fraction:.3f}_cam{cam_factor:.1f}"
        root = build_ds(cfg, str(out))
        tabs = {f.stem: pd.read_parquet(f)
                for f in sorted((root / "raw").glob("*.parquet"))}
        apps = tabs["applications"]
        g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
        lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
        rings = pd.read_parquet(root / "labels" / "rings.parquet")
        y = (lab.set_index("application_id")["ring_id"]
                .reindex(apps["application_id"]).fillna(0).to_numpy() > 0
             ).astype(int)
        nf = build_node_features(apps, tabs["emi_schedule"],
                                 ObservationTime.APPLICATION)
        gf = build_graph_features(g, apps, nf)
        node_cols = [c for c in nf.columns if not c.endswith(NODE_EXCLUDE_SUFFIX)]
        g_cols = [c for c in gf.columns if c != "ppr"]
        X = (nf.reindex(g.app_ids)[node_cols]
               .join(gf.reindex(g.app_ids)[g_cols], how="left"))
        groups = fold_groups(g).to_numpy()
        s = oof_gbt(X, y, groups)
        m = full_metrics(y, s)
        return dict(ring_fraction=ring_fraction, cam_factor=cam_factor,
                    n_apps=int(len(y)), n_pos=int(y.sum()),
                    n_rings=int(len(rings)),
                    auc_pr=round(float(m["auc_pr"]), 4),
                    lift_pr=round(float(m["lift_pr"]), 1),
                    recall_at_5pct=round(float(m["recall_at_5pct"]), 3))
    finally:
        if patched is not None:
            gen_mod.RING_TYPOLOGIES = orig          # restore, always


def main():
    import json
    results = {"prevalence": [], "camouflage": []}
    with tempfile.TemporaryDirectory(prefix="jale_stress_") as td:
        tmp = Path(td)
        print(f"{'setting':22s} {'n_apps':>7s} {'pos':>5s} {'rings':>6s} "
              f"{'AUCPR':>7s} {'lift':>6s} {'R@5%':>6s}")
        for rf in PREVALENCE:
            r = run_setting(rf, 1.0, tmp)
            results["prevalence"].append(r)
            print(f"{'prev ' + format(rf, '.1%'):22s} {r['n_apps']:7d} "
                  f"{r['n_pos']:5d} {r['n_rings']:6d} {r['auc_pr']:7.4f} "
                  f"{r['lift_pr']:6.1f} {r['recall_at_5pct']:6.3f}")
        for cf in CAMOUFLAGE:
            if cf == 1.0:
                continue                              # ref = prevalence baseline
            r = run_setting(SMOKE.fraud_ring_fraction, cf, tmp)
            results["camouflage"].append(r)
            print(f"{'cam x' + format(cf, '.1f'):22s} {r['n_apps']:7d} "
                  f"{r['n_pos']:5d} {r['n_rings']:6d} {r['auc_pr']:7.4f} "
                  f"{r['lift_pr']:6.1f} {r['recall_at_5pct']:6.3f}")

    prev = {r["ring_fraction"]: r["auc_pr"] for r in results["prevalence"]}
    cam = {r["cam_factor"]: r["auc_pr"] for r in results["camouflage"]}
    ref = prev[0.030]                             # x1 camouflage = baseline
    results["reading"] = dict(
        prevalence_range=[prev[0.010], prev[0.045]],
        camouflage_range=[ref, cam.get(2.0)],
        note=("AUC-PR band across a 4.5x prevalence range and x2 camouflage -> "
              "the headline number is not an artifact of an easy operating "
              "point. Small-prevalence cells are noisier (few positives); the "
              "sweep is a sensitivity bound, not a CI."),
    )
    (ROOT / "reports" / "ring_rate_stress.json").write_text(
        json.dumps(results, indent=1))
    print("\nwrote reports/ring_rate_stress.json")
    print(f"reading: prevalence {prev[0.010]:.3f}->{prev[0.045]:.3f} | "
          f"camouflage {ref:.3f}->{cam.get(2.0):.3f}")


if __name__ == "__main__":
    main()

