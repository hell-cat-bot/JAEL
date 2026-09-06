"""Lead time -- does the alarm fire BEFORE the ring finishes filing?

The experiment behind the "predicts formation, not just detection" claim.

Design (rolling-origin backtest -- how a deployed system actually runs):
  * every 14 days ("origin"), retrain on everything EXCEPT the last
    TRAIN_GAP_DAYS (models never see the future);
  * at every weekly snapshot, score the applications filed so far with the
    latest trained model and cluster them into connected components over the
    strong relations (device / account / person / guarantor);
  * score every >=3-application cluster with the FULL L4 ring score
    (jale/demo/l4_rings.py):  0.65 * model corroboration + 0.35 * structure,
    where structure = 0.8 * max(device/account/person/guarantor/burst
    concentration) + 0.2 * internal density;
  * a ring is DETECTED the first snapshot a cluster scoring >= threshold
    already contains >= 3 of its applications; it COMPLETES the day its last
    application is filed; lead time = completion - detection.

Why not structure alone?  First runs of this experiment showed why: a village
Common-Service-Centre kiosk that files 8 legitimate applications on one device
is structurally identical to a device farm. Structure alone flagged 10/10
rings but also ~4,400 benign clusters. The model's corroboration term is what
tells a kiosk from a farm -- which is exactly why the production score is
two-signal, and why a rules-only system cannot work.

Honesty: synthetic SMOKE data; internal validity, not deployment performance.
False alarms are counted as DISTINCT benign clusters ever crossing the
threshold (one cluster = one analyst review), not cluster-weeks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import STRONG_FOLD_RELATIONS, build_graph, fold_groups
from jale.features.builder import build_graph_features, build_node_features
from jale.models.models import fit_gbt

MIN_RING_SIZE = 3
BURST_WINDOW_DAYS = 14
TRAIN_CUTOFF_FRAC = 0.5      # train once on the first half of the timeline
STEP_DAYS = 7                # weekly snapshots
THRESHOLDS = (0.45, 0.50, 0.55, 0.60)
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def cluster_raw(members, work, gl_by_app, A, pos):
    """Raw per-cluster quantities (structure + corroboration inputs)."""
    n = len(members)
    if n < MIN_RING_SIZE:
        return None
    days = work.loc[members, "applied_day"].to_numpy(float)
    n_dev = work.loc[members, "device_id"].astype(str).nunique()
    n_acct = work.loc[members, "account_id"].astype(str).nunique()
    n_per = work.loc[members, "person_id"].astype(str).nunique()
    g_ids = [g for a in members for g in gl_by_app.get(a, [])]
    n_links, n_dist = len(g_ids), len(set(g_ids))

    axes = dict(
        device=1.0 - n_dev / n, account=1.0 - n_acct / n,
        person=1.0 - n_per / n,
        guarantor=(1.0 - n_dist / n_links) if n_links >= 2 else 0.0)
    d = np.sort(days)
    hi = np.searchsorted(d, d + BURST_WINDOW_DAYS, side="right")
    axes["burst"] = float((hi - np.arange(len(d))).max()) / n

    idx = pos.reindex(members).to_numpy()
    sub = A[idx][:, idx]
    density = min((sub.nnz / 2.0) / max(n * (n - 1) / 2.0, 1.0), 1.0)
    structural = 0.8 * max(axes.values()) + 0.2 * density

    counts = {int(k): int(v) for k, v in
              work.loc[members].groupby("ring").size().to_dict().items() if k > 0}
    return dict(n=n, structural=structural, axes=axes,
                mean_score=float(work.loc[members, "score"].mean()),
                frac_hot=float((work.loc[members, "score"] > p_hot_global[0]).mean()),
                counts=counts)


p_hot_global = [0.0]



def run(profile=SMOKE, data_dir="data/jale_smoke"):
    root = Path(build_ds(profile, data_dir))
    tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
    apps = tabs["applications"].copy()
    gl = tabs["guarantor_links"].copy()
    persons = tabs["persons"]
    lab = pd.read_parquet(root / "labels" / "application_labels.parquet")

    apps["application_id"] = apps["application_id"].astype(str)
    apps["ring"] = (apps["application_id"]
                    .map(lab.set_index("application_id")["ring_id"])
                    .fillna(0).astype(int))
    apps["y"] = (apps["ring"] > 0).astype(int)   # generator uses -1 for benign
    apps = apps.set_index("application_id", drop=False)
    nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
    NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
    HIST = ("n_missed", "dpd", "miss_rate", "ever_dpd30", "first_missed")

    ring_apps = {int(r): grp.index.tolist()
                 for r, grp in apps[apps.ring > 0].groupby("ring")}
    completion = {r: int(apps.loc[a, "applied_day"].max()) for r, a in ring_apps.items()}
    day_max = int(apps["applied_day"].max())
    cutoff = int(day_max * TRAIN_CUTOFF_FRAC)
    snapshots = list(range(cutoff, day_max + 1, STEP_DAYS))
    if snapshots[-1] != day_max:
        snapshots.append(day_max)
    gl_by_app = gl.groupby("application_id")["guarantor_person_id"].apply(list).to_dict()

    # ---- warm start: one model, trained on the FIRST HALF of the timeline ----
    # (a deployment always has history; the measured question is whether the
    #  two-signal alarm fires before the still-forming rings complete)
    tr = apps[apps.applied_day <= cutoff]
    g_tr = build_graph(tr, gl[gl.application_id.isin(tr.index)], persons)
    nf_tr = nf.reindex(tr.index)
    gf_tr = build_graph_features(g_tr, tr, nf_tr)
    Gtr = [c for c in gf_tr.columns if c != "ppr"]
    Xtr = nf_tr.join(gf_tr.reindex(tr.index)[Gtr], how="left")
    bad = [c for c in Xtr.columns if any(h in c for h in HIST)]
    assert not bad, f"repayment history leaked into features: {bad}"
    Xv = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    scaler = StandardScaler().fit(Xv)
    model = fit_gbt(scaler.transform(Xv), tr["y"].to_numpy(), seed=0)
    model_cols = list(Xtr.columns)   # physical cols (subset graphs can emit
    model_day = cutoff               # duplicate names -- track them literally)
    evaluable = [r for r in ring_apps if completion[r] > cutoff]

    cache: list[dict] = []
    ring_trace: dict[int, dict[int, float]] = {r: {} for r in ring_apps}

    for d in snapshots:
        sub = apps[apps.applied_day <= d]
        sub_gl = gl[gl.application_id.isin(sub.index)]

        # -- score the snapshot with the past-only model --
        g = build_graph(sub, sub_gl, persons)
        gf = build_graph_features(g, sub, nf)
        G = [c for c in gf.columns if c != "ppr"]
        X = nf.reindex(sub.index).join(gf.reindex(sub.index)[G], how="left")
        Xs = X.reindex(columns=model_cols)
        Xv = Xs.replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        scores = model.predict_proba(scaler.transform(Xv))[:, 1]

        work = sub[["applied_day", "device_id", "account_id", "person_id", "ring"]].copy()
        work["score"] = scores
        work["_comp"] = fold_groups(g).astype(str).reindex(sub.index)
        A = g.cooccurrence_union(tuple(STRONG_FOLD_RELATIONS)).tocsr()
        pos = pd.Series(np.arange(len(g.app_ids)), index=g.app_ids)
        p_hot_global[0] = float(np.percentile(scores, 95)) if len(scores) > 20 else np.inf

        raws = []
        for comp, members in work.groupby("_comp").groups.items():
            members = [str(m) for m in members]
            r = cluster_raw(members, work, gl_by_app, A, pos)
            if r:
                r["comp"] = str(comp)
                r["day"] = int(d)
                raws.append(r)
        if not raws:
            continue
        mm = _rank01(np.array([r["mean_score"] for r in raws]))
        for r, mn in zip(raws, mm):
            corrob = 0.5 * mn + 0.5 * r["frac_hot"]
            r["ring_score"] = 0.65 * corrob + 0.35 * r["structural"]
            cache.append(r)
        for r in raws:
            for rid, c in r["counts"].items():
                if c >= MIN_RING_SIZE and ring_trace[rid].get(d, 0.0) < r["ring_score"]:
                    ring_trace[rid][d] = r["ring_score"]




    evaluable = [r for r in ring_apps
                 if model_day >= 0 and completion[r] > model_day]

    def evaluate(thr: float, min_size: int) -> dict:
        detect: dict[int, int] = {}
        false_comps: set[str] = set()
        for c in cache:
            if c["n"] < min_size or c["ring_score"] < thr:
                continue
            for r, k in c["counts"].items():
                if k >= MIN_RING_SIZE and r not in detect and r in evaluable:
                    detect[r] = c["day"]
            if not c["counts"]:
                false_comps.add(c["comp"])
        leads = [completion[r] - detect[r] for r in detect]
        pre = [l for l in leads if l > 0]
        nuc = [len([a for a in ring_apps[r]
                    if apps.at[a, "applied_day"] <= detect[r]]) / len(ring_apps[r])
               for r in detect]
        return dict(detected=len(detect), before_completion=len(pre),
                    median_lead=float(np.median(pre)) if pre else None,
                    p25=float(np.percentile(pre, 25)) if pre else None,
                    p75=float(np.percentile(pre, 75)) if pre else None,
                    mean_nucleation=float(np.mean(nuc)) if nuc else None,
                    false_clusters=len(false_comps))

    sweep = []
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for ms in (3, 4, 5, 6, 8):
            row = evaluate(thr, ms)
            row.update(threshold=thr, min_size=ms)
            # operational load: distinct flagged clusters / week over horizon
            row["alerts_per_week"] = row["false_clusters"] / (day_max / STEP_DAYS)
            sweep.append(row)
    need = max(1, int(0.5 * len(evaluable))) if evaluable else 1
    # choose an honest frontier point: catch at least `need` rings before
    # completion, then minimise false alerts (analyst load)
    viable = [r for r in sweep
              if r["before_completion"] >= need and r["false_clusters"] <= 75]
    viable.sort(key=lambda r: (r["false_clusters"], -(r["median_lead"] or 0)))
    chosen = viable[0] if viable else min(sweep, key=lambda r: r["false_clusters"])

    out = dict(profile="SMOKE", n_applications=int(len(apps)),
               n_rings=len(ring_apps), n_evaluable_rings=len(evaluable),
               first_train_day=int(model_day), day_horizon=day_max,
               step_days=STEP_DAYS, n_snapshots=len(snapshots),
               chosen=chosen, sweep=sweep,
               ring_trace={str(r): {str(d): round(a, 3) for d, a in sorted(tr.items())}
                           for r, tr in ring_trace.items()},
               completion_day={str(r): int(c) for r, c in completion.items()})
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lead_time.json").write_text(json.dumps(out, indent=1))

    print(f"applications={len(apps):,}  rings={len(ring_apps)} "
          f"(evaluable after first training: {len(evaluable)})  "
          f"horizon={day_max}d  snapshots={len(snapshots)}")
    print(f"{'thr':>5} {'min_n':>5} {'det':>4} {'preCompl':>8} {'medLead':>8} "
          f"{'nuc':>5} {'false':>6}")
    for r in sweep:
        print(f"{r['threshold']:5.2f} {r['min_size']:5d} {r['detected']:4d} "
              f"{r['before_completion']:8d} "
              f"{('%.0f' % r['median_lead']) if r['median_lead'] is not None else '-':>8} "
              f"{(r['mean_nucleation'] or 0):5.0%} {r['false_clusters']:6d}")
    c = chosen
    lead_s = (f"{c['median_lead']:.0f}d (IQR {c['p25']:.0f}-{c['p75']:.0f})"
              if c["median_lead"] is not None else "n/a")
    print(f"\nCHOSEN: L4>={c['threshold']:.2f} & n>={c['min_size']} -> "
          f"{c['detected']}/{len(evaluable)} evaluable rings detected, "
          f"{c['before_completion']} BEFORE completion, median lead {lead_s}, "
          f"flagged at {(c['mean_nucleation'] or 0):.0%} of ring filed, "
          f"{c['false_clusters']} false clusters over {day_max}d "
          f"({c['alerts_per_week']:.1f}/wk)")
    return out


if __name__ == "__main__":
    run()

