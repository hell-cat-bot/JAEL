"""Entity resolution in the evaluation path -- measure the delta.

doubts.md threat #3 / HANDOFF open item 3: every model number assumes perfect
entity resolution because the graph is built from the generator's raw
``person_id``. This runs the pipeline twice -- once on raw ids, once on
Fellegi--Sunter-resolved ids -- and reports the gap. That gap is the honest cost
of not having a perfect identity column.

    PYTHONPATH=. python experiments/er_in_path.py
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.eval.metrics import full_metrics
from jale.features.builder import build_graph_features, build_node_features
from jale.graph.builder import build_graph, fold_groups
from jale.models.models import fit_gbt
from jale.resolution.apply import (apply_resolution, merge_diagnostics,
                                   resolved_person_map)

N_SPLITS = 5


def score(apps, gl, persons, y):
    g = build_graph(apps, gl, persons)
    groups = fold_groups(g).reindex(g.app_ids).to_numpy()
    nf = build_node_features(apps, None, ObservationTime.APPLICATION)
    gf = build_graph_features(g, apps, nf)
    node_cols = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
    g_cols = [c for c in gf.columns if c != "ppr"]
    X = nf.reindex(g.app_ids)[node_cols].join(gf.reindex(g.app_ids)[g_cols], how="left")
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    yy = pd.Series(y, index=list(range(len(y))))  # placeholder
    y_aligned = (pd.Series(y, index=apps["application_id"])
                 .reindex(g.app_ids).to_numpy())
    oof = np.zeros(len(y_aligned))
    for tr, te in GroupKFold(N_SPLITS).split(Xv, y_aligned, groups=groups):
        sc = StandardScaler().fit(Xv[tr])
        m = fit_gbt(sc.transform(Xv[tr]), y_aligned[tr], seed=0)
        oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return full_metrics(y_aligned, oof), len(np.unique(groups))


def main():
    root = Path(build_ds(SMOKE, "data/jale_smoke"))
    tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
    apps, gl, persons = tabs["applications"], tabs["guarantor_links"], tabs["persons"]
    lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
    truth = pd.read_parquet(root / "labels" / "person_identity_truth.parquet")
    y = (lab.set_index("application_id")["ring_id"]
            .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)

    print("=== raw person_id (perfect ER, what V1 reports) ===")
    m_raw, ng_raw = score(apps, gl, persons, y)
    print(f"  AUC-PR={m_raw['auc_pr']:.4f}  lift={m_raw['lift_pr']:.1f}x  "
          f"R@5%={m_raw['recall_at_5pct']:.3f}  fold groups={ng_raw}")

    print("\n=== Fellegi-Sunter resolved ids (what a deployment sees) ===")
    mapping, linker = resolved_person_map(persons)
    diag = merge_diagnostics(mapping, truth)
    print(f"  merges: {diag['clusters_merged']}  correct={diag['correct_merges']}  "
          f"false={diag['false_merges']}  records welded wrongly={diag['records_welded_wrongly']}")
    a2, g2, p2 = apply_resolution(apps, gl, persons, mapping)
    m_res, ng_res = score(a2, g2, p2, y)
    print(f"  AUC-PR={m_res['auc_pr']:.4f}  lift={m_res['lift_pr']:.1f}x  "
          f"R@5%={m_res['recall_at_5pct']:.3f}  fold groups={ng_res}")

    d = m_res["auc_pr"] - m_raw["auc_pr"]
    print(f"\n  delta AUC-PR = {d:+.4f}  "
          f"({'ER cost is small' if abs(d) < 0.03 else 'ER materially changes the result'})")


if __name__ == "__main__":
    main()
