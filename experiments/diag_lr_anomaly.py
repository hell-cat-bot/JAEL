"""Why does node-only LR score 0.144 (3.8x lift) while node-only GBT scores at chance?

Three competing hypotheses:
  H1  Extrapolation artefact: LR extrapolates past the training range on unbounded
      features; GBT cannot. -> winsorising / rank-transform should collapse LR to chance.
  H2  Real application-level signal that GBT misses at 131 positives.
      -> per-attribute tests should be significant.
  H3  Small-sample noise; the number is unstable. -> seeds should move it a lot.

Note the earlier "node-level parity" audit was run on PERSONS (income, dob_year,
employment, state, gender). It never tested APPLICATION-level attributes such as
loan_amount, down_payment or tenure. That gap is what this script closes.
"""
import sys; sys.path.insert(0, "/home/user/jale")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_logistic, fit_gbt

root = Path(build_ds(SMOKE, "data/jale_smoke"))
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
y = (lab.set_index("application_id")["ring_id"]
        .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
X = nf.reindex(g.app_ids)[NODE]
groups = fold_groups(g).to_numpy()

print("=== H2 TEST: is there real APPLICATION-level node signal? ===")
print("(the earlier parity audit was on PERSONS, not applications)")
for c in NODE:
    v = X[c].to_numpy(dtype=float)
    a, b = v[y == 1], v[y == 0]
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    d = (a.mean() - b.mean()) / (v.std() + 1e-12)
    flag = "   <-- SIGNIFICANT" if p < 0.01 else ""
    print(f"  {c:29s} fraud={a.mean():11.2f} benign={b.mean():11.2f} "
          f"d={d:+.3f} MWU p={p:.2e}{flag}")

print("\n=== H1 TEST: transform the inputs, does LR collapse to chance? ===")
def cv(Xin, kind, seed=0):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xin, y, groups=groups):
        Xv = Xin.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        sc = StandardScaler().fit(Xv[tr])
        Xtr, Xte = sc.transform(Xv[tr]), sc.transform(Xv[te])
        if kind == "lr":
            oof[te] = fit_logistic(Xtr, y[tr]).decision_function(Xte)
        else:
            oof[te] = fit_gbt(Xtr, y[tr], seed=seed).predict_proba(Xte)[:, 1]
    return full_metrics(y, oof)

lo, hi = X.quantile(0.01), X.quantile(0.99)
print(f"  LR raw             AUC-PR={cv(X, 'lr')['auc_pr']:.4f}")
print(f"  LR rank-transform  AUC-PR={cv(X.rank(pct=True), 'lr')['auc_pr']:.4f}")
print(f"  LR winsorised 1/99 AUC-PR={cv(X.clip(lower=lo, upper=hi, axis=1), 'lr')['auc_pr']:.4f}")

print("\n=== H3 TEST: seed stability of the GBT baseline ===")
s = [cv(X, "gbt", seed=k)["auc_pr"] for k in range(5)]
print(f"  GBT across 5 seeds: {[f'{v:.4f}' for v in s]}  mean={np.mean(s):.4f} "
      f"sd={np.std(s):.4f}")

print("\n=== per-feature univariate AUC (node features) ===")
for c in NODE:
    auc = roc_auc_score(y, X[c].to_numpy(float))
    print(f"  {c:29s} AUC={max(auc, 1 - auc):.4f}  (raw {auc:.4f})")
