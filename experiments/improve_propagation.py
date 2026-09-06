"""Two things:

(A) Quantify the `n_guarantors` node-level tell found by diag_lr_anomaly.py.
    Fraud applications average 0.85 guarantors vs 0.40 benign (MWU p=7e-15,
    univariate ROC-AUC 0.671). Root cause is NOT a hard-coded difference:
      - benign path:  P(guarantor) ~ 0.55 x (0.7 if a household sibling exists)
      - ring path:    0.35 for four typologies, but guarantor_star assigns
                      ~2 guarantors per member by construction.
    So the tell is intrinsic to the guarantor_star typology, which is legitimate
    -- that fraud type genuinely is about guarantor abuse. But it does falsify
    the claim "the generator injects no node-level signal". Measure how much.

(B) Test the highest-value model improvement: two-stage PPR propagation.
    Score with GBT, then diffuse suspicion along the graph with personalised
    PageRank seeded by the top-k scoring applications, then rescore. Fraud is
    clustered, so a partially-detected ring should pull in its undetected
    members. This is the L4 contribution in its simplest form.
"""
import sys; sys.path.insert(0, "/home/user/jale")
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from jale.config import SMOKE, ObservationTime
from jale.data.generator import build as build_ds
from jale.graph.builder import build_graph, fold_groups
from jale.features.builder import build_node_features, build_graph_features
from jale.eval.metrics import full_metrics
from jale.models.models import fit_gbt, personalised_pagerank_score

root = Path(build_ds(SMOKE, "data/jale_smoke"))
tabs = {f.stem: pd.read_parquet(f) for f in sorted((root / "raw").glob("*.parquet"))}
apps = tabs["applications"]
g = build_graph(apps, tabs["guarantor_links"], tabs["persons"])
A = g.cooccurrence_union()
lab = pd.read_parquet(root / "labels" / "application_labels.parquet")
y = (lab.set_index("application_id")["ring_id"]
        .reindex(apps["application_id"]).fillna(0).to_numpy() > 0).astype(int)
nf = build_node_features(apps, tabs["emi_schedule"], ObservationTime.APPLICATION)
gf = build_graph_features(g, apps, nf)
NODE = [c for c in nf.columns if not c.endswith(("_freq", "_code"))]
G = [c for c in gf.columns if c != "ppr"]
X_node = nf.reindex(g.app_ids)[NODE]
X_all = X_node.join(gf.reindex(g.app_ids)[G], how="left")
groups = fold_groups(g).to_numpy()

print("=== (A) how strong is the n_guarantors tell? ===")
for col in ["n_guarantors"]:
    v = X_node[col].to_numpy(float)
    auc = roc_auc_score(y, v)
    m = full_metrics(y, v)
    print(f"  {col} used as a raw score: ROC-AUC={auc:.4f} AUC-PR={m['auc_pr']:.4f} "
          f"(base {m['base_rate']:.4f}) lift={m['lift_pr']:.1f}x")
print("  -> compare node-only GBT AUC-PR=0.0376 (chance). The tree model is NOT")
print("     exploiting a feature with ROC-AUC 0.67, so the node-only baseline is")
print("     understated rather than the data being clean.")

print("\n=== (B) two-stage PPR propagation ===")
def prep(X, fit_mask):
    Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    return StandardScaler().fit(Xv[fit_mask]).transform(Xv)

def cv_base(X, y, groups):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
        m = np.zeros(len(y), bool); m[tr] = True
        oof[te] = fit_gbt(prep(X, m)[tr], y[tr], seed=0).predict_proba(prep(X, m)[te])[:, 1]
    return oof

s_base = cv_base(X_all, y, groups)
print(f"  stage 1 (GBT only)              AUC-PR={full_metrics(y, s_base)['auc_pr']:.4f}")

# Propagation uses only the model's own scores and the graph -- no labels.
# Seeded by the top-k scored applications, which is what an analyst would do.
for k_pct in (0.01, 0.02, 0.05):
    for alpha in (0.5, 0.7, 0.85):
        k = max(int(k_pct * len(y)), 1)
        seed_idx = np.argsort(-s_base)[:k]
        seed = np.zeros(len(y)); seed[seed_idx] = s_base[seed_idx]
        prop = personalised_pagerank_score(A, seed, alpha=alpha, iters=50)
        # combine: keep the original score, add the diffused evidence
        for w in (0.3, 0.5, 1.0):
            comb = s_base / max(s_base.max(), 1e-12) + w * prop / max(prop.max(), 1e-12)
            m = full_metrics(y, comb)
            print(f"  k={k_pct:.0%} alpha={alpha:.2f} w={w:.1f}   "
                  f"AUC-PR={m['auc_pr']:.4f}  R@5%={m['recall_at_5pct']:.3f}")

print("\n=== propagation alone (no GBT), seeded from a tiny known-fraud budget ===")
# Realistic cold start: an analyst confirms a handful of cases, then asks
# "who else does the graph implicate?". Uses labels only as the seed budget,
# which is the honest framing for this experiment.
rng = np.random.default_rng(0)
pos = np.flatnonzero(y == 1)
for nseed in (3, 5, 10):
    seed = np.zeros(len(y)); seed[rng.choice(pos, nseed, replace=False)] = 1.0
    prop = personalised_pagerank_score(A, seed, alpha=0.85, iters=50)
    m = full_metrics(y, prop)
    print(f"  {nseed} known cases seeded -> AUC-PR={m['auc_pr']:.4f} "
          f"R@5%={m['recall_at_5pct']:.3f}")
