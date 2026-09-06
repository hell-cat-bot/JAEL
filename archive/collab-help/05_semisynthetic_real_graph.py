"""Semi-synthetic evaluation: inject known rings into a REAL graph.

Why this exists
---------------
Every ring-level number we have comes from a portfolio we generated ourselves, so
we chose the fraud mechanism. That is the single biggest weakness in the project
and no public dataset fixes it directly, because none of them carry ring labels.

This is the next best thing. Take a real graph -- its actual node features, its
actual degree distribution, its actual community structure, its real benign
background -- and inject synthetic rings with known ground truth into it.

What that buys us:
  * The benign background is no longer ours. Threat #1 in doubts.md, reduced.
  * Graph density becomes realistic (T-Finance averages ~540 edges/node). Threat #6.
  * Feature distributions and correlations are real, so feature camouflage has to
    work against real statistics rather than against a distribution we wrote.

What it does NOT buy us:
  * The injected rings are still ours. This is not a substitute for TVS's own
    labelled cases, and it must never be described as one.

The key design rule
-------------------
Rings must be injected using the SAME mechanism as the real graph's structure, not
on top of it. Concretely: ring members are chosen from the real graph and then
given *extra* edges among themselves, drawn from the real graph's own degree
distribution. If we instead wired them with an obviously different pattern, the
model would detect our wiring rather than the ring, and the whole exercise would
be circular.

Run on Colab. Needs a real dataset; YelpChi is the smallest and loads
automatically.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from jale.eval.metrics import full_metrics, best_single_feature
from jale.data.public_datasets import load_yelpchi, subsample, ring_disjoint_folds_from_adjacency
from jale.models.models import fit_gbt


def inject_rings(A: sp.csr_matrix, n_rings: int, rng: np.random.default_rng,
                 size_lo: int = 8, size_hi: int = 20, extra_edges: int = 6,
                 camouflage: float = 0.0):
    """Add ring structure onto a real graph. Return (new_A, labels, ring_ids).

    Members are sampled from high-degree real nodes, because that is where a real
    fraud ring would hide -- a hub is the last place you would look. Members then
    receive `extra_edges` additional connections among themselves, so the ring is
    a *densification* of existing structure rather than a foreign pattern.

    `camouflage` adds benign nodes to each ring's neighbourhood without labelling
    them, so ring membership and unusualness come apart.
    """
    A = sp.coo_matrix(A).tocsr()
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    # sample from the upper half of the degree distribution
    pool = np.flatnonzero(deg >= np.median(deg))

    rows, cols, labels, ring_of = [], [], np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    used = set()
    rid = 0
    for _ in range(n_rings):
        k = int(rng.integers(size_lo, size_hi + 1))
        cand = [p for p in pool if p not in used]
        if len(cand) < k:
            break
        members = list(rng.choice(cand, k, replace=False))
        used.update(members)
        rid += 1
        for m in members:
            labels[m] = 1
            ring_of[m] = rid
        # extra intra-ring edges, undirected
        for _ in range(extra_edges * k // 2):
            a, b = rng.choice(members, 2, replace=False)
            rows += [a, b]; cols += [b, a]
        if camouflage > 0:
            n_cam = int(k * camouflage)
            if n_cam:
                cam = [c for c in np.flatnonzero(deg > 0) if c not in used]
                if len(cam) >= n_cam:
                    for c in rng.choice(cam, n_cam, replace=False):
                        a = int(rng.choice(members))
                        rows += [a, c]; cols += [c, a]

    extra = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    newA = (A + extra).tocsr()
    newA.data[:] = 1.0          # keep it unweighted and simple
    newA.setdiag(0.0); newA.eliminate_zeros()
    return newA, labels, ring_of


def neighbourhood_features(A: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
    """Graph features computed from the adjacency alone.

    Deliberately NOT importing jale.features.builder: that module is built around
    our own relational tables (device/dealer/account/person/guarantor), which a
    public benchmark does not have. These are the structural equivalents, and they
    are the ones that should transfer.
    """
    A = sp.csr_matrix(A, dtype=float)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    safe = np.where(deg > 0, deg, 1.0)
    feats = {"degree": deg}

    # co-occurrence: how many neighbours my neighbours have (2-hop reach)
    A2 = (A @ A).tocsr(); A2.setdiag(0.0)
    feats["two_hop"] = np.asarray(A2.sum(axis=1)).ravel()
    feats["clustering"] = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / safe

    # neighbour feature statistics -- the transferable part of our feature set
    for j in range(min(X.shape[1], 8)):
        v = X[:, j].astype(float)
        v = np.nan_to_num(v, nan=float(np.nanmean(v)) if np.isfinite(v).any() else 0.0)
        nbr_mean = np.asarray(A @ v).ravel() / safe
        has = (deg > 0).astype(float)
        feats[f"nbrmean_{j}"] = nbr_mean * has
        feats[f"nbrstd_{j}"] = np.sqrt(
            np.maximum(np.asarray(A @ (v ** 2)).ravel() / safe - nbr_mean ** 2, 0.0)) * has
        feats[f"nbrdiff_{j}"] = (v - nbr_mean) * has

    # component size
    ncomp, lab = sp.csgraph.connected_components(A, directed=False)
    sizes = np.bincount(lab)
    feats["component_size"] = sizes[lab].astype(float)

    return np.column_stack([feats[k] for k in feats])


def main():
    print("=== loading a REAL graph ===")
    g = load_yelpchi()
    print(" ", g.summary())
    if g.X.shape[0] > 60_000:
        g = subsample(g, 0.25)
        print("  subsampled:", g.summary())

    X = np.nan_to_num(g.X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    print(f"  real degree: mean {np.asarray(g.A.sum(1)).mean():.1f}, "
          f"median {np.median(np.asarray(g.A.sum(1))):.0f}")
    print(f"  real labels: {int(g.y.sum())} positive ({g.y.mean():.2%}) -- "
          f"UNUSED below except for the comparison at the end")

    print("\n=== injecting rings with known ground truth ===")
    rng = np.random.default_rng(0)
    A2, y_ring, ring_of = inject_rings(g.A, n_rings=max(n // 400, 20), rng=rng,
                                       camouflage=0.5)
    print(f"  injected {int(ring_of.max())} rings, {int(y_ring.sum())} members "
          f"({y_ring.mean():.2%})")
    print(f"  edges before {g.A.nnz:,} -> after {A2.nnz:,}")

    folds = ring_disjoint_folds_from_adjacency(A2, n_splits=5)
    print("  fold sizes:", np.bincount(folds))

    print("\n=== does our structural feature set detect them on a REAL graph? ===")
    Gf = neighbourhood_features(A2, X)
    nm, auc = best_single_feature(y_ring, Gf)
    print(f"  honest floor: best single structural feature {nm} "
          f"AUC-PR={auc:.4f} (base {y_ring.mean():.4f})")

    for lbl, XX in [("graph features only", Gf),
                    ("graph + real node features", np.hstack([Gf, X]))]:
        oof = np.zeros(len(y_ring))
        for tr, te in GroupKFold(n_splits=5).split(XX, y_ring, groups=folds):
            sc = StandardScaler().fit(XX[tr])      # training rows only
            oof[te] = fit_gbt(sc.transform(XX[tr]), y_ring[tr], seed=0).predict_proba(
                sc.transform(XX[te]))[:, 1]
        m = full_metrics(y_ring, oof)
        print(f"  {lbl:28s} AUC-PR={m['auc_pr']:.4f} lift={m['lift_pr']:.1f}x "
              f"R@5%={m['recall_at_5pct']:.3f}")

    print("\n=== control: the REAL labels, same features, same folds ===")
    print("  (different target, so this is a sanity check not a comparison)")
    yr = g.y.astype(int)
    if 0 < yr.sum() < len(yr):
        Xc = np.hstack([neighbourhood_features(g.A, X), X])
        oof = np.zeros(len(yr))
        for tr, te in GroupKFold(n_splits=5).split(Xc, yr, groups=folds):
            sc = StandardScaler().fit(Xc[tr])
            oof[te] = fit_gbt(sc.transform(Xc[tr]), yr[tr], seed=0).predict_proba(
                sc.transform(Xc[te]))[:, 1]
        m = full_metrics(yr, oof)
        print(f"  real YelpChi labels         AUC-PR={m['auc_pr']:.4f} "
              f"lift={m['lift_pr']:.1f}x R@5%={m['recall_at_5pct']:.3f}")

    print("\nDONE. Paste everything back verbatim, including tracebacks.")
    print("If a shape error appears it is a bug in this script -- record it, do not")
    print("work around it silently.")


if __name__ == "__main__":
    main()
