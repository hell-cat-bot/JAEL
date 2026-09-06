"""Loaders that put public graph-fraud benchmarks into JA-LE's own representation.

Why this module exists
----------------------
Our synthetic generator is the only source we have with *ring-level* ground truth,
so it is the only place we can measure ring-disjoint performance honestly. But a
model that works on data we generated ourselves proves much less than one that
works on someone else's. This module is the external-validity track.

Every loader returns the same shape: a feature matrix, a binary label vector, a
symmetric scipy affinity matrix, and ring-disjoint fold ids derived from
connected components. That last part matters -- we apply *our* evaluation
protocol to *their* data, rather than adopting their random splits, because the
whole point is to see what happens when a model cannot memorise part of a cluster.

Sizing (measured, from the papers and dataset cards -- verify on first download)
    YelpChi     45,954 nodes   3,846,979 edges   14.5% fraud   ~200 MB   fits free Colab
    Amazon      11,944 nodes   4,398,392 edges    9.5% fraud   ~250 MB   fits free Colab
    T-Finance   39,357 nodes  21,222,543 edges    4.6% fraud     ~2 GB   needs GPU runtime
    DGraph-Fin  3,700,550 nodes  4,300,999 edges  1.3% fraud     ~1 GB   needs high RAM
    T-Social     5,781,065 nodes  73,105,578 edges 3.0% fraud    ~8 GB   Pro tier only

DGraph-Fin is the one that actually matters for this problem: its edges are
"user lists another user as an emergency contact on a loan application", which is
structurally the guarantor relation. It is also the one most likely to break a
free Colab runtime. Start with YelpChi to prove the harness works.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

def _require_torch():
    """Import torch lazily.

    The fold logic and the subsample logic are pure numpy/scipy and have no
    business requiring a GPU library. Importing torch at module level would make
    them untestable anywhere torch is absent -- which is most of this project's
    CI surface. Only the loaders that read PyG objects need it.
    """
    try:
        import torch
        from torch_geometric.utils import to_scipy_sparse_matrix
        return torch, to_scipy_sparse_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "this loader requires torch and torch_geometric (Colab only).\n"
            f"original error: {exc}"
        ) from exc


@dataclass
class PublicGraph:
    name: str
    X: np.ndarray                     # (n, d) node features
    y: np.ndarray                     # (n,) binary label
    A: sp.csr_matrix                  # (n, n) symmetric affinity
    n_classes: int = 2

    def summary(self) -> str:
        n, d = self.X.shape
        rate = float(self.y.mean())
        return (f"{self.name}: {n:,} nodes, {d} features, "
                f"{int(self.y.sum()):,} positive ({rate:.2%}), "
                f"{self.A.nnz:,} directed edges, "
                f"avg degree {self.A.nnz / max(n, 1):.1f}")


def ring_disjoint_folds_from_adjacency(A: sp.spmatrix, n_splits: int = 5,
                                       seed: int = 0) -> np.ndarray:
    """Fold ids from connected components of the graph. Labels are never read.

    Same construction we use on synthetic data: anything in one connected
    component shares a fold, so no cluster is split across train and test. On a
    real benchmark this is stricter than anything published on that dataset, and
    that is deliberate -- it is the comparison we actually want.

    A single giant component is common on real graphs and would collapse the
    split. When that happens we fall back to grouping by *2-hop* neighbourhood
    size buckets, and we say so loudly rather than silently returning one fold.
    """
    n = A.shape[0]
    n_comp, labels = sp.csgraph.connected_components(sp.csr_matrix(A), directed=False)
    sizes = np.bincount(labels)
    largest = sizes.max()

    if largest > 0.5 * n:
        print(f"  WARNING {largest/n:.1%} of nodes are in one connected component; "
              f"component-wise folding would collapse to {n_splits} near-empty folds.")
        print("  Falling back to degree-bucket grouping. This is a weaker guarantee "
              "and must be disclosed wherever the number is reported.")
        deg = np.asarray(A.sum(axis=1)).ravel()
        buckets = np.clip(np.log1p(deg).astype(int), 0, 63)
        groups = buckets
    else:
        groups = labels

    from sklearn.model_selection import GroupKFold
    fold = np.zeros(n, dtype=int)
    if len(np.unique(groups)) >= n_splits:
        for i, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(groups, groups=groups)):
            fold[te] = i
    else:
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        assign = rng.permutation(len(uniq)) % n_splits
        fold = assign[np.searchsorted(uniq, groups)]
    return fold


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _load_dgl_fraud(name: str) -> PublicGraph:
    """YelpChi / Amazon fraud graphs from DGL's ``FraudDataset``.

    IMPORTANT: these are the anomaly-detection benchmarks from Dou et al.
    (CARE-GNN, CIKM 2020) -- NOT ``torch_geometric.datasets.Yelp`` (a GraphSAINT
    multi-label node dataset, ~717k nodes) or ``Amazon(name='Computers')`` (a
    co-purchase graph). An earlier version of this file loaded those by mistake;
    any number quoted from them against published fraud results was invalid.

    We collapse DGL's multi-relation graph to a single homogeneous adjacency
    (union of all relation types), which is what our co-occurrence graph is too.
    """
    try:
        import dgl
        from dgl.data import FraudDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("YelpChi / Amazon-fraud need DGL: pip install dgl "
                          f"(Colab only). {exc}") from exc
    ds = FraudDataset(name, train_size=0.4, val_size=0.1)
    gdgl = ds[0]
    y = gdgl.ndata["label"].numpy().astype(int)
    x = gdgl.ndata["feature"].numpy()
    n = len(y)
    A = sp.csr_matrix((n, n), dtype=np.float32)
    for et in gdgl.canonical_etypes:
        u, v = gdgl.edges(etype=et)
        A = A + sp.coo_matrix((np.ones(len(u), np.float32),
                               (u.numpy(), v.numpy())), shape=(n, n)).tocsr()
    A = A.maximum(A.T); A.data[:] = 1.0
    A.setdiag(0.0); A.eliminate_zeros()
    return PublicGraph({"yelp": "YelpChi", "amazon": "Amazon-Fraud"}[name], x, y, A)


def load_yelpchi(root: str = "/content/data") -> PublicGraph:
    """YelpChi review-fraud graph (Rayana & Akoglu; Dou et al. benchmark).

    ~45k reviews, 14.5% fraud -- much easier than a lending book, so it proves
    the harness runs, nothing more.
    """
    return _load_dgl_fraud("yelp")


def load_amazon(root: str = "/content/data") -> PublicGraph:
    """Amazon review-fraud graph (Dou et al.). ~11.9k users, 9.5% fraud."""
    return _load_dgl_fraud("amazon")


def load_dgraphfin_pyg(root: str = "/content/data") -> PublicGraph:
    """DGraph-Fin via PyG. Edges are 'user lists another as an emergency contact
    on a loan application' -- structurally the guarantor relation, from a real
    consumer lender (Finvolution). The one public dataset whose edge semantics
    match this problem. ~3.7M nodes: subsample before a GPU run."""
    torch, to_scipy_sparse_matrix = _require_torch()
    from torch_geometric.datasets import DGraphFin
    data = DGraphFin(root=root)[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    A = A.maximum(A.T); A.setdiag(0.0); A.eliminate_zeros()
    y = data.y.numpy().astype(int)
    # PyG labels: 0/1 valid, 2/3 = background/unlabelled -> treat >1 as negative-unknown
    return PublicGraph("DGraph-Fin", data.x.numpy(), (y == 1).astype(int), A)


def load_bwgnn_pt(path: str, name: str = "T-Finance") -> PublicGraph:
    """T-Finance / T-Social, distributed as .pt files by the BWGNN authors.

    Expected keys: 'feature', 'label', 'edge_index' (or 'edges'). Downloaded from
    the authors' Google Drive links in github.com/squareRoot3/Rethinking-Anomaly-Detection.
    """
    torch, _ = _require_torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    x = _to_np(blob.get("feature", blob.get("x")))
    y = _to_np(blob.get("label", blob.get("y"))).astype(int).ravel()
    ei = blob.get("edge_index", blob.get("edges"))
    ei = _to_np(ei).astype(np.int64)
    if ei.ndim == 1:                       # some dumps use a flat pair encoding
        ei = ei.reshape(2, -1)
    n = len(y)
    A = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T)                     # force symmetry, drop direction
    A.setdiag(0.0); A.eliminate_zeros()
    return PublicGraph(name, x, y, A)


def load_dgraphfin(path: str) -> PublicGraph:
    """DGraph-Fin (Finvolution Group / Xinye). 3.7M nodes -- handle with care.

    Distributed as a .npz with 'x', 'y', 'edge_index'. y has three classes
    (0 = unlabelled, 1 = fraud, 2 = normal); we keep only the labelled rows and
    remap, which is what published evaluations do. Subsampling is strongly
    advised before this touches a GPU runtime.
    """
    blob = np.load(path)
    x, y, ei = blob["x"], blob["y"], blob["edge_index"].astype(np.int64)
    keep = y != 0
    idx = np.flatnonzero(keep)
    remap = -np.ones(len(y), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    both = keep[ei[0]] & keep[ei[1]]
    ei = remap[ei[:, both]]
    n = len(idx)
    A = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0); A.eliminate_zeros()
    ylab = (y[keep] == 1).astype(int)
    print(f"  DGraph-Fin: dropped {int((~keep).sum()):,} unlabelled nodes "
          f"({(~keep).mean():.1%} of the graph)")
    return PublicGraph("DGraph-Fin", x[keep], ylab, A)


def subsample(g: PublicGraph, frac: float, seed: int = 0,
              keep_all_positive: bool = True) -> PublicGraph:
    """Random node subsample, optionally keeping every positive.

    Fraud is rare enough that uniform subsampling would leave too few positives
    to train on. Keeping all positives while subsampling negatives changes the
    base rate, which must be reported alongside any metric computed afterwards.
    """
    rng = np.random.default_rng(seed)
    n = len(g.y)
    pos = np.flatnonzero(g.y == 1)
    neg = np.flatnonzero(g.y == 0)
    keep = np.concatenate([pos if keep_all_positive else
                           rng.choice(pos, max(int(len(pos) * frac), 1), replace=False),
                           rng.choice(neg, max(int(len(neg) * frac), 1), replace=False)])
    keep = np.sort(keep)
    remap = -np.ones(n, dtype=np.int64); remap[keep] = np.arange(len(keep))
    A = g.A.tocoo()
    m = np.isin(A.row, keep) & np.isin(A.col, keep)
    A2 = sp.coo_matrix((A.data[m], (remap[A.row[m]], remap[A.col[m]])),
                       shape=(len(keep), len(keep))).tocsr()
    A2 = A2.maximum(A2.T); A2.setdiag(0.0); A2.eliminate_zeros()
    return PublicGraph(f"{g.name}[sub {frac:.0%}]", g.X[keep], g.y[keep], A2)


def _to_np(v):
    if "Tensor" in type(v).__name__ and hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


AVAILABLE = {
    "yelpchi": load_yelpchi,          # DGL FraudDataset('yelp')
    "amazon": load_amazon,            # DGL FraudDataset('amazon')
    "dgraphfin": load_dgraphfin_pyg,  # PyG DGraphFin -- the guarantor-relation match
}
