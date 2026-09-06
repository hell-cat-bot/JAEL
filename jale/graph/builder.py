"""Heterogeneous graph construction for JALE V1.

Prediction unit
---------------
The **application**. That is the decision a lender actually makes, and it is the
object that can be blocked, held or repriced.

Graph shape
-----------
A bipartite incidence graph over five node types:

    application -- APPLIED_BY    --> person
    application -- USED_DEVICE   --> device
    application -- SOURCED_BY    --> dealer
    application -- DISBURSED_TO  --> account
    application -- GUARANTEED_BY --> person (the guarantor)
    person      -- SAME_HOUSEHOLD--> person

The application--application relational signal is obtained by *projection* over
these shared entities. Projecting rather than materialising a dense
application-application graph matters: it keeps the edge set linear in the
number of shared-entity incidences instead of quadratic, and it lets every
relational feature be attributed to a specific named relation ("these two
applications share a device"), which is what makes the output explainable.

No labels are read anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


# The relations that can link two applications. Each entry maps a relation name
# to (source table, source key, target node type, target key).
RELATIONS: dict[str, dict[str, str]] = {
    "device":    {"table": "applications",     "key": "device_id",   "node": "device"},
    "dealer":    {"table": "applications",     "key": "dealer_id",   "node": "dealer"},
    "account":   {"table": "applications",     "key": "account_id",  "node": "account"},
    "person":    {"table": "applications",     "key": "person_id",   "node": "person"},
    "guarantor": {"table": "guarantor_links",  "key": "guarantor_person_id",
                  "node": "person", "app_key": "application_id"},
}

# Which relations are "strong" identity links for the purposes of constructing
# evaluation folds. Household and dealer links are deliberately EXCLUDED: a
# dealer legitimately serves thousands of unrelated customers, so folding on
# dealer would put almost the whole portfolio in one fold.
STRONG_FOLD_RELATIONS = ("device", "account", "person", "guarantor")


@dataclass
class LenderGraph:
    """Incidence structure plus the bookkeeping needed to invert it."""
    app_ids: pd.Index                     # prediction units, in canonical order
    node_index: dict[str, dict[str, int]]  # relation -> {node id -> column}
    incidence: dict[str, sp.csr_matrix]    # relation -> (n_apps x n_nodes)
    household_groups: pd.Series            # person_id -> household_id

    def n_apps(self) -> int:
        return len(self.app_ids)

    def relations(self) -> list[str]:
        return list(self.incidence)

    def cooccurrence_union(self, relations: tuple[str, ...] | None = None) -> sp.csr_matrix:
        """Application-to-application affinity over the chosen relations.

        Two applications are linked when they touch a common node on any of the
        given relations -- the projection of the bipartite incidence structure
        onto applications. The graph Laplacian, label propagation and the
        ring-disjoint folds all operate on this matrix.

        Weights accumulate across relations, so a pair sharing both a device and
        a guarantor scores higher than one sharing only a device.
        """
        rels = relations or tuple(self.relations())
        A: sp.csr_matrix | None = None
        for rel in rels:
            C = cooccurrence(self.incidence[rel])
            A = C if A is None else (A + C)
        if A is None:
            raise ValueError("no relations to union")
        A = A.tocsr()
        A.setdiag(0.0)
        A.eliminate_zeros()
        return A


def build_graph(applications: pd.DataFrame,
                guarantor_links: pd.DataFrame,
                persons: pd.DataFrame) -> LenderGraph:
    """Build the incidence matrices.

    Uses ``csr_matrix`` coordinate construction rather than networkx: at
    portfolio scale (millions of applications) a networkx graph does not fit in
    memory, whereas the sparse incidence representation does and still supports
    every operation we need.
    """
    app_ids = pd.Index(applications["application_id"].to_numpy(),
                       name="application_id")
    app_pos = pd.Series(np.arange(len(app_ids)), index=app_ids)

    incidence: dict[str, sp.csr_matrix] = {}
    node_index: dict[str, dict[str, int]] = {}

    for rel, spec in RELATIONS.items():
        if rel == "guarantor":
            src = guarantor_links
            app_key = spec.get("app_key", "application_id")
            node_key = spec["key"]
        else:
            src = applications
            app_key = "application_id"
            node_key = spec["key"]

        if src is None or len(src) == 0:
            incidence[rel] = sp.csr_matrix((len(app_ids), 0))
            node_index[rel] = {}
            continue

        sub = src[[app_key, node_key]].dropna()
        sub = sub[sub[app_key].isin(app_pos.index)]
        nodes = pd.Index(pd.unique(sub[node_key]))
        npos = pd.Series(np.arange(len(nodes)), index=nodes)

        rows = app_pos.loc[sub[app_key]].to_numpy()
        cols = npos.loc[sub[node_key]].to_numpy()
        data = np.ones(len(rows), dtype=np.float32)
        # Duplicate (app, node) incidences are collapsed by summing then binarising.
        M = sp.coo_matrix((data, (rows, cols)),
                          shape=(len(app_ids), len(nodes))).tocsr()
        M.data[:] = 1.0
        incidence[rel] = M
        node_index[rel] = {n: i for n, i in zip(nodes, np.arange(len(nodes)))}

    hh = persons.set_index("person_id")["household_id"] \
        if "household_id" in persons.columns else pd.Series(dtype=int)

    return LenderGraph(app_ids=app_ids, node_index=node_index,
                       incidence=incidence, household_groups=hh)


# --------------------------------------------------------------------------
# projection helpers
# --------------------------------------------------------------------------
def neighbour_counts(M: sp.csr_matrix) -> np.ndarray:
    """For each application, how many *other* applications share this node.

    ``M @ M.T`` gives, for each pair of applications, the number of shared nodes
    of this relation. Taking the row sums and subtracting the diagonal (self)
    yields the per-application co-occurrence count.
    """
    deg = np.asarray(M.sum(axis=1)).ravel()          # nodes per app
    node_deg = np.asarray(M.sum(axis=0)).ravel()     # apps per node
    # sum over shared nodes of (apps on that node - 1)
    counts = np.asarray(M.multiply(node_deg).sum(axis=1)).ravel() - deg
    return np.maximum(counts, 0.0)


def cooccurrence(M: sp.csr_matrix) -> sp.csr_matrix:
    """Sparse application x application shared-node counts for one relation."""
    C = (M @ M.T).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return C


def fold_groups(graph: LenderGraph) -> pd.Series:
    """Unsupervised connected components over STRONG relations -> fold ids.

    This is the mechanism that makes evaluation honest. Splitting applications
    at random lets members of the same ring land in both train and test; the
    model then memorises the ring and reports a score that means nothing.

    The groups are derived from the *observed graph only* -- no labels are used,
    so the split itself cannot leak. Fraud rings are densely connected clusters,
    so component-wise folding separates them as a side effect of separating
    anything densely connected.

    Household links are included via the person relation's household grouping,
    because a household is a genuine unit that must not straddle folds.
    """
    n = graph.n_apps()
    parent = list(range(n))

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for rel in STRONG_FOLD_RELATIONS:
        M = graph.incidence.get(rel)
        if M is None or M.shape[1] == 0:
            continue
        Mc = M.tocsc()
        for j in range(Mc.shape[1]):
            rows = Mc.indices[Mc.indptr[j]:Mc.indptr[j + 1]]
            if len(rows) < 2:
                continue
            if len(rows) > 500:
                # A node touched by hundreds of applications is a kiosk or a
                # missing-value sentinel, not an identity link. Folding on it
                # would collapse the portfolio into one giant fold, which
                # destroys the point of the exercise.
                continue
            first = int(rows[0])
            for r in rows[1:]:
                union(first, int(r))

    roots = np.array([find(i) for i in range(n)])
    remap = {r: k for k, r in enumerate(np.unique(roots))}
    return pd.Series([remap[r] for r in roots], index=graph.app_ids,
                     name="fold_group")
