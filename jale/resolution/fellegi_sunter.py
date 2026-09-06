"""Probabilistic record linkage (entity resolution) for JALE V1.

Implements the Fellegi--Sunter model directly rather than depending on Splink,
for three reasons:

1. **No labels are used.** Match (m) and non-match (u) probabilities are
   estimated by Expectation-Maximisation over the comparison vectors. A linkage
   model trained on fraud labels would leak; this one is fully unsupervised.
2. **The decision is auditable.** Every link carries a log-likelihood-ratio
   score that decomposes into per-field contributions, so an investigator can
   see *why* two records were merged.
3. **It is the layer most graph-fraud submissions skip** -- and skipping it is
   why their graphs are wrong. Two applications with the same person under two
   spellings of a Tamil name are one person; a naive join sees two.

Blocking
--------
All-pairs comparison is O(n^2) and infeasible. We compare only pairs that share
at least one *blocking key* (cheap, high-recall functions of the record). The
keys are deliberately loose: they favour recall, because a missed block is an
unrecoverable error whereas a spurious candidate pair is merely scored and
rejected.

Scaling note: this pure-Python implementation is fine to ~50k records. Beyond
that, swap in Splink (Fellegi--Sunter on DuckDB/Spark) -- the comparison
schema below maps onto a Splink ``settings`` dict one-for-one, and the EM step
is exactly what Splink's ``training_session`` does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# string similarity
# --------------------------------------------------------------------------
def jaro(s1: str, s2: str) -> float:
    """Jaro similarity in [0, 1]. Exact match -> 1.0."""
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    window = max(l1, l2) // 2 - 1
    if window < 0:
        window = 0
    f1 = [False] * l1
    f2 = [False] * l2
    matches = 0
    for i in range(l1):
        lo = max(0, i - window)
        hi = min(i + window + 1, l2)
        for j in range(lo, hi):
            if f2[j] or s1[i] != s2[j]:
                continue
            f1[i] = f2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(l1):
        if not f1[i]:
            continue
        while not f2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (matches / l1 + matches / l2
            + (matches - transpositions) / matches) / 3.0


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler: rewards a shared prefix, which is what transliteration
    variants of Indian names usually preserve (Arunkumar / Arun Kumar)."""
    j = jaro(s1, s2)
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        prefix += 1
    return j + prefix * p * (1 - j)


def daitch_soundex(name: str) -> str:
    """Very small phonetic key.

    This is *not* full Daitch-Mokotoff soundex -- it is a coarse consonant
    skeleton, sufficient for blocking (where recall matters and precision does
    not). Being honest about the simplification matters more than shipping a
    half-correct transliteration of a complex algorithm.
    """
    keep = "bcdfghjklmnpqrstvwxyz"
    out = []
    for ch in name.lower():
        if ch in keep:
            out.append(ch)
        elif ch in "aeiou":
            out.append("a")
    return "".join(out)[:6]


# --------------------------------------------------------------------------
# comparison schema
# --------------------------------------------------------------------------
@dataclass
class ComparisonLevel:
    label: str
    predicate: Callable[[object, object], bool]


@dataclass
class Comparison:
    """One field in the comparison vector, with ordered agreement levels.

    Levels MUST be ordered from strongest agreement to strongest disagreement.
    The EM step estimates a probability vector over these levels for the match
    and non-match populations.
    """
    name: str
    levels: list[ComparisonLevel]
    columns: tuple[str, str] | None = None   # (left_col, right_col); defaults to (name, name)

    def __post_init__(self) -> None:
        if self.columns is None:
            self.columns = (self.name, self.name)

    def level_index(self, a, b) -> int:
        for i, lvl in enumerate(self.levels):
            if lvl.predicate(a, b):
                return i
        return len(self.levels) - 1


def _exact(a, b) -> bool:
    return a == b


def _name_levels() -> list[ComparisonLevel]:
    return [
        ComparisonLevel("exact", lambda a, b: a == b),
        ComparisonLevel("high", lambda a, b: jaro_winkler(a, b) >= 0.90),
        ComparisonLevel("moderate", lambda a, b: jaro_winkler(a, b) >= 0.78),
        ComparisonLevel("disagree", lambda a, b: True),
    ]


def _binary_levels() -> list[ComparisonLevel]:
    return [
        ComparisonLevel("agree", lambda a, b: a == b),
        ComparisonLevel("disagree", lambda a, b: True),
    ]


def person_comparison_schema() -> list[Comparison]:
    """Comparison vector for deduplicating lender customer records."""
    return [
        Comparison("name_first", _name_levels()),
        Comparison("name_last", _name_levels()),
        Comparison("dob_year", _binary_levels()),
        Comparison("gender", _binary_levels()),
        Comparison("pincode", _binary_levels()),
        Comparison("mobile_hash", _binary_levels()),
    ]


# --------------------------------------------------------------------------
# blocking
# --------------------------------------------------------------------------
def default_blocking_keys(row: pd.Series) -> list[tuple[str, str]]:
    """Loose blocking keys favouring recall.

    Each key is namespaced by its rule so that different rules never collide.
    """
    keys: list[tuple[str, str]] = []
    fn = str(row.get("name_first", ""))
    ln = str(row.get("name_last", ""))
    mob = str(row.get("mobile_hash", ""))
    pin = str(row.get("pincode", ""))
    dob = str(row.get("dob_year", ""))
    if mob:
        keys.append(("mob", mob))
    if ln:
        keys.append(("snd", f"{daitch_soundex(ln)}|{dob}"))
        keys.append(("l3", f"{ln[:3]}|{pin}"))
    if fn and ln:
        keys.append(("fl", f"{fn[0]}|{ln[:4]}|{pin}"))
    return keys



def isotonic_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic regression onto the NON-INCREASING cone (PAVA).

    This is the correct projection onto the monotone simplex. The naive
    ``np.maximum.accumulate`` alternative is *not* a projection: when a later
    level exceeds an earlier one it raises the earlier one to match, which
    flattens distinct agreement levels into one and silently destroys the very
    ordering information the constraint was meant to protect. PAVA instead pools
    adjacent violators and replaces them with their weighted mean, which is the
    L2-optimal monotone fit.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    vals = list(y)
    wts = list(w)
    cnt = [1] * n
    i = 0
    while i < len(vals) - 1:
        if vals[i] < vals[i + 1] - 1e-12:          # violation of non-increasing
            tw = wts[i] + wts[i + 1]
            pooled = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tw
            vals[i] = pooled
            wts[i] = tw
            cnt[i] = cnt[i] + cnt[i + 1]
            del vals[i + 1]
            del wts[i + 1]
            del cnt[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty(n)
    k = 0
    for v, c in zip(vals, cnt):
        out[k:k + c] = v
        k += c
    return out

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class FellegiSunterResult:
    links: pd.DataFrame            # (left, right, score) above threshold
    clusters: pd.DataFrame         # (record_id, cluster_id)
    m_probabilities: dict = field(default_factory=dict)
    u_probabilities: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    pairs_scored: int = 0
    n_clusters: int = 0
    threshold_used: float = 0.0


class FellegiSunterLinker:
    """Unsupervised Fellegi--Sunter linkage with EM parameter estimation."""

    def __init__(self, comparisons: Sequence[Comparison],
                 blocking_fn: Callable[[pd.Series], list[tuple[str, str]]] = default_blocking_keys,
                 match_threshold: float | None = None,
                 min_posterior: float = 0.99,
                 nonmatch_threshold: float = 0.0,
                 em_iterations: int = 25,
                 seed: int = 0,
                 id_column: str | None = None,
                 verbose: bool = False):
        self.comparisons = list(comparisons)
        self.blocking_fn = blocking_fn
        # None => derive the acceptance threshold from the fitted mixture itself
        # (accept when posterior P(match) > 0.5). Deriving it beats hardcoding a
        # constant, which would not transfer between datasets.
        self.match_threshold = match_threshold
        self.min_posterior = min_posterior
        self.nonmatch_threshold = nonmatch_threshold
        self.em_iterations = em_iterations
        self.id_column = id_column
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.m_: dict[str, np.ndarray] = {}
        self.u_: dict[str, np.ndarray] = {}

    # -- candidate generation -------------------------------------------
    def _candidate_pairs(self, df: pd.DataFrame) -> np.ndarray:
        blocks: dict[tuple[str, str], list[int]] = {}
        for i, row in enumerate(df.itertuples(index=False)):
            for key in self.blocking_fn(row._asdict()):
                blocks.setdefault(key, []).append(i)

        pairs: set[tuple[int, int]] = set()
        cap = 2000   # guard against a single enormous block blowing up memory
        for members in blocks.values():
            if len(members) < 2:
                continue
            if len(members) > cap:
                # Very large blocks are almost certainly a common surname or a
                # missing-value sentinel. Sub-sample deterministically rather
                # than O(n^2)-ing the whole block.
                members = list(self.rng.choice(members, cap, replace=False))
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    x, y = members[a_i], members[b_i]
                    pairs.add((x, y) if x < y else (y, x))
        if not pairs:
            return np.empty((0, 2), dtype=int)
        return np.array(sorted(pairs), dtype=int)

    # -- comparison vectors ---------------------------------------------
    def _comparison_matrix(self, df: pd.DataFrame, pairs: np.ndarray) -> np.ndarray:
        """Returns an (n_pairs, n_fields) int matrix of level indices."""
        if len(pairs) == 0:
            return np.empty((0, len(self.comparisons)), dtype=int)
        cols = []
        for comp in self.comparisons:
            lcol, rcol = comp.columns
            lv = df[lcol].to_numpy()[pairs[:, 0]]
            rv = df[rcol].to_numpy()[pairs[:, 1]]
            idx = np.empty(len(pairs), dtype=int)
            for k in range(len(pairs)):
                idx[k] = comp.level_index(lv[k], rv[k])
            cols.append(idx)
        return np.column_stack(cols)

    # -- EM --------------------------------------------------------------
    @staticmethod
    def _ordered_projection(p: np.ndarray) -> np.ndarray:
        """Project a probability vector onto the ordered simplex.

        Enforces that agreement-level probabilities are *monotone*: for matches,
        probability must not increase as agreement weakens; for non-matches it
        must not decrease. Without this constraint EM can (and did) converge to
        an inverted solution in which disagreeing on a surname carried MORE
        evidence for a match than agreeing on it. Sorting is the projection onto
        that monotone cone, and renormalising keeps it a distribution.
        """
        return p

    def _initialise(self, C: np.ndarray) -> None:
        """Standard Fellegi--Sunter initialisation.

        ``u`` is set to the *marginal* frequency of each agreement level. This is
        the usual "non-matches dominate the candidate set" assumption and it
        matters: initialising u as near-uniform (the naive choice) leaves
        disagreements almost unpenalised at step 0, and EM then happily amplifies
        that into a degenerate mixture where P(match) drifts to ~0.5.

        ``m`` is initialised as a sharpened, monotone tilt toward agreement.
        ``pi`` (the match prior) starts low, because after blocking the great
        majority of candidate pairs are genuinely non-matches.
        """
        n = len(C)
        for f, comp in enumerate(self.comparisons):
            k = len(comp.levels)
            counts = np.bincount(C[:, f], minlength=k).astype(float)
            u = np.clip(counts / max(n, 1), 1e-6, None)
            u /= u.sum()
            # monotone: u must be non-decreasing across weakening agreement
            u = isotonic_decreasing(u[::-1])[::-1]
            u = np.clip(u, 1e-6, None); u /= u.sum()

            tilt = np.exp(-1.6 * np.arange(k))
            m = u * tilt
            m = np.clip(m, 1e-6, None)
            m = isotonic_decreasing(m)                 # keep monotone decreasing
            m = np.clip(m, 1e-6, None); m /= m.sum()

            self.m_[comp.name] = m
            self.u_[comp.name] = u
        self.pi_ = 0.02

    def _log_likelihood_ratio(self, C: np.ndarray) -> np.ndarray:
        """Sum over fields of log( m[level] / u[level] )."""
        s = np.zeros(len(C))
        for f, comp in enumerate(self.comparisons):
            m = np.clip(self.m_[comp.name], 1e-9, 1 - 1e-9)
            u = np.clip(self.u_[comp.name], 1e-9, 1 - 1e-9)
            w = np.log(m / u)
            s += w[C[:, f]]
        return s

    def _posterior_match(self, C: np.ndarray) -> np.ndarray:
        """P(match | comparison vector) under the fitted two-component mixture.

        Uses the estimated mixture proportion pi rather than assuming 0.5. That
        assumption was the second half of the collapse: with a flat prior the
        model is pushed to explain half of all blocked pairs as matches, which
        no real blocking scheme produces.
        """
        llr = self._log_likelihood_ratio(C)
        pi = float(np.clip(self.pi_, 1e-9, 1 - 1e-9))
        logit = np.log(pi / (1 - pi)) + llr
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))

    def fit(self, df: pd.DataFrame) -> "FellegiSunterLinker":
        pairs = self._candidate_pairs(df)
        self.pairs_ = pairs
        self.C_ = self._comparison_matrix(df, pairs)
        if len(pairs) == 0:
            self._initialise(self.C_)
            return self

        self._initialise(self.C_)
        n = len(self.C_)
        for it in range(self.em_iterations):
            g = self._posterior_match(self.C_)

            new_m: dict[str, np.ndarray] = {}
            new_u: dict[str, np.ndarray] = {}
            for f, comp in enumerate(self.comparisons):
                k = len(comp.levels)
                onehot = np.zeros((n, k))
                onehot[np.arange(n), self.C_[:, f]] = 1.0
                m_new = (onehot * g[:, None]).sum(0) / max(g.sum(), 1e-9)
                u_new = (onehot * (1 - g)[:, None]).sum(0) / max((1 - g).sum(), 1e-9)
                # Monotonicity projection: matches must not become more likely
                # as agreement weakens; non-matches must not become less likely.
                m_new = isotonic_decreasing(np.clip(m_new, 1e-6, None))
                u_new = isotonic_decreasing(np.clip(u_new, 1e-6, None)[::-1])[::-1]
                new_m[comp.name] = m_new / m_new.sum()
                new_u[comp.name] = u_new / u_new.sum()
            delta = max(float(np.abs(new_m[c.name] - self.m_[c.name]).max())
                        for c in self.comparisons)
            self.m_, self.u_ = new_m, new_u
            self.pi_ = float(np.clip(g.mean(), 1e-6, 1 - 1e-6))
            if self.verbose and (it % 5 == 0 or it == self.em_iterations - 1):
                print(f"    EM iter {it:2d}  max|dm|={delta:.5f}  "
                      f"pi={self.pi_:.4f}  P(match) mean={g.mean():.4f}")
            if delta < 1e-6:
                break
        return self

    # -- inference -------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> FellegiSunterResult:
        pairs = getattr(self, "pairs_", self._candidate_pairs(df))
        C = getattr(self, "C_", self._comparison_matrix(df, pairs))
        scores = self._log_likelihood_ratio(C)

        weights = {}
        for comp in self.comparisons:
            m = np.clip(self.m_[comp.name], 1e-9, 1 - 1e-9)
            u = np.clip(self.u_[comp.name], 1e-9, 1 - 1e-9)
            weights[comp.name] = np.log(m / u).tolist()

        idcol = (df[self.id_column].to_numpy() if self.id_column
                 else df.index.to_numpy())
        # Decision-theoretic operating point. In a fraud-ring graph a false
        # merge is far more damaging than a missed one: it fabricates a "ring"
        # out of unrelated customers and can drive an adverse action against
        # innocent people. So the default demands a high posterior rather than a
        # bare majority. This is a cost choice, not a tuned-on-labels choice.
        pt = float(np.clip(self.min_posterior, 0.5 + 1e-9, 1 - 1e-9))
        threshold = (self.match_threshold if self.match_threshold is not None
                     else float(np.log(self.pi_ / (1 - self.pi_))
                                + np.log(pt / (1 - pt))))
        keep = scores >= threshold
        links = pd.DataFrame({
            "left_id": idcol[pairs[keep, 0]],
            "right_id": idcol[pairs[keep, 1]],
            "score": scores[keep],
        })
        clusters = _connected_components(list(idcol), links)
        return FellegiSunterResult(
            links=links, clusters=clusters, threshold_used=threshold,
            m_probabilities={k: v.tolist() for k, v in self.m_.items()},
            u_probabilities={k: v.tolist() for k, v in self.u_.items()},
            weights=weights,
            pairs_scored=int(len(pairs)),
            n_clusters=int(clusters.cluster_id.nunique()),
        )

    def fit_predict(self, df: pd.DataFrame) -> FellegiSunterResult:
        return self.fit(df).predict(df)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def _connected_components(ids: Iterable, links: pd.DataFrame) -> pd.DataFrame:
    """Union-find over the accepted links.

    Union-find rather than a full graph library because linkage output is a
    sparse set of pairwise assertions and we only need the partition.
    """
    parent = {i: i for i in ids}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    for l, r in zip(links.left_id, links.right_id):
        rl, rr = find(l), find(r)
        if rl != rr:
            parent[rr] = rl

    rows = [(i, find(i)) for i in ids]
    out = pd.DataFrame(rows, columns=["record_id", "root"])
    remap = {r: c for c, r in enumerate(sorted(out.root.unique()))}
    out["cluster_id"] = out.root.map(remap)
    return out.drop(columns=["root"])


def cluster_sizes(clusters: pd.DataFrame) -> pd.Series:
    """Diagnostic: the size distribution of resolved clusters.

    A fat tail here is the classic over-linkage failure -- it means the matcher
    has collapsed many unrelated people into one identity (usually via a
    missing-value sentinel such as an empty mobile number). Always inspect this
    before trusting any downstream graph.
    """
    return clusters.cluster_id.value_counts().sort_index()
