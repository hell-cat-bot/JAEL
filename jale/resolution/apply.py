"""Put entity resolution *in the evaluation path*.

``scripts/run_v1.py`` builds the graph from the generator's raw ``person_id``,
which is already perfectly resolved -- so every model number it reports assumes
perfect entity resolution. In production there is no such column: the same person
appears under spelling variants, a new phone, a changed pincode. This module runs
the Fellegi--Sunter linker and rewrites ``person_id`` (in applications and in
guarantor links) to the *resolved* cluster id, so the graph downstream is the one
a real deployment would actually see -- false merges and all.

``resolved_person_map`` returns record_id -> resolved id. The 0.99-posterior
operating point means the linker under-merges rather than over-merges: a missed
merge splits one ring member into two graph nodes (recoverable by the neighbour
features), a false merge welds two strangers into a fabricated ring (not
recoverable). See ``jale/resolution/fellegi_sunter.py`` section 9.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fellegi_sunter import FellegiSunterLinker, person_comparison_schema


def resolved_person_map(persons: pd.DataFrame,
                        min_posterior: float = 0.99) -> tuple[dict[str, str], object]:
    """Return (person_id -> resolved_id, fitted_linker).

    ``resolved_id`` is ``rp_<cluster>`` for merged records and the original
    ``person_id`` otherwise, so the id space stays disjoint from the raw one and
    a diff is obvious.
    """
    df = persons.copy()
    df["dob_year"] = df["dob_year"].astype(str)
    linker = FellegiSunterLinker(person_comparison_schema(),
                                 id_column="person_id",
                                 min_posterior=min_posterior)
    res = linker.fit_predict(df)
    linker._last_threshold = float(res.threshold_used)  # for reporting
    # clusters: record_id -> cluster_id (singletons included)
    size = res.clusters.groupby("cluster_id")["record_id"].transform("size")
    mapping: dict[str, str] = {}
    for rec, cid, n in zip(res.clusters["record_id"], res.clusters["cluster_id"], size):
        mapping[str(rec)] = f"rp_{int(cid)}" if n > 1 else str(rec)
    # any person not seen by the linker maps to itself
    for pid in persons["person_id"].astype(str):
        mapping.setdefault(pid, pid)
    return mapping, linker


def apply_resolution(applications: pd.DataFrame,
                     guarantor_links: pd.DataFrame,
                     persons: pd.DataFrame,
                     mapping: dict[str, str]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rewrite ``person_id`` everywhere it identifies a customer."""
    apps = applications.copy()
    apps["person_id"] = apps["person_id"].astype(str).map(mapping).fillna(apps["person_id"])
    gl = guarantor_links.copy()
    if len(gl):
        gl["guarantor_person_id"] = (gl["guarantor_person_id"].astype(str)
                                     .map(mapping).fillna(gl["guarantor_person_id"]))
    ppl = persons.copy()
    ppl["person_id"] = ppl["person_id"].astype(str).map(mapping).fillna(ppl["person_id"])
    # collapse duplicate person rows created by a merge; keep the first
    ppl = ppl.drop_duplicates(subset="person_id", keep="first")
    return apps, gl, ppl


def merge_diagnostics(mapping: dict[str, str], identity_truth: pd.DataFrame) -> dict:
    """How many merges the linker made, and how many were wrong."""
    truth = identity_truth.set_index("person_id")["human_id"].to_dict()
    groups: dict[str, list[str]] = {}
    for pid, rid in mapping.items():
        groups.setdefault(rid, []).append(pid)
    merged = [g for g in groups.values() if len(g) > 1]
    correct = wrong = 0
    for g in merged:
        humans = {truth.get(p) for p in g}
        if len(humans) == 1:
            correct += 1
        else:
            wrong += 1
    n_records_in_wrong = sum(len(g) for g in merged
                             if len({truth.get(p) for p in g}) > 1)
    return {
        "clusters_merged": len(merged),
        "correct_merges": correct,
        "false_merges": wrong,
        "records_welded_wrongly": int(n_records_in_wrong),
    }
