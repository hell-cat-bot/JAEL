# JA·LE — Catch fraud ring before it is finished forming.

Built for the **TVS Credit E.P.I.C 8.0 IT Challenge, Problem (e) — Swarm
Intelligence Lending Network**: an AI-driven collective-intelligence platform
that identifies hidden relationships across loan applications, devices,
dealers, accounts, mobiles, locations, guarantors and payment behaviour to
**predict emerging fraud ecosystems before fraud occurs**.

JA·LE is a **rerunnable, leak-audited fraud-ecosystem early-warning engine** for
dealer-sourced retail lending. Every number in `PROPOSAL/JALE_Round2_deck.pptx`
traces back to a script in this repository or a verified public source.

---

## What we measure (the honest receipt)

| Metric | Value | Where it comes from |
|---|---|---|
| Ring-detection AUCPR (node + graph, ring-disjoint 5-fold CV) | **0.754** | `colab/JALE_Colab_Training_ran.ipynb` §4 |
| Honest nested-CV range | 0.68 – 0.75 | notebook §6b |
| Lift over random | **20.0×** | notebook §4 |
| Best single feature (`n_guarantors`) alone | 6.2× | notebook §4 |
| GraphSAGE GNN (same folds, same features) | 0.688 | notebook §5 |
| Leakage gap if you use a naive split (synthetic / real graph) | +0.100 / **+0.022** | notebook §6–7 |
| **Formation lead time (before the ring's last application)** | **median ~204 days, 0 false alarms** | `experiments/lead_time.py` |
| Cross-typology hold-out (why we flag novels, not "rings") | 0.273 AUCPR | `experiments/typology_generalisation.py` |
| Relation ablation: drop account / drop dealer | −0.22 / −0.10 AUCPR | notebook §8a |
| Prevalence & camouflage stress (D16) | 0.70–0.75 across 2–4.5% rings; 0.71 at ×2 camouflage | `experiments/ring_rate_stress.py` |
| Business-case anchors (audited AR Note 18) | target book Rs 19,029 cr @ 2.12% NPA; dealer advances **0.23% NPA** | `jale/data/tvs_data.py` (`verify()`) |

All synthetic data, all internal-validity results — stated up front, never
hidden. This is the point, not the caveat: the same protocol was applied to a
**real** public graph (YelpChi) and the measured leakage gap transfers.

> **Reproducibility note.** The notebook's 0.754 was executed on Colab
> (sklearn 1.8.x); a fresh local run on the pinned requirements gives 0.729 —
> the documented ±0.02–0.03 spread of HistGradientBoosting across sklearn
> builds (see `docs/doubts.md` D11). The *shape* never moves: graph ≫ node,
> ring-disjoint ≪ random, controls at chance, typology collapse.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_v1.py --profile SMOKE     # ~30 s — pipeline + leakage audits
python experiments/lead_time.py              # the "before fraud occurs" lead-time number
python PROPOSAL/make_charts.py               # regenerates all deck charts
python PROPOSAL/build_deck.py                # regenerates the PPTX
```

Colab (GPU, optional): `colab/JALE_Colab_Training_ran.ipynb` — contains the
executed GNN/benchmark runs. See the notebook's status table before trusting
any number in it.

## Repository map

| Path | What it is |
|---|---|
| `PROPOSAL/` | Round-2 deck builder + slide spec + chart pack + financial model |
| `experiments/` | The receipts: lead-time backtest, typology generalisation, ER-in-path, nested CV |
| `research/` | Evidence ledger (claim → source → confidence), case card |
| `colab/` | Executed notebook: GNNs on same folds, YelpChi/real-data protocol, ablations, leakage audits |
| `jale/` | The pipeline (generator · entity resolution · graph · features · models) |
| `demo/` | `jale_demo.html` — the analyst queue (ring-level scoring, evidence, cold start) |
| `docs/` | `doubts.md` — a running log of what we don't know, with verdicts |

## Why this is not "the usual graph + GNN deck"

1. **It runs.** ~30 s, one command, pinned environment. The Colab notebook was
   actually executed; its failures (e.g., BWGNN 0.072 — an implementation bug)
   are disclosed, not laundered.
2. **It is evaluated honestly.** Ring-disjoint CV, shuffled-label null, nested
   CV, and a measured leakage gap on both synthetic *and real* data.
3. **It says "before fraud occurs" with a number.** A formation lead time
   measured by a rolling backtest that only ever sees the past.
4. **It respects rural customers.** Creditworthiness ≠ fraud risk; dealer
   guilt is tested, not presumed; DPDP-safe by construction.
5. **Three verified citations, each mapped to a component.** No fabricated
   references.

The data has an honest ceiling: it is synthetic (no public dataset has
lending-ring ground truth) — which is exactly why the rollout plan begins with
a shadow backtest on TVS's own history, not with blocking decisions.

**One-paragraph pitch:** Most fraud-ring solutions detect after disbursement
and blame the busiest node. JA·LE scores the *cluster*, measures it under
leak-free evaluation, and — trained only on the past — flags forming
ecosystems a median of ~7 months before they finish filing, with zero false
alarms at its strict setting. Same folds, same features, honest about models
that lose, and every number regenerable in 30 seconds.
