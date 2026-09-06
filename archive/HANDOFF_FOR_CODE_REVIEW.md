# JA·LE — Code Review Handoff Brief

**Audience:** a code-reviewing agent (Claude Code) or human reviewer.
**Purpose:** give you enough context to review this project critically and propose fixes, without having to reverse-engineer intent from the source.
**Read order:** this file, then `doubts.md` (what we do not know), then `jale/README.md`, then the source.
**Companion:** `TVS_Credit_EPIC8_ProblemE_JALE_Implementation_Plan.docx` is the *pitch* document (written for a competition judge, plain-language first). It is not a substitute for this brief.

---

## 1. What this project is

TVS Credit E.P.I.C 8.0 competition, problem (e) *Swarm Intelligence Lending Network*.

Fraud rings file many loan applications that each look individually normal. The signal is **between** records — shared devices, shared disbursement accounts, shared guarantors — not in any single row. So we build a graph over lending entities and describe each application by its neighbourhood.

"Swarm intelligence" is a misleading title. In CS it means optimisation metaheuristics (ACO, PSO). The problem intends *swarming* = coordinated collective behaviour, which is a graph problem. This reframe is deliberate and is a scoring point.

**Novel contribution:** Layer L4, ring-level scoring — score the *ring*, not the *person*.
**Implemented** in `jale/demo/l4_rings.py` (structural formula, no learned weights, no
labels). L5 explanations (`jale/demo/explain.py`) and cold-start propagation
(`jale/demo/coldstart.py`) are implemented too, and entity resolution is now wired into
the evaluation path (`experiments/er_in_path.py`). The prototype UI is `demo/index.html`
→ `demo/jale_demo.html`. This section and §7 below predate that work — see the
2026-08-29/30 note in `doubts.md` (Session 4) for current status.

---

## 2. Verification status — the most important section in this file

Reviewing this code without knowing what has been executed will mislead you. Be precise about this:

| Component | File | Status |
|---|---|---|
| Config, profiles, typologies | `jale/config.py` | **Executed** |
| Synthetic generator | `jale/data/generator.py` | **Executed + audited** (parity, hard negatives, determinism) |
| Entity resolution (Fellegi–Sunter + EM) | `jale/resolution/fellegi_sunter.py` | **Executed + evaluated** against held-out truth |
| Graph construction, folds | `jale/graph/builder.py` | **Executed** |
| 86 features | `jale/features/builder.py` | **Executed** (2 bugs found and fixed, see §7) |
| sklearn/linear models, ring-disjoint CV, audits | `scripts/run_v1.py`, `jale/eval/*`, `jale/models/models.py` | **Executed end-to-end**, clean-slate rerun reproduces every number |
| Fold logic for public data | `jale/data/public_datasets.py::ring_disjoint_folds_from_adjacency` | **Executed** on a synthetic cluster graph; cluster-disjoint invariant verified |
| **PyTorch/PyG models** | `jale/models/torch_gnn.py` | **NEVER EXECUTED.** `py_compile` only. Syntax is valid; runtime behaviour, tensor shapes and PyG API compatibility are **entirely unverified**. |
| **Public dataset loaders** | `jale/data/public_datasets.py` (loaders) | **NEVER EXECUTED.** Schemas written from papers/dataset cards, not by inspecting files. |
| **Colab notebook** | `colab/JALE_Colab_Training.ipynb` | **NEVER RUN.** All 31 cells, 18 of them code, `compile()` — which proves nothing about execution. |
| Cross-typology generalisation | `experiments/typology_generalisation.py` | **Executed.** Mean held-out AUC-PR 0.273. See §6. |
| LR anomaly diagnosis | `experiments/diag_lr_anomaly.py` | **Executed.** Anomaly resolved; see §7. |
| Sweep + nested CV | `experiments/sweep_graph_model.py`, `experiments/nested_cv.py` | **Executed.** Reproduced inside `scripts/run_v1.py --nested`. |
| PPR propagation sweep | `experiments/improve_propagation.py` | **Executed.** No gain; cold-start works. See §7. |
| Colab-only experiments | `collab-help/01..04` | **NEVER RUN.** Syntax-checked only. Awaiting user's Colab output. |

The sandbox has **no torch, no torch_geometric, no xgboost, no lightgbm, no splink, no duckdb, no shap**, and **1 GB RAM**. Nothing torch-based can be tested here by design.

**Consequence for your review:** treat `torch_gnn.py`, the loaders, and the notebook as *draft code needing a first run*, not as working code. The verified half of the project is the scipy/sklearn pipeline.

---

## 3. Hard constraints — do not violate these

These came from the user as explicit requirements and are the reason several design choices look unusual.

1. **No cheating.** Nothing hard-coded to this dataset.
2. **No hard-coded logic that does not generalise.** Concretely: the entity-resolution threshold is *derived* from the fitted mixture, `log(π/(1−π)) + log(p/(1−p))` with `p = 0.99`, not tuned. A hard-coded threshold of 8.0 was tried and gave precision 0.452.
3. **No data leakage.** Enforced structurally, not by discipline:
   - `save_dataset(split_labels=True)` writes observables to `raw/`, ground truth to `labels/`.
   - `scripts/run_v1.py` loads **only** `raw/` and asserts no `label*`, `ring_id`, `human_id`, `is_kiosk` column exists in any frame.
   - `StandardScaler` is fitted on training rows only.
   - Folds come from unsupervised connected components; labels never touch split assignment.
4. **Quality over sandbox fit.** If something can't run here, target Colab and document — do not simplify the method.

If a proposed fix weakens any of these, it is a regression even if it improves a metric.

---

## 4. Architecture and data flow

```
jale/config.py            SMOKE (6,000 persons) / FULL (120,000); 5 ring typologies
        |
jale/data/generator.py    persons, devices, dealers, accounts, applications,
        |                 guarantor_links, emi_schedule, rings
        |                 -> save_dataset() splits raw/ vs labels/
        |
jale/resolution/fellegi_sunter.py   unsupervised identity merge (EM)
        |
jale/graph/builder.py     LenderGraph: 5 relations as scipy sparse incidence
        |                 cooccurrence_union() -> app-app affinity
        |                 fold_groups() -> unsupervised connected components
        |
jale/features/builder.py  11 node + 75 graph features (ObservationTime gated)
        |
scripts/run_v1.py         ring-disjoint GroupKFold over 4 models + 3 audits
        |
reports/v1_SMOKE.json
```

Five relations, with measured V1 sizes: `device` 2,404 nodes · `account` 3,450 · `person` 2,678 · `guarantor` 1,239 · `dealer` 172.

---

## 5. Measured results (SMOKE, ring-disjoint 5-fold CV, 18.0 s clean-slate)

3,474 applications · 131 fraud · **3.77 % base rate** · 10 rings (2 each of 5 typologies)

| Model | AUC-PR | Lift | AUC-ROC | R@1% | R@5% | R@10% |
|---|---|---|---|---|---|---|
| best single node feature (`n_guarantors`), raw | 0.233 | 6.2× | — | — | — | — |
| node-only logistic | 0.144 | 3.8× | 0.670 | 0.130 | 0.260 | 0.313 |
| node-only GBT (fixed params) | 0.243 | 6.5× | 0.571 | 0.198 | 0.214 | 0.244 |
| node-only GBT (nested CV) | 0.237 | 6.3× | 0.536 | 0.198 | 0.221 | 0.244 |
| node + graph GBT (fixed params) | 0.754 | 20.0× | 0.980 | 0.267 | 0.733 | 0.908 |
| graph-only GBT (nested CV) | 0.719 | 19.1× | 0.971 | 0.244 | 0.763 | 0.885 |
| **node + graph GBT (nested CV) — headline** | **0.717** | **19.0×** | **0.976** | 0.237 | **0.702** | 0.901 |
| node + graph, graph-reg. LR | 0.423 | 11.2× | 0.730 | 0.267 | 0.420 | 0.504 |

**The first row is the honest floor.** `n_guarantors` used as a raw score gives 6.2× lift, so
every "our model beats the simple thing" claim is measured against that rather than against
zero. `best_single_feature()` in `jale/eval/metrics.py` runs on every execution for exactly
this reason — see the `min_samples_leaf` bug in §7.

**Graph-only (0.719) matches node+graph (0.717).** Node features contribute nothing once the
neighbourhood is described numerically. Open decision: ship the smaller set, or keep node
features for explainability to underwriters.

**Report the nested number.** Fixed-param figures carry +0.037 AUC-PR of selection bias on
SMOKE, measured directly. `--nested` removes it at a cost of ~11 min.

### Audits — all pass

| Audit | Result |
|---|---|
| Shuffled-label control, node+graph GBT | AUC-PR 0.0413 vs base 0.0377 — **PASS** |
| Shuffled-label control, node-only GBT | AUC-PR 0.0392 vs base 0.0377 — **PASS** |
| Random split vs ring-disjoint | 0.854 vs 0.754 → **gap +0.100** (leakage quantified) |
| Node-level parity, person attributes | all p > 0.24 (see above) |
| Node-level parity, application attributes | **QUALIFIED** — `n_guarantors` 0.85 vs 0.40, MWU p = 7×10⁻¹⁵ |
| Entity resolution | P 0.888 / R 1.000 / F1 0.941 (TP 286, FP 36, FN 0), threshold derived at 2.04, π=0.0720 |
| Person-attribute parity | income KS p=0.70 · dob_year KS p=0.91 · employment χ² p=0.48 · state χ² p=0.24 · gender χ² p=0.66 (n=102 ring vs 6,270 benign) |
| Hard negatives present | 552 devices shared by >1 app (max 22) · 484 benign apps across 45 kiosks · 139 multi-loan guarantors (max 15) · 13 dealers >50 apps (max 589) |
| Label segregation | assertion clean; pipeline reads `raw/` only |
| Observation-time gate | no repayment-history column reaches an APPLICATION-time feature |

---

## 6. How we compare to published work — and why the comparison is weak

Verified figures from the literature (all on **random splits**):

T-Finance (39,357 nodes, 21.2 M edges, **4.6 % fraud** — close to our 3.77 %), from ConsisGAD, ICLR 2024:

| Method | AUROC | **AUPRC** | Macro-F1 |
|---|---|---|---|
| MLP | 92.17 | 52.79 | 82.33 |
| GraphSAGE | 89.42 | 49.08 | 77.62 |
| CARE-GNN | 91.45 | 72.27 | 83.68 |
| BWGNN | 93.08 | 77.79 | 86.97 |
| GAGA | 92.36 | 64.34 | 81.10 |
| ConsisGAD | 95.33 | 86.63 | 90.97 |

Ours: AUC-ROC **97.6**, AUC-PR **71.7**, ring-disjoint + nested. On a random split: AUC-ROC **98.4**, AUC-PR **85.4**.

**Four reasons this comparison cannot support a superiority claim:**

1. **Split protocol.** Every published number is a random split. Our 71.7 is ring-disjoint and nested. Like-for-like would be our 85.4 — which would beat everything in that table, and that is a red flag, not a victory.
2. **The gap is evidence against us.** Our random→ring-disjoint drop is 0.100, on top of a further 0.037 from tuning on the eval folds. It means our synthetic rings are *more cohesive and more separable* than real ones — the model can memorise a ring signature from a few members. Real rings are messier, which is why published random-split numbers are not as inflated.
3. **Self-generated data.** A model that performs well on data we wrote ourselves proves far less than one that performs well on someone else's. This is the single biggest weakness in the submission.
4. **Graph density.** T-Finance averages ~540 edges/node. Ours is far sparser. Neighbourhood aggregation has completely different behaviour at those densities.

**Honest verdict:** internally valid and well-controlled; externally unproven.

### The most uncomfortable result in this project

Train on four fraud typologies, test on the fifth (`experiments/typology_generalisation.py`,
runs in the sandbox in ~11 s):

| Held-out typology | n_test | positives | node-only | node+graph |
|---|---|---|---|---|
| identity_reuse | 296 | 11 | 0.032 | **0.889** |
| disbursement_sink | 671 | 25 | 0.055 | 0.211 |
| guarantor_star | 1,070 | 39 | 0.054 | 0.127 |
| dealer_collusion | 945 | 34 | 0.032 | 0.081 |
| device_farm | 594 | 22 | 0.045 | 0.060 |
| **mean** | | | 0.043 | **0.273** |

Mean held-out AUC-PR **0.273** against **0.754** in-distribution. The model appears to
memorise typology-specific signatures rather than learning what a ring *is*. The two
typologies that matter most in production — `device_farm` and `dealer_collusion` — are the
worst. Test sets are small (11–39 positives), so per-row numbers are noisy, but the pattern
is consistent.

Caveat worth stating: this is a harsh test — in production each typology would have labelled
examples. But it is the strongest available argument for building L4 ring scoring on
*structural* properties rather than learned feature weights, and for trying a GNN.
`collab-help/01_gnn_vs_gbt.py` measures exactly that.

---

## 7. Known bugs and open items

### Resolved during development (do not regress these)

| Bug | Fix |
|---|---|
| Fellegi–Sunter EM collapsed to a degenerate mixture (`surname` match scored −0.02, disagreement +6.17) | Marginal-frequency `u` init, EM-estimated π, monotonicity projection. **Do not revert to uniform `u`.** |
| `np.maximum.accumulate` used as a monotonicity "projection" | It is **not** a projection onto the monotone cone — it inflates values above the input maximum. Replaced with weighted isotonic regression (PAVA), `isotonic_decreasing()`, with a unit check. |
| Hard-coded `match_threshold = 8.0` → precision 0.452 | Derived from the fitted mixture. **Never hard-code an FS threshold.** |
| 40×40 name pool → 296 false merges | Distinct humans coincidentally sharing name+birth-year+gender. The *data* was unrealistic, not the algorithm. Expanded to 123×131 ≈ 16,113. |
| Shuffled-label control permuted within connected components (mostly size 1 → no-op) and reported a phantom leak at 0.242 | Permute within **true CV folds**; assert base rate preserved. |
| `nbrdiff` returned the raw value when there were no neighbours (safe division → mean 0) | Masked to zero + explicit `hasnbr_*` indicator. Without this the graph block's contribution is overstated. |
| `_time_windowed_burst` had a dead `if False else` branch and double-counted pairs | Rewritten with `np.searchsorted`, O(n log n), each pair counted once. |
| `emi_amount`/`emi_to_income` collided with the repayment-history leakage gate | Renamed `scheduled_emi_*` (they are computable at application time). |
| Callout boxes 17.19 cm on a 16.6 cm usable page | Pinned to 16.6 cm; wide tables auto-scaled. |
| **`min_samples_leaf=20` crippled the node-only baseline** — it scored 0.038 (chance), and we concluded the generator injects no per-applicant signal. That conclusion was wrong. | With 131 positives over 5 folds there are ~26 positives per fold, so a leaf needing 20 samples can barely split on the positive class. At leaf=10 node-only GBT scores 0.243. `GBT_DEFAULTS` in `jale/models/models.py` now uses leaf=10, lr=0.05. **The graph contribution is 6.3× → 19.0×, not 1.0× → 17.7×.** |
| Hyperparameters were tuned on the folds used for reporting | Added `select_gbt()` and a `--nested` flag: selection happens inside each outer training split only. Selection bias measured at +0.037 AUC-PR. |
| No guard against a misconfigured baseline flattering the model | `best_single_feature()` runs every execution and reports the strongest single column used raw. |

### Unresolved

1. **The generator has a real node-level tell: `n_guarantors`.** Fraud applications average 0.85 guarantors vs 0.40 benign (Mann–Whitney p = 7×10⁻¹⁵, univariate AUC-PR 0.233 = 6.2× lift). Root cause is *not* a hard-coded difference — the benign path draws a guarantor with p≈0.55 and rings with p≈0.35, but `guarantor_star` assigns ~2 guarantors per member by construction. So the tell is intrinsic to that typology and arguably realistic. **Open question for the reviewer: should it be reduced anyway, on the grounds that a real ring would avoid presenting it?** See `experiments/diag_lr_anomaly.py`.
2. **PPR propagation does not improve the score.** Swept k ∈ {1%, 2%, 5%} × α ∈ {0.5, 0.7, 0.85} × weight ∈ {0.3, 0.5, 1.0} on top of GBT scores: best 0.681 vs 0.667 baseline, within noise on 131 positives. Conclusion: once graph features are in the feature matrix, explicit diffusion adds nothing. **However**, cold-start propagation from only 3–10 *known* cases (no training at all) reaches AUC-PR 0.293–0.415, which is a genuinely useful analyst workflow and is not currently surfaced anywhere in the product. See `experiments/improve_propagation.py`.
3. **Entity resolution is NOT in the evaluation path.** `scripts/run_v1.py` builds the graph from the generator's raw `person_id`, which is already perfectly resolved. The Fellegi–Sunter layer is evaluated separately (precision 0.888, 36 false merges) and then never feeds the model — so **every model number reported assumes perfect ER**. The 36 false merges are the damaging kind: they fabricate an edge between unrelated customers, which is how a ring gets invented out of innocent people. Fix is cheap and should happen before any external claim. Logged as open item 1 in `doubts.md`.
4. **L4 ring-level scoring is not implemented.** Described in the plan; absent from the code. Every number reported is node-level.
5. **L5 explanation module does not exist.** Required for the live demo.
6. **`torch_gnn.py` has never run.** Specific things to check on first execution:
   - `BetaBandPass.get_coefficients` uses a hand-rolled DCT-II. Compare against the BWGNN authors' `BetaWavelet.get_filter`.
   - `BWGNN._cheb` initialises `T_prev = T_cur = x`, then for `k == 1` sets `T_next = Lx`. Standard Chebyshev on the Laplacian uses `T_0 = x`, `T_1 = Lx`, `T_k = 2LT_{k-1} - T_{k-2}`. Verify the indexing matches the filter's coefficient ordering.
   - `CAREGNNSimple._gated_agg` allocates `torch.zeros_like(x[:, :proj.out_features])` — this assumes `proj.out_features` exists and matches. Fragile.
   - `train_and_eval` splits the test fold in half for val/test (notebook §5). On a small fold this can leave too few positives for AUC-PR to be stable.
7. **`load_amazon` uses PyG's `Amazon(name='Computers')`** — a co-purchase graph, **not** the fraud graph from the anomaly-detection literature. Any number quoted against published Amazon results would be invalid. Needs replacing.
8. **No temporal / walk-forward split.** Production-realistic evaluation is missing.
9. **`ppr` is computed but excluded** from the 75 model features and unused. Either wire it into L4 or drop it.

---

## 8. Specific questions for the reviewer

Please answer these directly rather than giving general feedback:

1. **Is the ring-disjoint protocol actually airtight?** Folds come from connected components over `device`, `account`, `person`, `guarantor` — note `dealer` is deliberately **excluded** from `STRONG_FOLD_RELATIONS` (a dealer legitimately touches hundreds of unrelated customers, so including it would merge the whole portfolio). Is excluding dealer the right call, or does it leak?
2. **Is `cooccurrence_union()` the right graph?** It sums co-occurrence across all five relations, so a dealer with 589 applications creates a very dense subgraph. Should relations be kept separate (R-GCN / metapath) instead of unioned?
3. **Is the shuffled-label control sufficient?** It permutes within CV folds and preserves base rate. What leakage mode would it *miss*?
4. **Is the 0.100 random-vs-ring-disjoint gap being interpreted correctly?** We read it as "our synthetic rings are too cohesive". Alternative reading?
5. **Is the generator's camouflage realistic enough** that success on synthetic data means anything? Benign delinquency is real (7.1 % missed vs 37.4 % for rings) but gated out by `ObservationTime.APPLICATION`.
6. **`GraphRegularisedLogistic`** folds the Laplacian penalty into the ridge term and solves with Newton-IRLS. It underperforms GBT (0.423 vs 0.667). Is the formulation correct, or is λ/normalisation wrong?
7. **Which of the unresolved items is actually blocking** a credible submission, versus nice-to-have?
8. **What is the single highest-value experiment** to run on Colab first?

---

## 9. What NOT to change without discussion

- `min_posterior = 0.99` in entity resolution. This is a **cost decision**, not a tuning choice: a false merge welds two innocent strangers into a fabricated ring. Lowering it to the conventional 0.5 will improve recall and is the wrong trade.
- The `raw/` vs `labels/` split. It is the leakage defence.
- Ring-disjoint folds as the headline protocol. Reporting random-split numbers as the headline would be the single most damaging thing that could happen to this submission.
- `ObservationTime.APPLICATION` as the primary setting.

---

## 10. Reproducing

```bash
cd jale
pip install numpy pandas scipy scikit-learn pyarrow
python scripts/run_v1.py --profile SMOKE     # ~18 s, writes reports/v1_SMOKE.json
```

Expected: the four model rows and three audit results in §5, exactly. Any material difference means the environment changed — investigate before trusting anything downstream.

Colab: upload `colab/JALE_Colab_Training.ipynb` plus a zip of the project, then run top to bottom. Expect to fix `torch_gnn.py` on the first pass.
