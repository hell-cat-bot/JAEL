# doubts.md — a running log of what we do not know

Living document. Every session that produces a doubt, a challenge to our own
results, or a threat to validity gets recorded here with the evidence and a
verdict. The point is that a reader can see what we questioned, what we tested,
and what is still open — rather than only seeing the numbers we are happy with.

Format: **doubt → what we did → verdict → what is still open.**

---

# Session 1 — 2026-08-27

Prompted by the question: *"the results can't be blindly trusted; after Colab will
we be more sure; and is the current version actually better or was the earlier one
more robust?"*

## D1. Why the results cannot be blindly trusted

Nine threats, ordered by how much they should worry us. The first three are
structural and will not be fixed by running anything.

### 1. The data is self-generated (the dominant problem)

We wrote the generator, so we chose the fraud mechanism. A model that succeeds on
data we designed demonstrates internal consistency, not real-world capability.

This is not hypothetical. We found a concrete instance: `n_guarantors` carries
6.2× lift on its own (fraud 0.85 vs benign 0.40, Mann–Whitney p=7×10⁻¹⁵), and it
got there through the *construction* of one typology, not through any deliberate
design decision. It was found by accident while chasing an unrelated anomaly. If
one tell leaked in unintentionally and we caught it by luck, the reasonable
assumption is that others are present and uncaught.

**Verdict:** unfixable with public data. No public dataset has ring-level ground
truth. This is a permanent ceiling on how strongly the claim can be made.

### 2. Cross-typology generalisation fails

Measured: train on four typologies, test on the fifth → mean AUC-PR **0.273**
against **0.754** in-distribution. Worst on `device_farm` (0.060) and
`dealer_collusion` (0.081), which are the two patterns that matter most.

**Verdict:** direct evidence that the headline number overstates what the system
does. The model learns the five signatures we taught it, not what a ring is.
Reproduce with `experiments/typology_generalisation.py` (~11 s).

### 3. Entity resolution is not in the evaluation path

`scripts/run_v1.py` builds the graph from the generator's raw `person_id`, which
is already perfectly resolved. The Fellegi–Sunter layer is evaluated separately
(precision 0.888, **36 false merges**) and then never feeds the model.

So every model number assumes perfect entity resolution. In production it would
not have it, and the 36 false merges are the damaging kind — they fabricate an
edge between two unrelated customers, which is precisely how a ring gets
invented out of innocent people.

**Verdict:** real gap, optimistic by an unknown amount. **Not yet measured.**
Fixing it means running the model on the ER-resolved graph and reporting the
delta. This is cheap and should be done before any external claim.

### 4. 131 positives is a small number

Per-fold positive counts are 22, 10, 43, 13, 43. Bootstrap 95% CI on the
fixed-parameter AUC-PR (0.754) is **[0.688, 0.815]** — roughly ±0.06 — and the
nested headline of 0.717 sits near the bottom of that range, so the two estimates
are within each other's noise. The typology experiment is worse, with 11–39
positives per held-out set, so its individual rows are very noisy even though the
overall pattern is clear.

**Verdict:** the point estimates are softer than they look. Always quote the CI.

### 5. The improvement is concentrated in one fold

See D3 below for the full test. Per-fold deltas were +0.461, +0.076, +0.029,
+0.009, −0.001. One fold carries almost the entire gain, and the paired t-test
across 5 folds gives t=1.31 against a critical value of 2.78 — not significant.

### 6. Graph density is nothing like real fraud graphs

T-Finance averages ~540 edges per node; ours is far sparser. Neighbourhood
aggregation behaves completely differently at those densities, so no GNN result
transfers between the two.

### 7. We choose how hard the problem is

The camouflage rate is our own knob and we have never swept it. Until
`collab-help/04` runs, we do not know where the system stops working.

### 8. No temporal validation

Production would be walk-forward: train on the past, score the future. We
evaluate with random ring-disjoint folds, which is honest about rings but silent
about drift.

### 9. One dataset, one seed

A single generator run. We have no estimate of how much the results would move if
the portfolio were generated differently.

---

## D2. Will running the Colab version make us more sure?

**Partially — and it matters which script.** Being precise about this, because
"we ran it on Colab" is not the same as "we validated it".

| Script | What it would actually settle | What it would NOT settle |
|---|---|---|
| `02_real_benchmarks.py` | Whether our protocol and feature machinery work on **somebody else's** graph. This is the only one that touches threat #1. | Anything about **ring** detection — those datasets have no ring labels, so ring-disjoint folding and L4 scoring cannot be evaluated there. |
| `01_gnn_vs_gbt.py` | Whether a GNN beats gradient boosting on identical folds, and — more interestingly — whether a GNN generalises across typologies better than 0.273. | Still synthetic data. Does not touch threat #1. |
| `03_full_scale.py` | Engineering only: memory, runtime, whether the lift survives 120k persons. | Nothing about validity. |
| `04_camouflage_sweep.py` | Where the system breaks. Bounds the claim honestly. | Still synthetic. |

**So, honestly:**

- **Yes** for threat #1, but only for the *machinery*, not for ring detection. If
  our ring-disjoint protocol on YelpChi or DGraph-Fin produces numbers that sit
  sensibly below the published random-split figures, that is genuine evidence the
  approach is not an artefact of our generator.
- **Yes** for threat #2 — script 01 tests cross-typology generalisation with a
  different model class. This is the most scientifically interesting experiment
  available.
- **Yes** for threat #7.
- **No** for threats #3, #8, #9. Those need work here, not on Colab.

Two more cautions. `torch_gnn.py` has **never been executed**, so the first
Colab run is partly a debugging exercise rather than a measurement — expect shape
errors and record them. And `load_amazon` currently loads PyG's co-purchase
graph, which is *not* the fraud graph used in the anomaly-detection literature;
any Amazon number from it would be invalid against published results.

**The one thing that would settle the central question** — does this detect real
Indian lending fraud rings — is TVS's own labelled data. Nothing public can
substitute, and no amount of Colab will change that.

---

## D3. Is the current version better, or was the earlier one more robust?

Split into two separate claims, because the evidence is very different for each.

### The node-only fix (0.038 → 0.243) is definitely real

Not an opinion. Six configurations across three model families:

| Configuration | Node-only AUC-PR |
|---|---|
| GBT `leaf=20` (**OLD**) | **0.0355** |
| GBT `leaf=10` (NEW) | 0.2416 |
| GBT depth=3, leaf=5 | 0.2471 |
| GBT depth=2, leaf=5 | 0.2387 |
| RandomForest depth=4 | 0.2305 |
| ExtraTrees depth=6 | 0.2439 |

Five independent configurations land in 0.231–0.247. Exactly one lands at 0.036.
If `leaf=20` were merely suboptimal, the others would scatter around it. They do
not — they cluster tightly an order of magnitude above it. `leaf=20` was
structurally broken: with ~26 positives per fold, a leaf demanding 20 samples
cannot split on the positive class.

**Verdict: real. The old configuration was a defect, not a conservative choice.**

### The node+graph improvement (0.667 → 0.717) is directionally real but its size is uncertain

| Test | Result | Reading |
|---|---|---|
| Per-fold deltas | +0.461, +0.076, +0.029, +0.009, −0.001 | One fold carries nearly all of it |
| Paired t across 5 folds | t=1.31, df=4, critical 2.78 | **Not significant** at p=0.05 |
| Bootstrap, old | 95% CI [0.591, 0.739] | |
| Bootstrap, new | 95% CI [0.688, 0.815] | CIs overlap |
| Bootstrap, difference | +0.086, 95% CI [+0.049, +0.129] | Excludes zero |
| P(new > old) over 2000 resamples | **1.0000** | Ranking is robust |

The two tests disagree in spirit and both are correct. The bootstrap says the
*ranking* is stable — the new config beat the old on every one of 2000
resamples. The paired fold test says the *magnitude* is not established, because
fold-to-fold variance (sd 0.196) swamps the mean gain, and 5 folds is very low
power.

**Verdict:** the direction is safe to claim, the size is not. Quote it as
"consistently better, by somewhere in the region of 0.05–0.09 AUC-PR", not as
"+0.087".

### Was the earlier version more robust?

**No — but the question was sharp and it caught something.**

The earlier version was not conservative, it was broken: a baseline that scores
at chance because it is configured so it cannot look is worse than useless,
because it manufactures a false contrast ("the graph supplies all 17.7×") that
would not have survived a competent reviewer.

However, the instinct behind the question was right in one specific way. The old
headline of 0.667 was produced by untuned defaults and therefore carried **no
selection bias**. My first attempt at the improved version reported 0.754, which
*did* carry selection bias, because I swept hyperparameters on the folds I then
reported on. Had that gone out, it would have been exactly the "false show" the
question was worried about. It was caught and the reported number is now the
nested figure, 0.717.

So: the current version is better in substance (a real defect fixed, verified
across six model configurations) and better in method (nested CV, a
best-single-feature floor, and the typology failure published rather than buried).
It is not a false show. But the honest summary of the change is **"we fixed a
broken baseline and stopped over-claiming"**, not **"the model got much better"**.

---

## Open items carried forward

| # | Item | Why it matters | Where |
|---|---|---|---|
| 1 | **Put entity resolution in the evaluation path** and report the delta | Threat #3. Cheap, and currently every model number assumes perfect ER | sandbox, ~30 min |
| 2 | Add bootstrap CIs to `run_v1.py` output | Threat #4. Point estimates alone mislead | sandbox |
| 3 | Run `collab-help/01` and `02` | Threats #1, #2 | Colab |
| 4 | Run `collab-help/04` | Threat #7 | Colab |
| 5 | Decide whether to reduce the `n_guarantors` tell | Threat #1 instance. Arguably realistic, but it is a node-level shortcut | judgement call |
| 6 | Temporal / walk-forward split | Threat #8 | sandbox |
| 7 | Regenerate the dataset with several seeds and report the spread | Threat #9 | sandbox |

---

## Rules adopted from this session

1. **Never report a point estimate without its CI** when positives number in the
   low hundreds.
2. **Never sweep hyperparameters on the folds used for reporting.** Use `--nested`.
3. **Always report the best-single-feature floor.** A baseline at chance is a
   bug until proven otherwise.
4. **Publish the failure.** The cross-typology collapse (0.273) stays in every
   document. A submission reporting only good numbers deserves to be discounted.
5. **Distinguish "we ran it" from "we validated it."** Executing code on Colab is
   not the same as establishing external validity.

---

# Session 2 — 2026-08-27

Prompted by: *"if ring data isn't available online, why would TVS care — doesn't it
mean they'd have to change how they store their data? Does that make the proposal
useless? Does Colab run only on synthetic data? What are our chances of winning?
And what do these numbers actually mean?"*

## D4. "TVS would need ring labels, so the proposal is useless"

**This doubt is answered directly by the problem statement, and the answer is no.**

Problem (e) verbatim, from the PDF:

> Create an AI-driven collective intelligence platform that continuously learns and
> identifies hidden relationships across **loan applications, device fingerprints,
> dealer networks, bank accounts, mobile numbers, locations, guarantors, and
> payment behaviours** to predict emerging fraud ecosystems before fraud occurs.

That list is our graph. Mapping it field by field:

| TVS's words | Our table | Do they already store it? |
|---|---|---|
| loan applications | `applications` | Yes — it is their core system of record |
| device fingerprints | `device_id` | Yes — stated in the problem |
| dealer networks | `dealer_id` | Yes — origination is mandatory for NBFC audit |
| bank accounts | `account_id` | Yes — KYC and disbursement require it |
| mobile numbers | `mobile_hash` | Yes — stated in the problem |
| locations | `pincode` / `district` / `state` | Yes — stated in the problem |
| guarantors | `guarantor_links` | Yes — a contractual field on the loan agreement |
| payment behaviours | `emi_schedule` | Yes — it is their collections system |

**TVS is describing data they already hold.** They wrote that sentence. There is no
new data-collection exercise here, and no schema change: what is missing is a
*graph over* those fields plus entity resolution across applications, which is a
system to build rather than a way of storing data to change.

Independent confirmation that lenders do exactly this: **DGraph-Fin** is a
production graph from Finvolution Group, a real consumer lender, whose edges are
"one user lists another as an emergency contact on a loan application" — the
guarantor relation. A real lender built this graph and published it.

### The part of the doubt that *is* valid

Conflating two different things is the source of the worry, and separating them
dissolves most of it:

- **Ring labels in the data store** — nobody needs these. No NBFC stores a
  `ring_id` column.
- **Ring labels for training** — needed eventually, but the system *produces* the
  rings; humans confirm them. That confirmation is a byproduct of the existing
  investigations workflow, not new data entry.

The real weakness is therefore **cold start**: on day one there are few confirmed
ring labels, so a supervised L4 cannot train. Three mitigations, two of them
already built and one of them already measured:

1. **Most of the pipeline is unsupervised by design.** L1 entity resolution (EM,
   no labels), L2 graph construction, L3 all 86 features, and the ring-disjoint
   folds use no labels at all.
2. **Cold-start propagation works.** Measured: seeded with only 3, 5 or 10 known
   cases and *no training whatsoever*, personalised PageRank reaches AUC-PR 0.293
   / 0.394 / 0.415. That is a usable day-one system.
3. **Human-in-the-loop labelling** closes the loop: the system proposes clusters,
   investigators confirm, confirmations become labels, the supervised model
   improves.

Note also that TVS's own phrasing asks for a system that "continuously learns" and
predicts fraud "**before fraud occurs**" — that is a request for unsupervised and
early detection, which is precisely what does not need ring labels up front.

**Verdict: the doubt does not hold. Do not drop the ring framing, and do not
change the problem.** Dropping rings collapses this into problem (a) or (c),
which our own triage scored lower and which are far more crowded (crowding 5.0
and 4.0 against 2.0 for (e)). The ring framing is the reason (e) scored highest on
selection opportunity in the first place.

**What should change is the framing, not the scope:** present it as *surface
suspicious clusters for human confirmation*, not *classify rings using ring
labels*. That is what a real deployment does anyway, and it needs zero new labels
on day one.

## D5. "Does Colab run only on synthetic data? Can we use real datasets?"

**Partly true, and worth fixing.** The notebook already has both tracks —
sections 3–6 are synthetic, section 7 loads YelpChi / Amazon / T-Finance /
DGraph-Fin. But those real datasets have **no ring labels**, so on real data we
can only measure node-level detection, never ring-level.

The user's instinct — "use their parts and add the ring feature ourselves" — is a
real technique and the strongest available answer. Four options, worst first:

| Approach | Verdict |
|---|---|
| Label communities found by Louvain/Leiden as rings | **Circular.** You would validate against your own clustering. Reject. |
| Propagate from known fraud, take dense subgraphs as rings | Weak supervision, legitimate as a *signal*, invalid as ground truth. |
| Use temporal bursts in Elliptic (49 timesteps) as ring proxies | Plausible but speculative; the labels would be guesses. |
| **Semi-synthetic injection into a real graph** | **Best option. Build this.** |

**Semi-synthetic injection:** take a real graph — its actual node features, its
actual degree distribution, its actual community structure and its real benign
background — and inject synthetic rings with known ground truth into it. This
directly attacks two of the threats in D1: the *background* is no longer ours
(threat #1), and the graph density becomes realistic (threat #6). It does not
make the injected rings themselves real, so it is not a substitute for TVS's data,
but it is a large step up from a wholly synthetic portfolio.

Implemented as `collab-help/05_semisynthetic_real_graph.py`. Needs Colab because
it loads a real dataset.

**Honest limit:** none of this settles whether the system detects real Indian
lending fraud rings. Only TVS's own labelled cases can do that. Everything else
raises confidence; nothing replaces it.

## D6. What are the chances of winning?

I will not invent a probability — I do not know the field, and a number here would
be false precision. What I can do is separate what we control from what we do not,
because the answer is asymmetric.

### Judging criteria, verbatim from the PDF

> Round 2 — Teams to propose a solution(s) and showcase the problems they will be
> addressing... Present your solution through a framework/proof of concept through
> **PPT**, it can include explanation of your solution with a
> **Prototype/Screens/Wireframe** (if required).
>
> Note: This may require **visiting a two-wheeler / used car / consumer durable
> retailer and understanding the current loan process**... and define the
> innovative and radical way of changing the status quo.
>
> Round 3 — develop and present a **working/live demo**, address the functional
> aspects, and perform a **code walkthrough** to showcase the unique aspects.
>
> Submissions will be reviewed for plagiarism and **AI-generated content**.
> Original work, relevant research and **critical thinking** will be given greater
> weightage.

### Working in our favour

1. **Least-contested problem.** Our own triage put (e) at crowding 2.0 of 7,
   hardest tier with the lowest competition. Most teams avoid it.
2. **"Critical thinking" is explicitly weighted**, and that is where this project
   is unusually strong: ring-disjoint evaluation, shuffled-label controls, nested
   CV, a best-single-feature floor, and a published failure (0.273 cross-typology).
   Almost no hackathon submission self-reports a generalisation collapse.
3. **There is working code**, not slides. Round 3 requires a code walkthrough and
   we have ~2,600 lines that run in 19 seconds.
4. **The reframe insight** — "swarm intelligence" is an optimisation paradigm, the
   problem is graph fraud detection — is exactly the kind of original reading the
   rubric rewards.
5. **A genuine novelty claim** (L4 ring-level scoring) rather than a re-skinned
   classifier.

### Working against us — and these are serious

1. **We have none of the three required artefacts.** No PPT, no wireframe or
   screens, no live demo. We have a document and a repository. Round 2 asks for a
   PPT; Round 3 asks for a working demo. This is the single biggest risk and it is
   entirely fixable.
2. **L4 is not implemented.** Our headline novelty is described in the plan and
   absent from the code. If a judge says "show me the ring scoring", there is
   nothing to show.
3. **No explanation module (L5).** A case officer cannot read the output. For a
   live demo this is close to disqualifying — a demo of a score column is not a
   demo of a fraud system.
4. **The GNN has never executed.** If the demo depends on it, that is a live risk.
5. **Cross-typology generalisation is 0.273.** We disclose it, which is the right
   call, but a sharp judge will press on it.
6. **No retailer visit.** The PDF explicitly suggests visiting a two-wheeler
   retailer to understand the loan process. Every competing team that does this
   gains grounding we do not have. It is cheap and high-value.
7. **AI-generated content is screened.** Read that line again. This project has
   been substantially AI-authored. That is a genuine risk, and the mitigation is
   not to hide it — it is to *own* the work: understand every line, rewrite the
   narrative in your own voice, and be able to defend any decision in the
   walkthrough without notes. A team that can explain its own code beats a team
   that produced more of it.

### The honest summary

The technical substance is competitive and the evaluation rigour is above what
this kind of submission usually shows. **The packaging is not.** Right now the
project would lose on presentation despite winning on substance, and that is the
worst way to lose because it is the most fixable.

If you do only three things before submitting: **build the demo UI with the
explanation view, implement L4 so the novelty is demonstrable, and visit a
dealer.**

---

# Session 3 — 2026-08-28

Prompted by the first real Colab run, which produced a crash, plus two reviewer
observations: a shape error in CARE-GNN, and a scaler fitted on the whole dataset.

## D7. CARE-GNN crashed — and a second, silent bug was found behind it

**The crash.** `_gated_agg` aggregated neighbour messages in *input* space (86
dimensions) and then tried to add the result to `proj(x)`, which is 128
dimensions:

```
RuntimeError: The size of tensor a (128) must match the size of tensor b (86)
```

Fixed by projecting first, then aggregating in the projected space, so the
residual `h + out` is dimensionally consistent. This exact fragility was written
into the handoff brief as a thing to check on first execution — and then shipped
anyway. Worth naming: predicting a bug and not fixing it is worth nothing.

**The second bug, found by auditing the rest of the file after the crash.** BWGNN's
Chebyshev recursion applied the Laplacian to `T_{k-2}` instead of `T_{k-1}`:

```python
ax = torch.zeros_like(T_prev)              # T_prev is T_{k-2} -- wrong
ax.index_add_(0, col, T_prev[row] * ...)
T_next = 2.0 * Lx - T_prev
```

**This one does not raise.** It silently computes the wrong Chebyshev polynomials,
so the band-pass filter weights no longer correspond to the intended frequency
response — meaning BWGNN would have produced a plausible-looking number that did
not mean what the model claims to do. That is worse than a crash, and it would
have gone straight into the results table.

The recursion is now written out explicitly as `T_0 = x`, `T_1 = Lx`,
`T_k = 2·L·T_{k-1} − T_{k-2}`, with the Laplacian application factored into
`_apply_L` so the indexing cannot be gotten subtly wrong.

**Lesson adopted: a crash is an invitation to audit the whole file, not just the
line that failed.**

## D8. The scaler was fitted on the entire dataset

The reviewer was right. In the notebook's GNN section:

```python
Xv = StandardScaler().fit(Xv).transform(Xv)   # fitted on train AND test
```

Each test fold's mean and variance leaked into training. Note the sklearn path
never had this bug — `run_v1.py` uses `prep(X, fit_mask)` which fits on training
rows only. It was introduced only in the notebook and the Colab scripts, which is
a useful reminder that duplicated code paths drift.

Fixed in four places: notebook GNN cell, notebook real-benchmark cell,
`collab-help/01`, `02` and `05`. Verified no `fit_transform` on a full matrix
remains in any Colab script.

One honest nuance for the GNN specifically: `StandardScaler` is fitted on training
rows but must *transform* every node, because a transductive GNN needs features
for all of them to pass messages. That is unavoidable and standard; fitting on
training rows only is still the correct choice.

## D9. 79% of the graph's edges cross fold boundaries — a leak specific to GNNs

Found while checking whether the ring-disjoint guarantee actually holds for a
transductive model. Measured on SMOKE:

| Adjacency | Edges | Crossing fold boundaries |
|---|---|---|
| `cooccurrence_union()` — all 5 relations | 511,652 | **403,062 (78.8%)** |
| Strong relations only (no dealer) | 8,294 | **0 (0.0%)** |
| Dealer relation alone | 504,500 | 403,062 (79.9%) |

Two consequences.

**The leak.** Folds are built from connected components over the *strong*
relations (`device`, `account`, `person`, `guarantor`), but the graph handed to
the GNN included `dealer`. A transductive GNN passes messages along every edge, so
test nodes were receiving information from training nodes through dealer edges.
The ring-disjoint guarantee was intact for the feature-based model and **broken
for the GNN**. Fixed: message passing now uses the strong-relations graph, which
has zero cross-fold edges, so folds are genuinely disconnected components.

**The structural problem, which may be the more important finding.** The dealer
relation supplies **98.6% of all edges**. With only 172 dealers over 3,474
applications, "shares a dealer" connects huge numbers of unrelated customers, so
the graph the GNN sees is essentially one dense dealer blob rather than a set of
fraud structures. That is very likely a large part of why every GNN underperformed
gradient boosting:

| Model | AUC-PR |
|---|---|
| GraphSAGE | 0.564 |
| GCN | 0.407 |
| GAT | 0.254 |
| GBT (feature-based) | 0.754 |

Those GNN numbers were produced on the leaky, dealer-dominated graph **and** with
the buggy CARE-GNN and BWGNN code, so they should be treated as invalid and
re-run. Do not quote them.

**The distinction that resolves it:** dealer is a valuable *feature* and a useless
*edge*. Dropping the 16 dealer-derived features costs 0.098 AUC-PR (0.754 →
0.656), so they clearly carry signal. But as edges they swamp the graph. Keep
dealer in the feature set; keep it out of message passing.

## D10. What to hand a reviewer

`make_review_bundle.sh` builds a 120 KB zip with all source, the four documents,
the notebook, `collab-help/` and the results JSON. It excludes `jale/data/`
(1.2 MB of parquet, regenerable in ~6 s) and bytecode.

Reading order for the reviewer: `HANDOFF_FOR_CODE_REVIEW.md` → `doubts.md` →
`README.md` → source. The `.docx` plan is not in the bundle because it is a
binary pitch document aimed at a competition judge, not at a code reviewer; if
they need the framing, `metrics-explained.md` and the HANDOFF cover it.

## Results invalidated by this session

| Number | Status |
|---|---|
| All GNN figures (SAGE 0.564, GCN 0.407, GAT 0.254) | **Invalid.** Leaky graph + dealer domination. Re-run. |
| CARE-GNN | Never produced a result — crashed. |
| BWGNN | Never ran; would have been silently wrong. |
| Everything sklearn / feature-based | **Unaffected.** That path never had the scaler bug and does not do message passing. |

---

# Session 4 — 2026-08-29 / 30

Prompted by an external review before Round 2, and by the decision to build the
demo first and write the proposal around it.

## D11. The committed numbers did not reproduce

`reports/v1_SMOKE.json` had node+graph GBT AUC-PR **0.754** (fixed) / **0.717**
(nested, quoted as the headline everywhere). A clean rerun on the current
libraries (numpy 2.5 / pandas 3.0 / scikit-learn 1.8) gave **0.729** / **0.700**.
The `HistGradientBoosting` fits are version-sensitive by ~0.02–0.03 AUC-PR and
there was **no `requirements.txt`, no pin, no lockfile**.

**Verdict:** real gap. Fixed: `requirements.txt` pins the environment;
`reports/v1_SMOKE.json` regenerated and dated; every document updated to the
reproducible numbers. The *shape* (graph ≫ node, ring-disjoint ≪ random, control
at chance, typology collapse) never moved.

## D12. Nested CV re-introduced the `min_samples_leaf` bug

`GBT_GRID` still listed `min_samples_leaf ∈ {5, 10, 20}`. Nested selection runs
on a 3-fold inner split of the outer training rows → ~17 positives per inner
fold, so the inner AUC-PR is too noisy to reject `leaf = 20`, and nested
node-only collapsed to **0.049** — the exact failure README §8 is about, back
again through the tuning grid. Capped the grid at `leaf = 10`; nested node-only
is back to 0.238, node+graph to 0.700.

**Lesson:** a fix to a default is not a fix to the search space that can re-pick
the bad default.

## D13. Entity resolution in the path costs −0.08

`experiments/er_in_path.py`: rebuild the graph on Fellegi–Sunter-resolved ids
instead of the generator's perfect `person_id`.

| | AUC-PR | Recall@5% |
|---|---|---|
| perfect `person_id` (what V1 reports) | 0.729 | 0.725 |
| linker-resolved ids (34 false merges, 70 records welded) | 0.648 | 0.626 |

**Verdict:** threat #3 quantified at last. −0.08 is the honest cost of not having
a clean identity column. It is not tunable by the posterior threshold — the false
merges are coincidental all-field matches. A regulated NBFC's PAN/Aadhaar spine
would remove most of them; our synthetic duplicates model a worse case. Report
both numbers.

## D14. L4 / L5 / cold-start built; what the ring score can and cannot do

`jale/demo/l4_rings.py` — ring score = 0.65·(model corroboration) + 0.35·(max
structural concentration). No learned weights, no `ring_id`.

- At score ≥ 0.5: **all 12 real ring-clusters flagged**, 83% of fraud apps
  covered, 23 false-positive clusters (median 6 apps — repeat-guarantor families,
  shared household devices).
- Ranked precision is imperfect: heavily camouflaged rings sit in a component
  with many benign cover applications, so a few small benign clusters outrank
  them. This is a real synthetic-data limitation, disclosed in the demo.
- Cold start (`coldstart.py`): 3 / 5 / 10 confirmed seeds → AUC-PR
  0.21 / 0.26 / 0.29, averaged over 20 draws with seeds held out (more honest
  than Session 2's single favourable draw of 0.29 / 0.39 / 0.42).

## D15. The public loaders were loading the wrong datasets

`load_yelpchi` used `torch_geometric.datasets.Yelp` (GraphSAINT, ~717k nodes) and
`load_amazon` used `Amazon(name='Computers')` (co-purchase) — **not** the
anomaly-detection benchmarks. Any number quoted from them would have been invalid.
Replaced with DGL's `FraudDataset('yelp' / 'amazon')` and added a PyG `DGraphFin`
loader (the one dataset whose edge = the guarantor relation). Notebook stale cells
fixed (`Xr` crash, open-items list).

## Open items carried forward

| # | Item | Why |
|---|---|---|
| A | Temporal / walk-forward split | threat #8, still not done |
| B | Run `collab-help/05` (rings into a real graph) | the strongest external check; needs Colab |
| C | Run `collab-help/01` (GNN vs GBT on the fixed graph) | threats #1, #2 |
| D | Reduce or formally accept the `n_guarantors` tell | threat #1 instance |
| E | L4 ranked precision on camouflaged rings | synthetic-data limit; revisit if a real graph is available |

## D16. Does the headline survive prevalence and camouflage sweeps?

`experiments/ring_rate_stress.py` (SMOKE, ring-disjoint GroupKFold(5), node+graph
GBT — same protocol as run_v1; each cell is a freshly generated world):

| setting | pos / rings | AUCPR | lift | R@5% |
|---|---|---|---|---|
| prev 1.0% | 58 / 3 | 0.298 | 17.4x | 0.448 |
| prev 2.0% | 92 / 7 | 0.697 | 25.9x | 0.772 |
| prev 3.0% (baseline) | 131 / 10 | 0.729 | 19.3x | 0.725 |
| prev 4.5% | 199 / 16 | 0.747 | 13.4x | 0.613 |
| cam x1.5 | 167 / 12 | 0.658 | 14.0x | 0.611 |
| cam x2.0 | 194 / 11 | 0.706 | 13.0x | 0.639 |

**Verdict:** from 2% prevalence up the number sits in 0.70–0.75 and survives
2× camouflage at 0.706 — the headline is not an artifact of an easy operating
point. The 1.0% cell (3 rings only) is too small to read; disclosed, not
hidden. The sweep is a sensitivity bound, not a confidence interval, and it
varies our own simulator's knobs — it bounds what the reviewer should ask
about; it does not replace real-data validation. Written to
reports/ring_rate_stress.json.
