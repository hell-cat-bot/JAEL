# collab-help — experiments that need Colab

Everything here **cannot run in the sandbox**: no PyTorch, no PyG, 1 GB RAM.
Run them on Colab and send back the printed output — that is all that is needed.
Each script is self-contained and prints a compact summary at the end.

Upload the whole `jale/` project first (zip it, `Files → Upload`), then run:

```python
!pip install -q torch_geometric
```

and copy each script into a cell. Set `sys.path` to the project root if the
imports fail.

---

## Priority order

Run them in this order. If you only run one, run **01**. If you want the result that most strengthens the submission, run **05**.

| # | Script | Why it matters | Runtime |
|---|---|---|---|
| 01 | `01_gnn_vs_gbt.py` | The single biggest open question: does a GNN beat gradient boosting on the *same* graph, features, folds and metric? Everything in the plan assumes it might. Nobody has checked. | ~15 min GPU |
| 02 | `02_real_benchmarks.py` | Moves the claim from "works on our synthetic data" to "works on real financial data". Currently the biggest weakness in the whole submission. | ~10 min CPU |
| 03 | `03_full_scale.py` | Does anything break at 120,000 persons? Memory, runtime, and whether the lift survives. | ~30 min |
| 04 | `04_camouflage_sweep.py` | Where does the system stop working? Showing our own breaking point is more persuasive than hiding it. | ~20 min |
| 05 | `05_semisynthetic_real_graph.py` | Injects known rings into a **real** graph, so the benign background is not ours. The closest thing to external validation of ring detection that public data allows. | ~10 min CPU |

---

## What is already known (measured in the sandbox)

So you have the baseline to compare against. SMOKE profile, 3,474 applications,
131 fraud, 3.77% base rate, ring-disjoint 5-fold CV:

| Model | AUC-PR | Lift |
|---|---|---|
| best single node feature (`n_guarantors`), raw | 0.233 | 6.2× |
| node-only GBT (nested CV) | 0.237 | 6.3× |
| graph-only GBT (nested CV) | 0.719 | 19.1× |
| **node + graph GBT (nested CV)** | **0.717** | **19.0×** |
| random split (leaky, for reference) | 0.854 | — |

And the result that should worry you:

**Train on four fraud typologies, test on the fifth: mean AUC-PR 0.273, down from
0.754.** Per typology: identity_reuse 0.889, disbursement_sink 0.211,
guarantor_star 0.127, dealer_collusion 0.081, device_farm 0.060. The model
appears to memorise typology-specific signatures rather than learning what a ring
*is*. `experiments/typology_generalisation.py` reproduces this in the sandbox.

If a GNN generalises across typologies better than the feature-based model does,
that is the most interesting result available in this project. Script 01 reports
it.

---

## Sending results back

Paste the printed output verbatim. Do not summarise it — the exact numbers and
any tracebacks are the useful part. If a script crashes, that is a real finding:
`torch_gnn.py` has never been executed, so shape errors are expected and need
recording.
