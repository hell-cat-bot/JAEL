# 7 · Reproducibility · Re-run Every Number in 30 Seconds

> *"Nothing in this submission is a number we typed by hand. Clone, install, run — the pipeline regenerates the features, the models, the audit JSON, every chart, and the interactive console."*

---

## ⚡ Option A: 1-Click Master Execution (Fastest)

### On Windows:
Double-click on **`RUN_ALL_IN_30_SECONDS.bat`** (or run `RUNBOOK\RUN_ALL_IN_30_SECONDS.bat` from terminal).

### On Linux / macOS:
```bash
bash RUNBOOK/RUN_ALL_IN_30_SECONDS.sh
```

*(This automatically runs the pipeline, verifies all Note 18 math, regenerates the 8 charts, and opens the proposal slides and demo console in your default browser).*

---

## 💻 Option B: Step-by-Step Command Line Execution

```bash
# 1 · Clone and install pinned dependencies
git clone <repo-url> && cd TVS/V3
pip install -r requirements.txt

# 2 · Full pipeline: features, heterogeneous graphs, models, JSON audit files
python scripts/run_v1.py --profile SMOKE

# 3 · Regenerate all 8 presentation charts
python PROPOSAL/make_charts.py

# 4 · Rebuild & verify the slide artefacts
python make_slide1.py
python make_slide2.py

# 5 · Run independent mathematical audit of Note 18 & RBI macro numbers
python verify_audited_numbers.py
```

---

## 🔍 What a Reviewer Can Verify Independently

| Claim / Benchmark | Exact Reproduction Command | Source Document / File | Result to Check |
|---|---|---|---|
| **1. The Audited TVS Numbers** | `python verify_audited_numbers.py` | TVS Credit FY26 AR Note 18.9(iii) (pp. 136–137) | Target book: **₹19,029.23 Cr @ 2.12% NPA**<br>Dealer advance NPA: **0.23%** (Proof dealers repay) |
| **2. The RBI Macro Figures** | `python verify_audited_numbers.py` | RBI Annual Report 2025-26 | Lending fraud: **₹40,774 Cr (85%)**<br>Severity surge: **+118% in 2 years** (₹2.17 Cr → ₹4.72 Cr) |
| **3. Every Model Score** | `python scripts/run_v1.py --profile SMOKE` | `reports/v1_SMOKE.json` | Fixed GBT: **0.754 AUCPR** (20× lift)<br>Nested CV: **0.681 AUCPR** |
| **4. The Data Leakage Gap** | `python scripts/run_v1.py --profile SMOKE` | `reports/v1_SMOKE.json` | Random split: **0.854** vs Ring-disjoint: **0.729–0.754**<br>(**+10.0%** synthetic leakage inflation) |
| **5. The Real YelpChi Benchmark**| `python jale/data/public_datasets.py` | Public 45,954-node YelpChi graph | Random: **0.909** vs Disjoint: **0.887**<br>(**+2.2%** real-graph leakage gap) |
| **6. Lead Time & Early Warning** | `python experiments/lead_time.py` | `reports/lead_time.json` | Strict: **204 days median early warning**<br>**5/9 forming rings caught**, **0 false alarms** |
| **7. Causal Ablation Drop** | `python PROPOSAL/make_charts.py` | `PROPOSAL/charts/g3_ablation.png` | Bank Account: **−0.217** · Device: **−0.205**<br>Dealer: **−0.098 (least impact)** |
| **8. Financial Feasibility** | `python PROPOSAL/financial_model.py` | `reports/financial_model.json` | 5-Year Base NPV: **+₹39.2 Cr (4.2× ROI)**<br>Monte-Carlo: **89.3% positive runs** |

---

## 📂 Runnable Files Index

Inside this **`RUNBOOK/`** folder, you will find:
* **`RUN_ALL_IN_30_SECONDS.bat`**: Windows 1-click master runner.
* **`RUN_ALL_IN_30_SECONDS.sh`**: Linux/macOS master runner.
* **`run_all.py`**: Cross-platform Python runner.
* **`1_install_requirements.bat`**: Installs pip requirements.
* **`2_run_pipeline.bat`**: Executes `scripts/run_v1.py --profile SMOKE`.
* **`3_regenerate_charts.bat`**: Executes `PROPOSAL/make_charts.py`.
* **`4_audit_all_numbers.bat`**: Executes `verify_audited_numbers.py`.
* **`5_open_slides.bat`**: Opens Slide 1 & Slide 2 HTMLs in browser.
* **`6_open_interactive_demo.bat`**: Opens the 3-screen interactive console.
