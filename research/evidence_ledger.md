# Evidence ledger — TVS Credit EPIC 8.0 · Problem (e)

Every claim on a slide traces to a row here. Confidence:
**High** = multiple independent primary sources; **Med** = single credible source;
**Low** = signal/anecdote only (never a material claim).

## A. TVS Credit — company & financials

| ID | Claim | Type | Source | Date | Conf |
|---|---|---|---|---|---|
| E-01 | FY26 PAT Rs 913 cr (+19%); disbursements +26%; "reduction in total credit costs and GNPA as of Mar'26"; "prudent and proactive approach to underwriting and risk through calibrated credit policy restrictions"; 2.4 cr+ customers | Fact | TVS Credit press release | May 2026 | High |
| E-02 | FY25 PAT Rs 767 cr (+34%); AUM Rs 26,647 cr (+3%); income Rs 6,630 cr; PBT Rs 1,025 cr | Fact | TVS Credit PR / Hindu BusinessLine | 2025 | High |
| E-03 | AUM Rs 30,300 cr (Dec-25), +11.5% YTD vs 27,179 cr (Mar-25); mix 2W 29%, CD 19%, PL 15%, tractor 14%, usedCV 11%, usedCar 8%; 9M-FY26 disb 24,422 cr vs FY25 26,297 / FY24 25,018; cushion 2–4%; weakness "average, but improving earnings profile", "modest credit risk profile of borrowers" | Fact | CRISIL rationale | 09-Apr-2026 | High |
| E-12 | FY25 AUM growth limited to 3% "due to company limiting disbursements in new tractor and personal loan portfolios"; FY26 momentum driven by 2W + consumer durables | Fact | CRISIL (same as E-03) | Apr 2026 | High |
| E-04 | GNPA after write-offs ≈ 2.9% FY25 (2.8% Q4) | Fact | Hindu BusinessLine | Apr 2025 | High |
| E-10 | Saathi app 4.1★/209k ratings; complaints: payment failures, pressure calls, unreachable care; separate TVS Credit **Dealer App** exists | Fact | Play/App Store/MouthShut | 2025–26 | Med |
| E-14 | AUM 30,639 cr; loans net 30,285.47 cr (PY 26,298.84); PAT 913.17 cr (PY 767.25); NNPA 2.06% (PY 2.87%); write-offs 437.11 cr (PY 457.49); gross exposure 31,216.01 cr; gross NPA 1,141.36 cr (3.66%); complaints 5,625. **VERIFIED from the AR PDF text** (Note 8(e) / P&L / asset-quality note / Note 18.10; sector table at extraction lines 11610–11625) and arithmetic-audited: FY26 sector rows sum exactly to the printed totals (`python -c "from jale.data import tvs_data; tvs_data.verify()"`). Stage-3 ~758 cr and the TW/CD mix fractions remain CRISIL-secondary — superseded for the covered book by E-33 | Fact | TVS AR 2025-26 (audited primary) | FY26 | High |
| E-33 | Note 18.9(iii) sectoral exposures (pp. 136–137): Vehicles 12,963.80 cr @ 2.64% NPA; Consumer durable 6,065.43 cr @ 1.02% → **target dealer-sourced book Rs 19,029.23 cr @ 2.12% (gross NPA Rs 403.93 cr)**; Personal loans 4,774.84 @ 2.51%; Agriculture & allied (tractor) 4,306.66 @ 10.30%; Services 2,258.50 @ 6.88%; **Advance to dealers 228.17 cr @ 0.23% NPA (FY25: 184.25 @ 0.37%) — dealers repay**; Industry–MSME 454.18 @ 3.89%; grand total 31,216.01 @ 3.66% | Fact | TVS AR 2025-26 Note 18.9(iii) (audited primary) | FY26 (+FY25 comparators) | High |

## B. Macro & fraud context

| ID | Claim | Type | Source | Date | Conf |
|---|---|---|---|---|---|
| E-16 | FY26 reported fraud Rs 48,021 cr / 10,114 cases (FY25 32,803 / 23,722); advances-category 40,774 cr / 8,640 (~85%); FY25 advances 30,367 / 7,924; FY24 8,917 / 4,105 | Fact | RBI Annual Report 2025-26 (The Hindu/BS/Mint/ET/Fortune) | 29-May-2026 | High |
| E-17 | Derived: severity/case Rs 2.17 (FY24) → 3.83 (FY25) → 4.72 (FY26) cr = **+118% in 2 yrs**; cases falling, value rising → rings getting bigger | Computation | arithmetic on E-16 | — | High |
| E-05 | Mahindra Finance ~Rs 150 cr vehicle-loan fraud Q4-FY24: **KYC forgery, ~2,000 ghost customers, NE/Mizoram branch cluster**; board meeting postponed | Fact | BS / Indian Express / ET (exchange filing) | Apr 2024 | High |
| E-06 | Delhi: 35 bikes via fake papers by **misusing an NBFC's video-KYC system**; 3 arrested | Fact | Times of India | Jan 2024 | High |
| E-07 | Rajkot: 28 incl. NBFC staff; Rs 4 cr; falsified Panchayat records + fabricated machinery invoices | Fact | Times of India | Apr 2025 | High |
| E-08 | Hyderabad: NBFC employee siphoned Rs 30 L; forged NOCs | Fact | Times of India | Feb 2025 | High |
| E-09 | RBI mandates **Early Warning Systems** for Middle/Upper-layer NBFC fraud (transactional + behavioral); Master Directions FRM apply | Fact | Nishith Desai research; RBI Master Directions | 2024-26 | High-dir |
| E-15 | Ring anatomy (12 identities + 8 mules + 4 ghosts, multi-lender) | Signal | Innefu vendor blog | 2026 | Low
## C. Our measured results (repo pipeline + executed Colab)

| ID | Result | Type | Where | Conf |
|---|---|---|---|---|
| E-20 | Node+graph GBT ring-disjoint AUCPR 0.754 (20.0× lift, R@5% 0.733); graph-only 0.751; node-only 0.243; single feature n_guarantors 0.233 / 6.2× | Fact | Colab §4 | High |
| E-21 | Nested CV node+graph 0.681 (honest headline = range 0.68–0.75) | Fact | Colab §6b | High |
| E-22 | Leakage gap synthetic random 0.854 vs ring-disjoint 0.754 → **+0.100** | Fact | Colab §6 | High |
| E-23 | YelpChi real: ring-disjoint 0.887 vs random 0.909 → **+0.022**; SAGE 0.893 ≈ GBT 0.887 on dense real graph | Fact | Colab §7 | High (caveat: degree-bucket folds) |
| E-24 | GNN fair compare (same folds/features): SAGE 0.688, GAT 0.672, GCN 0.670, CARE-GNN 0.670, BWGNN 0.072 (impl. bug, disclosed) | Fact | Colab §5 | High |
| E-25 | Ablation (drop → AUCPR): account 0.537 (−0.217), device 0.549 (−0.205), dealer 0.656 (−0.098), guarantor 0.674 (−0.080), person 0.741 (−0.013) | Fact | Colab §8a | High |
| E-34 | Prevalence & camouflage stress: AUCPR 0.697–0.747 across fraud_ring_fraction 2.0–4.5% (baseline 0.729); 0.658–0.706 at ×1.5–2 camouflage; the 1.0% cell (3 rings) too small to read — disclosed. Headline not an artifact of an easy operating point | Fact | experiments/ring_rate_stress.py → reports/ring_rate_stress.json | High |
| E-26 | Typology hold-out mean AUCPR 0.273 vs 0.754 in-dist; identity_reuse 0.889 vs dealer_collusion 0.081 → model learns THE rings; motivates novelty flag | Fact | typology_generalisation.py | High |
| E-27 | **Lead-time backtest:** strict (L4≥0.70, n≥6): 5/9 forming rings caught pre-completion, median lead ~204 d (IQR 178–213), **0 false clusters** (~0/wk); loose (L4≥0.50, n≥3): 9/10 rings (6 pre-completion), lead ~173 d, ~5.2 false/wk | Fact | experiments/lead_time.py | High (synthetic, internal validity) |
| E-28 | L4 ring score ≥0.5: all 12 real rings flagged, 83% fraud apps covered, 23 FP clusters (median 6 apps) | Fact | doubts.md D14 | High |
| E-29 | ER-in-path costs −0.08 AUCPR (0.729→0.648); 34 false merges / 70 welded | Fact | er_in_path.py (D13) | High |
| E-30 | Cold-start 3/5/10 seeds → AUCPR 0.21/0.26/0.29 (held-out, 20 draws) | Fact | coldstart.py (D14) | High |
| E-31 | Shuffled-label null PASS (0.041 vs 0.038 base) → no memorisation | Fact | Colab §6 | High |
| E-32 | Message-passing needs strong-relations graph: full union 511,652 edges, 78.8% cross-fold (98.6% dealer); strong graph 8,294 edges, 0 cross-fold | Fact | Colab §5 note | High |

## D. Research anchors (verified)

| Paper | Verified? |
|---|---|
| Xu et al. "Towards Consumer Loan Fraud Detection: GNN with Role-Constrained CRF", AAAI 2021 | ✅ DOI 10.1609/aaai.v35i5.16582 |
| Pei et al. "Subgraph anomaly detection in financial transaction networks", ACM ICAIF 2020 | ✅ DOI 10.1145/3383455.3422548 |
| Luo "Robust Financial Fraud Detection via Causal Intervention … Dynamic Hypergraphs", Mathematics 13(24), MDPI 2025 | ✅ DOI 10.3390/math13244018 |
| Jiao et al. "Dynamic heterogeneous graph contrastive learning for … collusive financial fraud", Sci Rep 2026 | ✅ DOI 10.1038/s41598-026-58938-5 |
| GT-ACGL (Game-Theoretic Anticipatory Continual Graph Learning), Sci Rep 2026 | ✅ |
| "HGT-FD" AISTATS/PMLR 2026 | ❌ **NOT FOUND — do not cite**; real analog TROPICAL, Springer KAIS 2025 |
| MAFF-Bench, arXiv:2511.06448 (ICLR 2026) | ✅ title = "When AI Agents Collude Online…" |

## Source URLs (key items)

- CRISIL Apr-2026 rationale: references.crisilratings.com (TVSCreditServicesLimited_April 09_2026_RR_392849)
- TVS FY26 PR: tvscredit.com/press-releases/...rs-913-crores-for-fy-2025-26
- TVS FY25 PR: tvscredit.com/press-releases/...highest-ever-pat-of-rs-767-crore-for-the-year-ended-march25
- CRISIL Apr-2025 upgrade: ...TVSCreditServicesLimited_April 09_2025_RR_366539
- The Hindu BusinessLine FY25 GNPA: thehindubusinessline.com/money-and-banking/tvs-credit-posts-767-crore-pat-in-fy25
- RBI FY26 fraud: The Hindu (financial-institutions-report-over-10000-cases-of-fraud-involving-48000-crore-in-fy26-rbi-data); Mint; Business Standard
- Mahindra fraud: business-standard.com M&M Finance detects Rs 150 cr vehicle loan fraud (Apr 2024)
- TOI Delhi video-KYC bikes: timesofindia (Gang Dupes Nbfc, Procures 35 Bikes On Loan Using Fake Papers)
- Nishith Desai "Frauds in NBFCs"; MAFF-Bench arXiv:2511.06448
