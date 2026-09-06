# Round 3 plan — live demo + code walkthrough

## What Round 3 rewards (from the problem statement)
Working/live demo of the proposed solution · address functional aspects ·
**code walkthrough to showcase unique aspects**.

## The story we tell in Round 3
1. **Run it live** (`python scripts/run_v1.py --profile SMOKE` + Colab outputs)
   — ~30 s, every number in the deck regenerable in front of the judge.
2. **Walk the actual code** in order: generator (anti-cheating contract) →
   entity resolution (Fellegi–Sunter) → graph + 86 features → ring-disjoint CV
   + leakage audits → L4 cluster scoring → lead-time backtest.
3. **Open the failure files on purpose:** BWGNN 0.072 bug, typology collapse
   0.27 → the novelty flag, ER-in-path −0.08, cold-start curve. "We show you
   the cracks; that is why you can trust the rest."

## Demo enhancements to build (in priority order)
1. **Ecosystem Timeline tab** in `jale_demo.html` — weekly structure + ring
   score trajectory + lead-time badge (feeds from `reports/lead_time.json`).
2. **Formation mode (the "time machine")**: replay the portfolio week by week;
   show the strict-operating-point alarm firing while the ring is still filing
   → freeze frame at "median 204 days before the last application".
3. **Counterfactual slider**: per-relation ablation on a chosen cluster
   ("remove account links → risk 0.87 → 0.54") — reuse notebook §8a numbers,
   framed as *ablation attribution*, never "causal do() proof".
4. **Red-team ("tomorrow's fraud") toggle**: camouflage sweep from
   `collab-help/04_camouflage_sweep.py` — show where the detector stops
   working and what the novelty flag does in response.
5. **Dealer-risk view** wired to the dealer-ablation reading ("dealer signal is
   tested, not presumed") — the fairness exhibit.
6. **Synchronized-payment EWS prototype** (post-disbursal) on `emi_schedule`
   — the RBI Early-Warning-System hook (collections-side use case).

## Build order & ownership
- Do 1–3 first (Round-3 core: timeline + lead time + counterfactual).
- Add 4–5 one day before the finale.
- 6 only if time remains (it is the least demo-critical).

## Walkthrough talking points (anticipate these questions)
| Judge asks | Answer (plain words) |
|---|---|
| "Is this real data?" | No — synthetic + public, stated on every slide. That is why Phase 1 is a shadow backtest on TVS's own history before *any* blocking decision. |
| "Why not a GNN?" | Same folds, same features, GNN 0.688 vs tree model 0.754 — we report both. On a dense real graph they tie (YelpChi). The honest model won, and we showed our work. |
| "What would falsify this?" | A Phase-1 backtest on real TVS rings with no lead time, or a hub false-blame rate that hurts rural borrowers. Those are the two tests we set for ourselves. |
| "Where does it plug in?" | Between the LMS and disbursement: entity resolution links the app into the graph; the swarm/queue ranks its cluster; a human acts on the case-file. Never auto-reject. |
| "What's genuinely yours?" | The leak-free measurement discipline, the cluster-level (not person) scoring, the honest failure log, the lead-time backtest, and the TVS-anchored business framing. |
| "Why should Bank A care?" | 85% of reported fraud value is advances; rings are getting bigger (+118%/case in 2 yrs); and the same engine protects using (a) collections early-warning and (b) safer re-opening of throttled segments. |

## Guardrails for Round 3
- No new research claims. Every number must already exist in
  `reports/`, the notebook, or the demo data.
- Deeper ≠ better. If the judge is lost, we lost. Keep the walkthrough at the
  level any analyst can follow in 10 minutes.
- Never auto-reject; never claim causation; never hide a failure.