# Round-2 deck spec — JA·LE · Part 2/2 (slides 9–14 + presenter notes)

## S9 · Which connection carries the risk? [COLAB]
- Chart: **G3** ablation tornado.
- Plain words: "Remove the bank-account signal → risk detection drops most
  (−0.22). Remove 'who the dealer is' → barely moves (−0.10)."
- Design consequence: "We test every relationship's contribution and we do NOT
  presume the dealer is guilty. That protects real dealers and rural
  customers." (Rebut the hub-problem from S4.)

## S10 · Ring-vaccination, not ring-busting (the analyst's decision layer)
- The L4 ring queue: score the *cluster*, not the person; every flagged
  cluster carries an evidence case-file (which apps, which shared device /
  account / guarantor; dominant axis).
- Chart: **G7** analyst budget → recovery curve.
- Actions are **firebreaks** — minimal-friction, link-level, budget-aware:
  step-up verification · verify guarantor pool · hold disbursement to one
  account · dealer audit (when implicated) — never blanket blocking.
- Every action maps to a measured ablation: account −0.22, device −0.21,
  dealer −0.10 (least signal) [COLAB].
- Persona line: "Vaccinate the cheap link; the dealer relationship and honest
  borrowers stay untouched. A risk officer is told *which ring, which link,
  what to do* — in analyst minutes, not hours."

## S11 · The business case, anchored to TVS's audited numbers [PUBLIC+A#]
- Chart: **G5** financial hockey stick (base + upside + Monte-Carlo band).
- Plain words: "Anchored to TVS's own audited AR Note 18 (FY26): target
  dealer-sourced book Vehicles 12,964 + CD 6,065 = **Rs 19,029 cr**, target-book
  NPA Rs 404 cr @ 2.12%; write-offs Rs 437 cr; AUM Rs 30,639 cr. Dealer
  advances carry **0.23% NPA** — dealers repay [PUBLIC E-33]."
- Headline numbers: Year-1 pilot loss → Year-2 breakeven → Year-3 profit;
  base 5-yr NPV ≈ **Rs +39 cr**, upside ≈ **Rs +64 cr**; 89% chance of
  positive NPV.
- Strategic slide line: "Risk control is what lets TVS *grow* again:
  FY25 throttled tractor + personal loans for credit-cost reasons; a control
  layer is what re-opens them safely."
- Every assumption listed with an [A#] tag on the slide.

## S12 · Rural fairness is a feature, not a caveat
- Separate **creditworthiness ≠ fraud risk** (a thin-file honest borrower is not
  a fraud suspect).
- Hub false-blame KPI ≈ 0 (we measure it). ER false-merge audit: 34 welded
  strangers is why we do not auto-reject. DPDP 2023: hashed identifiers,
  no PII in edges. Explainable by construction (case-files, not black box).
- Chart: **G8** dealer-fairness (audited sector NPA ladder, dealer bar green
  at 0.23%).
- The two-ways line: "Tested two ways: TVS's audited books (dealer advances
  0.23% NPA — dealers repay) + our ablation (dealer link carries the LEAST
  signal, −0.10) — same verdict: the dealer is a conduit, not a cause."

## S13 · Roadmap & change management
- Phase 1 (0–3 mo) Shadow: one TN two-wheeler region, backtest lead time on
  historical rings, no blocking. Phase 2 (3–6) Advisory → Active: link-freezes
  in underwriting, analyst console. Phase 3 (6–12) Scale: CD, used car, PL,
  tractor; adversarial red-team; fairness monitoring.
- Change management: 2-week risk-analyst training, dealer SOP, kill-switch +
  rule fallback, pre-registered success metrics (precision at analyst capacity,
  lead time, hub false-blame).
- RICE slide (R high / I very high / C high-on-protocol / E low).

## S14 · Reproducibility & sources (the cliffhanger for Round 3)
- "Run it yourself in ~30 seconds" + QR / link to repo; Colab notebook with the
  executed numbers; `reports/` JSONs; evidence ledger (research/).
- Three verified citations (AAAI-21, ICAIF-20, MDPI-25), each mapped to a
  component. No fabricated references (state this explicitly).
- Round 3 promise: live demo + code walkthrough of the exact scripts that made
  these numbers.

---

## Notes for the presenter (plain-speaking rules)
- Never say "leveraging cutting-edge AI". Say what it does.
- Every number gets its tag spoken once ("0.754 — run on the executed notebook").
- "Is this real data?": "No — we are explicit: synthetic + public. That is
  exactly why Phase 1 is a shadow backtest on TVS's own history before any
  decision."
- "Why graph features not GNN?": "Same folds, same features — GNN 0.688, tree
  model 0.754. We report both. The honest model won."
- "What would falsify this?": "A Phase-1 backtest on real TVS rings that shows
  no lead time, or a false-blame rate that hurts rural borrowers. Those are the
  two tests we set for ourselves."