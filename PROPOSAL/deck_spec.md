# Round-2 deck spec — JA·LE (plain-language, evidence-first) · Part 1/2

**Deck order:** this spec is the source of truth. `build_deck.py` turns it into
a PPTX. Every number on a slide carries a tag: [REPO] rerunnable script,
[PUBLIC] verified public source, [COLAB] executed notebook, [SIM] synthetic.
**Keep it to ~14 slides.** One idea per slide; no more than 6 bullets.

---

## S1 · Title
- Headline: **JA·LE — Catch a fraud ring before it is finished forming.**
- Sub: TVS Credit EPIC 8.0 · Problem (e) · Swarm Intelligence Lending Network
- One line: "A rerunnable, leak-audited fraud-ecosystem early-warning engine for
  dealer-sourced retail lending."
- **Lead-time badge (say it first):** "Measured: flags a forming ring ~7 months
  before it finishes filing — median 204 days, zero false alarms at the strict
  setting [SIM]. It names the cheapest link to hold — and never blames the
  dealer [COLAB + PUBLIC]."
- Tag line: "Everyone can generate a fraud-detection architecture with AI.
  Nobody can generate a measured one."

## S2 · The problem where the money is
- Chart: **G6** (RBI lending-vs-digital + severity/case trend) [PUBLIC]
- Takeaway (plain words): "In FY26, lending fraud was Rs 40,774 crore — **85% of
  all reported fraud value**; card/internet fraud was Rs 29 crore. And each
  case is getting bigger: +118% value per case in two years."
- One line: "The fraud war TVS fights is a lending war."

## S3 · How rings actually get filed at TVS
- Map the dealer-sourced journey (lead → KYC → doc → underwriting → disbursal → EMI).
- Three real cases as proof rings exist and hurt NBFCs:
  - Mahindra Finance: ~Rs 150 cr, ~2,000 ghost customers, forged KYC [PUBLIC E-05]
  - Video-KYC abuse to get 35 bikes financed on fake papers [PUBLIC E-06]
  - Branch-staff collusion, fabricated invoices [PUBLIC E-07]
- Takeaway: "No single application looks suspicious. The ring lives **between**
  records — and it is usually already disbursed by the time anyone sees it."

## S4 · The generic answer, and its one weakness
- "Ask any AI tool for a fraud-ring solution and you get: graph + GNN + red-dot
  dashboard." (Expected; ~400 teams will show variants.)
- Its two failures (plain words):
  1. **Late:** by the time a ring is dense enough to detect, the money is gone.
  2. **Accusatory:** it blames the most *connected* node — usually a busy dealer
     or a village shop-phone — which punishes genuine rural customers.
- Positioning: "Our proposal asks a different question."

## S5 · What we built — one picture
- Simple layer diagram (no buzzwords): Data → Link records → Score clusters →
  Explain to an analyst → Decide (hold / verify / audit).
- Screenshot of the demo queue (artifact-first).
- Three honest labels under the diagram: "Synthetic + public data only · ~30 s
  to rerun · every number traceable to a script."

## S6 · What we measure (the receipt slide)
- Chart: **G4** model comparison [COLAB]
- Table of headline numbers with tags:
  | Metric | Value | Source |
  |---|---|---|
  | Ring-detection AUCPR (node + graph, ring-disjoint CV) | **0.754** | [COLAB] |
  | Honest nested-CV range | 0.68–0.75 | [COLAB] |
  | Lift over random | **20.0×** | [COLAB] |
  | Best single feature alone | 6.2× | [COLAB] |
  | GNN (GraphSAGE) baseline | 0.688 | [COLAB] |
- Takeaway: "The graph features, not the exotic model, carry the signal — and we
  prove it by running every model on the same folds."

## S7 · Why you can trust the number (the honesty slide)
- Three controls, all passing: shuffled-labels null (no memorisation),
  ring-disjoint CV, nested CV.
- Chart: **G2** leakage gap +0.100 synthetic / **+0.022 on a real graph**.
- One honest failure, told first: "Teach the model five fraud typologies and test
  it on a sixth → AUCPR drops to 0.27. So we do NOT claim it 'understands
  rings' — instead it flags *novel* structures it has never seen (built-in
  **novelty flag**), which is exactly what a lender needs for brand-new fraud."
- Takeaway: "We tell you where we fail. That is the point."

## S8 · The differentiator — before fraud occurs [SIM]
- Chart: **G1** (formation-velocity curves + recall-vs-load frontier).
- The measured claim (plain words): "Trained only on the past, the engine flags
  **5/9 still-forming rings before their last application arrives — a median of
  ~7 months of lead time — with zero false alarms** at the strict setting."
- "Loosen it and it catches 9/9, at a measurable analyst load." (Honest trade-off.)
- Map to the problem statement: this is *literally* "predict emerging fraud
  ecosystems before fraud occurs."