"""Financial model for the Round-2 deck -- every input is either an AUDITED
number (TVS Credit AR FY26, Note 18) or a labelled assumption.  Outputs
reports/financial_model.json.

Audited anchors (jale/data/tvs_data.py -- verify() reconciles the sector table
against the printed Note-18 totals):
  TVS Credit AR FY26 (Note 18.9(iii), pp. 136-137):
    AUM Rs 30,639 cr · loans net Rs 30,285.47 cr · PAT Rs 913.17 cr
    NNPA 2.06% · write-offs Rs 437.11 cr
    TARGET dealer-sourced retail book = Vehicles Rs 12,963.80 cr
      + Consumer durable Rs 6,065.43 cr = Rs 19,029.23 cr (61.1% of gross
      exposure), gross NPA Rs 403.93 cr (2.12%).
    Advance to dealers: Rs 228.17 cr at 0.23% NPA -- dealers repay (fairness
      anchor; never blanket-block the dealer).
  RBI AR FY26: total reported fraud Rs 48,021 cr; advances-category fraud
    Rs 40,774 cr (8,640 cases); FY25 Rs 30,367 cr (7,924); FY24 Rs 8,917 cr
    (4,105).

v2 change (AR-primary upgrade): the covered book previously used CRISIL
secondary mix fractions (2W 28% + CD 21% -> Rs 15,013 cr); it now uses the
audited Note-18 sector exposures directly (Rs 19,029 cr).  The Y4/Y5 extension
book and the growth-unlock book use the audited Personal-loans and Agriculture
(tractor) exposures instead of AUM-fraction approximations.

Model shape (5 years, crore rupees):
  BASE: fraud-loss avoidance + recovery uplift on the dealer-sourced
        Vehicles + consumer-durable book, ramped region-first then whole
        segment, then extended to tractor + personal loan in Y4-Y5.
        Y1 small-loss (build) -> Y2 ~breakeven -> Y3+ positive.
  UPSIDE: "growth unlock" -- the risk layer reduces the credit-cost drag that
        led TVS to throttle new tractor and personal-loan disbursals in FY25
        (CRISIL).  A controlled re-acceleration of those segments adds NII.
        Clearly labelled as a scenario, not the base.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jale.data import tvs_data as TVS  # audited AR Note-18 figures

REPORTS = Path(__file__).resolve().parents[1] / "reports"
DISCOUNT = 0.12

# ---- audited anchors (crore rupees) -------------------------------------------
VEH = TVS.SECTORS_FY26["Vehicles (2W/used-car/3W/CV)"]
CD = TVS.SECTORS_FY26["Consumer durable"]
PL = TVS.SECTORS_FY26["Personal loans"]
AGRI = TVS.SECTORS_FY26["Agriculture & allied (tractor)"]
A = dict(aum=TVS.AUM_CR, net_book=TVS.NET_LOAN_BOOK_CR, pat=TVS.PAT_CR,
         nnpa=TVS.NET_NPA_PCT / 100, writeoffs=TVS.WRITEOFF_FY26_CR,
         stage3=758.0,                       # CRISIL secondary (provenance only)
         gross_exposure=TVS.GROSS_EXPOSURE_CR,
         target_book=VEH[0] + CD[0],         # 19,029.23 -- audited
         target_book_npa=VEH[1] + CD[1],     # 403.93 -- audited
         dealer_npa_pct=TVS.DEALER_NPA_FY26_PCT)  # 0.23 -- fairness anchor
RBI = dict(total=48021.0, adv=40774.0, adv_cases=8640,
           hist=[("FY24", 8917.0, 4105), ("FY25", 30367.0, 7924),
                 ("FY26", 40774.0, 8640)])

# ---- derived, from anchors only --------------------------------------------
credit_cost = A["writeoffs"] / A["net_book"]                # 1.44% of net book
covered_base = A["target_book"]                             # audited Note-18 book
covered_ext = covered_base + PL[0] + AGRI[0]                # Y4/Y5 full-book ext
severity = [(y, v / c) for y, v, c in RBI["hist"]]

# ---- labelled assumptions (every one shown on the slide) --------------------
ASSUMPTIONS = dict(
    ring_share=0.20,          # [A2] organised-ring share of credit losses
    prevention_full=0.50,     # [A3] share of ring losses prevented at scale
    recovery_uplift=0.10,     # [A4] extra % of ring exposure recovered early
    ramp=(0.15, 0.52, 1.00),  # TW+CD coverage ramp Y1/Y2/Y3
    build_y1=8.0, opex_ramp=(2.5, 3.5, 4.5, 5.0, 5.5),
    change_mgmt_y1=1.5, friction_rate=0.0004,
    # upside (growth unlock) -- labelled scenario, not base
    throttled_book=PL[0] + AGRI[0],   # audited: personal loans + agri(tractor) = 9,081.50 cr
    unlock_yield_gain=0.005,  # [U1] net interest uplift enabled by lower risk drag
    unlock_start_y2=True)

def year_profile(prevention_full=None, ring_share=None, credit_cost=None,
                 recovery_uplift=None, build=None, opex_ramp=None,
                 extend=False):
    prevention_full = prevention_full or ASSUMPTIONS["prevention_full"]
    ring_share = ring_share or ASSUMPTIONS["ring_share"]
    credit_cost = credit_cost or globals()["credit_cost"]
    recovery_uplift = (ASSUMPTIONS["recovery_uplift"]
                       if recovery_uplift is None else recovery_uplift)
    build = ASSUMPTIONS["build_y1"] if build is None else build
    opex_ramp = ASSUMPTIONS["opex_ramp"] if opex_ramp is None else opex_ramp
    rows = []
    net_npv = 0.0
    for yr in range(1, 6):
        # covered book: TW+CD ramp in Y1-3, then tractor+PL added Y4-5
        if yr <= 3:
            book = covered_base * ASSUMPTIONS["ramp"][yr - 1]
        else:
            book = covered_ext
        prev = {1: 0.35, 2: 0.70, 3: 1.0, 4: 1.0, 5: 1.0}[yr] * prevention_full
        exposure = book * credit_cost * ring_share
        avoided = exposure * prev
        recovered = exposure * recovery_uplift * prev      # early-warning uplift
        cost = (build if yr == 1 else 0) + opex_ramp[yr - 1] + \
               (ASSUMPTIONS["change_mgmt_y1"] if yr == 1 else 0) + \
               ASSUMPTIONS["friction_rate"] * book
        net = avoided + recovered - cost
        rows.append(dict(year=f"Y{yr}", covered=round(book, 0),
                         avoided=round(avoided, 1),
                         recovered=round(recovered, 1),
                         cost=round(cost, 1), net=round(net, 1)))
        net_npv += net / ((1 + DISCOUNT) ** yr)
    return rows, net_npv


def upside_scenario():
    """Growth-unlock case (clearly labelled). Adds net interest on the
    throttled tractor + personal-loan book (FY25: disbursals limited for
    risk reasons). Conservative: only 30% of the throttled book re-opens,
    and the yield gain is capped at 50bps of that book.
    """
    rows, _ = year_profile()
    unlock = ASSUMPTIONS["throttled_book"] * 0.30     # re-opened at scale
    ramp = {1: 0.0, 2: 0.3, 3: 0.6, 4: 0.9, 5: 1.0}   # share unlocked by year
    for r in rows:
        yr = int(r["year"][1])
        extra = unlock * ASSUMPTIONS["unlock_yield_gain"] * ramp[yr]
        r["net_upside"] = round(r["net"] + extra, 1)
    npv_u = sum(r["net_upside"] / ((1 + DISCOUNT) ** i)
                for i, r in enumerate(rows, start=1))
    return rows, npv_u


def monte_carlo(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rows, npv = year_profile(
            prevention_full=rng.uniform(0.35, 0.60),
            ring_share=rng.uniform(0.10, 0.25),
            credit_cost=rng.uniform(0.012, 0.018),
            recovery_uplift=rng.uniform(0.05, 0.15),
            build=rng.uniform(6, 10))
        out.append([r["net"] for r in rows] + [npv])
    return np.array(out)


if __name__ == "__main__":
    rows, npv = year_profile()
    rows_u, npv_u = upside_scenario()
    mc = monte_carlo()
    pct = np.percentile(mc, [10, 50, 90], axis=0)     # (3, 6): last col = npv
    res = dict(
        anchors=A, rbi=RBI, assumptions=ASSUMPTIONS,
        derived=dict(credit_cost=credit_cost, covered_base=covered_base,
                     covered_ext=covered_ext,
                     severity=[(y, round(s, 2)) for y, s in severity]),
        base=rows, npv_base=round(npv, 1),
        upside=rows_u, npv_upside=round(npv_u, 1),
        mc=dict(net_y1=pct[:, 0].tolist(), net_y3=pct[:, 2].tolist(),
                net_y5=pct[:, 4].tolist(), npv_pct=pct[:, 5].tolist(),
                p_npv_positive=float((mc[:, 5] > 0).mean())))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "financial_model.json").write_text(json.dumps(res, indent=1, default=str))
    sev_s = " -> ".join(f"{y}: Rs {s:.2f} cr/case" for y, s in severity)
    print(f"credit-cost proxy: {credit_cost:.2%} of net book | "
          f"covered dealer-sourced book (Vehicles+CD, audited): Rs {covered_base:,.0f} cr | "
          f"RBI advance-fraud severity/case: {sev_s}")
    print("BASE   " + "  ".join(f"{r['year']}: {r['net']:+5.1f}" for r in rows) +
          f"   NPV {npv:+.1f}")
    print("UPSIDE " + "  ".join(f"{r['year']}: {r['net_upside']:+5.1f}" for r in rows_u) +
          f"   NPV {npv_u:+.1f}")
    print(f"MC NPV P10/P50/P90: {pct[0, 5]:.1f} / {pct[1, 5]:.1f} / {pct[2, 5]:.1f} cr "
          f"| P(NPV>0) = {(mc[:, 5] > 0).mean():.0%}")
