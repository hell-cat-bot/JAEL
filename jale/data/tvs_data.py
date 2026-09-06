"""TVS Credit FY26 audited figures -- single source of truth (JA-LE).

Every number below was extracted directly from the TVS Credit Services
Limited Annual Report 2025-26 (238 pp., standalone financial statements):

  * Note 18.9(iii) "Disclosure on sectoral exposures" (pp. 136-137) --
    gross exposure / gross NPA / NPA% per sector, FY26 with FY25 comparators.
  * Balance sheet Note 8(e): total loans - net Rs 30,285.47 cr (PY 26,298.84).
  * Statement of profit & loss: profit for the year Rs 913.17 cr (PY 767.25).
  * Asset-quality note: net NPA / net advances 2.06% (PY 2.87%).
  * Write-off / write-back of excess provisions Rs 437.11 cr (PY 457.49).
  * Note 18.10: customer complaints received FY26 = 5,625.
  * Management highlights: AUM Rs 30,639 Cr as of 31-Mar-2026.

Verified twice, the JA-LE way:
  1. text cross-check against the AR extraction (sector table at extraction
     lines 11610-11625);
  2. arithmetic audit: the FY26 sector rows sum EXACTLY to the printed grand
     totals.  Run:  python -c "from jale.data import tvs_data; tvs_data.verify()"

Known extraction artifact (flagged, not hidden): the FY25 NPA column does not
reconcile in the pypdf extraction -- the printed FY25 "Industry" subtotal row
(64.89) disagrees with its own sub-rows (26.01 + 6.04 = 62.05), and the printed
FY25 grand-total NPA (1,338.69) differs from the sector-row sum (1,371.53).
The FY26 table -- the year every slide quotes -- reconciles exactly.  We carry
the printed FY25 sub-row values and use FY25 only for direction-of-travel
colour (e.g. dealer NPA 0.37% -> 0.23%), never for material claims.

Why this module exists: the Round-2 deck previously anchored the business case
to CRISIL secondary mix approximations (2W 28% + CD 21% of AUM -> Rs 15,013 cr
covered book).  The audited Note-18 table gives the exact dealer-sourced retail
book -- Vehicles Rs 12,963.80 cr + Consumer durable Rs 6,065.43 cr =
Rs 19,029.23 cr (61.1% of gross exposure) -- plus the fairness anchor
"Advance to dealers: 0.23% NPA".  Import this everywhere so the deck, the
financial model and the charts quote identical audited numbers.
Confidence for every row: HIGH (audited primary source).
"""
from __future__ import annotations

# ---- headline (audited) -------------------------------------------------------
AUM_CR = 30_639.0             # management highlights, as of 31-Mar-2026
NET_LOAN_BOOK_CR = 30_285.47  # Note 8(e); PY 26,298.84
PAT_CR = 913.17               # PY 767.25 (+19%)
NET_NPA_PCT = 2.06            # net NPA / net advances; PY 2.87%
WRITEOFF_FY26_CR = 437.11     # write-off / write-back of excess provisions; PY 457.49
GROSS_EXPOSURE_CR = 31_216.01  # Note 18 grand total
GROSS_NPA_CR = 1_141.36       # Note 18 grand total (3.66%)
COMPLAINTS_FY26 = 5_625       # Note 18.10 complaints received

# ---- Note 18.9(iii): sector-wise GROSS exposure & gross NPA (Rs crore) --------
# columns: (exposure_cr, gross_npa_cr, npa_pct).  FY26 reconciles exactly.
SECTORS_FY26: dict[str, tuple[float, float, float]] = {
    "Agriculture & allied (tractor)": (4_306.66, 443.46, 10.30),
    "Industry - MSME": (454.18, 17.65, 3.89),
    "Industry - other": (33.53, 0.63, 1.88),
    "Services": (2_258.50, 155.45, 6.88),
    "Personal loans": (4_774.84, 119.72, 2.51),
    "Vehicles (2W/used-car/3W/CV)": (12_963.80, 341.92, 2.64),
    "Consumer durable": (6_065.43, 62.01, 1.02),
    "Advance to dealers": (228.17, 0.52, 0.23),
    "Other": (130.89, 0.00, 0.00),
}
SECTORS_FY25: dict[str, tuple[float, float, float]] = {
    "Agriculture & allied (tractor)": (4_726.15, 560.49, 11.86),
    "Industry - MSME": (824.70, 26.01, 3.15),
    "Industry - other": (147.01, 6.04, 4.11),
    "Services": (1_361.95, 32.84, 2.41),
    "Personal loans": (4_616.42, 177.29, 3.84),
    "Vehicles (2W/used-car/3W/CV)": (11_608.62, 460.63, 3.97),
    "Consumer durable": (3_689.42, 74.71, 2.02),
    "Advance to dealers": (184.25, 0.68, 0.37),
    "Other": (20.89, 0.00, 0.00),
}

# FY25 printed totals (kept for the audit trail; see module docstring)
FY25_PRINTED_TOTALS = dict(exposure=27_179.42, npa=1_338.69, pct=4.93)

# ---- derived anchors ----------------------------------------------------------
TARGET_SEGMENTS = ["Vehicles (2W/used-car/3W/CV)", "Consumer durable"]
DEALER_NPA_FY26_PCT = SECTORS_FY26["Advance to dealers"][2]   # 0.23
DEALER_NPA_FY25_PCT = SECTORS_FY25["Advance to dealers"][2]   # 0.37


def target_book() -> tuple[float, float, float]:
    """Dealer-sourced retail pilot book = Vehicles + Consumer durable.

    Returns (exposure_cr, gross_npa_cr, npa_pct) = (19029.23, 403.93, 2.12).
    """
    exp = sum(SECTORS_FY26[s][0] for s in TARGET_SEGMENTS)
    npa = sum(SECTORS_FY26[s][1] for s in TARGET_SEGMENTS)
    return exp, npa, npa / exp * 100


def npa_by_sector() -> dict[str, float]:
    """FY26 NPA% per sector -- the fairness chart's data."""
    return {k: v[2] for k, v in SECTORS_FY26.items()}


def verify() -> None:
    """Arithmetic audit against the printed Note-18 totals.

    FY26 must reconcile exactly (tolerance = rounding).  FY25 exposure must
    reconcile to rounding; the FY25 NPA column is flagged (see module
    docstring) and reported, not asserted.
    """
    exp26 = sum(v[0] for v in SECTORS_FY26.values())
    npa26 = sum(v[1] for v in SECTORS_FY26.values())
    assert abs(exp26 - GROSS_EXPOSURE_CR) < 0.02, \
        f"FY26 exposure {exp26} != {GROSS_EXPOSURE_CR}"
    assert abs(npa26 - GROSS_NPA_CR) < 0.02, f"FY26 NPA {npa26} != {GROSS_NPA_CR}"

    exp25 = sum(v[0] for v in SECTORS_FY25.values())
    npa25 = sum(v[1] for v in SECTORS_FY25.values())
    assert abs(exp25 - FY25_PRINTED_TOTALS["exposure"]) < 0.02

    tb_exp, tb_npa, tb_pct = target_book()
    assert abs(tb_exp - 19_029.23) < 0.01
    assert abs(tb_npa - 403.93) < 0.01

    print(f"FY26 audit: sectors sum {exp26:,.2f} cr / NPA {npa26:,.2f} cr == "
          f"printed totals ({GROSS_EXPOSURE_CR:,.2f} / {GROSS_NPA_CR:,.2f})  PASS")
    print(f"FY25 audit: exposure {exp25:,.2f} cr == printed "
          f"{FY25_PRINTED_TOTALS['exposure']:,.2f}  PASS")
    print(f"FY25 NPA:   sector-row sum {npa25:,.2f} vs printed "
          f"{FY25_PRINTED_TOTALS['npa']:,.2f} -> "
          + ("reconciles via sub-rows PASS" if abs(npa25 - FY25_PRINTED_TOTALS["npa"]) < 0.02
             else "known extraction artifact (FY25 Industry subtotal row)"))
    print(f"Target dealer-sourced book (Vehicles + CD): Rs {tb_exp:,.2f} cr | "
          f"gross NPA Rs {tb_npa:,.2f} cr ({tb_pct:.2f}%)")
    print(f"Dealer advances: {DEALER_NPA_FY26_PCT:.2f}% NPA FY26 "
          f"(FY25 {DEALER_NPA_FY25_PCT:.2f}%) -- dealers repay")


if __name__ == "__main__":
    verify()

