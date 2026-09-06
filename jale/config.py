"""Central configuration for JALE (Joint Anomaly & Linkage Engine) V1.

Every tunable lives here. Nothing downstream reads magic numbers out of thin
air, which matters because the whole point of V1 is that a reviewer can audit
exactly what was assumed.

Two profiles are provided:

``smoke``  -- small enough to run on a 2-core / 1 GB machine (the sandbox this
              was developed in). Use for CI, tests and logic verification.
``full``   -- the Colab profile. Scales the population up so that community
              detection and graph regularisation behave realistically.

Scale is a *parameter*, never a hard-coded assumption baked into an algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------
# Geographic / demographic pools
# --------------------------------------------------------------------------
# A deliberately small, coarse geography. Rural two-wheeler lending in India is
# concentrated in a handful of states with dense dealer networks; reproducing
# that concentration is what makes "many unrelated people in one pincode" a
# realistic hard negative rather than a modelling bug.
STATES: dict[str, list[str]] = {
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Erode"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Belagavi", "Shivamogga"],
    "Maharashtra": ["Pune", "Nashik", "Nagpur", "Kolhapur", "Aurangabad"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Gorakhpur", "Meerut"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "West Bengal": ["Kolkata", "Siliguri", "Durgapur", "Asansol", "Berhampore"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Udaipur", "Bikaner"],
    "Assam": ["Silchar", "Guwahati", "Dibrugarh", "Jorhat", "Nagaon"],
}

PRODUCTS: dict[str, dict[str, Any]] = {
    "two_wheeler": {"amount_mean": 85_000, "amount_sd": 25_000,
                    "tenure_choices": [12, 24, 36, 48], "share": 0.46},
    "three_wheeler": {"amount_mean": 210_000, "amount_sd": 55_000,
                      "tenure_choices": [24, 36, 48], "share": 0.11},
    "used_car": {"amount_mean": 340_000, "amount_sd": 90_000,
                 "tenure_choices": [24, 36, 48, 60], "share": 0.16},
    "consumer_durable": {"amount_mean": 42_000, "amount_sd": 18_000,
                         "tenure_choices": [6, 9, 12, 18], "share": 0.17},
    "used_commercial_vehicle": {"amount_mean": 420_000, "amount_sd": 110_000,
                                "tenure_choices": [36, 48, 60], "share": 0.10},
}

EMPLOYMENT_TYPES: list[str] = ["salaried", "self_employed", "agriculturist",
                               "gig_worker", "student_with_coapplicant"]

BANKS: list[str] = ["SBI", "PNB", "BoB", "Canara", "UnionBank", "IOB",
                    "HDFC", "ICICI", "Axis", "Kotak", "IPPB", "PPI-Wallet"]


# --------------------------------------------------------------------------
# Benign relational structure -- the source of hard negatives
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BenignStructure:
    """Parameters governing *legitimate* relational clustering.

    These are the knobs that make the problem hard. Rural India genuinely has
    Common Service Centres (CSCs) where one kiosk device files applications for
    hundreds of unrelated villagers; genuine families share an address and a
    phone; genuine repeat guarantors exist. A detector that flags all of these
    is useless, so they must be present in the data.
    """
    household_size_min: int = 1
    household_size_max: int = 5
    household_share_device_p: float = 0.35      # family members sharing one phone
    csc_kiosk_probability: float = 0.14         # P(application filed via shared kiosk)
    csc_apps_per_kiosk: tuple[int, int] = (8, 220)
    repeat_guarantor_p: float = 0.10            # P(a person has guaranteed before)
    repeat_guarantor_max: int = 4
    dealer_power_law_exponent: float = 1.55     # few dealers carry most volume
    duplicate_record_rate: float = 0.045        # fraction of customers with a 2nd CRM record
    branch_locality_p: float = 0.72             # P(bank branch in applicant's district)
    same_pincode_neighbour_p: float = 0.55      # within a household


# --------------------------------------------------------------------------
# Fraud ring typologies
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RingTypology:
    """One way a fraud ring manifests.

    ``benign_link_rate`` is the camouflage knob: the expected number of links
    each ring member additionally forms to *benign* entities. At 0.0 rings are
    trivially separable; raising it forces the detector to work. This is the
    parameter the CARE-GNN literature calls relation camouflage.
    """
    name: str
    ring_size: tuple[int, int]
    burst_days: tuple[int, int]
    benign_link_rate: float
    first_payment_default_p: float
    description: str


RING_TYPOLOGIES: dict[str, RingTypology] = {
    "device_farm": RingTypology(
        name="device_farm", ring_size=(4, 14), burst_days=(1, 6),
        benign_link_rate=0.6, first_payment_default_p=0.85,
        description="Several distinct identities applying from a small pool of "
                    "device fingerprints through one dealer inside a few days."),
    "guarantor_star": RingTypology(
        name="guarantor_star", ring_size=(5, 16), burst_days=(3, 18),
        benign_link_rate=0.8, first_payment_default_p=0.78,
        description="Many mutually-unrelated applicants nominating the same one "
                    "or two guarantors."),
    "dealer_collusion": RingTypology(
        name="dealer_collusion", ring_size=(6, 22), burst_days=(2, 14),
        benign_link_rate=1.0, first_payment_default_p=0.80,
        description="A single dealer sourcing a burst of applications with "
                    "correlated device and guarantor reuse."),
    "disbursement_sink": RingTypology(
        name="disbursement_sink", ring_size=(5, 15), burst_days=(4, 25),
        benign_link_rate=0.9, first_payment_default_p=0.88,
        description="Loans to nominally distinct borrowers disbursing into a "
                    "narrow set of bank accounts."),
    "identity_reuse": RingTypology(
        name="identity_reuse", ring_size=(2, 6), burst_days=(1, 40),
        benign_link_rate=0.4, first_payment_default_p=0.70,
        description="One real person re-applying under variant name spellings "
                    "and a shared device / account."),
}


# --------------------------------------------------------------------------
# Repayment behaviour
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RepaymentProcess:
    """Baseline (benign) delinquency process.

    Deliberately non-trivial: benign borrowers in this segment *do* miss EMIs.
    If benign borrowers never missed payments, "has a missed EMI" would leak
    fraud membership and the whole exercise would be circular.
    """
    benign_ever_dpd30_p: float = 0.22
    benign_first_payment_default_p: float = 0.04
    dpd_severity_mean: float = 18.0
    seasonality_amplitude: float = 0.30   # monsoon / harvest cycle


# --------------------------------------------------------------------------
# Observation time -- a first-class design decision, not an afterthought
# --------------------------------------------------------------------------
class ObservationTime:
    """When the detector is allowed to look.

    This exists because EMI repayment history is a *legitimate* feature but also
    a shortcut: in the generated population ring members miss installments at
    ~5x the benign rate, so any model given EMI history can score well without
    ever looking at the graph. That would let a V1 "prove" the graph matters
    while the graph was doing nothing.

    ``APPLICATION`` is the faithful setting for problem (e), which asks to
    "predict emerging fraud ecosystems *before fraud occurs*": at application
    time no EMI history exists, so the relational signal is the only signal.
    ``POST_DISBURSEMENT`` is reported alongside it as the harder, more realistic
    production setting where EMI history is available to everyone.

    Both are always reported. Never only the flattering one.
    """
    APPLICATION = "application"
    POST_DISBURSEMENT = "post_disbursement"


# --------------------------------------------------------------------------
# Run profiles
# --------------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    n_persons: int
    apps_per_person_lambda: float
    n_dealers: int
    n_devices_per_person_cap: int
    fraud_ring_fraction: float          # fraction of applications inside a ring
    history_days: int
    emi_schedule_max_months: int
    seed: int
    benign: BenignStructure = field(default_factory=BenignStructure)
    repayment: RepaymentProcess = field(default_factory=RepaymentProcess)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


SMOKE = Profile(
    name="smoke", n_persons=6_000, apps_per_person_lambda=0.55, n_dealers=180,
    n_devices_per_person_cap=2, fraud_ring_fraction=0.030, history_days=540,
    emi_schedule_max_months=24, seed=20260827,
)

FULL = Profile(
    name="full", n_persons=120_000, apps_per_person_lambda=0.60, n_dealers=2_400,
    n_devices_per_person_cap=3, fraud_ring_fraction=0.025, history_days=730,
    emi_schedule_max_months=48, seed=20260827,
)

PROFILES: dict[str, Profile] = {"smoke": SMOKE, "full": FULL}


def get_profile(name: str = "smoke") -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {list(PROFILES)}")
    return PROFILES[name]
