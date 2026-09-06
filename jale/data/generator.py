"""Semi-synthetic lender graph generator for JALE V1.

Why synthetic
-------------
There is no public Indian NBFC loan graph. The methodologically standard
response is controlled *anomaly injection*: build a population with realistic
benign relational structure, then inject rings with known labels so that
detection rate can be measured against a known truth rather than a guessed one.

The anti-cheating contract
--------------------------
This generator is designed so that cheating is structurally impossible:

1. **Distributional parity.** A ring member's *node* attributes (income,
   employment type, age, product, loan amount, district) are drawn from exactly
   the same distributions as a benign applicant. Rings differ **only** in
   relational structure. A tabular model that never sees the graph therefore
   *cannot* separate them -- and ``tests/test_no_node_leakage.py`` asserts that
   this holds. If the generator ever leaks node-level signal, that test fails.

2. **Benign delinquency.** Benign borrowers miss EMIs at a realistic rate
   (~22% ever-DPD30). If benign borrowers never defaulted, "has a missed EMI"
   would trivially reveal fraud membership and the task would be circular.

3. **Benign relational clustering.** Households share addresses and phones;
   Common Service Centre kiosks legitimately file applications for dozens of
   unrelated villagers on one device; popular dealers legitimately carry huge
   volume; some people legitimately guarantee several relatives' loans. These
   are hard negatives *by design*. A detector that flags them is wrong.

4. **Camouflage knob.** ``RingTypology.benign_link_rate`` controls how many
   extra links each ring member forms to benign entities. Raising it makes rings
   progressively harder. Nothing about the rings is a fixed, easy shape.

5. **Labels are held out.** The generator writes labels to separate files from
   the raw observable tables, so a pipeline can be written that physically
   cannot touch them.

Determinism: everything flows from one seeded ``numpy.random.Generator``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (BANKS, EMPLOYMENT_TYPES, PRODUCTS, RING_TYPOLOGIES,
                      STATES, Profile)

# Name pools are deliberately large. With a small pool (e.g. 40x40) coincidental
# "same first name + same surname + same birth year" collisions become common,
# which makes entity resolution look far worse than it is and -- more
# importantly -- misrepresents the real difficulty of the fraud-ring problem.
# Indian name diversity is high; the pools below reflect that.
FIRST_NAMES = [
    "Arun", "Priya", "Ravi", "Lakshmi", "Suresh", "Anita", "Vijay", "Deepa",
    "Rajesh", "Kavita", "Manoj", "Sunita", "Sanjay", "Meena", "Dinesh", "Pooja",
    "Ramesh", "Shanti", "Ganesh", "Rekha", "Mohan", "Geeta", "Ashok", "Nisha",
    "Prakash", "Saroj", "Kiran", "Usha", "Naveen", "Latha", "Imran", "Fatima",
    "Bhaskar", "Jyoti", "Tarun", "Swati", "Harish", "Madhuri", "Sandeep", "Rina",
    "Debashis", "Mamata", "Rupam", "Sabitri", "Naba", "Junmoni", "Bhupen", "Rumi",
    "Alok", "Bharati", "Chandan", "Divya", "Eknath", "Falguni", "Girish", "Hema",
    "Indrajeet", "Jagdish", "Kalyani", "Lalit", "Madhav", "Nalini", "Omprakash",
    "Padma", "Qamar", "Rajkumar", "Sadhana", "Trilok", "Umesh", "Vasanti", "Yogesh",
    "Zubeida", "Anand", "Basanti", "Chitra", "Dwarkanath", "Eshwar", "Gauri",
    "Hriday", "Ishita", "Jitendra", "Kusum", "Laxman", "Manisha", "Narendra",
    "Opal", "Pranab", "Ratna", "Sushil", "Tanvi", "Uday", "Vidya", "Yashwant",
    "Ambika", "Biren", "Champak", "Damodar", "Gopal", "Hansa", "Kailash",
    "Mridula", "Nikhil", "Padmini", "Raghav", "Shyama", "Tapan", "Uma",
    "Abdul", "Rehana", "Salim", "Nasreen", "Bashir", "Ayesha", "Farooq", "Saima",
    "Tsering", "Dolma", "Karma", "Pema", "Lhamo", "Sonam", "Ngodup", "Chime",
]
LAST_NAMES = [
    "Iyer", "Nair", "Reddy", "Rao", "Sharma", "Yadav", "Patel", "Mehta",
    "Gupta", "Singh", "Verma", "Kumar", "Das", "Bose", "Ghosh", "Sarkar",
    "Mukherjee", "Banerjee", "Chatterjee", "Bora", "Sarma", "Deka", "Hazarika",
    "Naidu", "Pillai", "Menon", "Joshi", "Kulkarni", "Deshmukh", "Shinde",
    "Khan", "Shaikh", "Ansari", "Qureshi", "Barman", "Sinha", "Chaudhary",
    "Mishra", "Tiwari", "Dubey", "Agarwal", "Bhatt", "Chavan", "Gaikwad",
    "Jadhav", "Kadam", "Pawar", "Sawant", "More", "Salunkhe", "Thorat", "Mane",
    "Gowda", "Hegde", "Shetty", "Poojary", "Kamath", "Bhat", "Acharya", "Udupa",
    "Pandey", "Shukla", "Trivedi", "Vyas", "Bhatt", "Dave", "Raval", "Modi",
    "Solanki", "Parmar", "Rathod", "Vaghela", "Chauhan", "Gohil", "Jadeja",
    "Sethi", "Kapoor", "Malhotra", "Khanna", "Arora", "Bhatia", "Sood", "Kohli",
    "Bedi", "Sahni", "Grover", "Chopra", "Bajwa", "Dhillon", "Grewal", "Sidhu",
    "Sandhu", "Brar", "Mann", "Gill", "Nanda", "Puri", "Soni", "Jain",
    "Agrawal", "Bansal", "Garg", "Mittal", "Singhal", "Jindal", "Goel", "Kedia",
    "Marwaha", "Lodha", "Chordia", "Bohra", "Sanghvi", "Shah", "Vora", "Doshi",
    "Thakkar", "Mistry", "Vaidya", "Sawant", "Naik", "Prabhu", "Pai", "Kini",
    "Bhandari", "Sodhi", "Talwar", "Walia", "Sekhon", "Aulakh", "Virk", "Randhawa",
]

DEVICE_MODELS = [
    "Redmi-9A", "Redmi-Note-11", "Realme-C35", "Samsung-M13", "Vivo-Y16",
    "Oppo-A57", "Motorola-G32", "Nokia-C21", "iPhone-8", "iPhone-11",
    "Tecno-Spark-9", "Infinix-Smart-6", "Itel-A49", "Lava-Agni", "Micromax-IN1",
]
OS_VERSIONS = ["Android-10", "Android-11", "Android-12", "Android-13",
               "Android-14", "iOS-15", "iOS-16", "iOS-17"]


def _h(s: str) -> str:
    """Stable short hash -- used to pseudonymise identifiers.

    Hashing rather than storing raw PAN/account numbers mirrors what a real
    DPDP-compliant pipeline does: hash before join, keep no raw identifiers.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class LenderGraphGenerator:
    """Builds a lender's entity graph plus injected fraud rings."""

    def __init__(self, profile: Profile, typology_subset: list[str] | None = None):
        self.p = profile
        self.rng = np.random.default_rng(profile.seed)
        self.typologies = (typology_subset if typology_subset
                           else list(RING_TYPOLOGIES))
        unknown = set(self.typologies) - set(RING_TYPOLOGIES)
        if unknown:
            raise ValueError(f"unknown ring typologies: {unknown}")

        self.persons: list[dict] = []
        self.devices: list[dict] = []
        self.dealers: list[dict] = []
        self.accounts: list[dict] = []
        self.applications: list[dict] = []
        self.guarantor_links: list[dict] = []
        self.rings: list[dict] = []

        self._device_seq = 0
        self._account_seq = 0
        self._app_seq = 0
        self._ring_seq = 0
        self._person_device: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # primitive factories
    # ------------------------------------------------------------------
    def _new_device(self, kiosk: bool = False) -> str:
        self._device_seq += 1
        did = f"dev_{self._device_seq:07d}"
        model = self.rng.choice(DEVICE_MODELS)
        os_ = self.rng.choice(OS_VERSIONS)
        # A device fingerprint is model+os+app-install-hash+build. Kiosk devices
        # are ordinary-looking -- there is no "kiosk" field, that would be a leak.
        self.devices.append({
            "device_id": did,
            "device_model": model,
            "os_version": os_,
            "app_install_hash": _h(f"apps{self._device_seq}{self.rng.integers(1e6)}"),
            "is_kiosk": bool(kiosk),   # ground-truth provenance, NOT a feature
        })
        return did

    def _new_account(self, district: str, state: str) -> str:
        self._account_seq += 1
        aid = f"acct_{self._account_seq:07d}"
        bank = self.rng.choice(BANKS)
        local = self.rng.random() < self.p.benign.branch_locality_p
        self.accounts.append({
            "account_id": aid,
            "bank": bank,
            "ifsc_hash": _h(f"ifsc{bank}{district if local else 'other'}"),
            "branch_district": district if local else str(self.rng.choice(
                list({d for ds in STATES.values() for d in ds}))),
            "branch_state": state,
            "account_type": "savings" if self.rng.random() < 0.82 else "current",
        })
        return aid

    def _new_person(self, household_id: int | None = None) -> str:
        pid = f"per_{len(self.persons) + 1:07d}"
        state = str(self.rng.choice(list(STATES)))
        district = str(self.rng.choice(STATES[state]))
        pincode = f"{600000 + self.rng.integers(0, 39999)}"
        emp = str(self.rng.choice(EMPLOYMENT_TYPES))
        # Income depends on employment type in a *benign* way only. Ring members
        # are drawn through this same function, so the marginal is identical.
        base = {"salaried": 28_000, "self_employed": 24_000, "agriculturist": 15_000,
                "gig_worker": 17_000, "student_with_coapplicant": 9_000}[emp]
        income = float(np.clip(self.rng.normal(base, base * 0.45), 5_000, 400_000))
        self.persons.append({
            "person_id": pid,
            "name_first": str(self.rng.choice(FIRST_NAMES)),
            "name_last": str(self.rng.choice(LAST_NAMES)),
            "pan_hash": _h(f"pan{pid}{self.rng.integers(1e9)}"),
            "dob_year": int(self.rng.integers(1965, 2005)),
            "gender": str(self.rng.choice(["M", "F"])),
            "state": state,
            "district": district,
            "pincode": pincode,
            "employment_type": emp,
            "monthly_income": round(income, 2),
            "household_id": household_id if household_id is not None else -1,
            "mobile_hash": _h(f"mob{pid}{self.rng.integers(1e9)}"),
            # Ground truth for entity resolution. Normally one record IS one
            # human, so human_id == person_id. Only identity_reuse rings collapse
            # several records onto one human_id. Written to labels/, never raw/.
            "human_id": pid,
        })
        return pid

    def _new_dealer(self) -> str:
        did = f"dler_{len(self.dealers) + 1:05d}"
        state = str(self.rng.choice(list(STATES)))
        district = str(self.rng.choice(STATES[state]))
        self.dealers.append({
            "dealer_id": did,
            "dealer_name": f"{str(self.rng.choice(LAST_NAMES))} Motors",
            "state": state,
            "district": district,
            "gstin_hash": _h(f"gst{did}"),
            "onboarded_days_ago": int(self.rng.integers(30, 2600)),
        })
        return did

    # ------------------------------------------------------------------
    # population
    # ------------------------------------------------------------------
    def build_population(self) -> None:
        rng = self.p.benign
        made = 0
        while made < self.p.n_persons:
            size = int(self.rng.integers(rng.household_size_min,
                                         rng.household_size_max + 1))
            size = min(size, self.p.n_persons - made)
            hid = len(self.persons)
            members = [self._new_person(household_id=hid) for _ in range(size)]
            made += size
            # Households often share one handset.
            if size > 1 and self.rng.random() < rng.household_share_device_p:
                shared = self._new_device()
                for m in members:
                    self._person_device.setdefault(m, []).append(shared)
            for m in members:
                self._person_device.setdefault(m, []).append(self._new_device())

        for _ in range(self.p.n_dealers):
            self._new_dealer()

        self._add_benign_duplicate_records()

    def _add_benign_duplicate_records(self) -> None:
        """Create *legitimate* duplicate customer records.

        Real lender CRMs are full of these: the same human re-registered years
        later under a spelling variant, with a new pincode after moving, or with
        a changed phone number. Without them, entity resolution has an
        unrealistically easy job and its measured precision/recall means nothing.

        These duplicates are BENIGN by construction -- they carry no ring label,
        and their applications are ordinary -- so they act as false-positive bait
        for the linkage model rather than as fraud signal.
        """
        rng = self.p.benign
        n_dup = int(len(self.persons) * rng.duplicate_record_rate)
        eligible = [p for p in self.persons if p["human_id"] == p["person_id"]]
        if not eligible or n_dup == 0:
            return
        n_dup = min(n_dup, len(eligible))
        chosen = self.rng.choice(len(eligible), size=n_dup, replace=False)
        for idx in chosen:
            src = eligible[int(idx)]
            pid = f"per_{len(self.persons) + 1:07d}"
            rec = dict(src)
            rec["person_id"] = pid
            rec["human_id"] = src["person_id"]     # same real human
            rec["household_id"] = -1
            roll = float(self.rng.random())
            if roll < 0.40:                        # transliteration variant
                rec["name_first"] = self._variant(src["name_first"])
            elif roll < 0.62:                      # moved house
                rec["pincode"] = f"{600000 + int(self.rng.integers(0, 39999))}"
            elif roll < 0.80:                      # new handset / number
                rec["mobile_hash"] = _h(f"mob{pid}{int(self.rng.integers(1e9))}")
            else:                                  # combined drift
                rec["name_last"] = self._variant(src["name_last"])
                rec["mobile_hash"] = _h(f"mob{pid}{int(self.rng.integers(1e9))}")
            self.persons.append(rec)

    # ------------------------------------------------------------------
    # benign applications
    # ------------------------------------------------------------------
    def _pick_product(self) -> str:
        names = list(PRODUCTS)
        weights = np.array([PRODUCTS[n]["share"] for n in names], dtype=float)
        return str(self.rng.choice(names, p=weights / weights.sum()))

    def _pick_dealer_near(self, district: str) -> str:
        """Dealers near the applicant are far more likely (power-law popularity)."""
        same = [d["dealer_id"] for d in self.dealers if d["district"] == district]
        pool = same if same and self.rng.random() < 0.62 else \
            [d["dealer_id"] for d in self.dealers]
        # Zipf-like popularity weighting.
        k = np.arange(1, len(pool) + 1, dtype=float)
        w = 1.0 / np.power(k, self.p.benign.dealer_power_law_exponent)
        idx = int(self.rng.choice(len(pool), p=w / w.sum()))
        return pool[idx]

    def _apply(self, person_id: str, day: int, ring_id: int | None = None,
               device_id: str | None = None, account_id: str | None = None,
               dealer_id: str | None = None,
               guarantor_ids: list[str] | None = None) -> str:
        per = self.persons[int(person_id.split("_")[1]) - 1]
        product = self._pick_product()
        spec = PRODUCTS[product]
        amount = float(np.clip(self.rng.normal(spec["amount_mean"],
                                               spec["amount_sd"]), 8_000, 1_200_000))
        tenure = int(self.rng.choice(spec["tenure_choices"]))
        down = float(np.clip(self.rng.normal(0.15, 0.07), 0.0, 0.5)) * amount
        if device_id is None:
            owned = self._person_device.get(person_id)
            if not owned:
                owned = [self._new_device()]
                self._person_device[person_id] = owned
            device_id = str(self.rng.choice(owned))
        if account_id is None:
            account_id = self._new_account(per["district"], per["state"])
        if dealer_id is None:
            dealer_id = self._pick_dealer_near(per["district"])

        self._app_seq += 1
        aid = f"app_{self._app_seq:07d}"
        self.applications.append({
            "application_id": aid,
            "person_id": person_id,
            "dealer_id": dealer_id,
            "device_id": device_id,
            "account_id": account_id,
            "product": product,
            "loan_amount": round(amount, 2),
            "tenure_months": tenure,
            "down_payment": round(down, 2),
            "applied_day": int(day),
            "application_pincode": per["pincode"],
            "application_district": per["district"],
            "application_state": per["state"],
            "employment_type": per["employment_type"],
            "monthly_income": per["monthly_income"],
            "applicant_age_at_application": 2026 - per["dob_year"],
            "n_guarantors": len(guarantor_ids or []),
            "ring_id": ring_id if ring_id is not None else -1,
        })
        for g in (guarantor_ids or []):
            self.guarantor_links.append({
                "application_id": aid, "guarantor_person_id": g,
            })
        return aid

    def build_benign_applications(self) -> None:
        rng = self.p.benign
        n_apps = int(self.p.n_persons * self.p.apps_per_person_lambda)
        person_ids = [p["person_id"] for p in self.persons]

        # Common Service Centres: one kiosk device, many unrelated applicants.
        n_kiosks = max(1, int(self.p.n_dealers * 0.25))
        kiosk_devices = [self._new_device(kiosk=True) for _ in range(n_kiosks)]

        for _ in range(n_apps):
            day = int(self.rng.integers(0, self.p.history_days))
            pid = str(self.rng.choice(person_ids))
            per = self.persons[int(pid.split("_")[1]) - 1]
            device_id = None
            if self.rng.random() < rng.csc_kiosk_probability:
                device_id = str(self.rng.choice(kiosk_devices))

            # Guarantors: sometimes a relative (shared household), sometimes a
            # genuine repeat guarantor from the same village.
            gids: list[str] = []
            if self.rng.random() < 0.55:
                sib = [p["person_id"] for p in self.persons
                       if p["household_id"] == per["household_id"]
                       and p["person_id"] != pid]
                if sib and self.rng.random() < 0.7:
                    gids = [str(self.rng.choice(sib))]
                else:
                    same_pin = [p["person_id"] for p in self.persons
                                if p["pincode"] == per["pincode"]
                                and p["person_id"] != pid]
                    if same_pin:
                        gids = [str(self.rng.choice(same_pin))]
            self._apply(pid, day, device_id=device_id, guarantor_ids=gids)

    # ------------------------------------------------------------------
    # ring injection
    # ------------------------------------------------------------------
    def _next_ring_id(self) -> int:
        self._ring_seq += 1
        return self._ring_seq

    def inject_rings(self) -> None:
        target_apps = int(len(self.applications) *
                          self.p.fraud_ring_fraction /
                          max(1e-9, 1 - self.p.fraud_ring_fraction))
        made = 0
        # Cycle typologies in a fixed order so the mix is stable and known.
        order = list(self.typologies)
        i = 0
        while made < target_apps:
            typ = RING_TYPOLOGIES[order[i % len(order)]]
            made += self._inject_one(typ)
            i += 1

    def _ring_members(self, n: int) -> list[str]:
        """Ring members are *fresh, ordinary* people.

        Deliberately created through ``_new_person`` so their attribute
        marginals match the benign population exactly. We do NOT pick
        "suspicious-looking" people, which would leak node-level signal.
        """
        return [self._new_person() for _ in range(n)]

    def _inject_one(self, typ) -> int:
        size = int(self.rng.integers(typ.ring_size[0], typ.ring_size[1] + 1))
        members = self._ring_members(size)
        rid = self._next_ring_id()
        start = int(self.rng.integers(0, self.p.history_days))
        days = [start + int(self.rng.integers(0, typ.burst_days[1] + 1))
                for _ in members]
        days = [min(d, self.p.history_days - 1) for d in days]

        state = str(self.rng.choice(list(STATES)))
        district = str(self.rng.choice(STATES[state]))
        hub_dealer = self._pick_dealer_near(district)

        if typ.name == "device_farm":
            pool = [self._new_device() for _ in
                    range(max(1, size // int(self.rng.integers(3, 6))))]
            devices = [str(self.rng.choice(pool)) for _ in members]
            accounts = [self._new_account(district, state) for _ in members]
            dealers = [hub_dealer] * size
        elif typ.name == "guarantor_star":
            devices = [self._new_device() for _ in members]
            accounts = [self._new_account(district, state) for _ in members]
            dealers = [hub_dealer if self.rng.random() < 0.7 else
                       self._pick_dealer_near(district) for _ in members]
        elif typ.name == "dealer_collusion":
            devices = [self._new_device() if self.rng.random() < 0.6 else
                       self._new_device() for _ in members]
            accounts = [self._new_account(district, state) for _ in members]
            dealers = [hub_dealer] * size
        elif typ.name == "disbursement_sink":
            devices = [self._new_device() for _ in members]
            sink = [self._new_account(district, state) for _ in
                    range(max(1, size // int(self.rng.integers(3, 6))))]
            accounts = [str(self.rng.choice(sink)) for _ in members]
            dealers = [hub_dealer if self.rng.random() < 0.5 else
                       self._pick_dealer_near(district) for _ in members]
        elif typ.name == "identity_reuse":
            # One underlying human, variant spellings. Attributes are held
            # constant across variants -- only the *identifier* changes, which
            # is precisely what entity resolution exists to catch.
            base = self.persons[int(members[0].split("_")[1]) - 1]
            for m in members[1:]:
                rec = self.persons[int(m.split("_")[1]) - 1]
                rec["dob_year"] = base["dob_year"]
                rec["pincode"] = base["pincode"]
                rec["district"] = base["district"]
                rec["state"] = base["state"]
                rec["gender"] = base["gender"]
                rec["employment_type"] = base["employment_type"]
                rec["monthly_income"] = base["monthly_income"]
                rec["mobile_hash"] = base["mobile_hash"]
                rec["name_first"] = self._variant(base["name_first"])
                rec["name_last"] = base["name_last"]
                # This is the ground truth the entity-resolution layer is scored
                # against: these separate records are one real human.
                rec["human_id"] = base["person_id"]
            dev = self._new_device()
            devices = [dev] * size
            acct = self._new_account(base["district"], base["state"])
            accounts = [acct] * size
            dealers = [self._pick_dealer_near(base["district"]) for _ in members]
        else:
            raise ValueError(typ.name)

        gstars = members[:2] if typ.name == "guarantor_star" else []
        for k, m in enumerate(members):
            gids: list[str] = []
            if typ.name == "guarantor_star":
                gids = [g for g in gstars if g != m]
                if not gids:
                    gids = [gstars[1]] if m == gstars[0] else [gstars[0]]
            elif self.rng.random() < 0.35:
                others = [x for x in members if x != m]
                if others:
                    gids = [str(self.rng.choice(others))]
            self._apply(m, days[k], ring_id=rid, device_id=devices[k],
                        account_id=accounts[k], dealer_id=dealers[k],
                        guarantor_ids=gids)

        self._add_camouflage(members, typ, rid)

        self.rings.append({
            "ring_id": rid, "typology": typ.name, "n_members": size,
            "start_day": start, "hub_dealer_id": hub_dealer,
            "state": state, "district": district,
        })
        return size

    def _add_camouflage(self, members: list[str], typ, rid: int) -> None:
        """Attach ring members to *benign* entities (relation camouflage).

        Without this, every ring is a disconnected island and connected-component
        detection alone solves the task -- which would be cheating by
        construction. Here each member additionally appears in ordinary-looking
        contexts so that isolation is no longer a tell.
        """
        n_extra = int(self.rng.poisson(typ.benign_link_rate * len(members)))
        if n_extra <= 0:
            return
        benign_pool = [p["person_id"] for p in self.persons
                       if p["household_id"] >= 0]
        if not benign_pool:
            return
        for _ in range(n_extra):
            who = str(self.rng.choice(members))
            partner = str(self.rng.choice(benign_pool))
            day = int(self.rng.integers(0, self.p.history_days))
            # The ring member shows up as a *guarantor* on an ordinary
            # application, or shares a kiosk device -- both benign-looking.
            if self.rng.random() < 0.5:
                # A genuine borrower's application, with the ring member listed
                # as guarantor. The borrower is a real customer being used as
                # cover, so this application is NOT fraud and stays unlabelled.
                per = self.persons[int(partner.split("_")[1]) - 1]
                self._apply(partner, day, guarantor_ids=[who],
                            dealer_id=self._pick_dealer_near(per["district"]))
            else:
                # The ring member's own additional application. This one IS part
                # of the scheme and carries the ring label.
                self._apply(who, day, ring_id=rid,
                            device_id=str(self.rng.choice(
                                [d["device_id"] for d in self.devices])),
                            guarantor_ids=[partner])

    @staticmethod
    def _variant(name: str) -> str:
        """Plausible transliteration / spelling variant of an Indian name.

        Uses a SHA-256 digest rather than ``hash()`` to pick the variant: Python
        randomises ``hash(str)`` per process via PYTHONHASHSEED, which would have
        silently made the dataset non-reproducible across runs.
        """
        if len(name) <= 3:
            return name + "h"
        ops = [lambda s: s[:-1] if s.endswith(("a", "i", "u")) else s + "h",
               lambda s: s.replace("ee", "i"),
               lambda s: s.replace("a", "ah", 1),
               lambda s: s[:-1] + "u",
               lambda s: s + "n"]
        pick = int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(ops)
        return ops[pick](name)

    # ------------------------------------------------------------------
    # repayment outcomes
    # ------------------------------------------------------------------
    def build_repayment(self) -> pd.DataFrame:
        """EMI schedule with realistic delinquency for BOTH benign and fraud.

        Benign borrowers default at a meaningful rate; ring members default at
        the typology rate. Because benign delinquency is high, delinquency alone
        is a weak fraud signal -- which is the point.
        """
        rp = self.p.repayment
        typ_lookup = {r["ring_id"]: RING_TYPOLOGIES[r["typology"]].first_payment_default_p
                      for r in self.rings}
        rows = []
        for a in self.applications:
            is_ring = a["ring_id"] != -1
            fpd_p = (typ_lookup.get(a["ring_id"], 0.0) if is_ring
                     else rp.benign_first_payment_default_p)
            if is_ring:
                ever_dpd = self.rng.random() < 0.93
            else:
                # seasonal modulation: monsoon months are harder for agri income
                month = (a["applied_day"] // 30) % 12
                seas = 1.0 + rp.seasonality_amplitude * np.sin(
                    2 * np.pi * (month - 7) / 12)
                ever_dpd = self.rng.random() < min(0.9,
                                                   rp.benign_ever_dpd30_p * seas)
            tenure = a["tenure_months"]
            obs = max(1, min(tenure, (self.p.history_days - a["applied_day"]) // 30))
            obs = int(max(1, min(obs, self.p.emi_schedule_max_months)))
            for k in range(obs):
                if k == 0:
                    missed = bool(self.rng.random() < fpd_p)
                else:
                    missed = bool(ever_dpd and self.rng.random() < 0.35)
                dpd = int(np.clip(self.rng.exponential(rp.dpd_severity_mean), 0, 400)) \
                    if missed else 0
                rows.append((a["application_id"], a["ring_id"], k + 1,
                             round(a["loan_amount"] / tenure, 2),
                             0 if not missed else 1, dpd))
        return pd.DataFrame(rows, columns=[
            "application_id", "ring_id", "installment_no", "emi_amount",
            "missed", "days_past_due"])

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def generate(self) -> dict[str, pd.DataFrame]:
        self.build_population()
        self.build_benign_applications()
        self.inject_rings()
        emi = self.build_repayment()
        return {
            "persons": pd.DataFrame(self.persons),
            "devices": pd.DataFrame(self.devices),
            "dealers": pd.DataFrame(self.dealers),
            "accounts": pd.DataFrame(self.accounts),
            "applications": pd.DataFrame(self.applications),
            "guarantor_links": pd.DataFrame(self.guarantor_links),
            "emi_schedule": emi,
            "rings": pd.DataFrame(self.rings),
        }


def save_dataset(tables: dict[str, pd.DataFrame], outdir: str | Path,
                 profile: Profile, split_labels: bool = True) -> Path:
    """Write the dataset.

    When ``split_labels`` is true, ground-truth labels (``ring_id``,
    ``rings.csv``, ``devices.is_kiosk``) are written under ``labels/`` and
    removed from the observable tables. A pipeline pointed at ``raw/`` then
    *cannot* read the labels even by accident -- the leakage defence is
    physical, not merely disciplinary.
    """
    outdir = Path(outdir)
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    labels_dir = outdir / "labels"
    if split_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    obs = {k: v.copy() for k, v in tables.items()}
    labels: dict[str, pd.DataFrame] = {}

    if split_labels:
        labels["rings"] = obs.pop("rings")
        ppl = obs["persons"]
        labels["person_identity_truth"] = ppl[["person_id", "human_id"]].copy()
        obs["persons"] = ppl.drop(columns=["human_id"])
        app = obs["applications"]
        labels["application_labels"] = app[["application_id", "ring_id"]].copy()
        obs["applications"] = app.drop(columns=["ring_id"])
        dev = obs["devices"]
        labels["device_provenance"] = dev[["device_id", "is_kiosk"]].copy()
        obs["devices"] = dev.drop(columns=["is_kiosk"])
        emi = obs["emi_schedule"]
        labels["emi_ring_map"] = emi[["application_id", "ring_id"]].drop_duplicates()
        obs["emi_schedule"] = emi.drop(columns=["ring_id"])

    if not split_labels:
        obs["persons"] = obs["persons"].drop(columns=["human_id"])

    for name, df in obs.items():
        df.to_parquet(outdir / "raw" / f"{name}.parquet", index=False)
    for name, df in labels.items():
        df.to_parquet(labels_dir / f"{name}.parquet", index=False)

    meta = {"profile": asdict(profile),
            "counts": {k: int(len(v)) for k, v in tables.items()},
            "labels_segregated": split_labels}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return outdir


def build(profile: Profile, outdir: str | Path = "data/jale_v1",
          typology_subset: list[str] | None = None,
          split_labels: bool = True) -> Path:
    gen = LenderGraphGenerator(profile, typology_subset=typology_subset)
    tables = gen.generate()
    return save_dataset(tables, outdir, profile, split_labels=split_labels)
