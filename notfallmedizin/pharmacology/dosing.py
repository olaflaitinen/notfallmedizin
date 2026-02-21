# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Drug dosing calculations for emergency medicine.

This module provides weight-based dosing calculators for common
emergency medicine drugs, continuous infusion rate calculators, and
organ-function-based dose adjustments. All calculations follow
current ACLS/PALS guidelines and peer-reviewed pharmacological
references.

Classes
-------
WeightBasedDosingCalculator
    Bolus and intermittent dose calculations with pediatric, renal,
    and hepatic adjustments.
ContinuousInfusionCalculator
    Continuous infusion rate calculations with titration ranges.

Dataclasses
-----------
DosingResult
    Result container for single-dose calculations.
InfusionResult
    Result container for continuous infusion calculations.
DrugProfile
    Internal representation of a drug's dosing parameters.

References
----------
.. [1] Link MS, Berkow LC, Kudenchuk PJ, et al. "Part 7: Adult
   Advanced Cardiovascular Life Support." Circulation. 2015;132(18
   Suppl 2):S444-S464.
.. [2] de Caen AR, Berg MD, Chameides L, et al. "Part 12: Pediatric
   Advanced Life Support." Circulation. 2015;132(18 Suppl 2):
   S526-S542.
.. [3] Cockcroft DW, Gault MH. "Prediction of creatinine clearance
   from serum creatinine." Nephron. 1976;16(1):31-41.
.. [4] Pugh RN, Murray-Lyon IM, Dawson JL, Pietroni MC, Williams R.
   "Transection of the oesophagus for bleeding oesophageal varices."
   Br J Surg. 1973;60(8):646-649.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    ValidationError,
)
from notfallmedizin.core.validators import validate_age


# ======================================================================
# Enumerations
# ======================================================================


class Route(Enum):
    """Administration route for a drug."""

    IV = "IV"
    IV_PUSH = "IV push"
    IV_INFUSION = "IV infusion"
    IM = "IM"
    IO = "IO"
    SC = "SC"
    PO = "PO"
    SL = "SL"
    IN = "intranasal"
    ET = "endotracheal"
    PR = "PR"
    NEBULIZED = "nebulized"


class ChildPughClass(Enum):
    """Child-Pugh hepatic function classification.

    References
    ----------
    .. [1] Pugh RN, et al. Br J Surg. 1973;60(8):646-649.
    """

    A = "A"
    B = "B"
    C = "C"


# ======================================================================
# Result dataclasses
# ======================================================================


@dataclass(frozen=True)
class DosingResult:
    """Container for a weight-based dosing calculation result.

    Attributes
    ----------
    drug_name : str
        Canonical drug name.
    dose_mg : float
        Calculated dose in milligrams.
    dose_per_kg : float
        Dose per kilogram of body weight (mg/kg).
    route : str
        Recommended administration route.
    frequency : str
        Dosing frequency (e.g., "once", "every 6 hours").
    max_single_dose : float or None
        Maximum allowable single dose in mg, or ``None`` if uncapped.
    max_daily_dose : float or None
        Maximum allowable daily dose in mg, or ``None`` if uncapped.
    warnings : list of str
        Clinical warnings or precautions.
    indication : str
        Clinical indication for the dose.
    adjusted_for : list of str
        List of adjustments applied (e.g., "pediatric", "renal").
    """

    drug_name: str
    dose_mg: float
    dose_per_kg: float
    route: str
    frequency: str
    max_single_dose: Optional[float] = None
    max_daily_dose: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    indication: str = ""
    adjusted_for: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InfusionResult:
    """Container for a continuous infusion calculation result.

    Attributes
    ----------
    drug_name : str
        Canonical drug name.
    rate_ml_hr : float
        Infusion rate in mL/hour.
    dose_mcg_kg_min : float
        Target dose in mcg/kg/min.
    concentration_mg_ml : float
        Drug concentration in mg/mL.
    dose_mcg_min : float
        Absolute dose rate in mcg/min.
    titration_min : float or None
        Minimum titration dose in mcg/kg/min.
    titration_max : float or None
        Maximum titration dose in mcg/kg/min.
    warnings : list of str
        Clinical warnings or precautions.
    """

    drug_name: str
    rate_ml_hr: float
    dose_mcg_kg_min: float
    concentration_mg_ml: float
    dose_mcg_min: float
    titration_min: Optional[float] = None
    titration_max: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# ======================================================================
# Internal drug profile
# ======================================================================


@dataclass
class _DosingEntry:
    """Dosing parameters for a single drug-indication pair."""

    dose_per_kg: float
    route: str
    frequency: str
    max_single_dose: Optional[float] = None
    max_daily_dose: Optional[float] = None
    pediatric_dose_per_kg: Optional[float] = None
    pediatric_max_single: Optional[float] = None
    renal_adjustment: Optional[Dict[str, float]] = None
    hepatic_adjustment: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class _InfusionEntry:
    """Infusion parameters for a continuously-infused drug."""

    standard_concentration_mg_ml: float
    titration_min_mcg_kg_min: float
    titration_max_mcg_kg_min: float
    usual_start_mcg_kg_min: float
    warnings: List[str] = field(default_factory=list)


# ======================================================================
# Built-in drug databases
# ======================================================================

# Renal adjustment keys represent CrCl thresholds in mL/min.
# The float value is a multiplier applied to the base dose.
# "severe" = CrCl < 15, "moderate" = CrCl 15-29, "mild" = CrCl 30-59.

_DRUG_DATABASE: Dict[str, Dict[str, _DosingEntry]] = {
    "epinephrine": {
        "cardiac_arrest": _DosingEntry(
            dose_per_kg=0.01,
            route="IV/IO",
            frequency="every 3-5 minutes",
            max_single_dose=1.0,
            pediatric_dose_per_kg=0.01,
            pediatric_max_single=1.0,
            warnings=["High-dose epinephrine (0.1 mg/kg) is not recommended routinely"],
        ),
        "anaphylaxis": _DosingEntry(
            dose_per_kg=0.01,
            route="IM",
            frequency="every 5-15 minutes as needed",
            max_single_dose=0.5,
            pediatric_dose_per_kg=0.01,
            pediatric_max_single=0.3,
            warnings=["Use anterolateral thigh for IM injection"],
        ),
        "bradycardia": _DosingEntry(
            dose_per_kg=0.0,
            route="IV infusion",
            frequency="continuous",
            max_single_dose=None,
            warnings=["Typically given as infusion 2-10 mcg/min"],
        ),
    },
    "amiodarone": {
        "cardiac_arrest": _DosingEntry(
            dose_per_kg=5.0,
            route="IV/IO",
            frequency="first dose; may repeat 2.5 mg/kg",
            max_single_dose=300.0,
            pediatric_dose_per_kg=5.0,
            pediatric_max_single=300.0,
            warnings=["May cause hypotension; infuse over 20-60 min if perfusing rhythm"],
        ),
        "stable_vt": _DosingEntry(
            dose_per_kg=0.0,
            route="IV infusion",
            frequency="150 mg over 10 min, then maintenance",
            max_single_dose=150.0,
            max_daily_dose=2200.0,
            warnings=["Use central line if possible", "Monitor QTc interval"],
        ),
    },
    "atropine": {
        "bradycardia": _DosingEntry(
            dose_per_kg=0.0,
            route="IV",
            frequency="every 3-5 minutes",
            max_single_dose=1.0,
            max_daily_dose=3.0,
            pediatric_dose_per_kg=0.02,
            pediatric_max_single=0.5,
            warnings=["Minimum pediatric dose 0.1 mg to avoid paradoxical bradycardia"],
        ),
        "organophosphate_poisoning": _DosingEntry(
            dose_per_kg=0.05,
            route="IV",
            frequency="every 5-10 minutes, titrate to secretions",
            max_single_dose=2.0,
            warnings=["No maximum total dose in organophosphate poisoning"],
        ),
    },
    "adenosine": {
        "svt": _DosingEntry(
            dose_per_kg=0.0,
            route="IV push",
            frequency="may repeat at 12 mg in 1-2 minutes",
            max_single_dose=12.0,
            pediatric_dose_per_kg=0.1,
            pediatric_max_single=6.0,
            warnings=[
                "Administer via rapid IV push followed by 20 mL NS flush",
                "Use proximal IV site",
                "Half-life approximately 6 seconds",
                "Contraindicated in 2nd/3rd degree heart block",
            ],
        ),
    },
    "ketamine": {
        "rsi": _DosingEntry(
            dose_per_kg=2.0,
            route="IV",
            frequency="once",
            max_single_dose=200.0,
            pediatric_dose_per_kg=2.0,
            pediatric_max_single=200.0,
            warnings=["May increase ICP in spontaneously breathing patients (debated)"],
        ),
        "procedural_sedation": _DosingEntry(
            dose_per_kg=1.5,
            route="IV",
            frequency="may repeat 0.5 mg/kg every 5-10 minutes",
            max_single_dose=200.0,
            pediatric_dose_per_kg=1.5,
            pediatric_max_single=200.0,
            warnings=["Have suction available for hypersalivation", "Consider glycopyrrolate pretreatment"],
        ),
        "analgesia": _DosingEntry(
            dose_per_kg=0.3,
            route="IV",
            frequency="over 15 minutes, may repeat",
            max_single_dose=30.0,
            warnings=["Sub-dissociative dose for analgesia"],
        ),
    },
    "propofol": {
        "rsi": _DosingEntry(
            dose_per_kg=1.5,
            route="IV",
            frequency="once",
            max_single_dose=200.0,
            pediatric_dose_per_kg=2.5,
            pediatric_max_single=200.0,
            warnings=["May cause significant hypotension", "Reduce dose in elderly and hemodynamically unstable"],
        ),
        "procedural_sedation": _DosingEntry(
            dose_per_kg=1.0,
            route="IV",
            frequency="0.5 mg/kg boluses every 3-5 minutes",
            max_single_dose=200.0,
            warnings=["Titrate to effect", "Have airway equipment ready"],
        ),
    },
    "rocuronium": {
        "rsi": _DosingEntry(
            dose_per_kg=1.2,
            route="IV",
            frequency="once",
            max_single_dose=200.0,
            pediatric_dose_per_kg=1.0,
            pediatric_max_single=200.0,
            warnings=["Onset 45-60 seconds at RSI dose", "Reversible with sugammadex"],
        ),
        "intubation": _DosingEntry(
            dose_per_kg=0.6,
            route="IV",
            frequency="once",
            max_single_dose=100.0,
            pediatric_dose_per_kg=0.6,
            pediatric_max_single=100.0,
            warnings=["Standard (non-RSI) intubating dose", "Onset 60-90 seconds"],
        ),
    },
    "succinylcholine": {
        "rsi": _DosingEntry(
            dose_per_kg=1.5,
            route="IV",
            frequency="once",
            max_single_dose=200.0,
            pediatric_dose_per_kg=2.0,
            pediatric_max_single=200.0,
            warnings=[
                "Contraindicated in hyperkalemia, burns >24h, crush injury >24h",
                "Risk of malignant hyperthermia",
                "Fasciculations may increase ICP/IOP",
            ],
        ),
    },
    "fentanyl": {
        "analgesia": _DosingEntry(
            dose_per_kg=1.0,
            route="IV",
            frequency="every 30-60 minutes as needed",
            max_single_dose=100.0,
            pediatric_dose_per_kg=1.0,
            pediatric_max_single=50.0,
            renal_adjustment={"severe": 0.5, "moderate": 0.75, "mild": 1.0},
            hepatic_adjustment={"C": 0.5, "B": 0.75, "A": 1.0},
            warnings=["Dose in mcg, not mg -- 1 mcg/kg", "Monitor for respiratory depression"],
        ),
        "procedural_sedation": _DosingEntry(
            dose_per_kg=1.0,
            route="IV",
            frequency="once, may repeat 0.5 mcg/kg",
            max_single_dose=100.0,
            warnings=["Administer slowly over 1-2 minutes"],
        ),
    },
    "morphine": {
        "analgesia": _DosingEntry(
            dose_per_kg=0.1,
            route="IV",
            frequency="every 2-4 hours as needed",
            max_single_dose=10.0,
            pediatric_dose_per_kg=0.1,
            pediatric_max_single=5.0,
            renal_adjustment={"severe": 0.25, "moderate": 0.5, "mild": 0.75},
            hepatic_adjustment={"C": 0.5, "B": 0.75, "A": 1.0},
            warnings=[
                "Active metabolite (M6G) accumulates in renal failure",
                "May cause histamine release and hypotension",
            ],
        ),
        "acute_pulmonary_edema": _DosingEntry(
            dose_per_kg=0.0,
            route="IV",
            frequency="may repeat every 5-15 minutes",
            max_single_dose=4.0,
            warnings=["Give 2-4 mg IV, titrate carefully", "Monitor respiratory status"],
        ),
    },
    "midazolam": {
        "seizure": _DosingEntry(
            dose_per_kg=0.2,
            route="IM/IN",
            frequency="once, may repeat once",
            max_single_dose=10.0,
            pediatric_dose_per_kg=0.2,
            pediatric_max_single=10.0,
            renal_adjustment={"severe": 0.5, "moderate": 0.75, "mild": 1.0},
            hepatic_adjustment={"C": 0.5, "B": 0.5, "A": 1.0},
            warnings=["IM route preferred when IV not available"],
        ),
        "procedural_sedation": _DosingEntry(
            dose_per_kg=0.05,
            route="IV",
            frequency="titrate every 2-3 minutes",
            max_single_dose=5.0,
            pediatric_dose_per_kg=0.05,
            pediatric_max_single=2.0,
            warnings=["Titrate slowly in elderly patients"],
        ),
    },
    "lorazepam": {
        "seizure": _DosingEntry(
            dose_per_kg=0.1,
            route="IV",
            frequency="may repeat once in 5-10 minutes",
            max_single_dose=4.0,
            pediatric_dose_per_kg=0.1,
            pediatric_max_single=4.0,
            renal_adjustment={"severe": 0.75, "moderate": 1.0, "mild": 1.0},
            hepatic_adjustment={"C": 0.5, "B": 0.75, "A": 1.0},
            warnings=["First-line benzodiazepine for status epilepticus"],
        ),
        "agitation": _DosingEntry(
            dose_per_kg=0.05,
            route="IV/IM",
            frequency="every 4-6 hours as needed",
            max_single_dose=2.0,
            warnings=["Avoid in patients with acute alcohol intoxication"],
        ),
    },
    "phenytoin": {
        "seizure": _DosingEntry(
            dose_per_kg=20.0,
            route="IV",
            frequency="loading dose, max infusion rate 50 mg/min",
            max_single_dose=1500.0,
            pediatric_dose_per_kg=20.0,
            pediatric_max_single=1500.0,
            renal_adjustment={"severe": 0.75, "moderate": 1.0, "mild": 1.0},
            hepatic_adjustment={"C": 0.5, "B": 0.75, "A": 1.0},
            warnings=[
                "Maximum infusion rate 50 mg/min in adults, 1 mg/kg/min in pediatrics",
                "Cardiac monitoring required during infusion",
                "Highly protein-bound; adjust in hypoalbuminemia",
                "Purple glove syndrome risk with peripheral IV",
            ],
        ),
    },
    "levetiracetam": {
        "seizure": _DosingEntry(
            dose_per_kg=60.0,
            route="IV",
            frequency="loading dose over 15 minutes",
            max_single_dose=4500.0,
            pediatric_dose_per_kg=40.0,
            pediatric_max_single=3000.0,
            renal_adjustment={"severe": 0.5, "moderate": 0.5, "mild": 0.75},
            warnings=["Favorable side-effect profile compared to phenytoin"],
        ),
    },
    "alteplase": {
        "acute_stroke": _DosingEntry(
            dose_per_kg=0.9,
            route="IV",
            frequency="10% as bolus, remainder over 60 minutes",
            max_single_dose=90.0,
            warnings=[
                "Must meet inclusion/exclusion criteria for thrombolysis",
                "Administer within 4.5 hours of symptom onset",
                "10% given as IV bolus over 1 minute",
                "Monitor for angioedema and hemorrhage",
            ],
        ),
        "pe": _DosingEntry(
            dose_per_kg=0.0,
            route="IV",
            frequency="100 mg over 2 hours",
            max_single_dose=100.0,
            warnings=[
                "For massive PE with hemodynamic instability",
                "Fixed dose 100 mg over 2 hours",
                "Consider half-dose (50 mg) in submassive PE",
            ],
        ),
    },
    "naloxone": {
        "opioid_reversal": _DosingEntry(
            dose_per_kg=0.0,
            route="IV/IM/IN",
            frequency="every 2-3 minutes, titrate to respiratory effort",
            max_single_dose=2.0,
            pediatric_dose_per_kg=0.1,
            pediatric_max_single=2.0,
            warnings=[
                "Start with 0.04-0.4 mg in opioid-dependent patients to avoid withdrawal",
                "Duration shorter than most opioids; monitor for re-sedation",
                "May need repeated doses or infusion",
            ],
        ),
    },
    "flumazenil": {
        "benzodiazepine_reversal": _DosingEntry(
            dose_per_kg=0.0,
            route="IV",
            frequency="0.2 mg every 1 minute, max total 1 mg",
            max_single_dose=0.2,
            max_daily_dose=1.0,
            pediatric_dose_per_kg=0.01,
            pediatric_max_single=0.2,
            warnings=[
                "Risk of seizures in chronic benzodiazepine users",
                "Contraindicated in known seizure disorder on benzodiazepines",
                "Contraindicated in mixed TCA/benzodiazepine overdose",
                "Short half-life; re-sedation may occur",
            ],
        ),
    },
    "norepinephrine": {
        "septic_shock": _DosingEntry(
            dose_per_kg=0.0,
            route="IV infusion",
            frequency="continuous titration",
            max_single_dose=None,
            warnings=[
                "First-line vasopressor for septic shock",
                "Administer via central line when possible",
                "Titrate to MAP >= 65 mmHg",
            ],
        ),
    },
}


_INFUSION_DATABASE: Dict[str, _InfusionEntry] = {
    "norepinephrine": _InfusionEntry(
        standard_concentration_mg_ml=0.016,
        titration_min_mcg_kg_min=0.01,
        titration_max_mcg_kg_min=3.0,
        usual_start_mcg_kg_min=0.1,
        warnings=["Extravasation can cause tissue necrosis", "Central line preferred"],
    ),
    "epinephrine": _InfusionEntry(
        standard_concentration_mg_ml=0.016,
        titration_min_mcg_kg_min=0.01,
        titration_max_mcg_kg_min=0.5,
        usual_start_mcg_kg_min=0.05,
        warnings=["May cause tachyarrhythmias and myocardial ischemia"],
    ),
    "dopamine": _InfusionEntry(
        standard_concentration_mg_ml=1.6,
        titration_min_mcg_kg_min=2.0,
        titration_max_mcg_kg_min=20.0,
        usual_start_mcg_kg_min=5.0,
        warnings=[
            "2-5 mcg/kg/min: dopaminergic effects",
            "5-10 mcg/kg/min: beta-1 effects",
            ">10 mcg/kg/min: alpha effects predominate",
        ],
    ),
    "dobutamine": _InfusionEntry(
        standard_concentration_mg_ml=1.0,
        titration_min_mcg_kg_min=2.5,
        titration_max_mcg_kg_min=20.0,
        usual_start_mcg_kg_min=5.0,
        warnings=["May cause hypotension via beta-2 vasodilation"],
    ),
    "vasopressin": _InfusionEntry(
        standard_concentration_mg_ml=0.0004,
        titration_min_mcg_kg_min=0.0,
        titration_max_mcg_kg_min=0.0,
        usual_start_mcg_kg_min=0.0,
        warnings=[
            "Fixed dose: 0.03-0.04 units/min (not weight-based)",
            "Used as adjunct to norepinephrine in septic shock",
        ],
    ),
    "phenylephrine": _InfusionEntry(
        standard_concentration_mg_ml=0.1,
        titration_min_mcg_kg_min=0.5,
        titration_max_mcg_kg_min=5.0,
        usual_start_mcg_kg_min=1.0,
        warnings=["Pure alpha-1 agonist; may cause reflex bradycardia"],
    ),
    "propofol": _InfusionEntry(
        standard_concentration_mg_ml=10.0,
        titration_min_mcg_kg_min=25.0,
        titration_max_mcg_kg_min=75.0,
        usual_start_mcg_kg_min=50.0,
        warnings=[
            "Monitor for propofol infusion syndrome with prolonged use",
            "Contains soybean oil and egg lecithin",
        ],
    ),
    "midazolam": _InfusionEntry(
        standard_concentration_mg_ml=0.5,
        titration_min_mcg_kg_min=0.5,
        titration_max_mcg_kg_min=6.0,
        usual_start_mcg_kg_min=1.0,
        warnings=["Active metabolite accumulates in renal failure"],
    ),
    "fentanyl": _InfusionEntry(
        standard_concentration_mg_ml=0.01,
        titration_min_mcg_kg_min=0.5,
        titration_max_mcg_kg_min=3.0,
        usual_start_mcg_kg_min=1.0,
        warnings=["Accumulates with prolonged infusion due to redistribution"],
    ),
    "ketamine": _InfusionEntry(
        standard_concentration_mg_ml=1.0,
        titration_min_mcg_kg_min=5.0,
        titration_max_mcg_kg_min=20.0,
        usual_start_mcg_kg_min=10.0,
        warnings=["Sub-dissociative analgesic infusion range"],
    ),
    "amiodarone": _InfusionEntry(
        standard_concentration_mg_ml=1.8,
        titration_min_mcg_kg_min=0.0,
        titration_max_mcg_kg_min=0.0,
        usual_start_mcg_kg_min=0.0,
        warnings=[
            "Post-arrest: 1 mg/min for 6 hours, then 0.5 mg/min for 18 hours",
            "Not weight-based; fixed-dose protocol",
        ],
    ),
    "nitroglycerin": _InfusionEntry(
        standard_concentration_mg_ml=0.4,
        titration_min_mcg_kg_min=0.0,
        titration_max_mcg_kg_min=0.0,
        usual_start_mcg_kg_min=0.0,
        warnings=[
            "Start 5-10 mcg/min, titrate by 5-10 mcg/min every 3-5 minutes",
            "Not weight-based",
        ],
    ),
    "nicardipine": _InfusionEntry(
        standard_concentration_mg_ml=0.1,
        titration_min_mcg_kg_min=0.0,
        titration_max_mcg_kg_min=0.0,
        usual_start_mcg_kg_min=0.0,
        warnings=[
            "Start 5 mg/hr, titrate by 2.5 mg/hr every 5-15 minutes",
            "Max 15 mg/hr",
            "Not weight-based",
        ],
    ),
}


def _cockcroft_gault(
    age: float,
    weight_kg: float,
    serum_creatinine: float,
    is_female: bool,
) -> float:
    """Estimate creatinine clearance using the Cockcroft-Gault formula.

    Parameters
    ----------
    age : float
        Patient age in years.
    weight_kg : float
        Patient weight in kilograms.
    serum_creatinine : float
        Serum creatinine in mg/dL.
    is_female : bool
        ``True`` if the patient is female.

    Returns
    -------
    float
        Estimated creatinine clearance in mL/min.

    References
    ----------
    .. [1] Cockcroft DW, Gault MH. Nephron. 1976;16(1):31-41.
    """
    if serum_creatinine <= 0:
        raise ValidationError(
            message="Serum creatinine must be positive.",
            parameter="serum_creatinine",
        )
    crcl = ((140.0 - age) * weight_kg) / (72.0 * serum_creatinine)
    if is_female:
        crcl *= 0.85
    return crcl


def _classify_renal_function(crcl: float) -> str:
    """Classify renal function by creatinine clearance thresholds.

    Parameters
    ----------
    crcl : float
        Creatinine clearance in mL/min.

    Returns
    -------
    str
        One of ``"normal"``, ``"mild"``, ``"moderate"``, or ``"severe"``.
    """
    if crcl >= 60.0:
        return "normal"
    elif crcl >= 30.0:
        return "mild"
    elif crcl >= 15.0:
        return "moderate"
    else:
        return "severe"


# ======================================================================
# WeightBasedDosingCalculator
# ======================================================================


class WeightBasedDosingCalculator:
    """Calculate weight-based drug doses for emergency medicine.

    This calculator uses a built-in database of common emergency
    medicine drugs to compute patient-specific doses with optional
    adjustments for pediatric patients, renal impairment, and hepatic
    impairment.

    Parameters
    ----------
    pediatric_age_cutoff : float, optional
        Age (in years) below which pediatric dosing is applied.
        Default is 18.0.
    elderly_age_cutoff : float, optional
        Age (in years) at or above which elderly dose reductions may
        be considered. Default is 65.0.

    Examples
    --------
    >>> calc = WeightBasedDosingCalculator()
    >>> result = calc.calculate_dose("epinephrine", weight_kg=70.0,
    ...                              indication="cardiac_arrest")
    >>> result.dose_mg
    0.7
    >>> result.route
    'IV/IO'

    References
    ----------
    .. [1] Link MS, et al. Circulation. 2015;132(18 Suppl 2):S444-S464.
    .. [2] de Caen AR, et al. Circulation. 2015;132(18 Suppl 2):S526-S542.
    """

    def __init__(
        self,
        pediatric_age_cutoff: float = 18.0,
        elderly_age_cutoff: float = 65.0,
    ) -> None:
        self.pediatric_age_cutoff = pediatric_age_cutoff
        self.elderly_age_cutoff = elderly_age_cutoff
        self._drug_db = _DRUG_DATABASE
        self._infusion_db = _INFUSION_DATABASE

    @property
    def available_drugs(self) -> List[str]:
        """Return a sorted list of drugs in the built-in database.

        Returns
        -------
        list of str
            Drug names.
        """
        return sorted(self._drug_db.keys())

    def get_indications(self, drug_name: str) -> List[str]:
        """Return available indications for a drug.

        Parameters
        ----------
        drug_name : str
            Drug name (case-insensitive).

        Returns
        -------
        list of str
            Available indication keys.

        Raises
        ------
        ValidationError
            If the drug is not in the database.
        """
        key = drug_name.strip().lower()
        if key not in self._drug_db:
            raise ValidationError(
                message=(
                    f"Drug '{drug_name}' not found in database. "
                    f"Available drugs: {self.available_drugs}."
                ),
                parameter="drug_name",
            )
        return sorted(self._drug_db[key].keys())

    def calculate_dose(
        self,
        drug_name: str,
        weight_kg: float,
        indication: str,
        age: Optional[float] = None,
        serum_creatinine: Optional[float] = None,
        is_female: Optional[bool] = None,
        child_pugh: Optional[str] = None,
    ) -> DosingResult:
        """Calculate a weight-based drug dose.

        Parameters
        ----------
        drug_name : str
            Drug name (case-insensitive).
        weight_kg : float
            Patient weight in kilograms.
        indication : str
            Clinical indication (case-insensitive). Call
            :meth:`get_indications` for available options.
        age : float or None, optional
            Patient age in years. Required for pediatric adjustments
            and Cockcroft-Gault renal dose adjustment.
        serum_creatinine : float or None, optional
            Serum creatinine in mg/dL for renal adjustment. If provided,
            ``age``, ``weight_kg``, and ``is_female`` are used for the
            Cockcroft-Gault calculation.
        is_female : bool or None, optional
            Patient sex for the Cockcroft-Gault calculation. Required
            when ``serum_creatinine`` is provided.
        child_pugh : {"A", "B", "C"} or None, optional
            Child-Pugh class for hepatic dose adjustment.

        Returns
        -------
        DosingResult
            Calculated dose with metadata.

        Raises
        ------
        ValidationError
            If the drug or indication is not found, or if required
            parameters are missing for organ-function adjustments.
        ClinicalRangeError
            If weight is out of plausible range.
        """
        drug_key = drug_name.strip().lower()
        indication_key = indication.strip().lower()

        if drug_key not in self._drug_db:
            raise ValidationError(
                message=(
                    f"Drug '{drug_name}' not found. "
                    f"Available: {self.available_drugs}."
                ),
                parameter="drug_name",
            )

        drug_entries = self._drug_db[drug_key]
        if indication_key not in drug_entries:
            raise ValidationError(
                message=(
                    f"Indication '{indication}' not found for '{drug_name}'. "
                    f"Available: {sorted(drug_entries.keys())}."
                ),
                parameter="indication",
            )

        self._validate_weight(weight_kg)
        if age is not None:
            validate_age(age)

        entry = drug_entries[indication_key]
        dose_per_kg = entry.dose_per_kg
        adjustments: List[str] = []
        warnings = list(entry.warnings)

        is_pediatric = age is not None and age < self.pediatric_age_cutoff
        if is_pediatric and entry.pediatric_dose_per_kg is not None:
            dose_per_kg = entry.pediatric_dose_per_kg
            adjustments.append("pediatric")

        renal_multiplier = 1.0
        if serum_creatinine is not None:
            if age is None:
                raise ValidationError(
                    message="Age is required for renal dose adjustment (Cockcroft-Gault).",
                    parameter="age",
                )
            if is_female is None:
                raise ValidationError(
                    message="Sex (is_female) is required for renal dose adjustment.",
                    parameter="is_female",
                )
            crcl = _cockcroft_gault(age, weight_kg, serum_creatinine, is_female)
            renal_class = _classify_renal_function(crcl)
            if renal_class != "normal" and entry.renal_adjustment is not None:
                renal_multiplier = entry.renal_adjustment.get(renal_class, 1.0)
                adjustments.append(f"renal (CrCl={crcl:.1f} mL/min, {renal_class})")
                warnings.append(
                    f"Renal adjustment applied: dose multiplied by {renal_multiplier}"
                )

        hepatic_multiplier = 1.0
        if child_pugh is not None:
            cp_upper = child_pugh.strip().upper()
            if cp_upper not in ("A", "B", "C"):
                raise ValidationError(
                    message=(
                        f"child_pugh must be 'A', 'B', or 'C', got '{child_pugh}'."
                    ),
                    parameter="child_pugh",
                )
            if entry.hepatic_adjustment is not None:
                hepatic_multiplier = entry.hepatic_adjustment.get(cp_upper, 1.0)
                adjustments.append(f"hepatic (Child-Pugh {cp_upper})")
                if hepatic_multiplier < 1.0:
                    warnings.append(
                        f"Hepatic adjustment applied: dose multiplied by {hepatic_multiplier}"
                    )

        if age is not None and age >= self.elderly_age_cutoff:
            warnings.append("Elderly patient: consider conservative dosing and closer monitoring")

        dose_mg = dose_per_kg * weight_kg * renal_multiplier * hepatic_multiplier

        max_single = entry.max_single_dose
        if is_pediatric and entry.pediatric_max_single is not None:
            max_single = entry.pediatric_max_single

        if max_single is not None and dose_mg > max_single:
            dose_mg = max_single
            warnings.append(f"Dose capped at maximum single dose of {max_single} mg")

        dose_mg = round(dose_mg, 4)
        actual_dose_per_kg = round(dose_mg / weight_kg, 4) if weight_kg > 0 else 0.0

        return DosingResult(
            drug_name=drug_key,
            dose_mg=dose_mg,
            dose_per_kg=actual_dose_per_kg,
            route=entry.route,
            frequency=entry.frequency,
            max_single_dose=entry.max_single_dose,
            max_daily_dose=entry.max_daily_dose,
            warnings=warnings,
            indication=indication_key,
            adjusted_for=adjustments,
        )

    @staticmethod
    def _validate_weight(weight_kg: float) -> None:
        """Validate patient weight.

        Parameters
        ----------
        weight_kg : float
            Weight in kilograms.

        Raises
        ------
        ClinicalRangeError
            If weight is outside [0.3, 500] kg.
        """
        if not isinstance(weight_kg, (int, float)):
            raise ValidationError(
                message=f"weight_kg must be numeric, got {type(weight_kg).__name__}.",
                parameter="weight_kg",
            )
        if weight_kg < 0.3 or weight_kg > 500.0:
            raise ClinicalRangeError(
                parameter="weight_kg",
                value=weight_kg,
                lower=0.3,
                upper=500.0,
            )


# ======================================================================
# ContinuousInfusionCalculator
# ======================================================================


class ContinuousInfusionCalculator:
    """Calculate continuous infusion rates for vasopressors and sedatives.

    Provides rate calculations in mL/hr given a target dose in
    mcg/kg/min and a known drug concentration, plus titration range
    information from a built-in database.

    Examples
    --------
    >>> calc = ContinuousInfusionCalculator()
    >>> result = calc.calculate_rate(
    ...     drug="norepinephrine",
    ...     concentration_mg_ml=0.016,
    ...     target_dose_mcg_kg_min=0.1,
    ...     weight_kg=70.0,
    ... )
    >>> round(result.rate_ml_hr, 2)
    26.25

    References
    ----------
    .. [1] Rhodes A, Evans LE, Alhazzani W, et al. "Surviving Sepsis
       Campaign: International Guidelines for Management of Sepsis
       and Septic Shock: 2016." Crit Care Med. 2017;45(3):486-552.
    """

    def __init__(self) -> None:
        self._infusion_db = _INFUSION_DATABASE

    @property
    def available_drugs(self) -> List[str]:
        """Return a sorted list of drugs with infusion profiles.

        Returns
        -------
        list of str
            Drug names.
        """
        return sorted(self._infusion_db.keys())

    def calculate_rate(
        self,
        drug: str,
        concentration_mg_ml: float,
        target_dose_mcg_kg_min: float,
        weight_kg: float,
    ) -> InfusionResult:
        """Calculate the infusion rate for a continuously-infused drug.

        The formula is::

            rate_ml_hr = (target_dose_mcg_kg_min * weight_kg * 60)
                         / (concentration_mg_ml * 1000)

        Parameters
        ----------
        drug : str
            Drug name (case-insensitive).
        concentration_mg_ml : float
            Drug concentration in mg/mL.
        target_dose_mcg_kg_min : float
            Desired dose in micrograms per kilogram per minute.
        weight_kg : float
            Patient weight in kilograms.

        Returns
        -------
        InfusionResult
            Calculated infusion parameters.

        Raises
        ------
        ValidationError
            If the drug is not found or inputs are invalid.
        """
        drug_key = drug.strip().lower()

        if concentration_mg_ml <= 0:
            raise ValidationError(
                message="concentration_mg_ml must be positive.",
                parameter="concentration_mg_ml",
            )
        if target_dose_mcg_kg_min < 0:
            raise ValidationError(
                message="target_dose_mcg_kg_min must be non-negative.",
                parameter="target_dose_mcg_kg_min",
            )
        WeightBasedDosingCalculator._validate_weight(weight_kg)

        dose_mcg_min = target_dose_mcg_kg_min * weight_kg
        rate_ml_hr = (dose_mcg_min * 60.0) / (concentration_mg_ml * 1000.0)

        titration_min: Optional[float] = None
        titration_max: Optional[float] = None
        warnings: List[str] = []

        if drug_key in self._infusion_db:
            entry = self._infusion_db[drug_key]
            titration_min = entry.titration_min_mcg_kg_min
            titration_max = entry.titration_max_mcg_kg_min
            warnings = list(entry.warnings)

            if titration_max > 0 and target_dose_mcg_kg_min > titration_max:
                warnings.append(
                    f"Target dose {target_dose_mcg_kg_min} mcg/kg/min exceeds "
                    f"usual maximum of {titration_max} mcg/kg/min"
                )
        else:
            warnings.append(
                f"Drug '{drug}' not in built-in database; "
                f"no titration range information available"
            )

        return InfusionResult(
            drug_name=drug_key,
            rate_ml_hr=round(rate_ml_hr, 4),
            dose_mcg_kg_min=target_dose_mcg_kg_min,
            concentration_mg_ml=concentration_mg_ml,
            dose_mcg_min=round(dose_mcg_min, 4),
            titration_min=titration_min,
            titration_max=titration_max,
            warnings=warnings,
        )

    def get_titration_range(self, drug: str) -> Tuple[float, float]:
        """Return the titration range for a drug.

        Parameters
        ----------
        drug : str
            Drug name (case-insensitive).

        Returns
        -------
        tuple of (float, float)
            ``(min_mcg_kg_min, max_mcg_kg_min)``.

        Raises
        ------
        ValidationError
            If the drug is not in the infusion database.
        """
        drug_key = drug.strip().lower()
        if drug_key not in self._infusion_db:
            raise ValidationError(
                message=(
                    f"Drug '{drug}' not found in infusion database. "
                    f"Available: {self.available_drugs}."
                ),
                parameter="drug",
            )
        entry = self._infusion_db[drug_key]
        return (entry.titration_min_mcg_kg_min, entry.titration_max_mcg_kg_min)

    def rate_table(
        self,
        drug: str,
        concentration_mg_ml: float,
        weight_kg: float,
        steps: int = 10,
    ) -> List[Tuple[float, float]]:
        """Generate a titration table of dose-to-rate mappings.

        Parameters
        ----------
        drug : str
            Drug name (case-insensitive).
        concentration_mg_ml : float
            Drug concentration in mg/mL.
        weight_kg : float
            Patient weight in kilograms.
        steps : int, optional
            Number of evenly spaced dose steps across the titration
            range. Default is 10.

        Returns
        -------
        list of tuple of (float, float)
            Each tuple is ``(dose_mcg_kg_min, rate_ml_hr)``.

        Raises
        ------
        ValidationError
            If the drug has no titration range or inputs are invalid.
        """
        tmin, tmax = self.get_titration_range(drug)
        if tmax <= 0:
            raise ValidationError(
                message=f"Drug '{drug}' uses a fixed-dose protocol, not weight-based titration.",
                parameter="drug",
            )

        doses = np.linspace(tmin, tmax, steps)
        table: List[Tuple[float, float]] = []
        for d in doses:
            result = self.calculate_rate(drug, concentration_mg_ml, float(d), weight_kg)
            table.append((round(float(d), 4), result.rate_ml_hr))
        return table
