# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Pharmacology module for the notfallmedizin library.

This package provides drug dosing calculations, drug interaction
checking, pharmacokinetic modeling, and a pharmaceutical alert
engine tailored to the emergency medicine setting.

Submodules
----------
dosing
    Weight-based and continuous infusion dosing calculators.
interactions
    Drug-drug interaction checker with QT and serotonin risk assessment.
kinetics
    One- and two-compartment pharmacokinetic models and renal function
    estimators (Cockcroft-Gault, MDRD, CKD-EPI).
alerts
    Pharmaceutical alert engine integrating allergy, interaction, dose
    range, duplicate therapy, organ-function, and special-population
    checks.

References:
    Cockcroft & Gault. Prediction of creatinine clearance. Nephron 1976.
    CKD-EPI 2021. Inker et al. N Engl J Med 2021;385(19):1737-1749.
"""

from notfallmedizin.pharmacology.alerts import (
    AlertCategory,
    AlertSeverity,
    PatientInfo,
    PharmacologicalAlertEngine,
    PharmAlert,
)
from notfallmedizin.pharmacology.dosing import (
    ContinuousInfusionCalculator,
    DosingResult,
    InfusionResult,
    Route,
    WeightBasedDosingCalculator,
)
from notfallmedizin.pharmacology.interactions import (
    CYP450Enzyme,
    DrugInteractionChecker,
    InteractionResult,
    InteractionSeverity,
)
from notfallmedizin.pharmacology.kinetics import (
    CKD_EPI,
    CockcroftGault,
    MDRD,
    OneCompartmentModel,
    TwoCompartmentModel,
)

__all__ = [
    # dosing
    "WeightBasedDosingCalculator",
    "ContinuousInfusionCalculator",
    "DosingResult",
    "InfusionResult",
    "Route",
    # interactions
    "DrugInteractionChecker",
    "InteractionResult",
    "InteractionSeverity",
    "CYP450Enzyme",
    # kinetics
    "OneCompartmentModel",
    "TwoCompartmentModel",
    "CockcroftGault",
    "MDRD",
    "CKD_EPI",
    # alerts
    "PharmacologicalAlertEngine",
    "PharmAlert",
    "PatientInfo",
    "AlertCategory",
    "AlertSeverity",
]
