# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Pharmaceutical alert system for emergency medicine.

This module implements a clinical decision support engine that
evaluates drug prescriptions against patient-specific factors to
generate safety alerts. It integrates dosing range checks,
interaction screening, allergy detection, duplicate therapy
identification, organ-function adjustments, and special population
warnings (pregnancy, elderly).

Classes
-------
PharmacologicalAlertEngine
    Main alert evaluation engine.

Dataclasses
-----------
PharmAlert
    Container for a single pharmaceutical alert.
PatientInfo
    Container for patient demographic and clinical data.

Enumerations
------------
AlertCategory
    Classification of alert types.
AlertSeverity
    Severity grading for alerts.

References
----------
.. [1] American Geriatrics Society 2019 Updated AGS Beers Criteria
   for Potentially Inappropriate Medication Use in Older Adults.
   J Am Geriatr Soc. 2019;67(4):674-694.
.. [2] FDA Pregnancy Categories and PLLR (Pregnancy and Lactation
   Labeling Rule).
.. [3] ACOG Committee Opinion No. 776: "Immune Modulating Therapies
   in Pregnancy and Lactation." Obstet Gynecol. 2019;133(4):e287-e295.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

from notfallmedizin.core.exceptions import ValidationError
from notfallmedizin.pharmacology.dosing import (
    WeightBasedDosingCalculator,
    _DRUG_DATABASE,
)
from notfallmedizin.pharmacology.interactions import (
    DrugInteractionChecker,
    InteractionSeverity,
)


# ======================================================================
# Enumerations
# ======================================================================


class AlertCategory(Enum):
    """Classification of pharmaceutical alert types.

    Attributes
    ----------
    ALLERGY : str
        Drug allergy or cross-reactivity alert.
    INTERACTION : str
        Drug-drug interaction alert.
    DOSE_RANGE : str
        Dose outside expected therapeutic range.
    DUPLICATE_THERAPY : str
        Two drugs from the same therapeutic class.
    CONTRAINDICATION : str
        Drug contraindicated given patient condition.
    RENAL_ADJUSTMENT : str
        Dose adjustment needed for renal impairment.
    HEPATIC_ADJUSTMENT : str
        Dose adjustment needed for hepatic impairment.
    PREGNANCY : str
        Drug with known teratogenic or fetal risk.
    ELDERLY : str
        Drug on the Beers Criteria list for older adults.
    QT_PROLONGATION : str
        Risk of QT prolongation.
    SEROTONIN_SYNDROME : str
        Risk of serotonin syndrome.
    """

    ALLERGY = "allergy"
    INTERACTION = "interaction"
    DOSE_RANGE = "dose_range"
    DUPLICATE_THERAPY = "duplicate_therapy"
    CONTRAINDICATION = "contraindication"
    RENAL_ADJUSTMENT = "renal_adjustment"
    HEPATIC_ADJUSTMENT = "hepatic_adjustment"
    PREGNANCY = "pregnancy"
    ELDERLY = "elderly"
    QT_PROLONGATION = "qt_prolongation"
    SEROTONIN_SYNDROME = "serotonin_syndrome"


class AlertSeverity(IntEnum):
    """Severity grading for pharmaceutical alerts.

    Higher numeric values indicate greater clinical urgency.

    Attributes
    ----------
    INFO : int
        Informational; no immediate action required.
    LOW : int
        Low severity; monitor patient.
    MODERATE : int
        Moderate severity; consider intervention.
    HIGH : int
        High severity; intervention recommended.
    CRITICAL : int
        Critical; immediate action required.
    """

    INFO = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


# ======================================================================
# Data containers
# ======================================================================


@dataclass(frozen=True)
class PharmAlert:
    """Container for a single pharmaceutical alert.

    Attributes
    ----------
    category : AlertCategory
        Type of alert.
    severity : AlertSeverity
        Severity grading.
    drug : str
        Drug that triggered the alert.
    message : str
        Human-readable alert description.
    recommendation : str
        Suggested clinical action.
    evidence_level : str
        Level of evidence supporting the alert.
    related_drug : str
        Second drug involved (for interaction/duplicate alerts).
    """

    category: AlertCategory
    severity: AlertSeverity
    drug: str
    message: str
    recommendation: str
    evidence_level: str = "expert consensus"
    related_drug: str = ""


@dataclass
class PatientInfo:
    """Patient demographic and clinical data for alert evaluation.

    Parameters
    ----------
    age : float
        Patient age in years.
    weight_kg : float
        Patient weight in kilograms.
    is_female : bool, optional
        ``True`` if patient is female. Default is ``False``.
    allergies : list of str, optional
        Known drug allergies (generic drug names, lowercase).
    current_medications : list of str, optional
        Current active medications (generic drug names, lowercase).
    serum_creatinine : float or None, optional
        Serum creatinine in mg/dL for renal function assessment.
    child_pugh : {"A", "B", "C"} or None, optional
        Child-Pugh hepatic function class.
    pregnancy_status : bool, optional
        ``True`` if the patient is pregnant. Default is ``False``.
    lactation_status : bool, optional
        ``True`` if the patient is breastfeeding. Default is ``False``.
    """

    age: float
    weight_kg: float
    is_female: bool = False
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    serum_creatinine: Optional[float] = None
    child_pugh: Optional[str] = None
    pregnancy_status: bool = False
    lactation_status: bool = False


# ======================================================================
# Internal databases
# ======================================================================

_DRUG_CLASSES: Dict[str, str] = {
    "epinephrine": "catecholamine_vasopressor",
    "norepinephrine": "catecholamine_vasopressor",
    "dopamine": "catecholamine_vasopressor",
    "dobutamine": "catecholamine_inotrope",
    "phenylephrine": "alpha_agonist_vasopressor",
    "vasopressin": "vasopressor",
    "fentanyl": "opioid_analgesic",
    "morphine": "opioid_analgesic",
    "hydromorphone": "opioid_analgesic",
    "meperidine": "opioid_analgesic",
    "tramadol": "opioid_analgesic",
    "midazolam": "benzodiazepine",
    "lorazepam": "benzodiazepine",
    "diazepam": "benzodiazepine",
    "propofol": "sedative_hypnotic",
    "ketamine": "dissociative_anesthetic",
    "etomidate": "sedative_hypnotic",
    "rocuronium": "neuromuscular_blocker",
    "succinylcholine": "depolarizing_neuromuscular_blocker",
    "vecuronium": "neuromuscular_blocker",
    "cisatracurium": "neuromuscular_blocker",
    "phenytoin": "antiepileptic",
    "levetiracetam": "antiepileptic",
    "valproic_acid": "antiepileptic",
    "lacosamide": "antiepileptic",
    "amiodarone": "antiarrhythmic_class_iii",
    "lidocaine": "antiarrhythmic_class_ib",
    "procainamide": "antiarrhythmic_class_ia",
    "adenosine": "antiarrhythmic_other",
    "alteplase": "thrombolytic",
    "tenecteplase": "thrombolytic",
    "naloxone": "opioid_antagonist",
    "flumazenil": "benzodiazepine_antagonist",
    "ondansetron": "antiemetic_5ht3",
    "metoclopramide": "antiemetic_prokinetic",
}

_ALLERGY_CROSS_REACTIVITY: Dict[str, List[str]] = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "nafcillin"],
    "amoxicillin": ["penicillin", "ampicillin", "piperacillin"],
    "cephalosporin": ["cefazolin", "ceftriaxone", "cefepime", "cephalexin"],
    "sulfonamide": ["sulfamethoxazole", "sulfasalazine"],
    "nsaid": ["ibuprofen", "ketorolac", "naproxen", "aspirin"],
    "ibuprofen": ["ketorolac", "naproxen"],
    "ketorolac": ["ibuprofen", "naproxen"],
    "morphine": ["codeine", "hydromorphone"],
    "codeine": ["morphine"],
    "egg": ["propofol"],
    "soy": ["propofol"],
    "propofol": ["egg", "soy"],
}

_PREGNANCY_WARNINGS: Dict[str, Tuple[str, str]] = {
    "warfarin": ("X", "Teratogenic. Causes warfarin embryopathy (nasal hypoplasia, stippled epiphyses)."),
    "phenytoin": ("D", "Fetal hydantoin syndrome risk. Use alternative antiepileptic if possible."),
    "valproic_acid": ("X", "Neural tube defects, cognitive impairment. Contraindicated in pregnancy."),
    "methotrexate": ("X", "Abortifacient and teratogenic. Absolutely contraindicated."),
    "alteplase": ("C", "Limited data. Use only if benefit clearly outweighs risk."),
    "amiodarone": ("D", "Fetal thyroid dysfunction and growth restriction. Avoid if possible."),
    "propofol": ("B", "Limited human data; animal studies show no fetal risk at clinical doses."),
    "ketamine": ("B", "Limited human pregnancy data. Appears safe for short procedural use."),
    "fentanyl": ("C", "Neonatal respiratory depression and withdrawal if used near delivery."),
    "morphine": ("C", "Neonatal respiratory depression and withdrawal syndrome."),
    "midazolam": ("D", "Risk of neonatal sedation and withdrawal. Avoid if possible."),
    "lorazepam": ("D", "Risk of neonatal sedation, hypotonia, and withdrawal."),
    "succinylcholine": ("C", "Crosses placenta minimally. Generally considered safe for RSI."),
    "rocuronium": ("C", "Minimal placental transfer. Considered safe for RSI."),
    "fluconazole": ("D", "Teratogenic at high doses (>=400 mg/day). Single low dose likely safe."),
    "metoclopramide": ("B", "No evidence of teratogenicity. Commonly used for hyperemesis."),
    "ondansetron": ("B", "Conflicting data on cardiac malformations. Generally considered safe."),
    "norepinephrine": ("C", "May reduce uterine blood flow. Use lowest effective dose."),
    "epinephrine": ("C", "May reduce uterine blood flow. Reserved for life-threatening situations."),
}

_BEERS_CRITERIA_DRUGS: Dict[str, Tuple[str, str]] = {
    "meperidine": (
        "Avoid",
        "Not effective at oral doses; neurotoxic metabolite normeperidine accumulates in elderly."
    ),
    "diazepam": (
        "Avoid",
        "Long-acting benzodiazepine. Increased sensitivity and risk of falls, fractures, cognitive impairment."
    ),
    "chlordiazepoxide": (
        "Avoid",
        "Long-acting benzodiazepine with active metabolites."
    ),
    "diphenhydramine": (
        "Avoid",
        "Highly anticholinergic. Confusion, urinary retention, constipation in elderly."
    ),
    "hydroxyzine": (
        "Avoid",
        "Anticholinergic effects. Increased fall risk."
    ),
    "promethazine": (
        "Avoid",
        "Highly anticholinergic. Respiratory depression risk."
    ),
    "metoclopramide": (
        "Avoid unless gastroparesis",
        "Extrapyramidal effects, tardive dyskinesia risk."
    ),
    "amitriptyline": (
        "Avoid",
        "Highly anticholinergic, sedating. Orthostatic hypotension, cardiac conduction disturbances."
    ),
    "nifedipine_ir": (
        "Avoid",
        "Immediate-release nifedipine. Risk of hypotension and myocardial ischemia."
    ),
    "sliding_scale_insulin": (
        "Avoid as sole regimen",
        "Higher risk of hypoglycemia without improved glycemic control."
    ),
    "chlorpropamide": (
        "Avoid",
        "Prolonged half-life; severe, prolonged hypoglycemia risk."
    ),
    "glyburide": (
        "Avoid",
        "Higher risk of prolonged hypoglycemia compared to other sulfonylureas."
    ),
    "indomethacin": (
        "Avoid",
        "Highest CNS adverse-effect risk among NSAIDs."
    ),
    "ketorolac": (
        "Avoid prolonged use",
        "GI bleeding risk. Limit to 5 days. Reduce dose in elderly."
    ),
    "muscle_relaxants": (
        "Avoid",
        "Cyclobenzaprine, methocarbamol, etc. Anticholinergic effects, sedation, fracture risk."
    ),
}

_DOSE_RANGES_MG_KG: Dict[str, Dict[str, Tuple[float, float]]] = {
    "epinephrine": {"cardiac_arrest": (0.005, 0.015), "anaphylaxis": (0.005, 0.015)},
    "amiodarone": {"cardiac_arrest": (3.0, 7.0)},
    "atropine": {"bradycardia": (0.01, 0.04), "organophosphate_poisoning": (0.02, 0.1)},
    "adenosine": {"svt": (0.05, 0.2)},
    "ketamine": {"rsi": (1.0, 3.0), "procedural_sedation": (0.5, 2.0), "analgesia": (0.1, 0.5)},
    "propofol": {"rsi": (1.0, 3.0), "procedural_sedation": (0.5, 2.0)},
    "rocuronium": {"rsi": (0.6, 1.5), "intubation": (0.3, 1.0)},
    "succinylcholine": {"rsi": (1.0, 2.5)},
    "fentanyl": {"analgesia": (0.5, 2.0)},
    "morphine": {"analgesia": (0.05, 0.2)},
    "midazolam": {"seizure": (0.1, 0.3), "procedural_sedation": (0.02, 0.1)},
    "lorazepam": {"seizure": (0.05, 0.15)},
    "phenytoin": {"seizure": (15.0, 25.0)},
    "levetiracetam": {"seizure": (20.0, 70.0)},
    "alteplase": {"acute_stroke": (0.7, 0.95)},
    "naloxone": {"opioid_reversal": (0.0, 0.15)},
    "flumazenil": {"benzodiazepine_reversal": (0.005, 0.02)},
}


# ======================================================================
# PharmacologicalAlertEngine
# ======================================================================


class PharmacologicalAlertEngine:
    """Evaluate drug prescriptions and generate pharmaceutical alerts.

    This engine checks a proposed prescription against patient-specific
    data to identify potential safety concerns including allergies,
    interactions, dose range violations, duplicate therapies,
    organ-function adjustments, and special population risks.

    Examples
    --------
    >>> from notfallmedizin.pharmacology.alerts import (
    ...     PharmacologicalAlertEngine, PatientInfo,
    ... )
    >>> engine = PharmacologicalAlertEngine()
    >>> patient = PatientInfo(
    ...     age=75, weight_kg=70, is_female=True,
    ...     allergies=["penicillin"],
    ...     current_medications=["amiodarone"],
    ...     serum_creatinine=2.0,
    ... )
    >>> alerts = engine.evaluate_prescription(
    ...     drug="morphine", dose_mg=5.0, patient=patient,
    ...     indication="analgesia",
    ... )
    >>> any(a.category.value == "renal_adjustment" for a in alerts)
    True

    References
    ----------
    .. [1] AGS 2019 Updated Beers Criteria. J Am Geriatr Soc.
       2019;67(4):674-694.
    """

    def __init__(self) -> None:
        self._interaction_checker = DrugInteractionChecker()
        self._dosing_calculator = WeightBasedDosingCalculator()

    def evaluate_prescription(
        self,
        drug: str,
        dose_mg: float,
        patient: PatientInfo,
        indication: Optional[str] = None,
    ) -> List[PharmAlert]:
        """Evaluate a drug prescription and return all triggered alerts.

        Parameters
        ----------
        drug : str
            Proposed drug name (case-insensitive).
        dose_mg : float
            Proposed dose in milligrams.
        patient : PatientInfo
            Patient demographic and clinical data.
        indication : str or None, optional
            Clinical indication. Required for dose-range checking
            against the built-in database.

        Returns
        -------
        list of PharmAlert
            All alerts triggered, sorted by severity (most severe
            first).
        """
        drug_key = drug.strip().lower()
        alerts: List[PharmAlert] = []

        alerts.extend(self._check_allergy(drug_key, patient))
        alerts.extend(self._check_interactions(drug_key, patient))
        alerts.extend(self._check_duplicate_therapy(drug_key, patient))
        alerts.extend(self._check_qt_risk(drug_key, patient))
        alerts.extend(self._check_serotonin_risk(drug_key, patient))

        if indication is not None:
            alerts.extend(
                self._check_dose_range(drug_key, dose_mg, patient.weight_kg, indication)
            )

        alerts.extend(self._check_renal(drug_key, patient))
        alerts.extend(self._check_hepatic(drug_key, patient))

        if patient.pregnancy_status:
            alerts.extend(self._check_pregnancy(drug_key))

        if patient.age >= 65:
            alerts.extend(self._check_elderly(drug_key, patient))

        alerts.sort(key=lambda a: a.severity, reverse=True)
        return alerts

    # ------------------------------------------------------------------
    # Allergy checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_allergy(drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Check for direct allergy or cross-reactivity."""
        alerts: List[PharmAlert] = []
        allergies_lower = {a.strip().lower() for a in patient.allergies}

        if drug in allergies_lower:
            alerts.append(PharmAlert(
                category=AlertCategory.ALLERGY,
                severity=AlertSeverity.CRITICAL,
                drug=drug,
                message=f"Patient has a documented allergy to {drug}.",
                recommendation="Do not administer. Select an alternative agent.",
                evidence_level="patient-reported",
            ))
            return alerts

        for allergy in allergies_lower:
            cross_reactive = _ALLERGY_CROSS_REACTIVITY.get(allergy, [])
            if drug in cross_reactive:
                alerts.append(PharmAlert(
                    category=AlertCategory.ALLERGY,
                    severity=AlertSeverity.HIGH,
                    drug=drug,
                    message=(
                        f"Potential cross-reactivity: patient is allergic to "
                        f"{allergy}, and {drug} shares structural/class similarity."
                    ),
                    recommendation=(
                        f"Assess specific allergy history. Consider allergy "
                        f"testing or select an alternative."
                    ),
                    evidence_level="established",
                    related_drug=allergy,
                ))

            if allergy in _ALLERGY_CROSS_REACTIVITY:
                if drug in _ALLERGY_CROSS_REACTIVITY[allergy]:
                    if not any(a.drug == drug and a.related_drug == allergy for a in alerts):
                        alerts.append(PharmAlert(
                            category=AlertCategory.ALLERGY,
                            severity=AlertSeverity.HIGH,
                            drug=drug,
                            message=(
                                f"Cross-reactivity warning: documented {allergy} "
                                f"allergy may cross-react with {drug}."
                            ),
                            recommendation="Verify allergy details. Consider alternative.",
                            evidence_level="established",
                            related_drug=allergy,
                        ))

        return alerts

    # ------------------------------------------------------------------
    # Drug interaction checks
    # ------------------------------------------------------------------

    def _check_interactions(self, drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Check for drug-drug interactions with current medications."""
        alerts: List[PharmAlert] = []
        for med in patient.current_medications:
            med_key = med.strip().lower()
            result = self._interaction_checker.check_interaction(drug, med_key)
            if result is not None:
                severity_map = {
                    InteractionSeverity.CONTRAINDICATED: AlertSeverity.CRITICAL,
                    InteractionSeverity.MAJOR: AlertSeverity.HIGH,
                    InteractionSeverity.MODERATE: AlertSeverity.MODERATE,
                    InteractionSeverity.MINOR: AlertSeverity.LOW,
                }
                alerts.append(PharmAlert(
                    category=AlertCategory.INTERACTION,
                    severity=severity_map.get(result.severity, AlertSeverity.MODERATE),
                    drug=drug,
                    message=(
                        f"Interaction with {med_key}: {result.clinical_effect} "
                        f"({result.severity.name})."
                    ),
                    recommendation=result.recommendation,
                    evidence_level=result.evidence_level,
                    related_drug=med_key,
                ))
        return alerts

    # ------------------------------------------------------------------
    # Duplicate therapy checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_duplicate_therapy(drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Detect duplicate therapeutic class prescribing."""
        alerts: List[PharmAlert] = []
        drug_class = _DRUG_CLASSES.get(drug)
        if drug_class is None:
            return alerts

        for med in patient.current_medications:
            med_key = med.strip().lower()
            if med_key == drug:
                continue
            med_class = _DRUG_CLASSES.get(med_key)
            if med_class is not None and med_class == drug_class:
                alerts.append(PharmAlert(
                    category=AlertCategory.DUPLICATE_THERAPY,
                    severity=AlertSeverity.MODERATE,
                    drug=drug,
                    message=(
                        f"Duplicate therapy: {drug} and {med_key} are both in "
                        f"class '{drug_class}'."
                    ),
                    recommendation=(
                        f"Review therapeutic intent. Discontinue one agent "
                        f"or document clinical rationale for dual therapy."
                    ),
                    evidence_level="clinical guideline",
                    related_drug=med_key,
                ))
        return alerts

    # ------------------------------------------------------------------
    # QT prolongation check
    # ------------------------------------------------------------------

    def _check_qt_risk(self, drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Assess QT prolongation risk with current medication list."""
        alerts: List[PharmAlert] = []
        all_drugs = [drug] + [m.strip().lower() for m in patient.current_medications]
        qt_drugs = self._interaction_checker.assess_qt_risk(all_drugs)

        qt_drug_names = {name for name, _ in qt_drugs}
        if drug not in qt_drug_names:
            return alerts

        qt_count = len(qt_drug_names)
        if qt_count >= 2:
            other_qt = [name for name in qt_drug_names if name != drug]
            alerts.append(PharmAlert(
                category=AlertCategory.QT_PROLONGATION,
                severity=AlertSeverity.HIGH if qt_count >= 3 else AlertSeverity.MODERATE,
                drug=drug,
                message=(
                    f"QT prolongation risk: {drug} combined with {', '.join(other_qt)} "
                    f"({qt_count} QT-prolonging agents total)."
                ),
                recommendation=(
                    "Obtain baseline ECG. Monitor QTc interval. "
                    "Consider alternative agents with lower QT risk."
                ),
                evidence_level="established",
                related_drug=", ".join(other_qt),
            ))
        return alerts

    # ------------------------------------------------------------------
    # Serotonin syndrome check
    # ------------------------------------------------------------------

    def _check_serotonin_risk(self, drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Assess serotonin syndrome risk."""
        alerts: List[PharmAlert] = []
        all_drugs = [drug] + [m.strip().lower() for m in patient.current_medications]
        risk_level = self._interaction_checker.serotonin_syndrome_risk_level(all_drugs)

        if risk_level in ("moderate", "high"):
            sero_drugs = self._interaction_checker.assess_serotonin_risk(all_drugs)
            sero_names = [name for name, _ in sero_drugs if name != drug]
            if sero_names:
                alerts.append(PharmAlert(
                    category=AlertCategory.SEROTONIN_SYNDROME,
                    severity=AlertSeverity.HIGH if risk_level == "high" else AlertSeverity.MODERATE,
                    drug=drug,
                    message=(
                        f"Serotonin syndrome risk ({risk_level}): {drug} combined "
                        f"with serotonergic agent(s) {', '.join(sero_names)}."
                    ),
                    recommendation=(
                        "Monitor for serotonin syndrome: agitation, clonus, "
                        "hyperthermia, diaphoresis, tachycardia, diarrhea. "
                        "Consider cyproheptadine if syndrome develops."
                    ),
                    evidence_level="established",
                    related_drug=", ".join(sero_names),
                ))
        return alerts

    # ------------------------------------------------------------------
    # Dose range checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dose_range(
        drug: str,
        dose_mg: float,
        weight_kg: float,
        indication: str,
    ) -> List[PharmAlert]:
        """Check whether a dose falls within the expected range."""
        alerts: List[PharmAlert] = []
        indication_key = indication.strip().lower()

        drug_ranges = _DOSE_RANGES_MG_KG.get(drug)
        if drug_ranges is None:
            return alerts

        ind_range = drug_ranges.get(indication_key)
        if ind_range is None:
            return alerts

        low_mg_kg, high_mg_kg = ind_range
        if weight_kg <= 0:
            return alerts

        actual_mg_kg = dose_mg / weight_kg

        if actual_mg_kg < low_mg_kg:
            alerts.append(PharmAlert(
                category=AlertCategory.DOSE_RANGE,
                severity=AlertSeverity.MODERATE,
                drug=drug,
                message=(
                    f"Dose appears low: {actual_mg_kg:.3f} mg/kg is below the "
                    f"typical range of {low_mg_kg}-{high_mg_kg} mg/kg for "
                    f"'{indication_key}'."
                ),
                recommendation=(
                    f"Verify intended dose. Typical range: "
                    f"{low_mg_kg}-{high_mg_kg} mg/kg."
                ),
                evidence_level="clinical guideline",
            ))

        if actual_mg_kg > high_mg_kg:
            severity = AlertSeverity.HIGH
            if actual_mg_kg > high_mg_kg * 2:
                severity = AlertSeverity.CRITICAL

            alerts.append(PharmAlert(
                category=AlertCategory.DOSE_RANGE,
                severity=severity,
                drug=drug,
                message=(
                    f"Dose appears high: {actual_mg_kg:.3f} mg/kg exceeds the "
                    f"typical range of {low_mg_kg}-{high_mg_kg} mg/kg for "
                    f"'{indication_key}'."
                ),
                recommendation=(
                    f"Verify intended dose. Maximum typical dose: "
                    f"{high_mg_kg} mg/kg. Reduce dose or document rationale."
                ),
                evidence_level="clinical guideline",
            ))

        return alerts

    # ------------------------------------------------------------------
    # Renal function checks
    # ------------------------------------------------------------------

    def _check_renal(self, drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Check whether renal dose adjustment is needed."""
        alerts: List[PharmAlert] = []
        if patient.serum_creatinine is None:
            return alerts

        drug_entries = _DRUG_DATABASE.get(drug, {})
        needs_adjustment = any(
            entry.renal_adjustment is not None
            for entry in drug_entries.values()
        )
        if not needs_adjustment:
            return alerts

        from notfallmedizin.pharmacology.kinetics import CockcroftGault

        try:
            crcl = CockcroftGault.calculate(
                age=patient.age,
                weight_kg=patient.weight_kg,
                serum_creatinine=patient.serum_creatinine,
                is_female=patient.is_female,
            )
        except Exception:
            return alerts

        stage = CockcroftGault.classify(crcl)
        if stage in ("normal", "mild"):
            return alerts

        alerts.append(PharmAlert(
            category=AlertCategory.RENAL_ADJUSTMENT,
            severity=AlertSeverity.HIGH if stage in ("severe", "dialysis") else AlertSeverity.MODERATE,
            drug=drug,
            message=(
                f"Renal impairment detected: CrCl = {crcl:.1f} mL/min "
                f"({stage}). {drug} requires renal dose adjustment."
            ),
            recommendation=(
                f"Adjust dose per renal dosing guidelines. "
                f"Consider using the WeightBasedDosingCalculator with "
                f"serum_creatinine parameter for automatic adjustment."
            ),
            evidence_level="pharmacokinetic data",
        ))
        return alerts

    # ------------------------------------------------------------------
    # Hepatic function checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hepatic(drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Check whether hepatic dose adjustment is needed."""
        alerts: List[PharmAlert] = []
        if patient.child_pugh is None:
            return alerts

        cp = patient.child_pugh.strip().upper()
        if cp not in ("A", "B", "C"):
            return alerts

        drug_entries = _DRUG_DATABASE.get(drug, {})
        needs_adjustment = any(
            entry.hepatic_adjustment is not None
            and entry.hepatic_adjustment.get(cp, 1.0) < 1.0
            for entry in drug_entries.values()
        )
        if not needs_adjustment:
            return alerts

        alerts.append(PharmAlert(
            category=AlertCategory.HEPATIC_ADJUSTMENT,
            severity=AlertSeverity.HIGH if cp == "C" else AlertSeverity.MODERATE,
            drug=drug,
            message=(
                f"Hepatic impairment (Child-Pugh {cp}): {drug} requires "
                f"hepatic dose adjustment."
            ),
            recommendation=(
                f"Reduce dose per hepatic dosing guidelines. "
                f"Consider using the WeightBasedDosingCalculator with "
                f"child_pugh parameter."
            ),
            evidence_level="pharmacokinetic data",
        ))
        return alerts

    # ------------------------------------------------------------------
    # Pregnancy checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pregnancy(drug: str) -> List[PharmAlert]:
        """Check for pregnancy-related drug warnings."""
        alerts: List[PharmAlert] = []
        warning = _PREGNANCY_WARNINGS.get(drug)
        if warning is None:
            return alerts

        category_letter, description = warning
        severity_map = {
            "X": AlertSeverity.CRITICAL,
            "D": AlertSeverity.HIGH,
            "C": AlertSeverity.MODERATE,
            "B": AlertSeverity.LOW,
            "A": AlertSeverity.INFO,
        }
        alerts.append(PharmAlert(
            category=AlertCategory.PREGNANCY,
            severity=severity_map.get(category_letter, AlertSeverity.MODERATE),
            drug=drug,
            message=(
                f"Pregnancy warning (Category {category_letter}): {description}"
            ),
            recommendation=(
                "Assess risk-benefit ratio. Consult OB/GYN if available. "
                "Document informed consent for category D/X drugs."
            ),
            evidence_level="FDA labeling / clinical data",
        ))
        return alerts

    # ------------------------------------------------------------------
    # Elderly (Beers Criteria) checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_elderly(drug: str, patient: PatientInfo) -> List[PharmAlert]:
        """Check Beers Criteria for potentially inappropriate medications in elderly."""
        alerts: List[PharmAlert] = []
        if patient.age < 65:
            return alerts

        beers_entry = _BEERS_CRITERIA_DRUGS.get(drug)
        if beers_entry is None:
            return alerts

        recommendation_type, rationale = beers_entry
        alerts.append(PharmAlert(
            category=AlertCategory.ELDERLY,
            severity=AlertSeverity.HIGH if recommendation_type == "Avoid" else AlertSeverity.MODERATE,
            drug=drug,
            message=(
                f"Beers Criteria alert for patient age {patient.age}: "
                f"{drug} -- {recommendation_type}. {rationale}"
            ),
            recommendation=(
                "Consider safer alternatives per AGS Beers Criteria. "
                "If no alternative, use lowest effective dose for shortest duration."
            ),
            evidence_level="AGS Beers Criteria 2019",
        ))
        return alerts
