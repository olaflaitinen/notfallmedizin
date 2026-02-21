# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Drug interaction checking for emergency medicine.

This module provides tools for detecting clinically significant drug
interactions in the emergency department setting. It includes a
built-in database of critical drug-drug interactions, QT prolongation
risk assessment, serotonin syndrome risk detection, and CYP450
interaction awareness.

Classes
-------
DrugInteractionChecker
    Check pairwise and multi-drug interactions against a built-in
    database of emergency medicine drug interactions.

Dataclasses
-----------
InteractionResult
    Container for a single drug-drug interaction finding.

Enumerations
------------
InteractionSeverity
    Severity classification for drug interactions.
CYP450Enzyme
    Major cytochrome P450 enzymes relevant to EM drug metabolism.

References
----------
.. [1] Woosley RL, Heise CW, Romero KA. "QTdrugs List."
   www.CredibleMeds.org, AZCERT, Inc.
.. [2] Boyer EW, Shannon M. "The serotonin syndrome." N Engl J Med.
   2005;352(11):1112-1120.
.. [3] Flockhart DA. "Drug Interactions: Cytochrome P450 Drug
   Interaction Table." Indiana University School of Medicine. 2007.
.. [4] Lexicomp Drug Interactions. Wolters Kluwer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from notfallmedizin.core.exceptions import ValidationError


# ======================================================================
# Enumerations
# ======================================================================


class InteractionSeverity(IntEnum):
    """Severity classification for drug-drug interactions.

    Numeric values enable ordering comparisons: higher values indicate
    more severe interactions.

    Attributes
    ----------
    MINOR : int
        Interaction of limited clinical significance.
    MODERATE : int
        Interaction that may require monitoring or dose adjustment.
    MAJOR : int
        Interaction that may cause serious adverse effects.
    CONTRAINDICATED : int
        Combination should be avoided entirely.
    """

    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CONTRAINDICATED = 4


class CYP450Enzyme(Enum):
    """Major cytochrome P450 enzyme isoforms.

    References
    ----------
    .. [1] Flockhart DA. Indiana University School of Medicine. 2007.
    """

    CYP1A2 = "CYP1A2"
    CYP2B6 = "CYP2B6"
    CYP2C9 = "CYP2C9"
    CYP2C19 = "CYP2C19"
    CYP2D6 = "CYP2D6"
    CYP2E1 = "CYP2E1"
    CYP3A4 = "CYP3A4"


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass(frozen=True)
class InteractionResult:
    """Container for a drug-drug interaction finding.

    Attributes
    ----------
    drug_a : str
        First drug in the interacting pair.
    drug_b : str
        Second drug in the interacting pair.
    severity : InteractionSeverity
        Severity classification.
    mechanism : str
        Pharmacological mechanism of the interaction.
    clinical_effect : str
        Expected clinical consequence.
    recommendation : str
        Clinical management recommendation.
    evidence_level : str
        Level of evidence supporting the interaction
        (e.g., "established", "probable", "suspected", "theoretical").
    cyp_enzymes : list of str
        CYP450 enzymes involved, if applicable.
    """

    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    recommendation: str
    evidence_level: str = "established"
    cyp_enzymes: List[str] = field(default_factory=list)


# ======================================================================
# Internal interaction database
# ======================================================================


@dataclass(frozen=True)
class _InteractionEntry:
    """Internal storage for a drug interaction record."""

    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    recommendation: str
    evidence_level: str = "established"
    cyp_enzymes: List[str] = field(default_factory=list)


def _pair_key(a: str, b: str) -> FrozenSet[str]:
    """Create an unordered pair key for two drug names."""
    return frozenset((a.strip().lower(), b.strip().lower()))


_INTERACTION_DB: Dict[FrozenSet[str], _InteractionEntry] = {
    # 1. Amiodarone + QT-prolonging agents
    _pair_key("amiodarone", "haloperidol"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive QT prolongation via potassium channel blockade",
        clinical_effect="Increased risk of torsades de pointes and fatal arrhythmias",
        recommendation="Avoid combination if possible. If required, continuous cardiac monitoring with serial QTc measurement.",
        evidence_level="established",
    ),
    _pair_key("amiodarone", "ondansetron"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive QT prolongation",
        clinical_effect="Increased risk of torsades de pointes",
        recommendation="Use alternative antiemetic (e.g., metoclopramide). If must use, monitor QTc.",
        evidence_level="established",
    ),
    _pair_key("amiodarone", "erythromycin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive QT prolongation; CYP3A4 inhibition by erythromycin increases amiodarone levels",
        clinical_effect="Increased risk of torsades de pointes and amiodarone toxicity",
        recommendation="Avoid combination. Use azithromycin if macrolide needed.",
        evidence_level="established",
        cyp_enzymes=["CYP3A4"],
    ),
    _pair_key("amiodarone", "fluconazole"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive QT prolongation; CYP3A4 and CYP2C9 inhibition by fluconazole",
        clinical_effect="Increased amiodarone levels and QT prolongation risk",
        recommendation="Avoid combination. Use alternative antifungal.",
        evidence_level="established",
        cyp_enzymes=["CYP3A4", "CYP2C9"],
    ),
    # 2. Amiodarone + digoxin
    _pair_key("amiodarone", "digoxin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Amiodarone inhibits P-glycoprotein and renal clearance of digoxin",
        clinical_effect="Digoxin toxicity (nausea, visual disturbances, arrhythmias)",
        recommendation="Reduce digoxin dose by 50% when initiating amiodarone. Monitor digoxin levels.",
        evidence_level="established",
    ),
    # 3. Amiodarone + warfarin
    _pair_key("amiodarone", "warfarin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="CYP2C9 inhibition by amiodarone decreases warfarin metabolism",
        clinical_effect="Elevated INR and increased bleeding risk",
        recommendation="Reduce warfarin dose by 30-50%. Monitor INR closely.",
        evidence_level="established",
        cyp_enzymes=["CYP2C9"],
    ),
    # 4. Succinylcholine contraindications
    _pair_key("succinylcholine", "rocuronium"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Sequential neuromuscular blockade with differing mechanisms",
        clinical_effect="Prolonged neuromuscular blockade with unpredictable recovery",
        recommendation="If succinylcholine used first, wait for clinical recovery before rocuronium. Sugammadex reverses rocuronium only.",
        evidence_level="established",
    ),
    # 5. Ketamine + MAOIs
    _pair_key("ketamine", "phenelzine"): _InteractionEntry(
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism="MAO inhibition potentiates sympathomimetic effects of ketamine",
        clinical_effect="Severe hypertensive crisis, hyperthermia, serotonin syndrome",
        recommendation="Contraindicated. Use alternative induction agent.",
        evidence_level="established",
    ),
    _pair_key("ketamine", "tranylcypromine"): _InteractionEntry(
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism="MAO inhibition potentiates sympathomimetic effects of ketamine",
        clinical_effect="Severe hypertensive crisis, hyperthermia",
        recommendation="Contraindicated. Use alternative induction agent.",
        evidence_level="established",
    ),
    # 6. Fentanyl + serotonergic agents (serotonin syndrome)
    _pair_key("fentanyl", "fluoxetine"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Combined serotonergic activity; fentanyl has weak serotonin reuptake inhibition",
        clinical_effect="Risk of serotonin syndrome (agitation, clonus, hyperthermia, diaphoresis)",
        recommendation="Monitor for serotonin syndrome signs. Use lowest effective opioid dose.",
        evidence_level="probable",
    ),
    _pair_key("fentanyl", "sertraline"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Combined serotonergic activity",
        clinical_effect="Risk of serotonin syndrome",
        recommendation="Monitor for serotonin syndrome. Consider alternative analgesic.",
        evidence_level="probable",
    ),
    _pair_key("fentanyl", "linezolid"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Linezolid is a reversible MAO inhibitor; combined with fentanyl serotonergic activity",
        clinical_effect="Risk of serotonin syndrome",
        recommendation="Avoid combination if possible. If required, monitor closely for 24+ hours.",
        evidence_level="established",
    ),
    # 7. Morphine + benzodiazepines
    _pair_key("morphine", "midazolam"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Synergistic CNS and respiratory depression",
        clinical_effect="Profound sedation, respiratory arrest, death",
        recommendation="If combination necessary, reduce doses of both agents. Ensure monitoring with capnography.",
        evidence_level="established",
    ),
    _pair_key("morphine", "lorazepam"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Synergistic CNS and respiratory depression",
        clinical_effect="Profound sedation, respiratory arrest",
        recommendation="Reduce doses of both. Continuous pulse oximetry and capnography.",
        evidence_level="established",
    ),
    _pair_key("fentanyl", "midazolam"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Synergistic CNS and respiratory depression",
        clinical_effect="Profound sedation, respiratory depression, apnea",
        recommendation="Reduce both doses by 25-50%. Continuous monitoring required.",
        evidence_level="established",
    ),
    _pair_key("fentanyl", "propofol"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Synergistic CNS depression and hemodynamic instability",
        clinical_effect="Profound sedation, hypotension, apnea",
        recommendation="Reduce propofol dose. Prepare for hemodynamic support.",
        evidence_level="established",
    ),
    # 8. Phenytoin interactions
    _pair_key("phenytoin", "fluconazole"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="CYP2C9 inhibition by fluconazole increases phenytoin levels",
        clinical_effect="Phenytoin toxicity (nystagmus, ataxia, confusion, seizures)",
        recommendation="Monitor phenytoin levels. Consider dose reduction or alternative antiepileptic.",
        evidence_level="established",
        cyp_enzymes=["CYP2C9"],
    ),
    _pair_key("phenytoin", "warfarin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Complex bidirectional CYP interaction; phenytoin induces CYP metabolism of warfarin",
        clinical_effect="Decreased warfarin efficacy initially; variable INR",
        recommendation="Monitor INR frequently. May need warfarin dose adjustment.",
        evidence_level="established",
        cyp_enzymes=["CYP2C9", "CYP3A4"],
    ),
    # 9. Alteplase + anticoagulants
    _pair_key("alteplase", "heparin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive thrombolytic and anticoagulant effects",
        clinical_effect="Markedly increased hemorrhagic risk",
        recommendation="Hold heparin during and for hours after alteplase per protocol. Monitor aPTT.",
        evidence_level="established",
    ),
    _pair_key("alteplase", "enoxaparin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Additive anticoagulant and fibrinolytic effects",
        clinical_effect="Significantly increased bleeding risk",
        recommendation="Delay LMWH for 24 hours post-thrombolysis per stroke guidelines.",
        evidence_level="established",
    ),
    _pair_key("alteplase", "warfarin"): _InteractionEntry(
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism="Additive hemorrhagic risk in anticoagulated patient",
        clinical_effect="High risk of symptomatic intracranial hemorrhage",
        recommendation="Alteplase contraindicated if INR > 1.7 for stroke indication.",
        evidence_level="established",
    ),
    # 10. Naloxone + opioids (expected therapeutic interaction)
    _pair_key("naloxone", "morphine"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Competitive mu-opioid receptor antagonism",
        clinical_effect="Reversal of opioid effect; may precipitate acute withdrawal",
        recommendation="Titrate naloxone to respiratory effort, not consciousness. Start with 0.04 mg in dependent patients.",
        evidence_level="established",
    ),
    _pair_key("naloxone", "fentanyl"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Competitive mu-opioid receptor antagonism",
        clinical_effect="Reversal of opioid effect; may need repeated doses due to fentanyl duration",
        recommendation="Titrate to respiratory effort. Monitor for re-narcotization (fentanyl t1/2 > naloxone t1/2).",
        evidence_level="established",
    ),
    # 11. Flumazenil + benzodiazepines
    _pair_key("flumazenil", "midazolam"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Competitive GABA-A receptor antagonism",
        clinical_effect="Reversal of benzodiazepine sedation; seizure risk in dependent patients",
        recommendation="Use only when benefits outweigh seizure risk. Avoid in chronic BZD users or mixed overdose with TCA.",
        evidence_level="established",
    ),
    _pair_key("flumazenil", "lorazepam"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Competitive GABA-A receptor antagonism",
        clinical_effect="Reversal of benzodiazepine sedation; re-sedation possible",
        recommendation="Lorazepam duration may exceed flumazenil. Monitor for re-sedation for >= 2 hours.",
        evidence_level="established",
    ),
    # 12. Epinephrine + beta-blockers
    _pair_key("epinephrine", "propranolol"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Beta-blockade leaves alpha-adrenergic vasoconstriction unopposed",
        clinical_effect="Severe hypertension and reflex bradycardia",
        recommendation="Use with extreme caution. Consider glucagon for beta-blocker reversal.",
        evidence_level="established",
    ),
    _pair_key("epinephrine", "metoprolol"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Beta-1 blockade may attenuate cardiac effects of epinephrine",
        clinical_effect="Reduced chronotropic/inotropic response to epinephrine; hypertension from unopposed alpha",
        recommendation="May need higher epinephrine doses. Monitor hemodynamics closely.",
        evidence_level="established",
    ),
    # 13. Norepinephrine + MAOIs
    _pair_key("norepinephrine", "phenelzine"): _InteractionEntry(
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism="MAO inhibition prevents norepinephrine degradation, potentiating pressor effects",
        clinical_effect="Severe hypertensive crisis, intracranial hemorrhage",
        recommendation="Contraindicated. Use vasopressin as alternative pressor if possible.",
        evidence_level="established",
    ),
    # 14. Adenosine + carbamazepine
    _pair_key("adenosine", "carbamazepine"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Carbamazepine potentiates adenosine receptor effects",
        clinical_effect="Enhanced and prolonged heart block, severe bradycardia",
        recommendation="Reduce adenosine dose. Start with 3 mg instead of 6 mg.",
        evidence_level="probable",
    ),
    # 15. Adenosine + dipyridamole
    _pair_key("adenosine", "dipyridamole"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Dipyridamole inhibits adenosine deaminase, prolonging adenosine half-life",
        clinical_effect="Prolonged AV block, severe bradycardia, asystole",
        recommendation="Reduce adenosine dose by 75% (start with 1 mg).",
        evidence_level="established",
    ),
    # 16. Propofol + fentanyl (already covered above as fentanyl+propofol)
    # 17. Midazolam + CYP3A4 inhibitors
    _pair_key("midazolam", "ketoconazole"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Potent CYP3A4 inhibition increases midazolam AUC 10-15 fold",
        clinical_effect="Prolonged and excessive sedation, respiratory depression",
        recommendation="Reduce midazolam dose by 50-75% or use lorazepam (glucuronidated, not CYP-dependent).",
        evidence_level="established",
        cyp_enzymes=["CYP3A4"],
    ),
    _pair_key("midazolam", "erythromycin"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="CYP3A4 inhibition increases midazolam levels",
        clinical_effect="Excessive sedation and respiratory depression",
        recommendation="Reduce midazolam dose or choose lorazepam.",
        evidence_level="established",
        cyp_enzymes=["CYP3A4"],
    ),
    # 18. Levetiracetam (few significant interactions, but note)
    _pair_key("levetiracetam", "phenytoin"): _InteractionEntry(
        severity=InteractionSeverity.MINOR,
        mechanism="Minimal pharmacokinetic interaction; additive antiepileptic effect",
        clinical_effect="Generally well tolerated in combination",
        recommendation="No dose adjustment typically needed. Can be combined for refractory status epilepticus.",
        evidence_level="established",
    ),
    # 19. Atropine + other anticholinergics
    _pair_key("atropine", "diphenhydramine"): _InteractionEntry(
        severity=InteractionSeverity.MODERATE,
        mechanism="Additive anticholinergic effects",
        clinical_effect="Anticholinergic toxicity: tachycardia, urinary retention, confusion, hyperthermia",
        recommendation="Monitor for anticholinergic burden. Avoid combination in elderly.",
        evidence_level="established",
    ),
    # 20. Propofol + midazolam
    _pair_key("propofol", "midazolam"): _InteractionEntry(
        severity=InteractionSeverity.MAJOR,
        mechanism="Synergistic GABA-A receptor modulation causing profound CNS depression",
        clinical_effect="Enhanced sedation, hypotension, respiratory depression",
        recommendation="Reduce propofol induction dose by 30-50% if midazolam pre-medication given.",
        evidence_level="established",
    ),
}


# ======================================================================
# QT-prolonging drug registry
# ======================================================================

_QT_PROLONGING_DRUGS: Dict[str, str] = {
    "amiodarone": "Known risk (Class III antiarrhythmic, potassium channel blocker)",
    "haloperidol": "Known risk (D2 antagonist with hERG channel blockade)",
    "ondansetron": "Known risk at IV doses > 16 mg; dose-dependent QT prolongation",
    "droperidol": "Known risk (FDA black box warning for QT prolongation)",
    "erythromycin": "Known risk (hERG channel blocker, CYP3A4 inhibitor)",
    "fluconazole": "Known risk at doses >= 400 mg",
    "methadone": "Known risk (dose-dependent, especially > 100 mg/day)",
    "procainamide": "Known risk (Class IA antiarrhythmic)",
    "sotalol": "Known risk (Class III antiarrhythmic)",
    "chlorpromazine": "Known risk (phenothiazine antipsychotic)",
    "quetiapine": "Conditional risk",
    "azithromycin": "Conditional risk (lower than erythromycin)",
    "ciprofloxacin": "Conditional risk (fluoroquinolone class effect)",
    "levofloxacin": "Conditional risk",
    "moxifloxacin": "Known risk (highest QT risk among fluoroquinolones)",
    "citalopram": "Known risk at doses > 40 mg",
    "escitalopram": "Conditional risk",
    "metoclopramide": "Conditional risk",
    "domperidone": "Known risk",
    "sumatriptan": "Conditional risk",
}


# ======================================================================
# Serotonergic drug registry
# ======================================================================

_SEROTONERGIC_DRUGS: Dict[str, str] = {
    "fluoxetine": "SSRI (potent serotonin reuptake inhibitor)",
    "sertraline": "SSRI",
    "paroxetine": "SSRI",
    "citalopram": "SSRI",
    "escitalopram": "SSRI",
    "fluvoxamine": "SSRI and CYP1A2/CYP2C19 inhibitor",
    "venlafaxine": "SNRI",
    "duloxetine": "SNRI",
    "tramadol": "Weak mu-opioid agonist and serotonin/norepinephrine reuptake inhibitor",
    "fentanyl": "Weak serotonin reuptake inhibitor at high doses",
    "meperidine": "Opioid with serotonin reuptake inhibition",
    "linezolid": "Reversible MAO-A inhibitor",
    "phenelzine": "Irreversible non-selective MAO inhibitor",
    "tranylcypromine": "Irreversible non-selective MAO inhibitor",
    "selegiline": "MAO-B inhibitor (non-selective at high doses)",
    "methylene_blue": "MAO-A inhibitor at doses > 1 mg/kg",
    "triptans": "5-HT1B/1D agonist (sumatriptan, etc.)",
    "ondansetron": "5-HT3 antagonist (low serotonin syndrome risk)",
    "lithium": "Enhances serotonergic neurotransmission",
    "st_johns_wort": "Herbal serotonin reuptake inhibitor",
    "dextromethorphan": "Sigma-1 and NMDA agonist with serotonergic activity",
    "cyclobenzaprine": "Structurally related to TCAs; serotonergic activity",
    "buspirone": "5-HT1A partial agonist",
    "trazodone": "Serotonin antagonist and reuptake inhibitor (SARI)",
    "mirtazapine": "Noradrenergic and specific serotonergic antidepressant",
}


# ======================================================================
# CYP450 substrate/inhibitor/inducer registry
# ======================================================================

_CYP_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "amiodarone": {
        "substrate": ["CYP3A4", "CYP2C8"],
        "inhibitor": ["CYP2C9", "CYP2D6", "CYP3A4"],
        "inducer": [],
    },
    "midazolam": {
        "substrate": ["CYP3A4"],
        "inhibitor": [],
        "inducer": [],
    },
    "fentanyl": {
        "substrate": ["CYP3A4"],
        "inhibitor": [],
        "inducer": [],
    },
    "phenytoin": {
        "substrate": ["CYP2C9", "CYP2C19"],
        "inhibitor": ["CYP2C19"],
        "inducer": ["CYP3A4", "CYP2C9", "CYP2C19"],
    },
    "propofol": {
        "substrate": ["CYP2B6", "CYP2C9"],
        "inhibitor": [],
        "inducer": [],
    },
    "ketamine": {
        "substrate": ["CYP3A4", "CYP2B6"],
        "inhibitor": [],
        "inducer": [],
    },
    "warfarin": {
        "substrate": ["CYP2C9", "CYP3A4", "CYP1A2"],
        "inhibitor": [],
        "inducer": [],
    },
    "fluconazole": {
        "substrate": [],
        "inhibitor": ["CYP2C9", "CYP2C19", "CYP3A4"],
        "inducer": [],
    },
    "erythromycin": {
        "substrate": ["CYP3A4"],
        "inhibitor": ["CYP3A4"],
        "inducer": [],
    },
    "ketoconazole": {
        "substrate": ["CYP3A4"],
        "inhibitor": ["CYP3A4"],
        "inducer": [],
    },
    "carbamazepine": {
        "substrate": ["CYP3A4"],
        "inhibitor": [],
        "inducer": ["CYP3A4", "CYP2C9", "CYP2C19"],
    },
    "rifampin": {
        "substrate": [],
        "inhibitor": [],
        "inducer": ["CYP3A4", "CYP2C9", "CYP2C19", "CYP2B6", "CYP1A2"],
    },
}


# ======================================================================
# DrugInteractionChecker
# ======================================================================


class DrugInteractionChecker:
    """Check for drug-drug interactions relevant to emergency medicine.

    The checker uses a curated database of critical EM drug
    interactions and can assess multi-drug regimens for QT
    prolongation risk, serotonin syndrome potential, and CYP450
    metabolic interactions.

    Examples
    --------
    >>> checker = DrugInteractionChecker()
    >>> result = checker.check_interaction("amiodarone", "haloperidol")
    >>> result.severity
    <InteractionSeverity.MAJOR: 3>
    >>> result.mechanism
    'Additive QT prolongation via potassium channel blockade'

    >>> results = checker.check_all_interactions(
    ...     ["fentanyl", "midazolam", "propofol"]
    ... )
    >>> len(results) >= 2
    True

    References
    ----------
    .. [1] Lexicomp Drug Interactions. Wolters Kluwer.
    .. [2] Woosley RL, et al. CredibleMeds.org.
    """

    def __init__(self) -> None:
        self._db = _INTERACTION_DB
        self._qt_drugs = _QT_PROLONGING_DRUGS
        self._serotonergic_drugs = _SEROTONERGIC_DRUGS
        self._cyp_profiles = _CYP_PROFILES

    def check_interaction(
        self,
        drug_a: str,
        drug_b: str,
    ) -> Optional[InteractionResult]:
        """Check for an interaction between two drugs.

        Parameters
        ----------
        drug_a : str
            First drug name (case-insensitive).
        drug_b : str
            Second drug name (case-insensitive).

        Returns
        -------
        InteractionResult or None
            The interaction finding, or ``None`` if no known interaction
            exists between the two drugs.

        Raises
        ------
        ValidationError
            If either drug name is empty.
        """
        a = drug_a.strip().lower()
        b = drug_b.strip().lower()
        if not a or not b:
            raise ValidationError(
                message="Drug names must not be empty.",
                parameter="drug_a" if not a else "drug_b",
            )
        if a == b:
            return None

        key = _pair_key(a, b)
        entry = self._db.get(key)
        if entry is None:
            return None

        return InteractionResult(
            drug_a=a,
            drug_b=b,
            severity=entry.severity,
            mechanism=entry.mechanism,
            clinical_effect=entry.clinical_effect,
            recommendation=entry.recommendation,
            evidence_level=entry.evidence_level,
            cyp_enzymes=list(entry.cyp_enzymes),
        )

    def check_all_interactions(
        self,
        drug_list: List[str],
    ) -> List[InteractionResult]:
        """Check all pairwise interactions among a list of drugs.

        Parameters
        ----------
        drug_list : list of str
            List of drug names (case-insensitive).

        Returns
        -------
        list of InteractionResult
            All identified interactions, sorted by severity (most
            severe first).

        Raises
        ------
        ValidationError
            If the drug list contains fewer than two drugs.
        """
        if len(drug_list) < 2:
            raise ValidationError(
                message="At least two drugs are required for interaction checking.",
                parameter="drug_list",
            )

        normalized = [d.strip().lower() for d in drug_list]
        results: List[InteractionResult] = []
        checked: Set[FrozenSet[str]] = set()

        for i, a in enumerate(normalized):
            for b in normalized[i + 1:]:
                key = _pair_key(a, b)
                if key in checked or a == b:
                    continue
                checked.add(key)
                result = self.check_interaction(a, b)
                if result is not None:
                    results.append(result)

        results.sort(key=lambda r: r.severity, reverse=True)
        return results

    def assess_qt_risk(
        self,
        drug_list: List[str],
    ) -> List[Tuple[str, str]]:
        """Identify QT-prolonging drugs in a medication list.

        Parameters
        ----------
        drug_list : list of str
            List of drug names (case-insensitive).

        Returns
        -------
        list of tuple of (str, str)
            Each tuple contains ``(drug_name, risk_description)`` for
            drugs with known QT prolongation risk.
        """
        results: List[Tuple[str, str]] = []
        for drug in drug_list:
            key = drug.strip().lower()
            if key in self._qt_drugs:
                results.append((key, self._qt_drugs[key]))
        return results

    def qt_interaction_count(self, drug_list: List[str]) -> int:
        """Count the number of QT-prolonging drugs in a medication list.

        Parameters
        ----------
        drug_list : list of str
            List of drug names.

        Returns
        -------
        int
            Number of QT-prolonging drugs identified.
        """
        return len(self.assess_qt_risk(drug_list))

    def assess_serotonin_risk(
        self,
        drug_list: List[str],
    ) -> List[Tuple[str, str]]:
        """Identify serotonergic drugs that may contribute to serotonin syndrome.

        Two or more serotonergic drugs in the same regimen
        significantly increase the risk of serotonin syndrome.

        Parameters
        ----------
        drug_list : list of str
            List of drug names (case-insensitive).

        Returns
        -------
        list of tuple of (str, str)
            Each tuple contains ``(drug_name, serotonergic_mechanism)``
            for drugs with serotonergic activity.

        References
        ----------
        .. [1] Boyer EW, Shannon M. N Engl J Med. 2005;352(11):1112-1120.
        """
        results: List[Tuple[str, str]] = []
        for drug in drug_list:
            key = drug.strip().lower()
            if key in self._serotonergic_drugs:
                results.append((key, self._serotonergic_drugs[key]))
        return results

    def serotonin_syndrome_risk_level(self, drug_list: List[str]) -> str:
        """Assess the overall serotonin syndrome risk for a drug regimen.

        Parameters
        ----------
        drug_list : list of str
            List of drug names.

        Returns
        -------
        str
            One of ``"none"``, ``"low"``, ``"moderate"``, or ``"high"``.
        """
        serotonergic = self.assess_serotonin_risk(drug_list)
        n = len(serotonergic)
        if n == 0:
            return "none"
        if n == 1:
            return "low"

        mechanisms = [desc for _, desc in serotonergic]
        has_maoi = any("MAO" in m for m in mechanisms)
        if has_maoi and n >= 2:
            return "high"
        if n >= 3:
            return "high"
        return "moderate"

    def get_cyp_profile(self, drug: str) -> Optional[Dict[str, List[str]]]:
        """Return the CYP450 metabolic profile for a drug.

        Parameters
        ----------
        drug : str
            Drug name (case-insensitive).

        Returns
        -------
        dict or None
            Dictionary with keys ``"substrate"``, ``"inhibitor"``, and
            ``"inducer"``, each mapping to a list of CYP enzyme names.
            Returns ``None`` if the drug is not in the CYP database.
        """
        key = drug.strip().lower()
        profile = self._cyp_profiles.get(key)
        if profile is None:
            return None
        return {k: list(v) for k, v in profile.items()}

    def check_cyp_interactions(
        self,
        drug_list: List[str],
    ) -> List[InteractionResult]:
        """Identify potential CYP450-mediated drug interactions.

        Checks whether any drug in the list inhibits or induces a
        CYP enzyme that metabolizes another drug in the list.

        Parameters
        ----------
        drug_list : list of str
            List of drug names.

        Returns
        -------
        list of InteractionResult
            Potential CYP-mediated interactions identified.
        """
        normalized = [d.strip().lower() for d in drug_list]
        profiles = {d: self._cyp_profiles[d] for d in normalized if d in self._cyp_profiles}
        results: List[InteractionResult] = []
        checked: Set[FrozenSet[str]] = set()

        for drug_a, prof_a in profiles.items():
            for drug_b, prof_b in profiles.items():
                if drug_a == drug_b:
                    continue
                key = frozenset((drug_a, drug_b))
                if key in checked:
                    continue

                shared_inhibit = set(prof_a.get("inhibitor", [])) & set(prof_b.get("substrate", []))
                shared_induce = set(prof_a.get("inducer", [])) & set(prof_b.get("substrate", []))
                shared_inhibit_rev = set(prof_b.get("inhibitor", [])) & set(prof_a.get("substrate", []))
                shared_induce_rev = set(prof_b.get("inducer", [])) & set(prof_a.get("substrate", []))

                enzymes_involved: List[str] = []
                mechanisms: List[str] = []

                if shared_inhibit:
                    enzymes_involved.extend(shared_inhibit)
                    mechanisms.append(
                        f"{drug_a} inhibits {', '.join(sorted(shared_inhibit))} "
                        f"which metabolizes {drug_b}"
                    )
                if shared_induce:
                    enzymes_involved.extend(shared_induce)
                    mechanisms.append(
                        f"{drug_a} induces {', '.join(sorted(shared_induce))} "
                        f"which metabolizes {drug_b}"
                    )
                if shared_inhibit_rev:
                    enzymes_involved.extend(shared_inhibit_rev)
                    mechanisms.append(
                        f"{drug_b} inhibits {', '.join(sorted(shared_inhibit_rev))} "
                        f"which metabolizes {drug_a}"
                    )
                if shared_induce_rev:
                    enzymes_involved.extend(shared_induce_rev)
                    mechanisms.append(
                        f"{drug_b} induces {', '.join(sorted(shared_induce_rev))} "
                        f"which metabolizes {drug_a}"
                    )

                if mechanisms:
                    checked.add(key)
                    any_inhibition = bool(shared_inhibit or shared_inhibit_rev)
                    any_induction = bool(shared_induce or shared_induce_rev)

                    if any_inhibition:
                        effect = "Increased levels of substrate drug; risk of toxicity"
                        rec = "Monitor for toxicity of substrate drug. Consider dose reduction."
                    elif any_induction:
                        effect = "Decreased levels of substrate drug; risk of therapeutic failure"
                        rec = "Monitor therapeutic efficacy. May need dose increase."
                    else:
                        effect = "Altered drug metabolism via CYP450 pathway"
                        rec = "Monitor drug levels if available."

                    results.append(InteractionResult(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        severity=InteractionSeverity.MODERATE,
                        mechanism="; ".join(mechanisms),
                        clinical_effect=effect,
                        recommendation=rec,
                        evidence_level="probable",
                        cyp_enzymes=sorted(set(enzymes_involved)),
                    ))

        return results

    @property
    def qt_prolonging_drugs(self) -> List[str]:
        """Return a sorted list of QT-prolonging drugs in the registry.

        Returns
        -------
        list of str
            Drug names.
        """
        return sorted(self._qt_drugs.keys())

    @property
    def serotonergic_drugs(self) -> List[str]:
        """Return a sorted list of serotonergic drugs in the registry.

        Returns
        -------
        list of str
            Drug names.
        """
        return sorted(self._serotonergic_drugs.keys())

    @property
    def interaction_count(self) -> int:
        """Return the number of interaction pairs in the database.

        Returns
        -------
        int
            Number of drug-drug interaction entries.
        """
        return len(self._db)
