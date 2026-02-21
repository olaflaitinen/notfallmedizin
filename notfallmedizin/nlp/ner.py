# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Named Entity Recognition for clinical text.

Provides two approaches to extracting clinical entities from
unstructured medical text:

1. :class:`RuleBasedNER` uses regular expressions and pattern
   matching.  It requires no training data and works as a lightweight
   fallback.
2. :class:`ClinicalNERModel` is a trainable model that uses a
   scikit-learn token classification pipeline by default, with an
   optional HuggingFace transformer backbone when the ``transformers``
   package is installed.

Supported Entity Types
----------------------
SYMPTOM, DIAGNOSIS, MEDICATION, PROCEDURE, ANATOMY, LAB_TEST,
LAB_VALUE, TEMPORAL, SEVERITY, NEGATION

Classes
-------
ClinicalEntity
    Dataclass representing a single extracted entity span.
RuleBasedNER
    Regex-based entity extractor (no training required).
ClinicalNERModel
    Trainable NER model with optional transformer backbone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from notfallmedizin.core.base import ClinicalModel
from notfallmedizin.core.exceptions import (
    DataFormatError,
    ModelNotFittedError,
    ValidationError,
)

try:
    from transformers import pipeline as hf_pipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "SYMPTOM",
        "DIAGNOSIS",
        "MEDICATION",
        "PROCEDURE",
        "ANATOMY",
        "LAB_TEST",
        "LAB_VALUE",
        "TEMPORAL",
        "SEVERITY",
        "NEGATION",
    }
)

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z'\-]*"
    r"|\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
    r"|[^\s]"
)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class ClinicalEntity:
    """A single clinical entity extracted from text.

    Parameters
    ----------
    text : str
        The surface form of the entity as it appears in the source text.
    entity_type : str
        One of the recognised entity types (e.g. ``"MEDICATION"``).
    start : int
        Character offset for the start of the entity span (inclusive).
    end : int
        Character offset for the end of the entity span (exclusive).
    confidence : float
        Model confidence in the prediction, in ``[0, 1]``.
    """

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise ValidationError(
                f"Unknown entity type '{self.entity_type}'. "
                f"Must be one of {sorted(ENTITY_TYPES)}.",
                parameter="entity_type",
            )
        if self.start < 0 or self.end < self.start:
            raise ValidationError(
                f"Invalid span: start={self.start}, end={self.end}.",
                parameter="start",
            )
        self.confidence = max(0.0, min(1.0, self.confidence))


# ------------------------------------------------------------------
# Tokenization helpers
# ------------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(token, start, end)`` triples.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of (str, int, int)
        Each element is ``(surface_form, char_start, char_end)``.
    """
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def _word_shape(token: str) -> str:
    """Return a coarse shape descriptor for *token*.

    Parameters
    ----------
    token : str
        A single token string.

    Returns
    -------
    str
        One of ``"digits"``, ``"UPPER"``, ``"Title"``, ``"lower"``,
        or ``"mixed"``.
    """
    if token.isdigit():
        return "digits"
    if token.isupper():
        return "UPPER"
    if token[0].isupper() and token[1:].islower():
        return "Title"
    if token.islower():
        return "lower"
    return "mixed"


# ------------------------------------------------------------------
# Rule-based NER
# ------------------------------------------------------------------


class RuleBasedNER:
    """Regex and pattern-based NER for clinical text.

    A lightweight alternative to the trainable :class:`ClinicalNERModel`
    that uses hand-crafted regular-expression patterns to identify
    clinical entities.  No training data is required.

    Parameters
    ----------
    confidence : float, optional
        Fixed confidence value assigned to every rule-based match.
        Default is ``0.85``.

    Examples
    --------
    >>> ner = RuleBasedNER()
    >>> entities = ner.predict("BP 120/80, HR 95. Started amoxicillin 500mg.")
    >>> [e.entity_type for e in entities]
    ['LAB_VALUE', 'LAB_VALUE', 'MEDICATION']
    """

    def __init__(self, confidence: float = 0.85) -> None:
        self.confidence = confidence
        self._patterns: dict[str, list[re.Pattern[str]]] = self._build_patterns()

    # ---- pattern compilation ------------------------------------

    @staticmethod
    def _build_patterns() -> dict[str, list[re.Pattern[str]]]:
        """Compile and return regex patterns grouped by entity type."""
        flags = re.IGNORECASE

        medication_names = (
            r"aspirin|clopidogrel|heparin|enoxaparin|warfarin|alteplase"
            r"|epinephrine|norepinephrine|dopamine|dobutamine|vasopressin"
            r"|amiodarone|lidocaine|adenosine|atropine|diltiazem"
            r"|morphine|fentanyl|ketamine|propofol|midazolam|lorazepam|diazepam"
            r"|acetaminophen|ibuprofen|ketorolac|naproxen"
            r"|ceftriaxone|azithromycin|vancomycin|piperacillin|metronidazole"
            r"|levofloxacin|ciprofloxacin|amoxicillin|ampicillin"
            r"|ondansetron|metoclopramide|promethazine"
            r"|albuterol|ipratropium|methylprednisolone|prednisone|dexamethasone"
            r"|nitroglycerin|labetalol|nicardipine|metoprolol|esmolol"
            r"|furosemide|mannitol|hydrochlorothiazide"
            r"|rocuronium|succinylcholine"
            r"|naloxone|flumazenil"
            r"|insulin|glucagon|dextrose"
            r"|magnesium\s*sulfate|calcium\s*chloride|calcium\s*gluconate"
            r"|potassium\s*chloride|sodium\s*bicarbonate"
            r"|tenecteplase|reteplase"
            r"|phenylephrine|pantoprazole|omeprazole"
        )

        patterns: dict[str, list[re.Pattern[str]]] = {}

        # MEDICATION: drug name optionally followed by a dose
        patterns["MEDICATION"] = [
            re.compile(
                rf"\b(?:{medication_names})"
                r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL|units?|IU|meq))?",
                flags,
            ),
        ]

        # LAB_VALUE: vital signs with numeric readings, labs with units
        patterns["LAB_VALUE"] = [
            re.compile(
                r"\b(?:BP|blood\s+pressure)\s*:?\s*\d{2,3}\s*/\s*\d{2,3}"
                r"(?:\s*mmHg)?",
                flags,
            ),
            re.compile(
                r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmHg\b",
                flags,
            ),
            re.compile(
                r"\b(?:HR|heart\s+rate|pulse)\s*:?\s*\d{2,3}"
                r"(?:\s*(?:bpm|/min))?",
                flags,
            ),
            re.compile(
                r"\b(?:RR|resp(?:iratory)?\s+rate)\s*:?\s*\d{1,2}"
                r"(?:\s*/min)?",
                flags,
            ),
            re.compile(
                r"\b(?:SpO2|O2\s*sat(?:uration)?)\s*:?\s*\d{2,3}\s*%?",
                flags,
            ),
            re.compile(
                r"\b(?:temp(?:erature)?)\s*:?\s*\d{2,3}(?:\.\d{1,2})?"
                r"\s*(?:[°]?\s*[CF])?",
                flags,
            ),
            re.compile(
                r"\bGCS\s*:?\s*(?:[3-9]|1[0-5])\b",
                flags,
            ),
            re.compile(
                r"\b(?:WBC|RBC|Hgb|Hct|plt|Na|K|Cl|CO2|BUN|Cr|"
                r"creatinine|glucose|AST|ALT|troponin|lactate|"
                r"BNP|INR|pH|pCO2|pO2|HCO3|lipase|amylase|"
                r"d-dimer|procalcitonin|CRP|ESR)"
                r"\s*:?\s*\d+(?:\.\d+)?",
                flags,
            ),
            re.compile(
                r"\b\d+(?:\.\d+)?\s*(?:mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|"
                r"ng/mL|pg/mL|mcg/dL|x\s*10\^[36]/[mu]L)\b",
                flags,
            ),
        ]

        # LAB_TEST: names of laboratory tests and panels
        patterns["LAB_TEST"] = [
            re.compile(
                r"\b(?:CBC|BMP|CMP|ABG|VBG|UA|urinalysis|"
                r"blood\s+culture|urine\s+culture|"
                r"troponin|d-dimer|lactate|procalcitonin|BNP|NT-proBNP|"
                r"PT|PTT|INR|fibrinogen|"
                r"lipase|amylase|LFTs|liver\s+function|"
                r"CRP|ESR|sed\s+rate)\b",
                flags,
            ),
        ]

        # TEMPORAL
        patterns["TEMPORAL"] = [
            re.compile(
                r"\b\d+\s*(?:hours?|hrs?|minutes?|mins?|days?|weeks?|"
                r"months?|years?)\s+(?:ago|prior|before|earlier)\b",
                flags,
            ),
            re.compile(
                r"\b(?:since|for|x)\s*\d+\s*(?:hours?|hrs?|minutes?|mins?|"
                r"days?|weeks?|months?|years?)\b",
                flags,
            ),
            re.compile(
                r"\bfor\s+the\s+(?:past|last)\s+\d+\s*(?:hours?|hrs?|"
                r"minutes?|mins?|days?|weeks?|months?|years?)\b",
                flags,
            ),
            re.compile(
                r"\b(?:yesterday|today|this\s+morning|last\s+night|"
                r"tonight|last\s+week|this\s+evening)\b",
                flags,
            ),
            re.compile(
                r"\b(?:sudden\s+onset|gradual\s+onset|acute\s+onset)\b",
                flags,
            ),
        ]

        # SEVERITY
        patterns["SEVERITY"] = [
            re.compile(
                r"\b(?:mild|moderate|severe|critical|"
                r"life[\s-]threatening)\b",
                flags,
            ),
            re.compile(
                r"\b(?:worsening|improving|stable|progressive|"
                r"intermittent|constant|persistent)\b",
                flags,
            ),
            re.compile(r"\b\d+\s*/\s*10\s*(?:pain)?\b", flags),
        ]

        # NEGATION
        patterns["NEGATION"] = [
            re.compile(
                r"\b(?:no|not|none|without|denies|denied|negative\s+for|"
                r"absence\s+of|no\s+evidence\s+of|rules?\s+out|"
                r"ruled\s+out|r/o|unlikely|unremarkable|"
                r"within\s+normal\s+limits|WNL)\b",
                flags,
            ),
        ]

        # SYMPTOM
        patterns["SYMPTOM"] = [
            re.compile(
                r"\b(?:chest\s+pain|shortness\s+of\s+breath|"
                r"difficulty\s+breathing|abdominal\s+pain|"
                r"back\s+pain|neck\s+pain|flank\s+pain|"
                r"pelvic\s+pain|epigastric\s+pain|"
                r"pleuritic\s+pain|substernal\s+pain)\b",
                flags,
            ),
            re.compile(
                r"\b(?:headache|dizziness|syncope|seizure|seizures|"
                r"nausea|vomiting|diarrhea|constipation|"
                r"confusion|weakness|numbness|tingling|"
                r"fever|chills|cough|sore\s+throat|congestion|"
                r"palpitations|edema|hemoptysis|hematuria|"
                r"dysuria|melena|hematemesis|"
                r"diaphoresis|tachycardia|bradycardia|"
                r"hypotension|hypertension|"
                r"dyspnea|wheezing|stridor|orthopnea|"
                r"diplopia|dysarthria|aphasia|ataxia|vertigo|"
                r"altered\s+mental\s+status|"
                r"loss\s+of\s+consciousness)\b",
                flags,
            ),
        ]

        # DIAGNOSIS
        patterns["DIAGNOSIS"] = [
            re.compile(
                r"\b(?:myocardial\s+infarction|STEMI|NSTEMI|"
                r"unstable\s+angina|acute\s+coronary\s+syndrome|ACS|"
                r"atrial\s+fibrillation|a-?fib|"
                r"heart\s+failure|CHF|"
                r"cardiac\s+arrest)\b",
                flags,
            ),
            re.compile(
                r"\b(?:stroke|CVA|TIA|"
                r"cerebrovascular\s+accident|"
                r"intracranial\s+hemorrhage|"
                r"subarachnoid\s+hemorrhage|SAH)\b",
                flags,
            ),
            re.compile(
                r"\b(?:pulmonary\s+embolism|PE|"
                r"deep\s+vein\s+thrombosis|DVT)\b",
                flags,
            ),
            re.compile(
                r"\b(?:pneumonia|COPD\s+exacerbation|"
                r"asthma\s+exacerbation|respiratory\s+failure|"
                r"ARDS|pleural\s+effusion)\b",
                flags,
            ),
            re.compile(
                r"\b(?:sepsis|septic\s+shock|bacteremia|"
                r"SIRS)\b",
                flags,
            ),
            re.compile(
                r"\b(?:appendicitis|cholecystitis|pancreatitis|"
                r"diverticulitis|bowel\s+obstruction|"
                r"GI\s+bleed|gastrointestinal\s+hemorrhage)\b",
                flags,
            ),
            re.compile(
                r"\b(?:aortic\s+dissection|aortic\s+aneurysm|AAA)\b",
                flags,
            ),
            re.compile(
                r"\b(?:DKA|diabetic\s+ketoacidosis|"
                r"hypoglycemia|hyperglycemia)\b",
                flags,
            ),
            re.compile(
                r"\b(?:anaphylaxis|allergic\s+reaction|"
                r"angioedema)\b",
                flags,
            ),
            re.compile(
                r"\b(?:meningitis|encephalitis|"
                r"status\s+epilepticus)\b",
                flags,
            ),
            re.compile(
                r"\b(?:urinary\s+tract\s+infection|UTI|"
                r"pyelonephritis|renal\s+calculi|"
                r"kidney\s+stone|renal\s+failure)\b",
                flags,
            ),
            re.compile(
                r"\b(?:fracture|dislocation|concussion|"
                r"laceration|contusion|sprain)\b",
                flags,
            ),
        ]

        # ANATOMY
        patterns["ANATOMY"] = [
            re.compile(
                r"\b(?:left|right|bilateral)\s+"
                r"(?:arm|leg|hand|foot|eye|ear|lung|kidney|"
                r"shoulder|hip|knee|ankle|wrist|elbow)\b",
                flags,
            ),
            re.compile(
                r"\b(?:head|neck|throat|chest|thorax|abdomen|"
                r"pelvis|back|spine|extremit(?:y|ies)|"
                r"heart|lungs?|liver|kidneys?|brain|spleen|"
                r"pancreas|gallbladder|"
                r"cervical\s+spine|thoracic\s+spine|"
                r"lumbar\s+spine|c-spine|"
                r"femur|tibia|fibula|humerus|radius|ulna|"
                r"clavicle|scapula|cranium|skull|ribs?|sternum)\b",
                flags,
            ),
        ]

        # PROCEDURE
        patterns["PROCEDURE"] = [
            re.compile(
                r"\b(?:intubation|extubation|"
                r"rapid\s+sequence\s+intubation|RSI|"
                r"CPR|cardiopulmonary\s+resuscitation|"
                r"defibrillation|cardioversion|"
                r"chest\s+tube|thoracostomy|thoracentesis|"
                r"central\s+line|central\s+venous\s+catheter|"
                r"arterial\s+line|IO\s+access|"
                r"lumbar\s+puncture|paracentesis|"
                r"incision\s+and\s+drainage|"
                r"fracture\s+reduction|closed\s+reduction|"
                r"splinting|suturing|wound\s+closure|"
                r"blood\s+transfusion|"
                r"CT\s+scan|MRI|X-ray|ultrasound|"
                r"ECG|EKG|echocardiogram)\b",
                flags,
            ),
        ]

        return patterns

    # ---- prediction ---------------------------------------------

    def predict(self, text: str) -> list[ClinicalEntity]:
        """Extract clinical entities from *text* using pattern matching.

        Parameters
        ----------
        text : str
            Clinical free-text input.

        Returns
        -------
        list of ClinicalEntity
            Extracted entities sorted by start offset.
        """
        if not text or not text.strip():
            return []

        entities: list[ClinicalEntity] = []
        for entity_type, pattern_list in self._patterns.items():
            for pattern in pattern_list:
                for match in pattern.finditer(text):
                    matched = match.group().strip()
                    if not matched:
                        continue
                    entities.append(
                        ClinicalEntity(
                            text=matched,
                            entity_type=entity_type,
                            start=match.start(),
                            end=match.start() + len(matched),
                            confidence=self.confidence,
                        )
                    )
        return self._resolve_overlaps(entities)

    # ---- overlap resolution -------------------------------------

    @staticmethod
    def _resolve_overlaps(
        entities: list[ClinicalEntity],
    ) -> list[ClinicalEntity]:
        """Remove overlapping entities, preferring longer spans.

        Entities are sorted by start offset and then by span length
        (descending).  A greedy sweep keeps only entities whose start
        offset is at or beyond the end of the last accepted entity.

        Parameters
        ----------
        entities : list of ClinicalEntity
            Possibly overlapping entities.

        Returns
        -------
        list of ClinicalEntity
            Non-overlapping entities sorted by start offset.
        """
        if not entities:
            return []
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        result: list[ClinicalEntity] = []
        last_end = -1
        for ent in entities:
            if ent.start >= last_end:
                result.append(ent)
                last_end = ent.end
        return result


# ------------------------------------------------------------------
# Trainable NER Model
# ------------------------------------------------------------------


class ClinicalNERModel(ClinicalModel):
    """Trainable NER model for clinical text.

    By default the model uses a scikit-learn token classification
    pipeline (character n-gram features with logistic regression).
    When ``use_transformer=True`` and the ``transformers`` package is
    installed, a pre-trained HuggingFace token-classification pipeline
    is used instead.

    Regardless of the backend, predictions are merged with results from
    an internal :class:`RuleBasedNER` instance to maximise recall.

    Parameters
    ----------
    use_transformer : bool, optional
        If ``True``, attempt to load a HuggingFace transformer model.
        Default is ``False``.
    model_name : str, optional
        HuggingFace model identifier for the transformer backend.
        Ignored when ``use_transformer`` is ``False``.  Default is
        ``"d4data/biomedical-ner-all"``.
    confidence_threshold : float, optional
        Minimum confidence for an entity to be included in results.
        Default is ``0.3``.
    max_iter : int, optional
        Maximum number of logistic-regression training iterations.
        Default is ``1000``.

    Examples
    --------
    >>> model = ClinicalNERModel()
    >>> model.fit(
    ...     ["Patient has chest pain and fever"],
    ...     [[{"entity_type": "SYMPTOM", "start": 12, "end": 22},
    ...       {"entity_type": "SYMPTOM", "start": 27, "end": 32}]],
    ... )
    >>> entities = model.predict("Patient reports chest pain")
    """

    _HF_LABEL_MAP: Dict[str, str] = {
        "Disease_disorder": "DIAGNOSIS",
        "Sign_symptom": "SYMPTOM",
        "Medication": "MEDICATION",
        "Biological_structure": "ANATOMY",
        "Diagnostic_procedure": "PROCEDURE",
        "Therapeutic_procedure": "PROCEDURE",
        "Lab_value": "LAB_VALUE",
        "Severity": "SEVERITY",
        "Negation": "NEGATION",
        "Date": "TEMPORAL",
        "Time": "TEMPORAL",
        "Duration": "TEMPORAL",
    }

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = "d4data/biomedical-ner-all",
        confidence_threshold: float = 0.3,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(
            description="Clinical Named Entity Recognition model",
        )
        if use_transformer and not _HAS_TRANSFORMERS:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required "
                "when use_transformer=True.  Install them with: "
                "pip install transformers torch"
            )
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter

        self._rule_ner = RuleBasedNER()

        # sklearn backend state (populated by fit)
        self._vectorizer: Optional[CountVectorizer] = None
        self._classifier: Optional[LogisticRegression] = None
        self._label_encoder: Optional[LabelEncoder] = None

        # transformer backend state (populated lazily)
        self._hf_pipeline: Optional[Any] = None

    # ---- feature engineering ------------------------------------

    @staticmethod
    def _build_token_feature(
        tokens: list[tuple[str, int, int]], idx: int
    ) -> str:
        """Build a feature string for the token at position *idx*.

        Parameters
        ----------
        tokens : list of (str, int, int)
            Tokenized text as ``(surface, start, end)`` triples.
        idx : int
            Index of the target token.

        Returns
        -------
        str
            Space-separated feature string suitable for vectorization.
        """
        tok = tokens[idx][0]
        parts: list[str] = [
            f"w={tok.lower()}",
            f"shape={_word_shape(tok) if len(tok) > 1 else 'single'}",
            f"pre2={tok[:2].lower()}",
            f"suf2={tok[-2:].lower()}",
            f"pre3={tok[:3].lower()}",
            f"suf3={tok[-3:].lower()}",
            f"has_digit={'1' if any(c.isdigit() for c in tok) else '0'}",
            f"has_hyphen={'1' if '-' in tok else '0'}",
            f"is_upper={'1' if tok.isupper() else '0'}",
            f"len={min(len(tok), 20)}",
        ]
        if idx > 0:
            parts.append(f"p1={tokens[idx - 1][0].lower()}")
        else:
            parts.append("p1=BOS")
        if idx > 1:
            parts.append(f"p2={tokens[idx - 2][0].lower()}")
        else:
            parts.append("p2=BOS2")
        if idx < len(tokens) - 1:
            parts.append(f"n1={tokens[idx + 1][0].lower()}")
        else:
            parts.append("n1=EOS")
        if idx < len(tokens) - 2:
            parts.append(f"n2={tokens[idx + 2][0].lower()}")
        else:
            parts.append("n2=EOS2")
        return " ".join(parts)

    @staticmethod
    def _annotations_to_bio(
        tokens: list[tuple[str, int, int]],
        annotations: list[dict[str, Any]],
    ) -> list[str]:
        """Convert entity annotations into BIO-format labels.

        Parameters
        ----------
        tokens : list of (str, int, int)
            Tokenized text.
        annotations : list of dict
            Each dict must contain ``"entity_type"`` (str),
            ``"start"`` (int), and ``"end"`` (int).

        Returns
        -------
        list of str
            BIO label for each token (e.g. ``"B-MEDICATION"``,
            ``"I-MEDICATION"``, ``"O"``).
        """
        labels = ["O"] * len(tokens)
        for ann in annotations:
            a_start = ann["start"]
            a_end = ann["end"]
            etype = ann["entity_type"]
            if etype not in ENTITY_TYPES:
                continue
            first = True
            for i, (_tok, ts, te) in enumerate(tokens):
                if ts >= a_start and te <= a_end:
                    labels[i] = f"B-{etype}" if first else f"I-{etype}"
                    first = False
        return labels

    @staticmethod
    def _bio_to_entities(
        text: str,
        tokens: list[tuple[str, int, int]],
        bio_labels: Sequence[str],
        confidences: Optional[Sequence[float]] = None,
    ) -> list[ClinicalEntity]:
        """Convert BIO label sequences back into entity spans.

        Parameters
        ----------
        text : str
            Original text (used to extract surface forms).
        tokens : list of (str, int, int)
            Tokenized text.
        bio_labels : sequence of str
            BIO label per token.
        confidences : sequence of float or None
            Per-token confidence scores.

        Returns
        -------
        list of ClinicalEntity
        """
        entities: list[ClinicalEntity] = []
        cur_type: Optional[str] = None
        cur_start: int = 0
        cur_confs: list[float] = []

        def _flush(end_idx: int) -> None:
            if cur_type is not None:
                end_offset = tokens[end_idx][2]
                entities.append(
                    ClinicalEntity(
                        text=text[cur_start:end_offset],
                        entity_type=cur_type,
                        start=cur_start,
                        end=end_offset,
                        confidence=float(np.mean(cur_confs)) if cur_confs else 1.0,
                    )
                )

        for i, label in enumerate(bio_labels):
            conf = confidences[i] if confidences is not None else 1.0
            if label.startswith("B-"):
                if cur_type is not None:
                    _flush(i - 1)
                cur_type = label[2:]
                cur_start = tokens[i][1]
                cur_confs = [conf]
            elif label.startswith("I-") and cur_type == label[2:]:
                cur_confs.append(conf)
            else:
                if cur_type is not None:
                    _flush(i - 1)
                cur_type = None
                cur_confs = []

        if cur_type is not None:
            _flush(len(tokens) - 1)

        return entities

    # ---- sklearn training / prediction --------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> ClinicalNERModel:
        """Train the NER model on annotated clinical texts.

        Parameters
        ----------
        X : list of str
            Training texts.
        y : list of list of dict
            Annotations for each text.  Each annotation dict must
            contain ``"entity_type"`` (str), ``"start"`` (int), and
            ``"end"`` (int).
        **kwargs
            Unused.

        Returns
        -------
        self

        Raises
        ------
        ValidationError
            If inputs are invalid or empty.
        DataFormatError
            If annotation dicts are missing required keys.
        """
        texts: list[str] = list(X)  # type: ignore[arg-type]
        if y is None:
            raise ValidationError(
                "Annotations (y) are required for NER training.",
                parameter="y",
            )
        annotations_list: list[list[dict[str, Any]]] = list(y)  # type: ignore[arg-type]

        if len(texts) == 0:
            raise ValidationError("At least one training text is required.")
        if len(texts) != len(annotations_list):
            raise ValidationError(
                f"Length mismatch: {len(texts)} texts vs "
                f"{len(annotations_list)} annotation lists.",
            )

        required_keys = {"entity_type", "start", "end"}
        for idx, anns in enumerate(annotations_list):
            for ann in anns:
                missing = required_keys - set(ann.keys())
                if missing:
                    raise DataFormatError(
                        f"Annotation at index {idx} is missing keys: {missing}."
                    )

        if self.use_transformer:
            self._load_transformer()
            self._set_fitted()
            return self

        all_features: list[str] = []
        all_labels: list[str] = []

        for text, anns in zip(texts, annotations_list):
            tokens = _tokenize(text)
            if not tokens:
                continue
            bio = self._annotations_to_bio(tokens, anns)
            for i in range(len(tokens)):
                all_features.append(self._build_token_feature(tokens, i))
                all_labels.append(bio[i])

        unique_labels = set(all_labels)
        if len(unique_labels) < 2:
            self._vectorizer = None
            self._classifier = None
            self._label_encoder = None
            self._set_fitted()
            return self

        self._vectorizer = CountVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=50000
        )
        X_train = self._vectorizer.fit_transform(all_features)

        self._label_encoder = LabelEncoder()
        y_train = self._label_encoder.fit_transform(all_labels)

        self._classifier = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            C=1.0,
        )
        self._classifier.fit(X_train, y_train)

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> Union[list[ClinicalEntity], list[list[ClinicalEntity]]]:
        """Extract clinical entities from one or more texts.

        Parameters
        ----------
        X : str or list of str
            A single clinical text or a list of texts.

        Returns
        -------
        list of ClinicalEntity
            If *X* is a single string.
        list of list of ClinicalEntity
            If *X* is a list of strings.

        Raises
        ------
        ModelNotFittedError
            If the sklearn backend is selected and :meth:`fit` has not
            been called (rule-based fallback is still applied).
        """
        single = isinstance(X, str)
        texts: list[str] = [X] if single else list(X)  # type: ignore[arg-type]

        results: list[list[ClinicalEntity]] = []
        for text in texts:
            results.append(self._predict_single(text))
        return results[0] if single else results

    def _predict_single(self, text: str) -> list[ClinicalEntity]:
        """Run NER on a single text string.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list of ClinicalEntity
        """
        if not text or not text.strip():
            return []

        all_entities: list[ClinicalEntity] = []

        # Transformer backend
        if self.use_transformer and self._hf_pipeline is not None:
            all_entities.extend(self._predict_transformer(text))

        # Sklearn backend
        elif self.is_fitted_ and self._classifier is not None:
            all_entities.extend(self._predict_sklearn(text))

        # Always add rule-based entities for recall
        all_entities.extend(self._rule_ner.predict(text))

        # Deduplicate and resolve overlaps
        filtered = [
            e for e in all_entities if e.confidence >= self.confidence_threshold
        ]
        return RuleBasedNER._resolve_overlaps(filtered)

    def _predict_sklearn(self, text: str) -> list[ClinicalEntity]:
        """Run the sklearn token classifier on *text*.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list of ClinicalEntity
        """
        tokens = _tokenize(text)
        if not tokens:
            return []

        features = [
            self._build_token_feature(tokens, i) for i in range(len(tokens))
        ]
        X_vec = self._vectorizer.transform(features)  # type: ignore[union-attr]
        y_pred = self._classifier.predict(X_vec)  # type: ignore[union-attr]
        y_proba = self._classifier.predict_proba(X_vec)  # type: ignore[union-attr]
        bio_labels = self._label_encoder.inverse_transform(y_pred)  # type: ignore[union-attr]
        confidences = [float(np.max(row)) for row in y_proba]

        return self._bio_to_entities(text, tokens, bio_labels, confidences)

    def _predict_transformer(self, text: str) -> list[ClinicalEntity]:
        """Run the HuggingFace NER pipeline on *text*.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list of ClinicalEntity
        """
        raw_results = self._hf_pipeline(text)  # type: ignore[misc]
        entities: list[ClinicalEntity] = []
        for item in raw_results:
            raw_label: str = item.get("entity_group", item.get("entity", ""))
            clean_label = raw_label.lstrip("B-").lstrip("I-")
            mapped = self._HF_LABEL_MAP.get(clean_label, None)
            if mapped is None:
                continue
            conf = float(item.get("score", 0.0))
            if conf < self.confidence_threshold:
                continue
            entities.append(
                ClinicalEntity(
                    text=item["word"],
                    entity_type=mapped,
                    start=item["start"],
                    end=item["end"],
                    confidence=conf,
                )
            )
        return entities

    def _load_transformer(self) -> None:
        """Load the HuggingFace NER pipeline if not yet initialised."""
        if self._hf_pipeline is not None:
            return
        self._hf_pipeline = hf_pipeline(
            "ner",
            model=self.model_name,
            aggregation_strategy="simple",
        )
