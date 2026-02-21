# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Medical coding utilities for emergency medicine.

Provides automated mapping of clinical text to standardized medical
codes:

- :class:`ICDCoder` maps clinical descriptions to ICD-10-CM codes.
- :class:`CPTCoder` maps procedures and visit descriptions to CPT
  codes.

Both coders use TF-IDF cosine similarity to rank candidate codes
against the input text.  Built-in dictionaries cover the most common
emergency-medicine codes (58 ICD-10 codes, 25 CPT codes).

Classes
-------
ICDCode
    Dataclass representing a single ICD-10-CM code match.
CPTCode
    Dataclass representing a single CPT code match.
ICDCoder
    TF-IDF-based ICD-10-CM coder.
CPTCoder
    TF-IDF-based CPT coder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from notfallmedizin.core.exceptions import ValidationError


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class ICDCode:
    """A single ICD-10-CM code match.

    Parameters
    ----------
    code : str
        ICD-10-CM code string (e.g. ``"I21.9"``).
    description : str
        Human-readable description of the code.
    confidence : float
        Similarity score between the input text and the code
        description, in ``[0, 1]``.
    chapter : str
        ICD-10 chapter label (e.g. ``"IX"`` for circulatory diseases).
    """

    code: str
    description: str
    confidence: float
    chapter: str


@dataclass
class CPTCode:
    """A single CPT code match.

    Parameters
    ----------
    code : str
        CPT code string (e.g. ``"99285"``).
    description : str
        Human-readable description of the procedure.
    confidence : float
        Similarity score between the input text and the code
        description, in ``[0, 1]``.
    category : str
        Broad CPT category (e.g. ``"E/M"``, ``"Surgery"``).
    """

    code: str
    description: str
    confidence: float
    category: str


# ------------------------------------------------------------------
# ICD-10-CM database  (58 common emergency codes)
# ------------------------------------------------------------------

_ICD10_DB: dict[str, tuple[str, str]] = {
    # Cardiac
    "I21.0": (
        "Acute transmural myocardial infarction of anterior wall",
        "IX",
    ),
    "I21.1": (
        "Acute transmural myocardial infarction of inferior wall",
        "IX",
    ),
    "I21.3": (
        "Acute transmural myocardial infarction of unspecified site",
        "IX",
    ),
    "I21.4": (
        "Acute subendocardial myocardial infarction",
        "IX",
    ),
    "I21.9": (
        "Acute myocardial infarction unspecified",
        "IX",
    ),
    "I20.0": (
        "Unstable angina",
        "IX",
    ),
    "I46.9": (
        "Cardiac arrest unspecified",
        "IX",
    ),
    "I48.0": (
        "Paroxysmal atrial fibrillation",
        "IX",
    ),
    "I48.91": (
        "Unspecified atrial fibrillation",
        "IX",
    ),
    "I50.9": (
        "Heart failure unspecified",
        "IX",
    ),
    # Pulmonary embolism / DVT
    "I26.99": (
        "Other pulmonary embolism without acute cor pulmonale",
        "IX",
    ),
    "I26.09": (
        "Other pulmonary embolism with acute cor pulmonale",
        "IX",
    ),
    "I82.40": (
        "Acute embolism and thrombosis of deep veins of lower extremity unspecified",
        "IX",
    ),
    # Cerebrovascular
    "I63.9": (
        "Cerebral infarction unspecified",
        "IX",
    ),
    "I61.9": (
        "Nontraumatic intracerebral hemorrhage unspecified",
        "IX",
    ),
    "I60.9": (
        "Nontraumatic subarachnoid hemorrhage unspecified",
        "IX",
    ),
    "G45.9": (
        "Transient cerebral ischemic attack unspecified TIA",
        "VI",
    ),
    # Respiratory
    "J18.9": (
        "Pneumonia unspecified organism",
        "X",
    ),
    "J44.1": (
        "Chronic obstructive pulmonary disease COPD with acute exacerbation",
        "X",
    ),
    "J45.901": (
        "Unspecified asthma with status asthmaticus",
        "X",
    ),
    "J96.00": (
        "Acute respiratory failure unspecified whether with hypoxia or hypercapnia",
        "X",
    ),
    "J06.9": (
        "Acute upper respiratory infection unspecified",
        "X",
    ),
    # Sepsis
    "A41.9": (
        "Sepsis unspecified organism",
        "I",
    ),
    "R65.20": (
        "Severe sepsis without septic shock",
        "XVIII",
    ),
    "R65.21": (
        "Severe sepsis with septic shock",
        "XVIII",
    ),
    # Abdominal / GI
    "K35.80": (
        "Unspecified acute appendicitis",
        "XI",
    ),
    "K81.0": (
        "Acute cholecystitis",
        "XI",
    ),
    "K85.9": (
        "Acute pancreatitis unspecified",
        "XI",
    ),
    "K57.20": (
        "Diverticulitis of large intestine with perforation and abscess without bleeding",
        "XI",
    ),
    "K92.0": (
        "Hematemesis",
        "XI",
    ),
    "K92.1": (
        "Melena",
        "XI",
    ),
    "K92.2": (
        "Gastrointestinal hemorrhage unspecified",
        "XI",
    ),
    "K56.60": (
        "Unspecified intestinal obstruction",
        "XI",
    ),
    # Genitourinary
    "N39.0": (
        "Urinary tract infection site not specified UTI",
        "XIV",
    ),
    "N20.0": (
        "Calculus of kidney renal calculus nephrolithiasis kidney stone",
        "XIV",
    ),
    "N17.9": (
        "Acute kidney failure unspecified acute renal failure",
        "XIV",
    ),
    # Trauma / Fractures
    "S72.001A": (
        "Fracture of unspecified part of neck of right femur initial encounter hip fracture",
        "XIX",
    ),
    "S82.001A": (
        "Unspecified fracture of right patella initial encounter",
        "XIX",
    ),
    "S52.501A": (
        "Unspecified fracture of lower end of right radius initial encounter wrist fracture",
        "XIX",
    ),
    "S42.001A": (
        "Fracture of unspecified part of right clavicle initial encounter",
        "XIX",
    ),
    "S06.0X0A": (
        "Concussion without loss of consciousness initial encounter",
        "XIX",
    ),
    "S06.9X0A": (
        "Unspecified intracranial injury without loss of consciousness initial encounter head injury",
        "XIX",
    ),
    "S22.31XA": (
        "Fracture of one rib right side initial encounter rib fracture",
        "XIX",
    ),
    "S62.001A": (
        "Unspecified fracture of right navicular bone of wrist initial encounter",
        "XIX",
    ),
    "S82.101A": (
        "Unspecified fracture of upper end of right tibia initial encounter",
        "XIX",
    ),
    "S42.201A": (
        "Unspecified fracture of upper end of right humerus initial encounter shoulder fracture",
        "XIX",
    ),
    # Allergic / Anaphylaxis
    "T78.2XXA": (
        "Anaphylactic shock unspecified initial encounter anaphylaxis",
        "XIX",
    ),
    "T78.40XA": (
        "Allergy unspecified initial encounter allergic reaction",
        "XIX",
    ),
    # Endocrine / Metabolic
    "E11.65": (
        "Type 2 diabetes mellitus with hyperglycemia",
        "IV",
    ),
    "E10.10": (
        "Type 1 diabetes mellitus with ketoacidosis without coma DKA",
        "IV",
    ),
    "E16.2": (
        "Hypoglycemia unspecified",
        "IV",
    ),
    # Toxicology / Poisoning
    "T39.1X1A": (
        "Poisoning by 4-aminophenol derivatives accidental acetaminophen overdose",
        "XIX",
    ),
    "T40.2X1A": (
        "Poisoning by other opioids accidental opioid overdose",
        "XIX",
    ),
    "T43.011A": (
        "Poisoning by tricyclic antidepressants accidental TCA overdose",
        "XIX",
    ),
    "T50.901A": (
        "Poisoning by other drugs medicaments and biological substances accidental",
        "XIX",
    ),
    # Neurological
    "G40.901": (
        "Epilepsy unspecified not intractable with status epilepticus seizure",
        "VI",
    ),
    # Signs and symptoms
    "R55": (
        "Syncope and collapse fainting",
        "XVIII",
    ),
    "R06.02": (
        "Shortness of breath dyspnea",
        "XVIII",
    ),
    "R07.9": (
        "Chest pain unspecified",
        "XVIII",
    ),
    "R10.9": (
        "Unspecified abdominal pain",
        "XVIII",
    ),
    "R11.2": (
        "Nausea with vomiting unspecified",
        "XVIII",
    ),
    "M54.5": (
        "Low back pain lumbago",
        "XIII",
    ),
    "T79.4XXA": (
        "Traumatic shock initial encounter",
        "XIX",
    ),
}


# ------------------------------------------------------------------
# CPT database  (25 common emergency codes)
# ------------------------------------------------------------------

_CPT_DB: dict[str, tuple[str, str]] = {
    "99281": (
        "Emergency department visit level 1 self-limited or minor problem",
        "E/M",
    ),
    "99282": (
        "Emergency department visit level 2 low to moderate severity",
        "E/M",
    ),
    "99283": (
        "Emergency department visit level 3 moderate severity",
        "E/M",
    ),
    "99284": (
        "Emergency department visit level 4 high severity urgent evaluation",
        "E/M",
    ),
    "99285": (
        "Emergency department visit level 5 high severity immediate significant threat to life",
        "E/M",
    ),
    "99291": (
        "Critical care evaluation and management first 30 to 74 minutes",
        "E/M",
    ),
    "99292": (
        "Critical care evaluation and management each additional 30 minutes",
        "E/M",
    ),
    "12001": (
        "Simple repair of superficial wounds of scalp neck axillae external genitalia trunk extremities 2.5 cm or less",
        "Surgery",
    ),
    "12011": (
        "Simple repair of superficial wounds of face ears eyelids nose lips mucous membranes 2.5 cm or less",
        "Surgery",
    ),
    "31500": (
        "Intubation endotracheal emergency procedure",
        "Surgery",
    ),
    "36556": (
        "Insertion of non-tunneled centrally inserted central venous catheter",
        "Surgery",
    ),
    "32551": (
        "Tube thoracostomy insertion chest tube with or without water seal",
        "Surgery",
    ),
    "62270": (
        "Lumbar puncture spinal tap diagnostic",
        "Surgery",
    ),
    "49083": (
        "Abdominal paracentesis diagnostic or therapeutic",
        "Surgery",
    ),
    "10060": (
        "Incision and drainage of abscess simple or single",
        "Surgery",
    ),
    "10061": (
        "Incision and drainage of abscess complicated or multiple",
        "Surgery",
    ),
    "93010": (
        "Electrocardiogram ECG interpretation and report only 12-lead",
        "Diagnostic",
    ),
    "93005": (
        "Electrocardiogram ECG tracing only without interpretation 12-lead",
        "Diagnostic",
    ),
    "71046": (
        "Radiologic examination chest 2 views frontal and lateral X-ray",
        "Radiology",
    ),
    "72110": (
        "Radiologic examination lumbosacral spine complete including bending views X-ray",
        "Radiology",
    ),
    "36415": (
        "Collection of venous blood by venipuncture phlebotomy",
        "Lab",
    ),
    "96374": (
        "Therapeutic prophylactic or diagnostic injection intravenous push single or initial substance",
        "Medicine",
    ),
    "96375": (
        "Therapeutic prophylactic or diagnostic injection intravenous push each additional sequential substance",
        "Medicine",
    ),
    "29125": (
        "Application of short arm splint forearm to hand static",
        "Surgery",
    ),
    "24600": (
        "Treatment of closed elbow dislocation without anesthesia",
        "Surgery",
    ),
}


# ------------------------------------------------------------------
# Base coder (shared logic)
# ------------------------------------------------------------------


class _BaseCoder:
    """Internal base class for TF-IDF-based medical code matching.

    Not part of the public API.
    """

    def __init__(
        self,
        database: dict[str, tuple[str, str]],
        top_k: int = 5,
        confidence_threshold: float = 0.1,
    ) -> None:
        self._db = database
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        self._codes: list[str] = list(database.keys())
        self._descriptions: list[str] = [v[0] for v in database.values()]
        self._chapters: list[str] = [v[1] for v in database.values()]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            lowercase=True,
        )
        self._desc_matrix = self._vectorizer.fit_transform(self._descriptions)

    def _match(self, text: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """Return the *top_k* best-matching description indices.

        Parameters
        ----------
        text : str
            Query text.
        top_k : int or None
            Override for the number of results.

        Returns
        -------
        list of (int, float)
            ``(index_into_database, cosine_similarity)`` pairs sorted
            by similarity descending.
        """
        if not text or not text.strip():
            return []
        k = top_k if top_k is not None else self.top_k
        query_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._desc_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (int(idx), float(sims[idx]))
            for idx in top_indices
            if sims[idx] >= self.confidence_threshold
        ]

    def _get_description(self, code: str) -> str:
        """Look up the description for *code*.

        Parameters
        ----------
        code : str
            Code string.

        Returns
        -------
        str
            Description, or ``""`` if not found.
        """
        entry = self._db.get(code)
        return entry[0] if entry else ""

    def _search(self, query: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """Search descriptions by free text query.

        Identical to :meth:`_match` but provided as a semantic alias.
        """
        return self._match(query, top_k=top_k)


# ------------------------------------------------------------------
# ICD-10-CM Coder
# ------------------------------------------------------------------


class ICDCoder(_BaseCoder):
    """Map clinical text to ICD-10-CM codes using TF-IDF similarity.

    Contains a built-in dictionary of 58 common emergency-medicine
    ICD-10-CM codes.  For each query, the coder computes the cosine
    similarity between the query's TF-IDF vector and every code
    description, returning the top matches.

    Parameters
    ----------
    top_k : int, optional
        Number of top matches to return.  Default is ``5``.
    confidence_threshold : float, optional
        Minimum cosine similarity for a code to be included.
        Default is ``0.1``.

    Examples
    --------
    >>> coder = ICDCoder()
    >>> results = coder.encode("patient with acute chest pain and ST elevation")
    >>> results[0].code
    'R07.9'
    """

    def __init__(
        self,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            database=_ICD10_DB,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

    def encode(self, clinical_text: str) -> list[ICDCode]:
        """Map *clinical_text* to ICD-10-CM codes.

        Parameters
        ----------
        clinical_text : str
            Free-text clinical description.

        Returns
        -------
        list of ICDCode
            Matched codes sorted by confidence (descending).

        Raises
        ------
        ValidationError
            If *clinical_text* is empty.
        """
        if not clinical_text or not clinical_text.strip():
            raise ValidationError(
                "clinical_text must not be empty.",
                parameter="clinical_text",
            )
        matches = self._match(clinical_text)
        return [
            ICDCode(
                code=self._codes[idx],
                description=self._descriptions[idx],
                confidence=score,
                chapter=self._chapters[idx],
            )
            for idx, score in matches
        ]

    def get_code_description(self, code: str) -> str:
        """Look up the description for a specific ICD-10-CM code.

        Parameters
        ----------
        code : str
            ICD-10-CM code (e.g. ``"I21.9"``).

        Returns
        -------
        str
            Description string, or ``""`` if the code is not in the
            built-in dictionary.
        """
        return self._get_description(code)

    def search_codes(
        self, query: str, top_k: Optional[int] = None
    ) -> list[ICDCode]:
        """Search for ICD-10-CM codes matching a free-text query.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int or None, optional
            Override for number of results.  Uses the instance default
            if ``None``.

        Returns
        -------
        list of ICDCode
            Matching codes sorted by relevance.
        """
        matches = self._search(query, top_k=top_k)
        return [
            ICDCode(
                code=self._codes[idx],
                description=self._descriptions[idx],
                confidence=score,
                chapter=self._chapters[idx],
            )
            for idx, score in matches
        ]

    def list_codes(self) -> list[ICDCode]:
        """Return every code in the built-in dictionary.

        Returns
        -------
        list of ICDCode
            All codes with confidence set to ``1.0``.
        """
        return [
            ICDCode(
                code=code,
                description=desc,
                confidence=1.0,
                chapter=chapter,
            )
            for code, (desc, chapter) in self._db.items()
        ]


# ------------------------------------------------------------------
# CPT Coder
# ------------------------------------------------------------------


class CPTCoder(_BaseCoder):
    """Map clinical text to CPT codes using TF-IDF similarity.

    Contains a built-in dictionary of 25 common emergency-medicine CPT
    codes covering evaluation and management, surgery, diagnostics,
    radiology, lab, and medicine categories.

    Parameters
    ----------
    top_k : int, optional
        Number of top matches to return.  Default is ``5``.
    confidence_threshold : float, optional
        Minimum cosine similarity for a code to be included.
        Default is ``0.1``.

    Examples
    --------
    >>> coder = CPTCoder()
    >>> results = coder.encode("endotracheal intubation")
    >>> results[0].code
    '31500'
    """

    def __init__(
        self,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            database=_CPT_DB,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

    def encode(self, clinical_text: str) -> list[CPTCode]:
        """Map *clinical_text* to CPT codes.

        Parameters
        ----------
        clinical_text : str
            Free-text clinical or procedural description.

        Returns
        -------
        list of CPTCode
            Matched codes sorted by confidence (descending).

        Raises
        ------
        ValidationError
            If *clinical_text* is empty.
        """
        if not clinical_text or not clinical_text.strip():
            raise ValidationError(
                "clinical_text must not be empty.",
                parameter="clinical_text",
            )
        matches = self._match(clinical_text)
        return [
            CPTCode(
                code=self._codes[idx],
                description=self._descriptions[idx],
                confidence=score,
                category=self._chapters[idx],
            )
            for idx, score in matches
        ]

    def get_code_description(self, code: str) -> str:
        """Look up the description for a specific CPT code.

        Parameters
        ----------
        code : str
            CPT code (e.g. ``"99285"``).

        Returns
        -------
        str
            Description string, or ``""`` if the code is not in the
            built-in dictionary.
        """
        return self._get_description(code)

    def search_codes(
        self, query: str, top_k: Optional[int] = None
    ) -> list[CPTCode]:
        """Search for CPT codes matching a free-text query.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int or None, optional
            Override for number of results.

        Returns
        -------
        list of CPTCode
            Matching codes sorted by relevance.
        """
        matches = self._search(query, top_k=top_k)
        return [
            CPTCode(
                code=self._codes[idx],
                description=self._descriptions[idx],
                confidence=score,
                category=self._chapters[idx],
            )
            for idx, score in matches
        ]

    def list_codes(self) -> list[CPTCode]:
        """Return every code in the built-in dictionary.

        Returns
        -------
        list of CPTCode
            All codes with confidence set to ``1.0``.
        """
        return [
            CPTCode(
                code=code,
                description=desc,
                confidence=1.0,
                category=cat,
            )
            for code, (desc, cat) in self._db.items()
        ]
