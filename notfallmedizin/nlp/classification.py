# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Clinical text classification for emergency medicine.

Provides classifiers for two common emergency-department NLP tasks:

1. :class:`ChiefComplaintClassifier` maps free-text chief complaints
   to ESI-compatible clinical categories (cardiac, respiratory, etc.).
2. :class:`TriageNotesClassifier` maps triage nursing notes to an
   acuity / severity level (1 through 5).

Both classifiers use a TF-IDF plus logistic-regression pipeline by
default.  When the ``transformers`` package is installed, an optional
transformer backbone can be selected.

Classes
-------
AcuityLevel
    Enum of acuity levels 1 through 5.
ChiefComplaintClassifier
    Maps chief complaints to clinical categories.
TriageNotesClassifier
    Maps triage notes to acuity levels.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_extraction.text import TfidfVectorizer
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
# Constants and enums
# ------------------------------------------------------------------


class AcuityLevel(enum.IntEnum):
    """Emergency Severity Index acuity levels.

    Attributes
    ----------
    RESUSCITATION : int
        Level 1 -- immediate life-saving intervention required.
    EMERGENT : int
        Level 2 -- high risk, confused/lethargic, or severe pain.
    URGENT : int
        Level 3 -- multiple resources needed.
    LESS_URGENT : int
        Level 4 -- one resource expected.
    NON_URGENT : int
        Level 5 -- no resources expected.
    """

    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


COMPLAINT_CATEGORIES: list[str] = [
    "cardiac",
    "respiratory",
    "neurological",
    "trauma",
    "abdominal",
    "psychiatric",
    "pediatric",
    "obstetric",
    "toxicological",
    "other",
]

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cardiac": [
        "chest pain", "palpitations", "syncope", "cardiac arrest",
        "heart attack", "mi", "stemi", "nstemi", "angina",
        "atrial fibrillation", "afib", "arrhythmia", "heart failure",
        "chf", "bradycardia", "tachycardia", "aortic",
        "hypertensive emergency", "cardiac",
    ],
    "respiratory": [
        "shortness of breath", "dyspnea", "sob", "difficulty breathing",
        "wheezing", "asthma", "copd", "pneumonia", "cough",
        "hemoptysis", "respiratory distress", "stridor",
        "pulmonary embolism", "pe", "pleurisy", "respiratory failure",
        "oxygen", "desaturation", "hypoxia",
    ],
    "neurological": [
        "headache", "seizure", "stroke", "cva", "tia", "weakness",
        "numbness", "tingling", "altered mental status", "confusion",
        "dizziness", "vertigo", "aphasia", "dysarthria", "diplopia",
        "meningitis", "encephalitis", "facial droop", "unresponsive",
        "loss of consciousness", "loc",
    ],
    "trauma": [
        "fall", "fell", "mva", "motor vehicle", "accident", "laceration",
        "fracture", "dislocation", "head injury", "concussion",
        "wound", "assault", "gunshot", "gsw", "stab", "burn",
        "crush injury", "amputation", "blunt trauma",
        "penetrating trauma", "trauma", "injury", "injuries",
    ],
    "abdominal": [
        "abdominal pain", "nausea", "vomiting", "diarrhea",
        "constipation", "gi bleed", "melena", "hematemesis",
        "appendicitis", "cholecystitis", "pancreatitis",
        "bowel obstruction", "flank pain", "rectal bleeding",
        "abdominal", "epigastric",
    ],
    "psychiatric": [
        "suicidal", "si", "homicidal", "psychosis", "hallucination",
        "anxiety", "panic attack", "depression", "agitation",
        "psychiatric", "bipolar", "schizophrenia", "overdose",
        "self harm", "self-harm",
    ],
    "pediatric": [
        "pediatric", "child", "infant", "neonate", "newborn",
        "febrile seizure", "croup", "bronchiolitis", "rsv",
        "failure to thrive", "neonatal", "pediatric fever",
    ],
    "obstetric": [
        "pregnant", "pregnancy", "contractions", "labor", "delivery",
        "vaginal bleeding", "ectopic", "preeclampsia", "eclampsia",
        "placenta", "postpartum", "obstetric", "ob",
        "gestational", "trimester",
    ],
    "toxicological": [
        "overdose", "ingestion", "poisoning", "toxic", "intoxication",
        "alcohol", "drug abuse", "substance abuse",
        "acetaminophen overdose", "opioid overdose", "withdrawal",
        "toxicological", "envenomation", "bite",
    ],
    "other": [],
}

_SEVERITY_KEYWORDS: dict[int, list[str]] = {
    1: [
        "cardiac arrest", "unresponsive", "intubated", "pulseless",
        "apneic", "code blue", "resuscitation", "cpr",
        "hemodynamically unstable", "active hemorrhage",
        "status epilepticus", "anaphylaxis",
    ],
    2: [
        "chest pain", "stroke", "stemi", "altered mental status",
        "severe pain", "high risk", "respiratory distress",
        "acute abdomen", "hemodynamic compromise", "sepsis",
        "suicidal ideation", "overdose", "severe",
        "new onset weakness", "acute",
    ],
    3: [
        "moderate pain", "abdominal pain", "fever", "vomiting",
        "dehydration", "laceration", "fracture", "infection",
        "urinary", "pneumonia", "asthma exacerbation",
        "multiple resources", "moderate",
    ],
    4: [
        "minor", "simple laceration", "sprain", "earache",
        "sore throat", "rash", "urinary symptoms",
        "prescription refill", "mild pain", "mild",
        "chronic complaint", "one resource",
    ],
    5: [
        "medication refill", "suture removal", "recheck",
        "minor complaint", "insect bite", "cold symptoms",
        "no resources", "non-urgent", "minor",
    ],
}


# ------------------------------------------------------------------
# Keyword-based fallback helpers
# ------------------------------------------------------------------


def _keyword_classify_category(text: str) -> tuple[str, float]:
    """Classify *text* into a complaint category using keyword matching.

    Parameters
    ----------
    text : str
        Chief complaint text.

    Returns
    -------
    tuple of (str, float)
        ``(category, confidence)``.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {cat: 0 for cat in COMPLAINT_CATEGORIES}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                scores[cat] += 1
    best_cat = max(scores, key=lambda c: scores[c])
    total = sum(scores.values())
    if total == 0:
        return "other", 0.1
    return best_cat, min(scores[best_cat] / max(total, 1), 1.0)


def _keyword_classify_severity(text: str) -> tuple[int, float]:
    """Classify *text* into an acuity level using keyword matching.

    Parameters
    ----------
    text : str
        Triage note text.

    Returns
    -------
    tuple of (int, float)
        ``(acuity_level, confidence)``.
    """
    text_lower = text.lower()
    scores: dict[int, int] = {lvl: 0 for lvl in range(1, 6)}
    for level, kws in _SEVERITY_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                scores[level] += 1
    best = max(scores, key=lambda lv: scores[lv])
    total = sum(scores.values())
    if total == 0:
        return 3, 0.1
    return best, min(scores[best] / max(total, 1), 1.0)


# ------------------------------------------------------------------
# Chief Complaint Classifier
# ------------------------------------------------------------------


class ChiefComplaintClassifier(ClinicalModel):
    """Classify free-text chief complaints into clinical categories.

    Categories
    ----------
    cardiac, respiratory, neurological, trauma, abdominal,
    psychiatric, pediatric, obstetric, toxicological, other

    The default backend is TF-IDF with logistic regression.  When
    ``use_transformer=True`` and the ``transformers`` package is
    installed, a HuggingFace zero-shot classification pipeline is used.

    Parameters
    ----------
    use_transformer : bool, optional
        Use a HuggingFace transformer backend.  Default is ``False``.
    model_name : str, optional
        HuggingFace model identifier for the transformer backend.
        Default is ``"facebook/bart-large-mnli"``.
    max_iter : int, optional
        Maximum logistic-regression iterations.  Default is ``1000``.
    ngram_range : tuple of (int, int), optional
        N-gram range for the TF-IDF vectorizer.  Default is ``(1, 2)``.

    Examples
    --------
    >>> clf = ChiefComplaintClassifier()
    >>> clf.fit(
    ...     ["chest pain radiating to left arm", "fell from ladder"],
    ...     ["cardiac", "trauma"],
    ... )
    >>> clf.predict(["difficulty breathing"])
    array(['respiratory'], dtype='<U...')
    """

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = "facebook/bart-large-mnli",
        max_iter: int = 1000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__(
            description="Chief complaint classifier for emergency triage",
        )
        if use_transformer and not _HAS_TRANSFORMERS:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required "
                "when use_transformer=True.  Install them with: "
                "pip install transformers torch"
            )
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.max_iter = max_iter
        self.ngram_range = ngram_range

        self._tfidf: Optional[TfidfVectorizer] = None
        self._clf: Optional[LogisticRegression] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._hf_pipeline: Optional[Any] = None

    # ---- fit / predict ------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> ChiefComplaintClassifier:
        """Train the classifier on labelled chief complaints.

        Parameters
        ----------
        X : list of str
            Chief complaint texts.
        y : list of str
            Category labels.  Each label must be one of
            :data:`COMPLAINT_CATEGORIES`.
        **kwargs
            Unused.

        Returns
        -------
        self

        Raises
        ------
        ValidationError
            If inputs are invalid or labels are unrecognized.
        """
        texts: list[str] = list(X)  # type: ignore[arg-type]
        if y is None:
            raise ValidationError(
                "Labels (y) are required for training.",
                parameter="y",
            )
        labels: list[str] = list(y)  # type: ignore[arg-type]

        if len(texts) == 0:
            raise ValidationError("At least one training sample is required.")
        if len(texts) != len(labels):
            raise ValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(labels)} labels."
            )
        invalid = set(labels) - set(COMPLAINT_CATEGORIES)
        if invalid:
            raise ValidationError(
                f"Unrecognized categories: {invalid}. "
                f"Valid categories: {COMPLAINT_CATEGORIES}."
            )

        if self.use_transformer:
            self._load_transformer()
            self._set_fitted()
            return self

        self._tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        X_train = self._tfidf.fit_transform(texts)

        self._label_encoder = LabelEncoder()
        y_train = self._label_encoder.fit_transform(labels)

        self._clf = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            C=1.0,
        )
        self._clf.fit(X_train, y_train)

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict clinical categories for chief complaints.

        Parameters
        ----------
        X : list of str
            Chief complaint texts.

        Returns
        -------
        numpy.ndarray of str
            Predicted category for each input text.
        """
        texts: list[str] = list(X) if not isinstance(X, str) else [X]  # type: ignore[arg-type]

        if self.use_transformer and self._hf_pipeline is not None:
            return self._predict_transformer(texts)

        if self.is_fitted_ and self._clf is not None:
            X_vec = self._tfidf.transform(texts)  # type: ignore[union-attr]
            y_pred = self._clf.predict(X_vec)
            return self._label_encoder.inverse_transform(y_pred)  # type: ignore[union-attr]

        # Keyword fallback when not fitted
        return np.array(
            [_keyword_classify_category(t)[0] for t in texts], dtype=object
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for chief complaints.

        Parameters
        ----------
        X : list of str
            Chief complaint texts.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_categories)
            Probability estimates.  Column order matches
            ``self._label_encoder.classes_`` when fitted, or
            :data:`COMPLAINT_CATEGORIES` otherwise.
        """
        texts: list[str] = list(X) if not isinstance(X, str) else [X]  # type: ignore[arg-type]

        if self.is_fitted_ and self._clf is not None:
            X_vec = self._tfidf.transform(texts)  # type: ignore[union-attr]
            return self._clf.predict_proba(X_vec)

        n = len(texts)
        n_cats = len(COMPLAINT_CATEGORIES)
        proba = np.full((n, n_cats), 1.0 / n_cats)
        for i, text in enumerate(texts):
            cat, conf = _keyword_classify_category(text)
            if cat in COMPLAINT_CATEGORIES:
                idx = COMPLAINT_CATEGORIES.index(cat)
                proba[i, :] = (1.0 - conf) / max(n_cats - 1, 1)
                proba[i, idx] = conf
        return proba

    def predict_single(self, text: str) -> tuple[str, float]:
        """Classify a single chief complaint with confidence.

        Parameters
        ----------
        text : str
            Chief complaint text.

        Returns
        -------
        tuple of (str, float)
            ``(category, confidence)``.
        """
        preds = self.predict([text])
        cat = str(preds[0])
        proba = self.predict_proba([text])
        conf = float(np.max(proba[0]))
        return cat, conf

    # ---- transformer helpers ------------------------------------

    def _load_transformer(self) -> None:
        """Load the HuggingFace zero-shot classification pipeline."""
        if self._hf_pipeline is not None:
            return
        self._hf_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=self.model_name,
        )

    def _predict_transformer(self, texts: list[str]) -> np.ndarray:
        """Classify *texts* using the transformer pipeline.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        numpy.ndarray of str
        """
        results: list[str] = []
        for text in texts:
            out = self._hf_pipeline(text, candidate_labels=COMPLAINT_CATEGORIES)
            results.append(out["labels"][0])
        return np.array(results, dtype=object)


# ------------------------------------------------------------------
# Triage Notes Classifier
# ------------------------------------------------------------------


class TriageNotesClassifier(ClinicalModel):
    """Classify nursing triage notes by acuity / severity level.

    Predicts an :class:`AcuityLevel` (1 through 5) from free-text
    triage notes.  The default backend is TF-IDF with logistic
    regression.  When ``use_transformer=True``, a HuggingFace zero-shot
    classification pipeline is used.

    Parameters
    ----------
    use_transformer : bool, optional
        Use a HuggingFace transformer backend.  Default is ``False``.
    model_name : str, optional
        HuggingFace model identifier.  Default is
        ``"facebook/bart-large-mnli"``.
    max_iter : int, optional
        Maximum logistic-regression iterations.  Default is ``1000``.
    ngram_range : tuple of (int, int), optional
        N-gram range for TF-IDF.  Default is ``(1, 2)``.

    Examples
    --------
    >>> clf = TriageNotesClassifier()
    >>> clf.fit(
    ...     ["unresponsive, no pulse", "minor laceration on finger"],
    ...     [1, 5],
    ... )
    >>> clf.predict(["chest pain, diaphoresis, nausea"])
    array([2])
    """

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = "facebook/bart-large-mnli",
        max_iter: int = 1000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__(
            description="Triage notes severity classifier",
        )
        if use_transformer and not _HAS_TRANSFORMERS:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required "
                "when use_transformer=True.  Install them with: "
                "pip install transformers torch"
            )
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.max_iter = max_iter
        self.ngram_range = ngram_range

        self._tfidf: Optional[TfidfVectorizer] = None
        self._clf: Optional[LogisticRegression] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._hf_pipeline: Optional[Any] = None

    # ---- fit / predict ------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> TriageNotesClassifier:
        """Train the classifier on labelled triage notes.

        Parameters
        ----------
        X : list of str
            Triage note texts.
        y : list of int
            Acuity levels (1 through 5).
        **kwargs
            Unused.

        Returns
        -------
        self

        Raises
        ------
        ValidationError
            If inputs are invalid.
        """
        texts: list[str] = list(X)  # type: ignore[arg-type]
        if y is None:
            raise ValidationError(
                "Labels (y) are required for training.",
                parameter="y",
            )
        labels: list[int] = [int(lbl) for lbl in y]  # type: ignore[arg-type]

        if len(texts) == 0:
            raise ValidationError("At least one training sample is required.")
        if len(texts) != len(labels):
            raise ValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(labels)} labels."
            )
        invalid = {lbl for lbl in labels if lbl not in range(1, 6)}
        if invalid:
            raise ValidationError(
                f"Invalid acuity levels: {invalid}. Must be in [1, 5]."
            )

        if self.use_transformer:
            self._load_transformer()
            self._set_fitted()
            return self

        self._tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        X_train = self._tfidf.fit_transform(texts)

        self._label_encoder = LabelEncoder()
        y_train = self._label_encoder.fit_transform(labels)

        self._clf = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            C=1.0,
        )
        self._clf.fit(X_train, y_train)

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict acuity levels for triage notes.

        Parameters
        ----------
        X : list of str
            Triage note texts.

        Returns
        -------
        numpy.ndarray of int
            Predicted acuity level (1 through 5) for each text.
        """
        texts: list[str] = list(X) if not isinstance(X, str) else [X]  # type: ignore[arg-type]

        if self.use_transformer and self._hf_pipeline is not None:
            return self._predict_transformer(texts)

        if self.is_fitted_ and self._clf is not None:
            X_vec = self._tfidf.transform(texts)  # type: ignore[union-attr]
            y_pred = self._clf.predict(X_vec)
            decoded = self._label_encoder.inverse_transform(y_pred)  # type: ignore[union-attr]
            return np.array(decoded, dtype=int)

        return np.array(
            [_keyword_classify_severity(t)[0] for t in texts], dtype=int
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict acuity-level probabilities for triage notes.

        Parameters
        ----------
        X : list of str
            Triage note texts.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_levels)
            Probability estimates.  Columns correspond to levels 1..5.
        """
        texts: list[str] = list(X) if not isinstance(X, str) else [X]  # type: ignore[arg-type]

        if self.is_fitted_ and self._clf is not None:
            X_vec = self._tfidf.transform(texts)  # type: ignore[union-attr]
            return self._clf.predict_proba(X_vec)

        n = len(texts)
        proba = np.full((n, 5), 0.2)
        for i, text in enumerate(texts):
            level, conf = _keyword_classify_severity(text)
            col = level - 1
            proba[i, :] = (1.0 - conf) / 4.0
            proba[i, col] = conf
        return proba

    def predict_single(self, text: str) -> tuple[AcuityLevel, float]:
        """Classify a single triage note with confidence.

        Parameters
        ----------
        text : str
            Triage note text.

        Returns
        -------
        tuple of (AcuityLevel, float)
            ``(acuity_level, confidence)``.
        """
        preds = self.predict([text])
        level = AcuityLevel(int(preds[0]))
        proba = self.predict_proba([text])
        conf = float(np.max(proba[0]))
        return level, conf

    # ---- transformer helpers ------------------------------------

    def _load_transformer(self) -> None:
        """Load the HuggingFace zero-shot classification pipeline."""
        if self._hf_pipeline is not None:
            return
        self._hf_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=self.model_name,
        )

    def _predict_transformer(self, texts: list[str]) -> np.ndarray:
        """Classify *texts* using the transformer pipeline.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        numpy.ndarray of int
        """
        labels = [
            "level 1 resuscitation",
            "level 2 emergent",
            "level 3 urgent",
            "level 4 less urgent",
            "level 5 non-urgent",
        ]
        results: list[int] = []
        for text in texts:
            out = self._hf_pipeline(text, candidate_labels=labels)
            top = out["labels"][0]
            level = int(top.split()[1])
            results.append(level)
        return np.array(results, dtype=int)
