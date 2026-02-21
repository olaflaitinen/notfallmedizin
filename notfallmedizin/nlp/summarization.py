# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Clinical text summarization for emergency medicine.

Provides extractive summarization of clinical text using the TextRank
algorithm, implemented from scratch with numpy.  Also supports
structured handoff summary generation in SBAR format (Situation,
Background, Assessment, Recommendation).

The TextRank implementation works as follows:

1. Split text into sentences.
2. Compute TF-IDF vectors for each sentence (from scratch, no sklearn).
3. Build a cosine-similarity graph over sentences.
4. Run the PageRank algorithm to rank sentences by importance.
5. Select the top-ranked sentences as the summary.

Classes
-------
SBARSummary
    Dataclass holding the four SBAR components.
ClinicalSummarizer
    Extractive summarization engine and SBAR generator.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from notfallmedizin.core.exceptions import ValidationError


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class SBARSummary:
    """Structured summary in SBAR format.

    Parameters
    ----------
    situation : str
        Brief statement of the current clinical situation.
    background : str
        Relevant medical history and context.
    assessment : str
        Current assessment, findings, and working diagnosis.
    recommendation : str
        Recommended plan of care and pending actions.
    """

    situation: str
    background: str
    assessment: str
    recommendation: str

    def to_text(self) -> str:
        """Render the SBAR summary as a single text block.

        Returns
        -------
        str
            Formatted multi-line string.
        """
        lines = [
            f"S (Situation): {self.situation}",
            f"B (Background): {self.background}",
            f"A (Assessment): {self.assessment}",
            f"R (Recommendation): {self.recommendation}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, str]:
        """Return the summary as a dictionary.

        Returns
        -------
        dict of str to str
        """
        return {
            "situation": self.situation,
            "background": self.background,
            "assessment": self.assessment,
            "recommendation": self.recommendation,
        }


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?])\s*\n+"
    r"|\n{2,}"
)

_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "vs", "etc", "approx", "dept",
    "pt", "hr", "min", "sec", "mg", "ml", "dl", "kg", "lb",
    "iv", "im", "po", "prn", "bid", "tid", "qid",
})


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Handles common clinical abbreviations to avoid spurious splits.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        Non-empty sentence strings.
    """
    rough = _SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for chunk in rough:
        chunk = chunk.strip()
        if not chunk:
            continue
        if buffer:
            chunk = buffer + " " + chunk
            buffer = ""
        words = chunk.split()
        if words and words[-1].rstrip(".").lower() in _ABBREVIATIONS:
            buffer = chunk
            continue
        sentences.append(chunk)
    if buffer:
        sentences.append(buffer)
    return [s for s in sentences if len(s.split()) >= 2]


# ------------------------------------------------------------------
# TF-IDF from scratch
# ------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z][a-z\-]*[a-z]|[a-z]", re.IGNORECASE)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because",
    "but", "and", "or", "if", "while", "about", "up", "its", "it",
    "he", "she", "they", "them", "his", "her", "this", "that",
    "which", "who", "whom", "what",
})


def _tokenize_words(text: str) -> list[str]:
    """Lowercase word tokenization, filtering stopwords.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        Lowercased, filtered tokens.
    """
    return [
        w.lower()
        for w in _WORD_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 1
    ]


def _build_tfidf_matrix(sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute a TF-IDF matrix for *sentences* from scratch.

    Parameters
    ----------
    sentences : list of str
        Input sentences.

    Returns
    -------
    tfidf : numpy.ndarray of shape (n_sentences, vocab_size)
        TF-IDF matrix.
    vocab : list of str
        Vocabulary (column labels).
    """
    n = len(sentences)
    if n == 0:
        return np.empty((0, 0)), []

    tokenized = [_tokenize_words(s) for s in sentences]

    vocab_set: set[str] = set()
    for tokens in tokenized:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)

    if v == 0:
        return np.zeros((n, 1)), [""]

    # Term frequency (normalized by sentence length)
    tf = np.zeros((n, v), dtype=np.float64)
    for i, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        length = len(tokens) if tokens else 1
        for word, count in counts.items():
            j = word_to_idx[word]
            tf[i, j] = count / length

    # Inverse document frequency
    df = np.zeros(v, dtype=np.float64)
    for tokens in tokenized:
        seen: set[str] = set()
        for w in tokens:
            if w not in seen:
                df[word_to_idx[w]] += 1.0
                seen.add(w)

    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    tfidf = tf * idf[np.newaxis, :]

    # L2 normalize each row
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = tfidf / norms

    return tfidf, vocab


# ------------------------------------------------------------------
# Cosine similarity and PageRank
# ------------------------------------------------------------------


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for L2-normalized rows.

    Parameters
    ----------
    matrix : numpy.ndarray of shape (n, d)
        L2-normalized row vectors.

    Returns
    -------
    numpy.ndarray of shape (n, n)
        Symmetric similarity matrix with values in ``[0, 1]``.
    """
    sim = matrix @ matrix.T
    np.clip(sim, 0.0, 1.0, out=sim)
    return sim


def _pagerank(
    similarity: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Run the PageRank algorithm on a similarity matrix.

    Parameters
    ----------
    similarity : numpy.ndarray of shape (n, n)
        Non-negative adjacency / similarity matrix.
    damping : float, optional
        Damping factor.  Default is ``0.85``.
    max_iter : int, optional
        Maximum iterations.  Default is ``100``.
    tol : float, optional
        Convergence tolerance (L1 norm of score change).
        Default is ``1e-6``.

    Returns
    -------
    numpy.ndarray of shape (n,)
        PageRank scores summing to 1.
    """
    n = similarity.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Zero out the diagonal (no self-loops)
    adj = similarity.copy()
    np.fill_diagonal(adj, 0.0)

    # Row-normalize to build a transition matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = adj / row_sums

    scores = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        new_scores = damping * (M.T @ scores) + teleport
        delta = float(np.abs(new_scores - scores).sum())
        scores = new_scores
        if delta < tol:
            break

    total = scores.sum()
    if total > 0:
        scores /= total

    return scores


# ------------------------------------------------------------------
# Clinical Summarizer
# ------------------------------------------------------------------


class ClinicalSummarizer:
    """Extractive summarization engine for clinical text.

    Uses the TextRank algorithm (PageRank on a sentence similarity
    graph) to identify the most important sentences, and can also
    generate structured handoff summaries in SBAR format.

    Parameters
    ----------
    num_sentences : int, optional
        Default number of sentences to include in a summary.
        Default is ``3``.
    damping : float, optional
        PageRank damping factor.  Default is ``0.85``.
    max_iter : int, optional
        Maximum PageRank iterations.  Default is ``100``.
    tol : float, optional
        PageRank convergence tolerance.  Default is ``1e-6``.

    Examples
    --------
    >>> cs = ClinicalSummarizer()
    >>> text = (
    ...     "Patient is a 65-year-old male presenting with chest pain. "
    ...     "Pain started 2 hours ago. Radiates to left arm. "
    ...     "History of hypertension and diabetes. "
    ...     "ECG shows ST elevation in leads II, III, aVF. "
    ...     "Troponin elevated at 0.5. Started on heparin drip."
    ... )
    >>> summary = cs.summarize(text)
    """

    def __init__(
        self,
        num_sentences: int = 3,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        if num_sentences < 1:
            raise ValidationError(
                "num_sentences must be >= 1.",
                parameter="num_sentences",
            )
        self.num_sentences = num_sentences
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    # ---- core TextRank ------------------------------------------

    def _textrank(
        self, sentences: list[str], n: int
    ) -> list[tuple[int, float]]:
        """Rank sentences using TextRank.

        Parameters
        ----------
        sentences : list of str
            Input sentences.
        n : int
            Number of top sentences to return.

        Returns
        -------
        list of (int, float)
            ``(sentence_index, score)`` pairs sorted by score
            (descending).
        """
        if not sentences:
            return []

        tfidf, _ = _build_tfidf_matrix(sentences)
        sim = _cosine_similarity_matrix(tfidf)
        scores = _pagerank(
            sim,
            damping=self.damping,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked[:n]

    # ---- public API ---------------------------------------------

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Produce an extractive summary of *text*.

        Sentences are ranked by TextRank and selected in their
        original order until *max_length* words are reached.

        Parameters
        ----------
        text : str
            Clinical text to summarize.
        max_length : int, optional
            Approximate maximum number of words in the summary.
            Default is ``150``.

        Returns
        -------
        str
            The extractive summary.
        """
        if not text or not text.strip():
            return ""

        sentences = _split_sentences(text)
        if not sentences:
            return text.strip()
        if len(sentences) <= self.num_sentences:
            return " ".join(sentences)

        ranked = self._textrank(sentences, len(sentences))
        top_indices = {idx for idx, _ in ranked[:self.num_sentences]}

        selected: list[str] = []
        word_count = 0
        for i, sent in enumerate(sentences):
            if i in top_indices:
                words = sent.split()
                if word_count + len(words) > max_length and selected:
                    break
                selected.append(sent)
                word_count += len(words)

        if not selected:
            selected.append(sentences[0])

        return " ".join(selected)

    def extract_key_findings(self, text: str) -> list[str]:
        """Extract the most important clinical findings from *text*.

        Sentences are ranked by TextRank, and each top sentence is
        returned as a key finding.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list of str
            Key clinical findings, ordered by importance (most
            important first).
        """
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return [text.strip()] if text.strip() else []

        n = min(self.num_sentences + 2, len(sentences))
        ranked = self._textrank(sentences, n)

        return [sentences[idx] for idx, _ in ranked]

    def generate_handoff_summary(
        self, patient_data: dict[str, Any]
    ) -> str:
        """Generate a structured SBAR handoff summary.

        The summary is assembled from the fields in *patient_data*.
        Missing fields are handled gracefully with placeholder text.

        Parameters
        ----------
        patient_data : dict
            Expected keys (all optional):

            - ``"name"`` (str): patient name or identifier
            - ``"age"`` (int or str): patient age
            - ``"sex"`` or ``"gender"`` (str)
            - ``"chief_complaint"`` (str)
            - ``"history"`` (str or list of str): medical history
            - ``"medications"`` (str or list of str): current medications
            - ``"allergies"`` (str or list of str)
            - ``"vitals"`` (str or dict): current vital signs
            - ``"findings"`` (str or list of str): exam / lab findings
            - ``"assessment"`` (str): working diagnosis / assessment
            - ``"plan"`` or ``"recommendation"`` (str or list of str)
            - ``"disposition"`` (str)
            - ``"pending"`` (str or list of str): pending results

        Returns
        -------
        str
            Formatted SBAR summary text.
        """
        sbar = self._build_sbar(patient_data)
        return sbar.to_text()

    def build_sbar(self, patient_data: dict[str, Any]) -> SBARSummary:
        """Build a structured :class:`SBARSummary` object.

        Parameters
        ----------
        patient_data : dict
            See :meth:`generate_handoff_summary` for expected keys.

        Returns
        -------
        SBARSummary
        """
        return self._build_sbar(patient_data)

    # ---- SBAR assembly ------------------------------------------

    @staticmethod
    def _join(value: Any) -> str:
        """Coerce a value to a summary-friendly string.

        Lists are joined with ``"; "``.  ``None`` becomes ``""``.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return "; ".join(str(v) for v in value if v)
        if isinstance(value, dict):
            return "; ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    def _build_sbar(self, data: dict[str, Any]) -> SBARSummary:
        """Assemble SBAR components from *data*.

        Parameters
        ----------
        data : dict
            Patient data fields.

        Returns
        -------
        SBARSummary
        """
        j = self._join

        # -- Situation --
        parts_s: list[str] = []
        name = data.get("name", "")
        age = data.get("age", "")
        sex = data.get("sex", data.get("gender", ""))
        cc = data.get("chief_complaint", "")
        ident_parts = [p for p in [str(age), str(sex)] if p]
        ident = ", ".join(ident_parts)
        if name:
            parts_s.append(f"{name} ({ident})" if ident else str(name))
        elif ident:
            parts_s.append(f"Patient is {ident}")
        if cc:
            parts_s.append(f"presenting with {cc}")
        situation = " ".join(parts_s) if parts_s else "No situation data provided."

        # -- Background --
        parts_b: list[str] = []
        hx = j(data.get("history", ""))
        if hx:
            parts_b.append(f"History: {hx}")
        meds = j(data.get("medications", ""))
        if meds:
            parts_b.append(f"Medications: {meds}")
        allergy = j(data.get("allergies", ""))
        if allergy:
            parts_b.append(f"Allergies: {allergy}")
        background = ". ".join(parts_b) if parts_b else "No relevant background available."

        # -- Assessment --
        parts_a: list[str] = []
        vitals = j(data.get("vitals", ""))
        if vitals:
            parts_a.append(f"Vitals: {vitals}")
        findings = j(data.get("findings", ""))
        if findings:
            parts_a.append(f"Findings: {findings}")
        dx = data.get("assessment", data.get("diagnosis", ""))
        if dx:
            parts_a.append(f"Assessment: {dx}")
        assessment = ". ".join(parts_a) if parts_a else "Assessment pending."

        # -- Recommendation --
        parts_r: list[str] = []
        plan = j(data.get("plan", data.get("recommendation", "")))
        if plan:
            parts_r.append(f"Plan: {plan}")
        pending = j(data.get("pending", ""))
        if pending:
            parts_r.append(f"Pending: {pending}")
        dispo = data.get("disposition", "")
        if dispo:
            parts_r.append(f"Disposition: {dispo}")
        recommendation = ". ".join(parts_r) if parts_r else "Awaiting further evaluation."

        return SBARSummary(
            situation=situation,
            background=background,
            assessment=assessment,
            recommendation=recommendation,
        )
