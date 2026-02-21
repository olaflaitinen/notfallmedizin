# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""NLP module for the notfallmedizin library.

This package provides natural language processing tools tailored to
emergency-medicine clinical text, including named entity recognition,
chief-complaint and triage-note classification, extractive text
summarization, and automated medical coding.

Submodules
----------
ner
    Named Entity Recognition for clinical text.  Rule-based and
    trainable (sklearn / transformer) approaches.
classification
    Chief-complaint classification into ESI-compatible categories and
    triage-note severity classification.
summarization
    Extractive summarization using TextRank (implemented from scratch
    with numpy) and structured SBAR handoff summaries.
coding
    ICD-10-CM and CPT code mapping from free-text clinical
    descriptions via TF-IDF similarity.

References:
    Mihalcea & Tarau. TextRank. EMNLP 2004.
"""

from notfallmedizin.nlp.classification import (
    AcuityLevel,
    ChiefComplaintClassifier,
    TriageNotesClassifier,
)
from notfallmedizin.nlp.coding import (
    CPTCode,
    CPTCoder,
    ICDCode,
    ICDCoder,
)
from notfallmedizin.nlp.ner import (
    ENTITY_TYPES,
    ClinicalEntity,
    ClinicalNERModel,
    RuleBasedNER,
)
from notfallmedizin.nlp.summarization import (
    ClinicalSummarizer,
    SBARSummary,
)

__all__ = [
    # ner
    "ClinicalEntity",
    "ClinicalNERModel",
    "RuleBasedNER",
    "ENTITY_TYPES",
    # classification
    "ChiefComplaintClassifier",
    "TriageNotesClassifier",
    "AcuityLevel",
    # summarization
    "ClinicalSummarizer",
    "SBARSummary",
    # coding
    "ICDCode",
    "ICDCoder",
    "CPTCode",
    "CPTCoder",
]
