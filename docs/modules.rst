Module Overview
===============

notfallmedizin has 13 main modules and 57 submodules.

Core
----
Base classes, configuration, validators, exceptions. Submodules: base, config, validators, exceptions.

Triage
------
ED triage: ESI, MTS, CTAS, ML triage, pediatric (PAT, PEWS). Submodules: esi, mts, ctas, ml_triage, pediatric.

Scoring
-------
SOFA, qSOFA, SIRS; HEART, TIMI, CHA2DS2-VASc; ISS, RTS, TRISS; GCS, NIHSS; PEWS, APGAR; CURB-65, ROX. Submodules: sepsis, cardiac, trauma, neurological, pediatric, respiratory.

Vitals
------
Monitor, anomaly detection, trends, NEWS2 alerts. Submodules: monitor, anomaly, trends, alerts.

Imaging
-------
Preprocessing, chest X-ray, CT hemorrhage (ABC/2), FAST/eFAST, ejection fraction. Submodules: preprocessing, xray, ct, ultrasound.

NLP
---
NER, classification, TextRank summarisation, SBAR, ICD-10/CPT coding. Submodules: ner, classification, summarization, coding.

Pharmacology
------------
Dosing, interactions, kinetics (Cockcroft-Gault, CKD-EPI), alerts. Submodules: dosing, interactions, kinetics, alerts.

Timeseries
---------
Forecasting, decomposition, features, real-time (BOCD). Submodules: forecasting, decomposition, features, realtime.

Prediction
----------
Mortality, readmission (LACE), deterioration (MEWS), LOS, disposition. Submodules: mortality, readmission, deterioration, los, disposition.

Trauma
------
Primary/secondary survey, burns (Parkland), hemorrhage (ATLS), TBI. Submodules: assessment, burns, hemorrhage, tbi.

Cardiac
-------
ECG (Pan-Tompkins, HRV), arrhythmia, STEMI, Framingham, Wells PE/DVT. Submodules: ecg, arrhythmia, stemi, risk.

Statistics
----------
Kaplan-Meier, Cox PH, log-rank; Bayesian A/B and diagnostic; power; meta-analysis; ROC/DeLong, NRI, IDI. Submodules: survival, bayesian, power, meta_analysis, diagnostic.

Benchmarks
----------
Synthetic data, metrics, model comparison, reporting. Submodules: datasets, metrics, comparison, reporting.
