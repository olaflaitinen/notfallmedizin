# Changelog

All notable changes to the notfallmedizin project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] (2026-02-21)

### Added

- Initial release of notfallmedizin.
- **core**: Base classes (`BaseEstimator`, `BaseScorer`, `BaseTransformer`, `ClinicalModel`), configuration (`get_config`, `set_config`, `config_context`), validators (vitals, age, GCS, lab values, probability, DataFrame), and exception hierarchy.
- **triage**: ESI v4, MTS, CTAS calculators; ML triage classifier and feature extractor; pediatric triage with PAT and PEWS.
- **vitals**: Vital signs monitor (MAP, shock index), anomaly detection (Isolation Forest, statistical), trend analysis (Mann-Kendall, CUSUM), and clinical alerts with NEWS2.
- **imaging**: Image preprocessing (normalize, resize, CT windowing, denoise, augment), chest X-ray classifier with Grad-CAM, CT hemorrhage detection and ABC/2 volume, FAST/eFAST and ejection fraction calculators.
- **nlp**: Clinical NER (rule-based and trainable), chief complaint and triage note classification, TextRank summarisation, SBAR handoff, ICD-10 and CPT coding.
- **pharmacology**: Weight-based and continuous infusion dosing (18+ EM drugs), drug interaction checker (30+ pairs), QT and serotonin risk, one- and two-compartment PK models, Cockcroft-Gault and CKD-EPI, pharmaceutical alert engine.
- **scoring**: Sepsis (SOFA, qSOFA, SIRS), cardiac (HEART, TIMI, CHA2DS2-VASc), trauma (ISS, RTS, TRISS), neurological (GCS, NIHSS), pediatric (PEWS, APGAR), respiratory (CURB-65, ROX index).
- **timeseries**: Exponential smoothing and ARIMA forecasting, seasonal and wavelet decomposition, clinical time-series feature extraction, streaming processor and Bayesian online changepoint detection.
- **prediction**: Mortality predictor (ensemble) and APACHE II mortality, LACE index and readmission predictor, MEWS and deterioration predictor, LOS predictor and ED throughput (M/M/c), disposition predictor and bed availability estimator.
- **trauma**: Primary and secondary survey, burn assessment (Rule of Nines, Lund-Browder, Parkland), hemorrhage classification (ATLS), massive transfusion (ABC score), TBI and concussion assessment.
- **cardiac**: ECG processor (Pan-Tompkins, HRV), arrhythmia detector and rhythm analyzer, STEMI detector and protocol, Framingham and Wells PE/DVT risk calculators.
- **statistics**: Kaplan-Meier and Cox PH survival, log-rank test, Bayesian A/B and diagnostic test updating, power analysis and multiplicity correction, fixed/random-effects meta-analysis and Egger test, diagnostic metrics and ROC/DeLong, NRI and IDI.
- **benchmarks**: Synthetic ED dataset generator, classification/regression/clinical metrics, model comparison with cross-validation, and benchmark report generation.
- Test suite for core (validators, config, exceptions).
- Sphinx documentation (docs/).
- AUTHORS, CONTRIBUTING.md, and project metadata for GitHub.

### References

- Author: Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
- Repository: https://github.com/olaflaitinen/notfallmedizin
- License: Apache-2.0.

[0.1.0]: https://github.com/olaflaitinen/notfallmedizin/releases/tag/v0.1.0
