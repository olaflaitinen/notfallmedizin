.. notfallmedizin documentation master file.
   You can adapt this file completely to your layout.

notfallmedizin
==============

Emergency Medicine meets Artificial Intelligence: a Python library for
clinical decision support, predictive analytics, and real-time patient
monitoring in the emergency department.

**Author:** Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov  
**Copyright:** 2026  
**Repository:** https://github.com/olaflaitinen/notfallmedizin

Overview
--------

notfallmedizin provides 13 main modules and 57 submodules covering:

- **Triage:** ESI, MTS, CTAS, ML-based triage, pediatric triage
- **Scoring:** Sepsis (SOFA, qSOFA, SIRS), cardiac (HEART, TIMI, CHA2DS2-VASc), trauma (ISS, RTS, TRISS), neurological (GCS, NIHSS), pediatric (PEWS, APGAR), respiratory (CURB-65, ROX)
- **Vitals:** Monitoring, anomaly detection, trend analysis, NEWS2 alerts
- **Imaging:** Preprocessing, chest X-ray, CT hemorrhage, FAST/eFAST
- **NLP:** NER, classification, summarisation, ICD/CPT coding
- **Pharmacology:** Dosing, interactions, kinetics, alerts
- **Timeseries:** Forecasting, decomposition, features, real-time streaming
- **Prediction:** Mortality, readmission, deterioration, LOS, disposition
- **Trauma:** Primary/secondary survey, burns, hemorrhage, TBI
- **Cardiac:** ECG, arrhythmia, STEMI, risk scores
- **Statistics:** Survival, Bayesian, power, meta-analysis, diagnostic tests
- **Benchmarks:** Synthetic data, metrics, model comparison, reporting

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   modules
   formulas
   api
   references
   glossary
   license
   disclaimer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
