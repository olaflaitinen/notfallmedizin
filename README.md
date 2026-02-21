<p align="center">
  <img src="https://img.shields.io/badge/notfallmedizin-Emergency%20Medicine%20%2B%20AI-2980b9?style=for-the-badge" alt="notfallmedizin" />
</p>

<p align="center">
  <a href="https://pypi.org/project/notfallmedizin/"><img src="https://img.shields.io/pypi/v/notfallmedizin?style=flat-square" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/notfallmedizin/"><img src="https://img.shields.io/pypi/pyversions/notfallmedizin?style=flat-square" alt="Python versions" /></a>
  <a href="https://pypi.org/project/notfallmedizin/"><img src="https://img.shields.io/pypi/dm/notfallmedizin?style=flat-square" alt="PyPI downloads" /></a>
  <a href="https://pypi.org/project/notfallmedizin/"><img src="https://img.shields.io/pypi/l/notfallmedizin?style=flat-square" alt="License" /></a>
  <a href="https://github.com/olaflaitinen/notfallmedizin"><img src="https://img.shields.io/github/repo-size/olaflaitinen/notfallmedizin?style=flat-square" alt="Repo size" /></a>
  <a href="https://github.com/olaflaitinen/notfallmedizin/issues"><img src="https://img.shields.io/github/issues/olaflaitinen/notfallmedizin?style=flat-square" alt="Open issues" /></a>
  <a href="https://github.com/olaflaitinen/notfallmedizin"><img src="https://img.shields.io/github/last-commit/olaflaitinen/notfallmedizin?style=flat-square" alt="Last commit" /></a>
  <a href="https://github.com/olaflaitinen/notfallmedizin/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square" alt="Apache 2.0" /></a>
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776ab?style=flat-square&logo=python" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Code%20style-Ruff-000000?style=flat-square" alt="Ruff" />
  <img src="https://img.shields.io/badge/Type%20checker-mypy-blue?style=flat-square" alt="mypy" />
  <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?style=flat-square&logo=pytest" alt="pytest" />
  <img src="https://img.shields.io/badge/Docs-Sphinx%20%7C%20RTD-0A9EDC?style=flat-square" alt="Sphinx" />
  <img src="https://img.shields.io/badge/Status-Beta-yellow?style=flat-square" alt="Beta" />
  <img src="https://img.shields.io/badge/Platform-OS%20Independent-lightgrey?style=flat-square" alt="Platform" />
  <img src="https://img.shields.io/badge/PEP%20561-typed-green?style=flat-square" alt="PEP 561" />
  <img src="https://img.shields.io/badge/Modules-13%20%7C%2057%20submodules-2980b9?style=flat-square" alt="Modules" />
</p>

---

# notfallmedizin

**Emergency Medicine meets Artificial Intelligence.** A comprehensive Python library for clinical decision support, predictive analytics, and real-time patient monitoring in the emergency department.

| | |
|---|---|
| **Author** | Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov |
| **Copyright** | 2026 |
| **Repository** | [github.com/olaflaitinen/notfallmedizin](https://github.com/olaflaitinen/notfallmedizin) |
| **Project start** | February 2026 |

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Clinical formulas (LaTeX)](#clinical-formulas-latex)
- [Module architecture](#module-architecture)
- [Scoring systems reference](#scoring-systems-reference)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Design principles](#design-principles)
- [Author and license](#author-and-license)
- [Disclaimer](#disclaimer)

---

## Overview

`notfallmedizin` provides a unified, modular framework for machine learning, statistical modelling, and signal processing in emergency medicine. It targets researchers, clinicians, and engineers building data-driven ED solutions.

### Capability matrix

| Domain | Components | Submodules |
|--------|------------|------------|
| **Triage** | ESI v4, MTS, CTAS, ML triage, pediatric (PAT, PEWS) | `esi`, `mts`, `ctas`, `ml_triage`, `pediatric` |
| **Scoring** | SOFA, qSOFA, SIRS, HEART, TIMI, CHA2DS2-VASc, ISS, RTS, TRISS, GCS, NIHSS, CURB-65, PEWS, APGAR, ROX | 6 submodules |
| **Vitals** | Monitor, anomaly detection, trends (Mann-Kendall, CUSUM), NEWS2 | `monitor`, `anomaly`, `trends`, `alerts` |
| **Imaging** | Preprocessing, chest X-ray, CT (ABC/2), FAST/eFAST, EF | `preprocessing`, `xray`, `ct`, `ultrasound` |
| **NLP** | NER, classification, TextRank, SBAR, ICD-10/CPT | `ner`, `classification`, `summarization`, `coding` |
| **Pharmacology** | Dosing (18+ drugs), interactions (30+), PK, Cockcroft-Gault, CKD-EPI, alerts | `dosing`, `interactions`, `kinetics`, `alerts` |
| **Timeseries** | Forecasting (ETS, ARIMA, VAR), decomposition, features, BOCD | `forecasting`, `decomposition`, `features`, `realtime` |
| **Prediction** | Mortality, readmission (LACE), deterioration (MEWS), LOS, disposition | 5 submodules |
| **Trauma** | ABCDE, burns (Parkland), hemorrhage (ATLS), TBI, concussion | `assessment`, `burns`, `hemorrhage`, `tbi` |
| **Cardiac** | ECG (Pan-Tompkins, HRV), arrhythmia, STEMI, Framingham, Wells | `ecg`, `arrhythmia`, `stemi`, `risk` |
| **Statistics** | KM, Cox PH, log-rank, Bayesian, power, meta-analysis, ROC/DeLong, NRI, IDI | 5 submodules |
| **Benchmarks** | Synthetic data, metrics, model comparison, reporting | `datasets`, `metrics`, `comparison`, `reporting` |

---

## Installation

```bash
pip install notfallmedizin
```

### Optional extras

| Extra | Purpose |
|-------|---------|
| `[imaging]` | PyTorch, torchvision, Pillow (medical imaging) |
| `[nlp]` | transformers, tokenizers (clinical NLP) |
| `[timeseries]` | statsmodels (ARIMA, advanced TS) |
| `[full]` | All optional dependencies |
| `[dev]` | pytest, mypy, ruff, sphinx (development and docs) |

```bash
pip install notfallmedizin[imaging]
pip install notfallmedizin[nlp]
pip install notfallmedizin[timeseries]
pip install notfallmedizin[full]
pip install notfallmedizin[dev]
```

---

## Quick start

### Clinical scoring

```python
from notfallmedizin.scoring.sepsis import SOFAScore, qSOFAScore

sofa = SOFAScore()
result = sofa.calculate(
    pao2_fio2_ratio=200, platelets=80, bilirubin=3.5,
    map_value=65, gcs=13, creatinine=2.1, mechanical_ventilation=True,
)
print(result.total_score, result.interpretation)

q = qSOFAScore()
r = q.calculate(systolic_bp=95, respiratory_rate=24, altered_mentation=True)
print(r.total_score)
```

### Triage

```python
from notfallmedizin.triage.esi import ESITriageCalculator

esi = ESITriageCalculator()
result = esi.calculate(
    chief_complaint="chest pain",
    vital_signs={"heart_rate": 110, "systolic_bp": 85, "spo2": 91},
    resource_estimate=3,
    mental_status="alert",
    severe_pain_distress=False,
    requires_immediate_intervention=False,
)
print(result.level, result.reasoning)
```

### Vital signs and alerts

```python
from notfallmedizin.vitals.monitor import VitalSignsMonitor
from notfallmedizin.vitals.alerts import ClinicalAlertEngine

monitor = VitalSignsMonitor()
monitor.add_observation(
    timestamp=1700000000.0,
    heart_rate=88, systolic_bp=120, diastolic_bp=75,
    respiratory_rate=16, spo2=98, temperature=36.8,
)
print(monitor.get_current_state().shock_index)

engine = ClinicalAlertEngine()
for a in engine.evaluate({"heart_rate": 135, "spo2": 88}):
    print(a.severity, a.message)
```

### Survival analysis

```python
import numpy as np
from notfallmedizin.statistics.survival import KaplanMeierEstimator

km = KaplanMeierEstimator()
km.fit(durations=np.array([1,2,3,4,5,6,7,8]),
       event_observed=np.array([1,0,1,0,1,1,0,1]))
print(km.median_survival_time())
```

---

## Clinical formulas (LaTeX)

Key equations implemented in the library (display math supported on GitHub).

### Hemodynamics

**Mean arterial pressure (MAP):**

$$
\text{MAP} = \text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})
$$

**Shock index (SI):**

$$
\text{SI} = \frac{\text{HR}}{\text{SBP}}
$$

Normal range approximately 0.5–0.7; >1.0 suggests hemodynamic compromise.

**Modified shock index (MSI):**

$$
\text{MSI} = \frac{\text{HR}}{\text{MAP}}
$$

### Resuscitation (burns)

**Parkland formula (24 h crystalloid):**

$$
V = 4 \times \text{weight (kg)} \times \text{TBSA (\%)}
$$

Half in the first 8 hours, half over the next 16 hours.

### Hemorrhage volume (CT)

**ABC/2 (ellipsoid approximation):**

$$
V = \frac{A \times B \times C}{2}
$$

with \(A\), \(B\), \(C\) the three largest perpendicular diameters in cm.

### Renal function

**Cockcroft-Gault (creatinine clearance, mL/min):**

$$
\text{CrCl} = \frac{(140 - \text{age}) \times \text{weight (kg)}}{\text{SCr} \times 72} \times (0.85 \text{ if female})
$$

**CKD-EPI 2021 (GFR, mL/min/1.73 m²):**

$$
\text{GFR} = 142 \times \min(\text{SCr}/\kappa, 1)^\alpha \times \max(\text{SCr}/\kappa, 1)^{-1.200} \times 0.9938^{\text{age}} \times 1.012
$$

with \(\kappa\), \(\alpha\) sex-dependent.

### Pharmacokinetics (one-compartment)

**Concentration after IV bolus:**

$$
C(t) = \frac{D}{V_d} \, e^{-k_e t}
$$

**Half-life:**

$$
t_{1/2} = \frac{\ln 2}{k_e} = \frac{0.693}{k_e}
$$

### Survival analysis

**Kaplan-Meier estimator:**

$$
\hat{S}(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

**Cox proportional hazards:**

$$
h(t \mid X) = h_0(t) \exp(X^\top \beta)
$$

**APACHE II predicted mortality (logistic):**

$$
\ln \frac{R}{1-R} = -3.517 + 0.146 \times \text{APACHE II} + \text{diagnostic weight}
$$

### Diagnostic accuracy

**Sensitivity and specificity:**

$$
\text{Sens} = \frac{TP}{TP+FN}, \qquad \text{Spec} = \frac{TN}{TN+FP}
$$

**Likelihood ratio positive:**

$$
LR^+ = \frac{\text{Sensitivity}}{1 - \text{Specificity}}
$$

**DeLong test (AUC comparison):** compares two correlated ROC curves via covariance of placement values.

---

## Module architecture

| Module | Submodules | Description |
|--------|------------|-------------|
| `core` | `base`, `config`, `validators`, `exceptions` | Base classes, config, validation, exceptions |
| `triage` | `esi`, `mts`, `ctas`, `ml_triage`, `pediatric` | Triage scoring and ML |
| `vitals` | `monitor`, `anomaly`, `trends`, `alerts` | Vital signs and NEWS2 |
| `imaging` | `preprocessing`, `xray`, `ct`, `ultrasound` | Medical image analysis |
| `nlp` | `ner`, `classification`, `summarization`, `coding` | Clinical text and coding |
| `pharmacology` | `dosing`, `interactions`, `kinetics`, `alerts` | Dosing and safety |
| `scoring` | `sepsis`, `cardiac`, `trauma`, `neurological`, `pediatric`, `respiratory` | Clinical scores |
| `timeseries` | `forecasting`, `decomposition`, `features`, `realtime` | Time series and BOCD |
| `prediction` | `mortality`, `readmission`, `deterioration`, `los`, `disposition` | Outcome prediction |
| `trauma` | `assessment`, `burns`, `hemorrhage`, `tbi` | Trauma and burns |
| `cardiac` | `ecg`, `arrhythmia`, `stemi`, `risk` | ECG and risk scores |
| `statistics` | `survival`, `bayesian`, `power`, `meta_analysis`, `diagnostic` | Biostatistics |
| `benchmarks` | `datasets`, `metrics`, `comparison`, `reporting` | Evaluation framework |

**Total: 13 main modules, 57 submodules.**

---

## Scoring systems reference

| Score | Range | Use | Key reference |
|-------|--------|-----|----------------|
| **SOFA** | 0–24 | Organ failure / sepsis | Vincent et al., ICM 1996 |
| **qSOFA** | 0–3 | Sepsis screening (≥2 positive) | Sepsis-3 |
| **SIRS** | 0–4 | Systemic inflammation (≥2) | ACCP/SCCM |
| **HEART** | 0–10 | Chest pain risk | Six et al., Neth Heart J 2008 |
| **TIMI** | 0–7 | UA/NSTEMI | Antman et al., JAMA 2000 |
| **CHA2DS2-VASc** | 0–9 | Stroke risk (AF) | ESC guidelines |
| **ISS** | 0–75 | Trauma severity | Baker et al. |
| **RTS** | 0–7.84 | Trauma (coded) | Champion et al. |
| **TRISS** | \(P_s\) | Trauma survival probability | Boyd et al. |
| **GCS** | 3–15 | Consciousness | Teasdale & Jennett, Lancet 1974 |
| **NIHSS** | 0–42 | Stroke severity | Brott et al. |
| **CURB-65** | 0–5 | Pneumonia severity | BTS |
| **ROX** | \(\geq 4.88\) | HFNC success | Roca et al. |
| **PEWS** | 0–9 | Pediatric deterioration | Monaghan |
| **APGAR** | 0–10 | Neonatal status | Apgar, 1953 |

### SOFA mortality bands (approximate)

| SOFA total | 30-day mortality (approx.) |
|------------|-----------------------------|
| 0–6 | &lt; 10% |
| 7–9 | 15–20% |
| 10–12 | 40–50% |
| &gt; 12 | &gt; 80% |

### HEART risk strata

| HEART total | Risk | MACE probability |
|-------------|------|-------------------|
| 0–3 | Low | ~1–2% |
| 4–6 | Moderate | ~12–17% |
| 7–10 | High | ~50%+ |

---

## Dependencies

### Core (required)

| Package | Version | Role |
|---------|---------|------|
| numpy | ≥ 1.24 | Arrays and numerics |
| scipy | ≥ 1.10 | Stats, signal, optimize |
| pandas | ≥ 2.0 | DataFrames and I/O |
| scikit-learn | ≥ 1.3 | ML and preprocessing |

### Optional

| Extra | Packages |
|-------|----------|
| `imaging` | torch, torchvision, Pillow |
| `nlp` | transformers, tokenizers |
| `timeseries` | statsmodels |
| `full` | All of the above + matplotlib, seaborn |
| `dev` | pytest, pytest-cov, pytest-benchmark, mypy, ruff, sphinx, sphinx-rtd-theme |

### Requirements summary

| Requirement | Minimum |
|-------------|---------|
| Python | 3.9 |
| NumPy | 1.24 |
| SciPy | 1.10 |
| pandas | 2.0 |
| scikit-learn | 1.3 |

---

## Documentation

| Resource | Description |
|----------|-------------|
| **Sphinx docs** | `pip install -e ".[dev]"` then `sphinx-build -b html docs docs/_build/html`. Open `docs/_build/html/index.html`. |
| **Sections** | Installation, Quick start, Module overview, API reference, References, Glossary, License, Disclaimer |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [AUTHORS](AUTHORS) | Author and copyright |

---

## Design principles

| Principle | Description |
|-----------|-------------|
| **Clinical accuracy** | Scoring and algorithms follow published guidelines with references. |
| **Modular design** | Submodules are independently importable with minimal coupling. |
| **Consistent API** | Estimators: `fit` / `predict` / `transform`. Scorers: `calculate` / `interpret`. |
| **Type safety** | Full type annotations; mypy strict compatible. |
| **Minimal core deps** | Core uses only NumPy, SciPy, pandas, scikit-learn. |
| **Reproducibility** | Global random state via `notfallmedizin.core.config`. |

---

## Author and license

| | |
|---|---|
| **Author** | Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov |
| **License** | Apache License 2.0. See [LICENSE](LICENSE). |
| **Repository** | [github.com/olaflaitinen/notfallmedizin](https://github.com/olaflaitinen/notfallmedizin) |

---

## Disclaimer

This library is for **research and educational use only**. It is not a certified medical device and must not be used as the sole basis for clinical decisions. Always follow local protocols and consult qualified healthcare professionals for patient care.
