Quick Start
===========

Minimal examples for the main use cases.

Clinical scoring
----------------

qSOFA and SOFA (sepsis)::

   from notfallmedizin.scoring.sepsis import qSOFAScore, SOFAScore

   q = qSOFAScore()
   r = q.calculate(systolic_bp=95, respiratory_rate=24, altered_mentation=True)
   print(r.total_score)

   sofa = SOFAScore()
   r = sofa.calculate(pao2_fio2_ratio=200, platelets=80, bilirubin=3.5,
                      map_value=65, gcs=13, creatinine=2.1, mechanical_ventilation=True)
   print(r.total_score, r.interpretation)

HEART score::

   from notfallmedizin.scoring.cardiac import HEARTScore
   heart = HEARTScore()
   r = heart.calculate(history=1, ecg=1, age=0, risk_factors=1, troponin=0)
   print(r.total_score, r.risk_category)

GCS and CURB-65::

   from notfallmedizin.scoring.neurological import GCSCalculator
   from notfallmedizin.scoring.respiratory import CURB65Score

   gcs = GCSCalculator()
   r = gcs.calculate(eye=4, verbal=5, motor=6)

   curb = CURB65Score()
   r = curb.calculate(confusion=False, bun=15, respiratory_rate=18,
                      systolic_bp=120, diastolic_bp=80, age=40)

Triage
------

::

   from notfallmedizin.triage.esi import ESITriageCalculator
   esi = ESITriageCalculator()
   result = esi.calculate(chief_complaint="chest pain",
                          vital_signs={"heart_rate": 110, "systolic_bp": 85, "spo2": 91},
                          resource_estimate=3, mental_status="alert",
                          severe_pain_distress=False, requires_immediate_intervention=False)
   print(result.level, result.reasoning)

Vital signs and alerts
----------------------

::

   from notfallmedizin.vitals.monitor import VitalSignsMonitor
   from notfallmedizin.vitals.alerts import ClinicalAlertEngine

   monitor = VitalSignsMonitor()
   monitor.add_observation(timestamp=1700000000.0, heart_rate=88, systolic_bp=120,
                           diastolic_bp=75, respiratory_rate=16, spo2=98, temperature=36.8)
   state = monitor.get_current_state()
   print(state.shock_index)

   engine = ClinicalAlertEngine()
   alerts = engine.evaluate({"heart_rate": 135, "spo2": 88})

Survival analysis
-----------------

::

   import numpy as np
   from notfallmedizin.statistics.survival import KaplanMeierEstimator
   km = KaplanMeierEstimator()
   km.fit(durations=np.array([1,2,3,4,5,6,7,8]),
          event_observed=np.array([1,0,1,0,1,1,0,1]))
   print(km.median_survival_time())

Benchmarking
------------

::

   from notfallmedizin.benchmarks.datasets import SyntheticEDDatasetGenerator
   gen = SyntheticEDDatasetGenerator(n_samples=500)
   X, y = gen.generate_mortality_dataset()
   split = gen.split(X, y)
