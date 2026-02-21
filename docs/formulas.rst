.. _formulas:

Clinical Formulas
=================

This page lists key clinical and statistical formulas implemented or referenced
in notfallmedizin. All equations use standard notation; units are given where
relevant.

Hemodynamics
------------

Mean arterial pressure (MAP, mmHg):

.. math::
   \text{MAP} = \text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})

Shock index (SI, dimensionless). Normal range approximately 0.5--0.7; >1.0 suggests hemodynamic compromise:

.. math::
   \text{SI} = \frac{\text{HR}}{\text{SBP}}

Modified shock index (MSI):

.. math::
   \text{MSI} = \frac{\text{HR}}{\text{MAP}}

Resuscitation (burns)
--------------------

Parkland formula for 24-hour crystalloid volume (mL). Half in first 8 hours, half over next 16 hours; weight in kg, TBSA as percentage:

.. math::
   V = 4 \times \text{weight (kg)} \times \text{TBSA (\%)}

Hemorrhage volume (CT)
----------------------

ABC/2 ellipsoid approximation for intracerebral hemorrhage volume (mL); *A*, *B*, *C* are the three largest perpendicular diameters in cm:

.. math::
   V = \frac{A \times B \times C}{2}

Renal function
--------------

Cockcroft-Gault creatinine clearance (mL/min). Use 0.85 multiplier for females:

.. math::
   \text{CrCl} = \frac{(140 - \text{age}) \times \text{weight (kg)}}{\text{SCr} \times 72} \times (0.85 \text{ if female})

CKD-EPI 2021 (GFR, mL/min/1.73 m²) uses sex-specific :math:`\kappa` and :math:`\alpha`:

.. math::
   \text{GFR} = 142 \times \min(\text{SCr}/\kappa, 1)^\alpha \times \max(\text{SCr}/\kappa, 1)^{-1.200} \times 0.9938^{\text{age}} \times 1.012

Pharmacokinetics (one-compartment)
----------------------------------

Concentration after IV bolus (dose *D*, volume of distribution :math:`V_d`, elimination rate :math:`k_e`):

.. math::
   C(t) = \frac{D}{V_d} \, e^{-k_e t}

Elimination half-life:

.. math::
   t_{1/2} = \frac{\ln 2}{k_e} = \frac{0.693}{k_e}

Survival analysis
-----------------

Kaplan-Meier survival estimator (:math:`d_i` events, :math:`n_i` at risk at time :math:`t_i`):

.. math::
   \hat{S}(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)

Cox proportional hazards model (baseline hazard :math:`h_0(t)`, covariates :math:`X`, coefficients :math:`\beta`):

.. math::
   h(t \mid X) = h_0(t) \exp(X^\top \beta)

Severity and mortality
---------------------

APACHE II predicted mortality (logistic; *R* = probability of death):

.. math::
   \ln \frac{R}{1-R} = -3.517 + 0.146 \times \text{APACHE II} + \text{diagnostic weight}

Diagnostic accuracy
-------------------

Sensitivity and specificity:

.. math::
   \text{Sensitivity} = \frac{TP}{TP+FN}, \qquad \text{Specificity} = \frac{TN}{TN+FP}

Positive likelihood ratio:

.. math::
   LR^+ = \frac{\text{Sensitivity}}{1 - \text{Specificity}}

References for these formulas are given in :doc:`references`.
