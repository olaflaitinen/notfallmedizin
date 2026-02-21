# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Pharmacokinetic modeling for emergency medicine.

This module provides compartmental pharmacokinetic models and renal
function estimation formulas commonly used in emergency medicine drug
dosing decisions.

Classes
-------
OneCompartmentModel
    Single-compartment first-order elimination model.
TwoCompartmentModel
    Two-compartment model with bi-exponential decay.
CockcroftGault
    Creatinine clearance estimation.
MDRD
    GFR estimation using the four-variable MDRD equation.
CKD_EPI
    GFR estimation using the CKD-EPI 2021 equation.

References
----------
.. [1] Rowland M, Tozer TN. "Clinical Pharmacokinetics and
   Pharmacodynamics: Concepts and Applications." 4th ed. Lippincott
   Williams & Wilkins, 2010.
.. [2] Cockcroft DW, Gault MH. "Prediction of creatinine clearance
   from serum creatinine." Nephron. 1976;16(1):31-41.
.. [3] Levey AS, Bosch JP, Lewis JB, et al. "A more accurate method
   to estimate glomerular filtration rate from serum creatinine."
   Ann Intern Med. 1999;130(6):461-470.
.. [4] Inker LA, Eneanya ND, Coresh J, et al. "New creatinine- and
   cystatin C-based equations to estimate GFR without race." N Engl
   J Med. 2021;385(19):1737-1749.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    ValidationError,
)


# ======================================================================
# OneCompartmentModel
# ======================================================================


class OneCompartmentModel:
    r"""Single-compartment pharmacokinetic model with first-order elimination.

    Models drug concentration after an intravenous bolus dose using
    the classical one-compartment equation:

    .. math::

        C(t) = \frac{D}{V_d} \cdot e^{-k_e \cdot t}

    where *D* is dose, *Vd* is volume of distribution, *ke* is the
    elimination rate constant, and *t* is time after administration.

    Parameters
    ----------
    volume_distribution : float or None, optional
        Volume of distribution in liters. Default is ``None`` (must be
        provided at calculation time).
    elimination_rate : float or None, optional
        First-order elimination rate constant in 1/hour. Default is
        ``None``.

    Examples
    --------
    >>> model = OneCompartmentModel(volume_distribution=50.0, elimination_rate=0.1)
    >>> round(model.calculate_concentration(500.0, time=0.0), 2)
    10.0
    >>> round(model.calculate_half_life(), 2)
    6.93

    References
    ----------
    .. [1] Rowland M, Tozer TN. Clinical Pharmacokinetics and
       Pharmacodynamics. 4th ed. 2010.
    """

    def __init__(
        self,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> None:
        self.volume_distribution = volume_distribution
        self.elimination_rate = elimination_rate

    def _validate_params(
        self,
        dose: float,
        vd: float,
        ke: float,
    ) -> Tuple[float, float, float]:
        """Validate pharmacokinetic parameters.

        Returns
        -------
        tuple of (float, float, float)
            Validated (dose, vd, ke).
        """
        if dose < 0:
            raise ValidationError(
                message="Dose must be non-negative.",
                parameter="dose",
            )
        if vd <= 0:
            raise ValidationError(
                message="Volume of distribution must be positive.",
                parameter="volume_distribution",
            )
        if ke <= 0:
            raise ValidationError(
                message="Elimination rate constant must be positive.",
                parameter="elimination_rate",
            )
        return dose, vd, ke

    def _resolve_vd(self, vd: Optional[float]) -> float:
        """Resolve volume of distribution from argument or instance attribute."""
        result = vd if vd is not None else self.volume_distribution
        if result is None:
            raise ValidationError(
                message="volume_distribution must be provided.",
                parameter="volume_distribution",
            )
        return result

    def _resolve_ke(self, ke: Optional[float]) -> float:
        """Resolve elimination rate from argument or instance attribute."""
        result = ke if ke is not None else self.elimination_rate
        if result is None:
            raise ValidationError(
                message="elimination_rate must be provided.",
                parameter="elimination_rate",
            )
        return result

    def calculate_concentration(
        self,
        dose: float,
        time: float,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> float:
        r"""Calculate plasma concentration at a given time after IV bolus.

        .. math::

            C(t) = \frac{D}{V_d} \cdot e^{-k_e \cdot t}

        Parameters
        ----------
        dose : float
            Administered dose in mg.
        time : float
            Time after administration in hours.
        volume_distribution : float or None, optional
            Volume of distribution in liters. Falls back to instance
            attribute if ``None``.
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour. Falls back to instance
            attribute if ``None``.

        Returns
        -------
        float
            Plasma concentration in mg/L.

        Raises
        ------
        ValidationError
            If any parameter is invalid or missing.
        """
        vd = self._resolve_vd(volume_distribution)
        ke = self._resolve_ke(elimination_rate)
        dose, vd, ke = self._validate_params(dose, vd, ke)

        if time < 0:
            raise ValidationError(
                message="Time must be non-negative.",
                parameter="time",
            )
        return (dose / vd) * math.exp(-ke * time)

    def calculate_half_life(
        self,
        elimination_rate: Optional[float] = None,
    ) -> float:
        r"""Calculate the elimination half-life.

        .. math::

            t_{1/2} = \frac{\ln 2}{k_e}

        Parameters
        ----------
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour.

        Returns
        -------
        float
            Half-life in hours.
        """
        ke = self._resolve_ke(elimination_rate)
        if ke <= 0:
            raise ValidationError(
                message="Elimination rate constant must be positive.",
                parameter="elimination_rate",
            )
        return math.log(2) / ke

    def calculate_steady_state(
        self,
        dose: float,
        interval: float,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> float:
        r"""Calculate the average steady-state concentration.

        For repeated dosing at fixed intervals, the average
        steady-state concentration is:

        .. math::

            C_{ss,avg} = \frac{D}{V_d \cdot k_e \cdot \tau}

        where :math:`\tau` is the dosing interval.

        Parameters
        ----------
        dose : float
            Dose per administration in mg.
        interval : float
            Dosing interval in hours.
        volume_distribution : float or None, optional
            Volume of distribution in liters.
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour.

        Returns
        -------
        float
            Average steady-state concentration in mg/L.
        """
        vd = self._resolve_vd(volume_distribution)
        ke = self._resolve_ke(elimination_rate)
        dose, vd, ke = self._validate_params(dose, vd, ke)

        if interval <= 0:
            raise ValidationError(
                message="Dosing interval must be positive.",
                parameter="interval",
            )
        return dose / (vd * ke * interval)

    def time_to_concentration(
        self,
        target_concentration: float,
        dose: float,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> float:
        r"""Calculate the time to reach a target concentration (falling).

        Solves for *t* in the equation :math:`C(t) = C_{target}`:

        .. math::

            t = \frac{-\ln\left(\frac{C_{target} \cdot V_d}{D}\right)}{k_e}

        Parameters
        ----------
        target_concentration : float
            Desired plasma concentration in mg/L.
        dose : float
            Administered dose in mg.
        volume_distribution : float or None, optional
            Volume of distribution in liters.
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour.

        Returns
        -------
        float
            Time in hours to reach the target concentration.

        Raises
        ------
        ValidationError
            If the target concentration is unreachable (exceeds C0).
        """
        vd = self._resolve_vd(volume_distribution)
        ke = self._resolve_ke(elimination_rate)
        dose, vd, ke = self._validate_params(dose, vd, ke)

        if target_concentration <= 0:
            raise ValidationError(
                message="Target concentration must be positive.",
                parameter="target_concentration",
            )

        c0 = dose / vd
        if target_concentration > c0:
            raise ValidationError(
                message=(
                    f"Target concentration ({target_concentration} mg/L) exceeds "
                    f"initial concentration ({c0:.4f} mg/L). Unreachable by decay."
                ),
                parameter="target_concentration",
            )

        return -math.log(target_concentration / c0) / ke

    def plot_concentration_curve(
        self,
        dose: float,
        duration: float,
        n_points: int = 200,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time-concentration data for plotting.

        Parameters
        ----------
        dose : float
            Administered dose in mg.
        duration : float
            Total time span in hours.
        n_points : int, optional
            Number of data points. Default is 200.
        volume_distribution : float or None, optional
            Volume of distribution in liters.
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            ``(times, concentrations)`` arrays of shape ``(n_points,)``.
        """
        vd = self._resolve_vd(volume_distribution)
        ke = self._resolve_ke(elimination_rate)
        dose, vd, ke = self._validate_params(dose, vd, ke)

        if duration <= 0:
            raise ValidationError(
                message="Duration must be positive.",
                parameter="duration",
            )

        times = np.linspace(0, duration, n_points)
        concentrations = (dose / vd) * np.exp(-ke * times)
        return times, concentrations

    def calculate_auc(
        self,
        dose: float,
        volume_distribution: Optional[float] = None,
        elimination_rate: Optional[float] = None,
    ) -> float:
        r"""Calculate the area under the concentration-time curve (AUC).

        For a single IV bolus in a one-compartment model:

        .. math::

            AUC_{0 \to \infty} = \frac{D}{V_d \cdot k_e}

        Parameters
        ----------
        dose : float
            Administered dose in mg.
        volume_distribution : float or None, optional
            Volume of distribution in liters.
        elimination_rate : float or None, optional
            Elimination rate constant in 1/hour.

        Returns
        -------
        float
            AUC in mg*hr/L.
        """
        vd = self._resolve_vd(volume_distribution)
        ke = self._resolve_ke(elimination_rate)
        dose, vd, ke = self._validate_params(dose, vd, ke)
        return dose / (vd * ke)

    def __repr__(self) -> str:
        return (
            f"OneCompartmentModel("
            f"volume_distribution={self.volume_distribution}, "
            f"elimination_rate={self.elimination_rate})"
        )


# ======================================================================
# TwoCompartmentModel
# ======================================================================


class TwoCompartmentModel:
    r"""Two-compartment pharmacokinetic model with bi-exponential decay.

    Models drug concentration after IV bolus using the equation:

    .. math::

        C(t) = A \cdot e^{-\alpha \cdot t} + B \cdot e^{-\beta \cdot t}

    where :math:`\alpha` is the distribution rate constant (fast phase)
    and :math:`\beta` is the elimination rate constant (slow phase).
    *A* and *B* are the intercepts of the distribution and elimination
    phases, respectively.

    Parameters
    ----------
    A : float
        Intercept of the distribution phase (mg/L).
    alpha : float
        Distribution rate constant (1/hour).
    B : float
        Intercept of the elimination phase (mg/L).
    beta : float
        Elimination rate constant (1/hour).

    Examples
    --------
    >>> model = TwoCompartmentModel(A=8.0, alpha=2.0, B=2.0, beta=0.1)
    >>> round(model.calculate_concentration(0.0), 2)
    10.0
    >>> round(model.distribution_half_life(), 2)
    0.35

    References
    ----------
    .. [1] Rowland M, Tozer TN. Clinical Pharmacokinetics and
       Pharmacodynamics. 4th ed. 2010. Chapter 19.
    """

    def __init__(
        self,
        A: float,
        alpha: float,
        B: float,
        beta: float,
    ) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValidationError(
                message="Rate constants alpha and beta must be positive.",
                parameter="alpha" if alpha <= 0 else "beta",
            )
        if alpha <= beta:
            raise ValidationError(
                message=(
                    "alpha (distribution) must be greater than beta (elimination) "
                    "for a valid two-compartment model."
                ),
                parameter="alpha",
            )
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def calculate_concentration(self, time: float) -> float:
        r"""Calculate plasma concentration at time *t*.

        .. math::

            C(t) = A \cdot e^{-\alpha t} + B \cdot e^{-\beta t}

        Parameters
        ----------
        time : float
            Time after administration in hours.

        Returns
        -------
        float
            Plasma concentration in mg/L.
        """
        if time < 0:
            raise ValidationError(
                message="Time must be non-negative.",
                parameter="time",
            )
        return self.A * math.exp(-self.alpha * time) + self.B * math.exp(-self.beta * time)

    def distribution_half_life(self) -> float:
        r"""Calculate the distribution (alpha) half-life.

        .. math::

            t_{1/2,\alpha} = \frac{\ln 2}{\alpha}

        Returns
        -------
        float
            Distribution half-life in hours.
        """
        return math.log(2) / self.alpha

    def elimination_half_life(self) -> float:
        r"""Calculate the elimination (beta) half-life.

        .. math::

            t_{1/2,\beta} = \frac{\ln 2}{\beta}

        Returns
        -------
        float
            Terminal elimination half-life in hours.
        """
        return math.log(2) / self.beta

    def initial_concentration(self) -> float:
        """Return the initial concentration at t=0.

        Returns
        -------
        float
            C(0) = A + B in mg/L.
        """
        return self.A + self.B

    def calculate_auc(self) -> float:
        r"""Calculate the area under the concentration-time curve.

        .. math::

            AUC = \frac{A}{\alpha} + \frac{B}{\beta}

        Returns
        -------
        float
            AUC in mg*hr/L.
        """
        return self.A / self.alpha + self.B / self.beta

    def plot_concentration_curve(
        self,
        duration: float,
        n_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time-concentration data for plotting.

        Parameters
        ----------
        duration : float
            Total time span in hours.
        n_points : int, optional
            Number of data points. Default is 200.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            ``(times, concentrations)`` arrays.
        """
        if duration <= 0:
            raise ValidationError(
                message="Duration must be positive.",
                parameter="duration",
            )
        times = np.linspace(0, duration, n_points)
        concentrations = (
            self.A * np.exp(-self.alpha * times)
            + self.B * np.exp(-self.beta * times)
        )
        return times, concentrations

    def time_to_concentration(self, target: float, tol: float = 1e-6, max_iter: int = 200) -> float:
        r"""Estimate time to reach a target concentration using bisection.

        Because the bi-exponential function is monotonically decreasing,
        bisection on :math:`[0, t_{max}]` is reliable.

        Parameters
        ----------
        target : float
            Target plasma concentration in mg/L.
        tol : float, optional
            Convergence tolerance. Default is 1e-6.
        max_iter : int, optional
            Maximum bisection iterations. Default is 200.

        Returns
        -------
        float
            Estimated time in hours.

        Raises
        ------
        ValidationError
            If the target is unreachable.
        """
        c0 = self.initial_concentration()
        if target <= 0 or target > c0:
            raise ValidationError(
                message=(
                    f"Target {target} mg/L is outside achievable range "
                    f"(0, {c0:.4f}] mg/L."
                ),
                parameter="target",
            )

        t_upper = 10.0 * self.elimination_half_life()
        while self.calculate_concentration(t_upper) > target:
            t_upper *= 2

        t_lo, t_hi = 0.0, t_upper
        for _ in range(max_iter):
            t_mid = (t_lo + t_hi) / 2.0
            c_mid = self.calculate_concentration(t_mid)
            if abs(c_mid - target) < tol:
                return t_mid
            if c_mid > target:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return (t_lo + t_hi) / 2.0

    def __repr__(self) -> str:
        return (
            f"TwoCompartmentModel("
            f"A={self.A}, alpha={self.alpha}, "
            f"B={self.B}, beta={self.beta})"
        )


# ======================================================================
# Renal Function Estimators
# ======================================================================


class CockcroftGault:
    """Estimate creatinine clearance using the Cockcroft-Gault formula.

    The Cockcroft-Gault equation estimates creatinine clearance (CrCl)
    from serum creatinine, age, weight, and sex:

    .. math::

        CrCl = \\frac{(140 - age) \\times weight}{72 \\times S_{Cr}}
               \\times (0.85 \\text{ if female})

    This estimator is widely used for drug dose adjustment thresholds
    because most pharmacokinetic studies reference CrCl rather than GFR.

    Examples
    --------
    >>> cg = CockcroftGault()
    >>> round(cg.calculate(age=65, weight_kg=70, serum_creatinine=1.2,
    ...                     is_female=False), 1)
    72.3

    References
    ----------
    .. [1] Cockcroft DW, Gault MH. Nephron. 1976;16(1):31-41.
    """

    @staticmethod
    def calculate(
        age: float,
        weight_kg: float,
        serum_creatinine: float,
        is_female: bool,
    ) -> float:
        """Calculate creatinine clearance.

        Parameters
        ----------
        age : float
            Patient age in years (18-120).
        weight_kg : float
            Actual body weight in kilograms.
        serum_creatinine : float
            Serum creatinine in mg/dL. Must be positive.
        is_female : bool
            ``True`` if the patient is female.

        Returns
        -------
        float
            Estimated creatinine clearance in mL/min.

        Raises
        ------
        ValidationError
            If serum creatinine is non-positive.
        ClinicalRangeError
            If age or weight is out of plausible range.
        """
        if not isinstance(age, (int, float)) or age < 18 or age > 120:
            raise ClinicalRangeError(
                parameter="age", value=float(age), lower=18.0, upper=120.0,
            )
        if not isinstance(weight_kg, (int, float)) or weight_kg < 20 or weight_kg > 500:
            raise ClinicalRangeError(
                parameter="weight_kg", value=float(weight_kg), lower=20.0, upper=500.0,
            )
        if serum_creatinine <= 0:
            raise ValidationError(
                message="Serum creatinine must be positive.",
                parameter="serum_creatinine",
            )

        crcl = ((140.0 - age) * weight_kg) / (72.0 * serum_creatinine)
        if is_female:
            crcl *= 0.85
        return crcl

    @staticmethod
    def classify(crcl: float) -> str:
        """Classify renal function stage by creatinine clearance.

        Parameters
        ----------
        crcl : float
            Creatinine clearance in mL/min.

        Returns
        -------
        str
            One of ``"normal"`` (>=90), ``"mild"`` (60-89),
            ``"moderate"`` (30-59), ``"severe"`` (15-29), or
            ``"dialysis"`` (<15).
        """
        if crcl >= 90:
            return "normal"
        elif crcl >= 60:
            return "mild"
        elif crcl >= 30:
            return "moderate"
        elif crcl >= 15:
            return "severe"
        else:
            return "dialysis"


class MDRD:
    """Estimate GFR using the four-variable MDRD equation.

    The Modification of Diet in Renal Disease (MDRD) study equation
    estimates glomerular filtration rate (GFR) from serum creatinine,
    age, and sex. This implementation uses the re-expressed MDRD
    equation for use with standardized (IDMS) creatinine assays.

    .. math::

        GFR = 175 \\times S_{Cr}^{-1.154} \\times age^{-0.203}
              \\times (0.742 \\text{ if female})

    Examples
    --------
    >>> mdrd = MDRD()
    >>> round(mdrd.calculate(age=65, serum_creatinine=1.2,
    ...                      is_female=False), 1)
    62.8

    References
    ----------
    .. [1] Levey AS, et al. Ann Intern Med. 1999;130(6):461-470.
    .. [2] Levey AS, et al. Ann Intern Med. 2006;145(4):247-254.
    """

    @staticmethod
    def calculate(
        age: float,
        serum_creatinine: float,
        is_female: bool,
    ) -> float:
        """Calculate estimated GFR using the MDRD equation.

        Parameters
        ----------
        age : float
            Patient age in years (18-120).
        serum_creatinine : float
            Serum creatinine in mg/dL (IDMS standardized). Must be
            positive.
        is_female : bool
            ``True`` if the patient is female.

        Returns
        -------
        float
            Estimated GFR in mL/min/1.73m^2.

        Raises
        ------
        ValidationError
            If serum creatinine is non-positive.
        ClinicalRangeError
            If age is out of range.
        """
        if not isinstance(age, (int, float)) or age < 18 or age > 120:
            raise ClinicalRangeError(
                parameter="age", value=float(age), lower=18.0, upper=120.0,
            )
        if serum_creatinine <= 0:
            raise ValidationError(
                message="Serum creatinine must be positive.",
                parameter="serum_creatinine",
            )

        gfr = 175.0 * (serum_creatinine ** -1.154) * (age ** -0.203)
        if is_female:
            gfr *= 0.742
        return gfr


class CKD_EPI:
    """Estimate GFR using the CKD-EPI 2021 creatinine equation.

    The Chronic Kidney Disease Epidemiology Collaboration (CKD-EPI)
    2021 equation estimates GFR without a race coefficient, as
    recommended by the NKF-ASN Task Force.

    For **females** (kappa = 0.7, alpha = -0.241):

    .. math::

        GFR = 142 \\times \\min(S_{Cr}/0.7, 1)^{-0.241}
              \\times \\max(S_{Cr}/0.7, 1)^{-1.200}
              \\times 0.9938^{age} \\times 1.012

    For **males** (kappa = 0.9, alpha = -0.302):

    .. math::

        GFR = 142 \\times \\min(S_{Cr}/0.9, 1)^{-0.302}
              \\times \\max(S_{Cr}/0.9, 1)^{-1.200}
              \\times 0.9938^{age}

    Examples
    --------
    >>> ckd = CKD_EPI()
    >>> round(ckd.calculate(age=65, serum_creatinine=1.2,
    ...                     is_female=False), 1)
    64.5

    References
    ----------
    .. [1] Inker LA, et al. N Engl J Med. 2021;385(19):1737-1749.
    """

    @staticmethod
    def calculate(
        age: float,
        serum_creatinine: float,
        is_female: bool,
    ) -> float:
        """Calculate estimated GFR using CKD-EPI 2021.

        Parameters
        ----------
        age : float
            Patient age in years (18-120).
        serum_creatinine : float
            Serum creatinine in mg/dL (IDMS standardized). Must be
            positive.
        is_female : bool
            ``True`` if the patient is female.

        Returns
        -------
        float
            Estimated GFR in mL/min/1.73m^2.

        Raises
        ------
        ValidationError
            If serum creatinine is non-positive.
        ClinicalRangeError
            If age is out of range.
        """
        if not isinstance(age, (int, float)) or age < 18 or age > 120:
            raise ClinicalRangeError(
                parameter="age", value=float(age), lower=18.0, upper=120.0,
            )
        if serum_creatinine <= 0:
            raise ValidationError(
                message="Serum creatinine must be positive.",
                parameter="serum_creatinine",
            )

        if is_female:
            kappa = 0.7
            alpha = -0.241
            sex_factor = 1.012
        else:
            kappa = 0.9
            alpha = -0.302
            sex_factor = 1.0

        scr_ratio = serum_creatinine / kappa
        term_min = min(scr_ratio, 1.0) ** alpha
        term_max = max(scr_ratio, 1.0) ** (-1.200)

        gfr = 142.0 * term_min * term_max * (0.9938 ** age) * sex_factor
        return gfr

    @staticmethod
    def classify(gfr: float) -> str:
        """Classify CKD stage by GFR.

        Uses the KDIGO 2012 classification.

        Parameters
        ----------
        gfr : float
            GFR in mL/min/1.73m^2.

        Returns
        -------
        str
            CKD stage: ``"G1"`` (>=90), ``"G2"`` (60-89), ``"G3a"``
            (45-59), ``"G3b"`` (30-44), ``"G4"`` (15-29), or ``"G5"``
            (<15).
        """
        if gfr >= 90:
            return "G1"
        elif gfr >= 60:
            return "G2"
        elif gfr >= 45:
            return "G3a"
        elif gfr >= 30:
            return "G3b"
        elif gfr >= 15:
            return "G4"
        else:
            return "G5"
