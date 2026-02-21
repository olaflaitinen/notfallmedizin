# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Bayesian statistical methods for emergency medicine.

Implements Bayesian A/B testing with Beta-Binomial conjugacy and
Bayesian diagnostic test updating via Bayes' theorem.

References
----------
Berry, D. A. (2006). Bayesian clinical trials. Nature Reviews Drug
    Discovery, 5(1), 27-36.
Fagan, T. J. (1975). Nomogram for Bayes theorem. NEJM, 293(5), 257.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class BayesianResult:
    """Result of a Bayesian diagnostic test update.

    Attributes
    ----------
    posterior_probability : float
    likelihood_ratio : float
    prior : float
    evidence : float
    """

    posterior_probability: float
    likelihood_ratio: float
    prior: float
    evidence: float


class BayesianABTest:
    """Bayesian A/B testing with Beta-Binomial conjugate model.

    Each group's success rate is modelled as Beta(alpha, beta) posterior
    given observed successes and trials under a conjugate prior.
    """

    def __init__(self) -> None:
        self._groups: Dict[str, Tuple[int, int]] = {}
        self._posteriors: Dict[str, Tuple[float, float]] = {}

    def add_group(
        self, name: str, successes: int, trials: int
    ) -> "BayesianABTest":
        """Register a group with observed data.

        Parameters
        ----------
        name : str
        successes : int
        trials : int

        Returns
        -------
        self
        """
        if trials < 0 or successes < 0 or successes > trials:
            raise ValidationError(
                f"Invalid data for group '{name}': successes={successes}, "
                f"trials={trials}."
            )
        self._groups[name] = (successes, trials)
        return self

    def compute_posterior(
        self, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> Dict[str, Tuple[float, float]]:
        """Compute Beta posterior parameters for all groups.

        Parameters
        ----------
        prior_alpha : float
        prior_beta : float

        Returns
        -------
        dict
            Mapping of group name to (posterior_alpha, posterior_beta).
        """
        self._posteriors = {}
        for name, (s, n) in self._groups.items():
            self._posteriors[name] = (
                prior_alpha + s,
                prior_beta + (n - s),
            )
        return dict(self._posteriors)

    def probability_b_beats_a(
        self,
        group_a: str = "",
        group_b: str = "",
        n_simulations: int = 100_000,
    ) -> float:
        """Estimate P(group_b > group_a) via Monte Carlo.

        Parameters
        ----------
        group_a : str
        group_b : str
        n_simulations : int

        Returns
        -------
        float
        """
        if not self._posteriors:
            self.compute_posterior()

        names = list(self._posteriors.keys())
        a_name = group_a or names[0]
        b_name = group_b or (names[1] if len(names) > 1 else names[0])

        a_alpha, a_beta = self._posteriors[a_name]
        b_alpha, b_beta = self._posteriors[b_name]

        rng = np.random.default_rng(42)
        samples_a = rng.beta(a_alpha, a_beta, size=n_simulations)
        samples_b = rng.beta(b_alpha, b_beta, size=n_simulations)

        return float(np.mean(samples_b > samples_a))

    def expected_loss(
        self, n_simulations: int = 100_000
    ) -> Dict[str, float]:
        """Compute expected loss for choosing each group.

        Parameters
        ----------
        n_simulations : int

        Returns
        -------
        dict
            Mapping group name to expected loss.
        """
        if not self._posteriors:
            self.compute_posterior()

        rng = np.random.default_rng(42)
        samples: Dict[str, np.ndarray] = {}
        for name, (a, b) in self._posteriors.items():
            samples[name] = rng.beta(a, b, size=n_simulations)

        all_samples = np.stack(list(samples.values()), axis=0)
        best = np.max(all_samples, axis=0)

        losses: Dict[str, float] = {}
        for i, name in enumerate(samples):
            losses[name] = float(np.mean(best - all_samples[i]))

        return losses

    def credible_interval(
        self, group: str, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Return the (1 - alpha) credible interval for a group.

        Parameters
        ----------
        group : str
        alpha : float

        Returns
        -------
        tuple of (lower, upper)
        """
        if not self._posteriors:
            self.compute_posterior()
        a, b = self._posteriors[group]
        lower = float(sp_stats.beta.ppf(alpha / 2, a, b))
        upper = float(sp_stats.beta.ppf(1 - alpha / 2, a, b))
        return (round(lower, 6), round(upper, 6))


class BayesianDiagnosticTest:
    """Sequential Bayesian diagnostic probability updating.

    Applies Bayes' theorem to update disease probability given
    sequential test results and known test characteristics.
    """

    def __init__(self) -> None:
        self._current_prior: Optional[float] = None

    def calculate_posterior(
        self,
        prior_probability: float,
        sensitivity: float,
        specificity: float,
        test_positive: bool,
    ) -> BayesianResult:
        """Compute posterior probability after a single test.

        Parameters
        ----------
        prior_probability : float
        sensitivity : float
        specificity : float
        test_positive : bool

        Returns
        -------
        BayesianResult
        """
        self._validate_probability(prior_probability, "prior_probability")
        self._validate_probability(sensitivity, "sensitivity")
        self._validate_probability(specificity, "specificity")

        if test_positive:
            lr = sensitivity / (1.0 - specificity) if specificity < 1 else float("inf")
            numerator = sensitivity * prior_probability
            denominator = (
                sensitivity * prior_probability
                + (1 - specificity) * (1 - prior_probability)
            )
        else:
            lr = (1.0 - sensitivity) / specificity if specificity > 0 else 0.0
            numerator = (1 - sensitivity) * prior_probability
            denominator = (
                (1 - sensitivity) * prior_probability
                + specificity * (1 - prior_probability)
            )

        posterior = numerator / denominator if denominator > 0 else 0.0
        self._current_prior = posterior

        return BayesianResult(
            posterior_probability=round(posterior, 6),
            likelihood_ratio=round(lr, 4),
            prior=prior_probability,
            evidence=round(denominator, 6),
        )

    def update(
        self,
        sensitivity: float,
        specificity: float,
        test_positive: bool,
    ) -> BayesianResult:
        """Update the current posterior with a new test result.

        Uses the most recent posterior as the new prior.

        Parameters
        ----------
        sensitivity : float
        specificity : float
        test_positive : bool

        Returns
        -------
        BayesianResult
        """
        if self._current_prior is None:
            raise ValidationError(
                "No prior set. Call calculate_posterior first."
            )
        return self.calculate_posterior(
            self._current_prior, sensitivity, specificity, test_positive
        )

    def fagan_nomogram_data(
        self,
        prior_probability: float,
        sensitivity: float,
        specificity: float,
    ) -> Dict[str, float]:
        """Generate data points for a Fagan nomogram.

        Parameters
        ----------
        prior_probability : float
        sensitivity : float
        specificity : float

        Returns
        -------
        dict
            Keys: pre_test_odds, lr_positive, lr_negative,
            post_test_odds_positive, post_test_odds_negative,
            post_test_prob_positive, post_test_prob_negative.
        """
        pre_odds = prior_probability / (1 - prior_probability) if prior_probability < 1 else float("inf")
        lr_pos = sensitivity / (1 - specificity) if specificity < 1 else float("inf")
        lr_neg = (1 - sensitivity) / specificity if specificity > 0 else 0.0

        post_odds_pos = pre_odds * lr_pos
        post_odds_neg = pre_odds * lr_neg

        post_prob_pos = post_odds_pos / (1 + post_odds_pos) if post_odds_pos < float("inf") else 1.0
        post_prob_neg = post_odds_neg / (1 + post_odds_neg)

        return {
            "pre_test_odds": round(pre_odds, 4),
            "lr_positive": round(lr_pos, 4),
            "lr_negative": round(lr_neg, 4),
            "post_test_odds_positive": round(post_odds_pos, 4),
            "post_test_odds_negative": round(post_odds_neg, 4),
            "post_test_prob_positive": round(post_prob_pos, 6),
            "post_test_prob_negative": round(post_prob_neg, 6),
        }

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not 0 <= value <= 1:
            raise ValidationError(
                f"{name} must be in [0, 1], got {value}."
            )
