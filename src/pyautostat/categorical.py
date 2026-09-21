"""Independent-sample association tests for categorical columns."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidTestError


def _interval(estimates, level, requested):
    if len(estimates) < max(50, requested // 2):
        return None
    alpha = (1 - level) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return {
        "level": level,
        "lower": float(lower),
        "upper": float(upper),
        "method": "paired-row percentile bootstrap",
        "valid_resamples": len(estimates),
    }


def _cohens_h(first, second):
    return float(2 * (np.arcsin(np.sqrt(first)) - np.arcsin(np.sqrt(second))))


def categorical_association(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    *,
    success_value=None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict:
    """Pearson chi-square with Cramer's V for two categorical columns.

    Pairwise missing rows are excluded. Group and outcome order follows first
    appearance. Cohen's h is available only for a 2x2 table and an explicit
    success_value; positive h means the first group has a higher success rate.
    Expected counts must all be at least five for the chi-square approximation.
    """
    if not isinstance(group_col, str) or not group_col.strip():
        raise InvalidTestError("group_col must be a non-empty column name string.")
    if not isinstance(outcome_col, str) or not outcome_col.strip():
        raise InvalidTestError("outcome_col must be a non-empty column name string.")
    if group_col == outcome_col:
        raise InvalidTestError("group_col and outcome_col must name different columns.")
    for name in (group_col, outcome_col):
        if name not in df.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise InvalidTestError("confidence_level must be a finite number between 0 and 1.")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, (int, np.integer))
        or (bootstrap_samples != 0 and bootstrap_samples < 100)
    ):
        raise InvalidTestError("bootstrap_samples must be 0 or an integer of at least 100.")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise InvalidTestError("random_state must be a nonnegative integer or None.")

    usable = df[[group_col, outcome_col]].dropna()
    group_codes, groups = pd.factorize(usable[group_col], sort=False)
    outcome_codes, outcomes = pd.factorize(usable[outcome_col], sort=False)
    if len(groups) < 2 or len(outcomes) < 2:
        raise InsufficientDataError(
            "Chi-square association needs at least 2 observed categories in each column "
            "after excluding missing rows."
        )
    if success_value is not None and (len(groups) != 2 or len(outcomes) != 2):
        raise InvalidTestError("Cohen's h needs exactly 2 groups and 2 outcome categories.")
    if success_value is not None and (
        not pd.api.types.is_scalar(success_value) or pd.isna(success_value)
    ):
        raise InvalidTestError("success_value must be one non-missing scalar outcome category.")
    observed = np.zeros((len(groups), len(outcomes)), dtype=int)
    np.add.at(observed, (group_codes, outcome_codes), 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        chi2, p_value, degrees_of_freedom, expected = chi2_contingency(observed, correction=False)
    if not np.isfinite(chi2) or not np.isfinite(p_value):
        raise InsufficientDataError("Chi-square returned an undefined result; review the table.")
    if np.any(expected < 5):
        raise InsufficientDataError(
            "Chi-square needs expected counts of at least 5 in every cell for its "
            "approximation. Combine sparse categories or use an exact method."
        )
    n = int(observed.sum())
    cramer_v = float(np.sqrt(chi2 / (n * min(len(groups) - 1, len(outcomes) - 1))))

    success_index = None
    if success_value is not None:
        matches = [index for index, value in enumerate(outcomes) if value == success_value]
        if len(matches) != 1:
            raise InvalidTestError("success_value must match one observed outcome category.")
        success_index = matches[0]

    v_estimates = []
    h_estimates = []
    if bootstrap_samples:
        rng = np.random.default_rng(random_state)
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, n, size=n)
            sampled = np.bincount(
                group_codes[indices] * len(outcomes) + outcome_codes[indices],
                minlength=observed.size,
            ).reshape(observed.shape)
            if np.any(sampled.sum(axis=0) == 0) or np.any(sampled.sum(axis=1) == 0):
                continue
            sampled_chi2 = chi2_contingency(sampled, correction=False).statistic
            estimate = np.sqrt(sampled_chi2 / (n * min(len(groups) - 1, len(outcomes) - 1)))
            if np.isfinite(estimate):
                v_estimates.append(float(estimate))
            if success_index is not None:
                rates = sampled[:, success_index] / sampled.sum(axis=1)
                h_estimates.append(_cohens_h(*rates))

    result = {
        "test": "Pearson chi-square independence",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "groups": list(groups),
        "outcomes": list(outcomes),
        "observed_counts": observed.tolist(),
        "expected_counts": expected.tolist(),
        "sample_size": n,
        "assumptions": {
            "independent_observations": "Required by study design; cannot be verified from values.",
            "minimum_expected_count": float(expected.min()),
            "expected_count_status": "met",
        },
        "effect_size": {
            "name": "Cramer's V",
            "value": cramer_v,
            "confidence_interval": (
                _interval(v_estimates, float(confidence_level), int(bootstrap_samples))
                if bootstrap_samples
                else None
            ),
        },
    }
    if success_index is not None:
        rates = observed[:, success_index] / observed.sum(axis=1)
        result["cohens_h"] = {
            "success_value": success_value,
            "group_proportions": [float(value) for value in rates],
            "value": _cohens_h(*rates),
            "confidence_interval": (
                _interval(h_estimates, float(confidence_level), int(bootstrap_samples))
                if bootstrap_samples
                else None
            ),
        }
    return result
