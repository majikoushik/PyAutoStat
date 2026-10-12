"""Canonical linkage layer for statistical methods and analytical quantities.

This module is the canonical source of truth for cross-cutting method and quantity
metadata, linking sensitivity analysis, practical significance, method contracts,
and execution dispatch without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantityDefinition:
    """Canonical definition of an analytical or inferential quantity."""

    quantity: str
    shape: str  # "scalar" | "coefficient_vector" | "vector"
    # e.g. "difference", "correlation", "variance_ratio", "association", "reliability", etc.
    domain: str
    signed: bool | None  # True if signed, False if nonnegative, None if vector/unconstrained
    supports_practical_threshold: bool = False


@dataclass(frozen=True)
class MethodRegistration:
    """Canonical linkage record for one supported analytical method."""

    method_id: str
    name: str
    sensitivity_quantity: str | None = None
    sensitivity_shape: str = "scalar"  # "scalar" | "coefficient_vector" | "vector"
    practical_significance_quantities: tuple[str, ...] = ()
    requires_contract: bool = True
    is_inferential: bool = True
    hypothesis_test: bool = True


# ---------------------------------------------------------------------------
# Canonical Quantities
# ---------------------------------------------------------------------------

QUANTITY_REGISTRY: dict[str, QuantityDefinition] = {
    # Signed scalar difference quantities (support practical significance thresholds)
    "mean_difference": QuantityDefinition(
        quantity="mean_difference",
        shape="scalar",
        domain="difference",
        signed=True,
        supports_practical_threshold=True,
    ),
    "cohens_d": QuantityDefinition(
        quantity="cohens_d",
        shape="scalar",
        domain="difference",
        signed=True,
        supports_practical_threshold=True,
    ),
    "cohens_dz": QuantityDefinition(
        quantity="cohens_dz",
        shape="scalar",
        domain="difference",
        signed=True,
        supports_practical_threshold=False,
    ),
    "rank_biserial": QuantityDefinition(
        quantity="rank_biserial",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    "matched_pairs_rank_biserial": QuantityDefinition(
        quantity="matched_pairs_rank_biserial",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    # Signed correlation quantities (support practical significance thresholds)
    "pearson_r": QuantityDefinition(
        quantity="pearson_r",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    "spearman_rho": QuantityDefinition(
        quantity="spearman_rho",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    "point_biserial_r": QuantityDefinition(
        quantity="point_biserial_r",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    "kendall_tau_b": QuantityDefinition(
        quantity="kendall_tau_b",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    "partial_pearson_r": QuantityDefinition(
        quantity="partial_pearson_r",
        shape="scalar",
        domain="correlation",
        signed=True,
        supports_practical_threshold=True,
    ),
    # Nonnegative association and variance-ratio quantities (support practical thresholds)
    "cramers_v": QuantityDefinition(
        quantity="cramers_v",
        shape="scalar",
        domain="association",
        signed=False,
        supports_practical_threshold=True,
    ),
    "eta_squared": QuantityDefinition(
        quantity="eta_squared",
        shape="scalar",
        domain="variance_ratio",
        signed=False,
        supports_practical_threshold=True,
    ),
    "epsilon_squared": QuantityDefinition(
        quantity="epsilon_squared",
        shape="scalar",
        domain="variance_ratio",
        signed=False,
        supports_practical_threshold=True,
    ),
    "partial_eta_squared": QuantityDefinition(
        quantity="partial_eta_squared",
        shape="scalar",
        domain="variance_ratio",
        signed=False,
        supports_practical_threshold=True,
    ),
    # Ratio, proportion, and concordance quantities
    "odds_ratio": QuantityDefinition(
        quantity="odds_ratio",
        shape="scalar",
        domain="ratio",
        signed=False,
        supports_practical_threshold=False,
    ),
    "paired_proportion_difference": QuantityDefinition(
        quantity="paired_proportion_difference",
        shape="scalar",
        domain="proportion",
        signed=True,
        supports_practical_threshold=True,
    ),
    "kendalls_w": QuantityDefinition(
        quantity="kendalls_w",
        shape="scalar",
        domain="concordance",
        signed=False,
        supports_practical_threshold=True,
    ),
    # Reliability quantities
    "cronbach_alpha": QuantityDefinition(
        quantity="cronbach_alpha",
        shape="scalar",
        domain="reliability",
        signed=True,
        supports_practical_threshold=False,
    ),
    "intraclass_correlation": QuantityDefinition(
        quantity="intraclass_correlation",
        shape="scalar",
        domain="reliability",
        signed=True,
        supports_practical_threshold=False,
    ),
    # Vector and structured quantities
    "group_mean_vector": QuantityDefinition(
        quantity="group_mean_vector",
        shape="vector",
        domain="vector",
        signed=None,
        supports_practical_threshold=False,
    ),
    "coefficient_vector": QuantityDefinition(
        quantity="coefficient_vector",
        shape="coefficient_vector",
        domain="regression",
        signed=None,
        supports_practical_threshold=False,
    ),
    # Term-level regression quantities
    "regression_coefficient": QuantityDefinition(
        quantity="regression_coefficient",
        shape="scalar",
        domain="regression",
        signed=True,
        supports_practical_threshold=False,
    ),
    "log_odds_coefficient": QuantityDefinition(
        quantity="log_odds_coefficient",
        shape="scalar",
        domain="regression",
        signed=True,
        supports_practical_threshold=False,
    ),
}


# ---------------------------------------------------------------------------
# Canonical Method Registrations
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, MethodRegistration] = {
    "welch_t": MethodRegistration(
        method_id="welch_t",
        name="Welch independent-samples t-test",
        sensitivity_quantity="mean_difference",
        sensitivity_shape="scalar",
        practical_significance_quantities=("mean_difference", "cohens_d"),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "student_t": MethodRegistration(
        method_id="student_t",
        name="Student independent-samples t-test",
        sensitivity_quantity="mean_difference",
        sensitivity_shape="scalar",
        practical_significance_quantities=("mean_difference", "cohens_d"),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "paired_t": MethodRegistration(
        method_id="paired_t",
        name="Paired-samples t-test",
        sensitivity_quantity="mean_difference",
        sensitivity_shape="scalar",
        practical_significance_quantities=("mean_difference",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "one_sample_t": MethodRegistration(
        method_id="one_sample_t",
        name="One-sample Student's t-test",
        sensitivity_quantity="mean_difference",
        sensitivity_shape="scalar",
        practical_significance_quantities=("mean_difference",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "mann_whitney_u": MethodRegistration(
        method_id="mann_whitney_u",
        name="Mann-Whitney U rank-sum test",
        sensitivity_quantity="rank_biserial",
        sensitivity_shape="scalar",
        practical_significance_quantities=("rank_biserial",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "wilcoxon_signed_rank": MethodRegistration(
        method_id="wilcoxon_signed_rank",
        name="Wilcoxon signed-rank test",
        sensitivity_quantity="matched_pairs_rank_biserial",
        sensitivity_shape="scalar",
        practical_significance_quantities=("matched_pairs_rank_biserial",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "welch_anova": MethodRegistration(
        method_id="welch_anova",
        name="Welch one-way ANOVA",
        sensitivity_quantity="group_mean_vector",
        sensitivity_shape="vector",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "one_way_anova": MethodRegistration(
        method_id="one_way_anova",
        name="One-way ANOVA (equal variance)",
        sensitivity_quantity="eta_squared",
        sensitivity_shape="scalar",
        practical_significance_quantities=("eta_squared",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "kruskal_wallis": MethodRegistration(
        method_id="kruskal_wallis",
        name="Kruskal-Wallis one-way rank ANOVA",
        sensitivity_quantity="epsilon_squared",
        sensitivity_shape="scalar",
        practical_significance_quantities=("epsilon_squared",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "pearson_correlation": MethodRegistration(
        method_id="pearson_correlation",
        name="Pearson product-moment correlation",
        sensitivity_quantity="pearson_r",
        sensitivity_shape="scalar",
        practical_significance_quantities=("pearson_r",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "spearman_correlation": MethodRegistration(
        method_id="spearman_correlation",
        name="Spearman rank-order correlation",
        sensitivity_quantity="spearman_rho",
        sensitivity_shape="scalar",
        practical_significance_quantities=("spearman_rho",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "kendall_tau_b": MethodRegistration(
        method_id="kendall_tau_b",
        name="Kendall's tau-b rank correlation",
        sensitivity_quantity="kendall_tau_b",
        sensitivity_shape="scalar",
        practical_significance_quantities=("kendall_tau_b",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "point_biserial_correlation": MethodRegistration(
        method_id="point_biserial_correlation",
        name="Point-biserial correlation",
        sensitivity_quantity="point_biserial_r",
        sensitivity_shape="scalar",
        practical_significance_quantities=("point_biserial_r",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "partial_pearson_correlation": MethodRegistration(
        method_id="partial_pearson_correlation",
        name="Partial Pearson correlation",
        sensitivity_quantity="partial_pearson_r",
        sensitivity_shape="scalar",
        practical_significance_quantities=("partial_pearson_r",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "pearson_chi_square": MethodRegistration(
        method_id="pearson_chi_square",
        name="Pearson chi-square test of independence",
        sensitivity_quantity="cramers_v",
        sensitivity_shape="scalar",
        practical_significance_quantities=("cramers_v",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "fisher_exact": MethodRegistration(
        method_id="fisher_exact",
        name="Fisher's exact test",
        sensitivity_quantity="odds_ratio",
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "mcnemar": MethodRegistration(
        method_id="mcnemar",
        name="McNemar test of paired binary proportions",
        sensitivity_quantity="paired_proportion_difference",
        sensitivity_shape="scalar",
        practical_significance_quantities=("paired_proportion_difference",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "linear_regression": MethodRegistration(
        method_id="linear_regression",
        name="Ordinary least-squares linear regression",
        sensitivity_quantity="coefficient_vector",
        sensitivity_shape="coefficient_vector",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "logistic_regression": MethodRegistration(
        method_id="logistic_regression",
        name="Binary logistic regression",
        sensitivity_quantity="coefficient_vector",
        sensitivity_shape="coefficient_vector",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "cronbach_alpha": MethodRegistration(
        method_id="cronbach_alpha",
        name="Cronbach's alpha scale reliability",
        sensitivity_quantity="cronbach_alpha",
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=False,
    ),
    "repeated_measures_anova": MethodRegistration(
        method_id="repeated_measures_anova",
        name="One-way repeated-measures ANOVA",
        sensitivity_quantity="partial_eta_squared",
        sensitivity_shape="scalar",
        practical_significance_quantities=("partial_eta_squared",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "friedman_test": MethodRegistration(
        method_id="friedman_test",
        name="Friedman rank-sum test",
        sensitivity_quantity="kendalls_w",
        sensitivity_shape="scalar",
        practical_significance_quantities=("kendalls_w",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "two_way_anova": MethodRegistration(
        method_id="two_way_anova",
        name="Two-way factorial ANOVA",
        sensitivity_quantity="partial_eta_squared",
        sensitivity_shape="scalar",
        practical_significance_quantities=("partial_eta_squared",),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    "intraclass_correlation": MethodRegistration(
        method_id="intraclass_correlation",
        name="Intraclass correlation coefficient",
        sensitivity_quantity="intraclass_correlation",
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=True,
        is_inferential=True,
        hypothesis_test=True,
    ),
    # Non-inferential dataset profiling and coefficient-only capabilities
    "dataset_profile": MethodRegistration(
        method_id="dataset_profile",
        name="Dataset profile",
        sensitivity_quantity=None,
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=False,
        is_inferential=False,
        hypothesis_test=False,
    ),
    "spearman_coefficient": MethodRegistration(
        method_id="spearman_coefficient",
        name="Spearman correlation coefficient",
        sensitivity_quantity=None,
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=False,
        is_inferential=False,
        hypothesis_test=False,
    ),
    "kendall_coefficient": MethodRegistration(
        method_id="kendall_coefficient",
        name="Kendall tau-b correlation coefficient",
        sensitivity_quantity=None,
        sensitivity_shape="scalar",
        practical_significance_quantities=(),
        requires_contract=False,
        is_inferential=False,
        hypothesis_test=False,
    ),
}


# ---------------------------------------------------------------------------
# Registry Accessors
# ---------------------------------------------------------------------------


def get_method_registration(method_id: str) -> MethodRegistration | None:
    """Return the registration for a method ID or None if unregistered."""
    return METHOD_REGISTRY.get(method_id)


def get_quantity_definition(quantity: str) -> QuantityDefinition | None:
    """Return the definition for a quantity identifier or None if unregistered."""
    return QUANTITY_REGISTRY.get(quantity)


def list_method_registrations() -> tuple[MethodRegistration, ...]:
    """Return all canonical method registrations."""
    return tuple(METHOD_REGISTRY.values())


def list_quantity_definitions() -> tuple[QuantityDefinition, ...]:
    """Return all canonical quantity definitions."""
    return tuple(QUANTITY_REGISTRY.values())


def estimate_quantity_for_method(method_id: str) -> str | None:
    """Return the stable primary sensitivity quantity identifier for a method."""
    reg = METHOD_REGISTRY.get(method_id)
    return reg.sensitivity_quantity if reg is not None else None


def sensitivity_shape_for_method(method_id: str) -> str:
    """Return the sensitivity result shape ('scalar', 'coefficient_vector', 'vector')."""
    reg = METHOD_REGISTRY.get(method_id)
    return reg.sensitivity_shape if reg is not None else "scalar"


# ---------------------------------------------------------------------------
# Cross-Registry Invariant Validation
# ---------------------------------------------------------------------------


def validate_registry_invariants() -> list[str]:
    """Check cross-registry consistency and detect metadata drift.

    Returns a list of error descriptions. An empty list indicates all
    invariants hold.
    """
    errors: list[str] = []

    # 1. Method ID uniqueness is guaranteed by dict, but check key matches method_id
    for key, reg in METHOD_REGISTRY.items():
        if key != reg.method_id:
            errors.append(
                f"Method key {key!r} does not match registration method_id {reg.method_id!r}."
            )

    # 2. Quantity ID uniqueness and key match
    for key, qdef in QUANTITY_REGISTRY.items():
        if key != qdef.quantity:
            errors.append(
                f"Quantity key {key!r} does not match definition quantity {qdef.quantity!r}."
            )

    # 3. Sensitivity quantities exist in QUANTITY_REGISTRY
    for reg in METHOD_REGISTRY.values():
        if (
            reg.sensitivity_quantity is not None
            and reg.sensitivity_quantity not in QUANTITY_REGISTRY
        ):
            errors.append(
                f"Method {reg.method_id!r} references unknown sensitivity quantity "
                f"{reg.sensitivity_quantity!r}."
            )

    # 4. Coefficient-vector methods must use structured/coefficient_vector shape
    for reg in METHOD_REGISTRY.values():
        if reg.sensitivity_shape == "coefficient_vector":
            if reg.sensitivity_quantity != "coefficient_vector":
                errors.append(
                    f"Method {reg.method_id!r} has sensitivity_shape='coefficient_vector' "
                    f"but sensitivity_quantity is {reg.sensitivity_quantity!r}."
                )

    # 5. Practical significance quantities exist in QUANTITY_REGISTRY and support thresholds
    for reg in METHOD_REGISTRY.values():
        for q in reg.practical_significance_quantities:
            if q not in QUANTITY_REGISTRY:
                errors.append(
                    f"Method {reg.method_id!r} practical significance quantity {q!r} is unknown."
                )
            elif not QUANTITY_REGISTRY[q].supports_practical_threshold:
                errors.append(
                    f"Method {reg.method_id!r} practical significance quantity {q!r} does not "
                    "support practical thresholds."
                )

    # 6. Check against METHOD_CONTRACTS (lazy import)
    try:
        from .method_contracts import METHOD_CONTRACTS

        for reg in METHOD_REGISTRY.values():
            if reg.requires_contract and reg.method_id not in METHOD_CONTRACTS:
                errors.append(
                    f"Runnable inferential method {reg.method_id!r} requires a contract but is "
                    "missing from METHOD_CONTRACTS."
                )
        for method_id in METHOD_CONTRACTS:
            if method_id not in METHOD_REGISTRY:
                errors.append(
                    f"Method contract exists for {method_id!r} which is not registered in "
                    "METHOD_REGISTRY."
                )
    except ImportError as exc:
        errors.append(f"Could not import METHOD_CONTRACTS: {exc}")

    # 7. Check against METHOD_CAPABILITIES (lazy import)
    try:
        from .recommendation import METHOD_CAPABILITIES

        for cap_id, _cap in METHOD_CAPABILITIES.items():
            if cap_id not in METHOD_REGISTRY:
                errors.append(f"MethodCapability {cap_id!r} is not registered in METHOD_REGISTRY.")
    except ImportError as exc:
        errors.append(f"Could not import METHOD_CAPABILITIES: {exc}")

    # 8. Check against practical_significance.SUPPORTED_QUANTITIES (lazy import)
    try:
        from .practical_significance import SUPPORTED_QUANTITIES

        for q in SUPPORTED_QUANTITIES:
            if q not in QUANTITY_REGISTRY:
                errors.append(
                    f"practical_significance.SUPPORTED_QUANTITIES contains {q!r} which is "
                    "missing from QUANTITY_REGISTRY."
                )
            elif not QUANTITY_REGISTRY[q].supports_practical_threshold:
                errors.append(
                    f"Quantity {q!r} is in SUPPORTED_QUANTITIES but "
                    "supports_practical_threshold is False."
                )
    except ImportError as exc:
        errors.append(f"Could not import practical_significance: {exc}")

    # 9. Check against execution dispatched methods (lazy import)
    try:
        from .execution import _GROUP_BACKENDS, _NULL_HYPOTHESES

        for method_id in _GROUP_BACKENDS:
            if method_id not in METHOD_REGISTRY:
                errors.append(f"Execution group backend {method_id!r} is not in METHOD_REGISTRY.")
        for method_id in _NULL_HYPOTHESES:
            if method_id not in METHOD_REGISTRY:
                errors.append(f"Execution null hypothesis {method_id!r} is not in METHOD_REGISTRY.")
    except ImportError as exc:
        errors.append(f"Could not import execution mappings: {exc}")

    return errors
