"""Invariant tests for the canonical method and quantity registry."""

from pyautostat.execution import _GROUP_BACKENDS, _NULL_HYPOTHESES
from pyautostat.method_contracts import METHOD_CONTRACTS
from pyautostat.method_registry import (
    METHOD_REGISTRY,
    QUANTITY_REGISTRY,
    get_method_registration,
    get_quantity_definition,
    list_method_registrations,
    list_quantity_definitions,
    validate_registry_invariants,
)
from pyautostat.practical_significance import (
    NONNEGATIVE_QUANTITIES,
    SIGNED_QUANTITIES,
    SUPPORTED_QUANTITIES,
)
from pyautostat.recommendation import METHOD_CAPABILITIES
from pyautostat.sensitivity import _QUANTITIES


def test_registry_invariants_pass():
    """Canonical validation function reports zero drift across modules."""
    errors = validate_registry_invariants()
    assert errors == []


def test_all_runnable_inferential_methods_are_registered():
    runnable_caps = {
        cid
        for cid, cap in METHOD_CAPABILITIES.items()
        if cap.availability == "runnable" and cid != "dataset_profile"
    }
    registered_inferential = {m.method_id for m in list_method_registrations() if m.is_inferential}
    assert runnable_caps <= registered_inferential


def test_required_methods_have_method_contracts():
    required = {m.method_id for m in list_method_registrations() if m.requires_contract}
    assert required == set(METHOD_CONTRACTS.keys())


def test_sensitivity_methods_are_registered():
    for method_id in _QUANTITIES:
        reg = get_method_registration(method_id)
        assert reg is not None, f"Sensitivity method {method_id!r} is not registered."
        assert reg.sensitivity_quantity == _QUANTITIES[method_id]


def test_sensitivity_quantities_are_known():
    known_quantities = set(QUANTITY_REGISTRY.keys())
    for method_id, qty in _QUANTITIES.items():
        assert qty in known_quantities, f"Unknown sensitivity quantity {qty!r} for {method_id!r}."


def test_execution_methods_are_known_to_registry():
    known_methods = set(METHOD_REGISTRY.keys())
    for method_id in _GROUP_BACKENDS:
        assert method_id in known_methods, f"Group backend {method_id!r} not in registry."
    for method_id in _NULL_HYPOTHESES:
        assert method_id in known_methods, (
            f"Execution null hypothesis {method_id!r} not in registry."
        )


def test_method_ids_unique():
    method_ids = [m.method_id for m in list_method_registrations()]
    assert len(method_ids) == len(set(method_ids))
    for key, reg in METHOD_REGISTRY.items():
        assert key == reg.method_id


def test_coefficient_vector_methods_use_structured_sensitivity():
    for reg in list_method_registrations():
        if reg.sensitivity_shape == "coefficient_vector":
            assert reg.method_id in {"linear_regression", "logistic_regression"}
            assert reg.sensitivity_quantity == "coefficient_vector"
            qdef = get_quantity_definition(reg.sensitivity_quantity)
            assert qdef is not None
            assert qdef.shape == "coefficient_vector"


def test_practical_significance_quantities_are_known():
    known_quantities = set(QUANTITY_REGISTRY.keys())
    for q in SUPPORTED_QUANTITIES:
        assert q in known_quantities
        qdef = get_quantity_definition(q)
        assert qdef is not None
        assert qdef.supports_practical_threshold is True
    assert set(SIGNED_QUANTITIES) <= known_quantities
    assert set(NONNEGATIVE_QUANTITIES) <= known_quantities
    for q in SIGNED_QUANTITIES:
        assert QUANTITY_REGISTRY[q].signed is True
    for q in NONNEGATIVE_QUANTITIES:
        assert QUANTITY_REGISTRY[q].signed is False


def test_registry_accessors():
    reg = get_method_registration("linear_regression")
    assert reg is not None
    assert reg.name == "Ordinary least-squares linear regression"

    qdef = get_quantity_definition("cronbach_alpha")
    assert qdef is not None
    assert qdef.domain == "reliability"

    assert get_method_registration("nonexistent") is None
    assert get_quantity_definition("nonexistent") is None
    assert len(list_method_registrations()) == len(METHOD_REGISTRY)
    assert len(list_quantity_definitions()) == len(QUANTITY_REGISTRY)
