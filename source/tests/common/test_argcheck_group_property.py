# SPDX-License-Identifier: LGPL-3.0-or-later
"""``group_reduce`` was implemented on GroupPropertyFittingNet but the
argcheck schema for the ``group_property`` fitting type reused the
``property`` schema verbatim, so dargs' strict-mode key check (the same
check ``dp --pt train`` runs on every ``input.json``) rejected
``group_reduce`` as an unknown key -- even though ``group_reduce="sum"`` is
a supported, tested code path once past validation.

The same reused schema also let a config set ``numb_aparam``,
``default_fparam``, ``dim_case_embd``, ``resnet_dt``, ``intensive``, or
``distinguish_types`` -- fields GroupPropertyFittingNet has no wiring for --
and pass strict validation while silently doing nothing. Those fields are
removed from the group_property schema so setting one is now a validation
error like any other typo, not a silent no-op.
"""

from __future__ import (
    annotations,
)

from dargs import (
    Argument,
)
from dargs.dargs import (
    ArgumentKeyError,
)

from deepmd.utils.argcheck import (
    fitting_group_property,
    fitting_variant_type_args,
)


def _fitting_net_schema() -> Argument:
    # Mirrors how model_args() builds the real "fitting_net" field: a
    # type-selected variant, not a bare argument list, so "type" is handled
    # by the Variant dispatch rather than treated as an unknown key.
    return Argument("fitting_net", dict, [], [fitting_variant_type_args()])


def _normalize_and_check(value: dict) -> dict:
    schema = _fitting_net_schema()
    normalized = schema.normalize_value(value)
    schema.check_value(normalized, strict=True)
    return normalized


def test_group_reduce_argument_is_declared():
    names = [arg.name for arg in fitting_group_property()]
    assert "group_reduce" in names


def test_group_reduce_sum_passes_strict_validation():
    out = _normalize_and_check(
        {"type": "group_property", "property_name": "y", "group_reduce": "sum"}
    )
    assert out["group_reduce"] == "sum"


def test_group_reduce_defaults_to_mean_when_omitted():
    out = _normalize_and_check({"type": "group_property", "property_name": "y"})
    assert out["group_reduce"] == "mean"


def test_group_property_still_defaults_activation_to_gelu():
    # guard against the group_reduce addition disturbing the existing
    # gelu-default override.
    out = _normalize_and_check({"type": "group_property", "property_name": "y"})
    assert out["activation_function"] == "gelu"


def test_unsupported_property_fields_are_removed_from_the_schema():
    names = {arg.name for arg in fitting_group_property()}
    for unsupported in (
        "numb_aparam",
        "default_fparam",
        "dim_case_embd",
        "resnet_dt",
        "intensive",
        "distinguish_types",
    ):
        assert unsupported not in names


def test_setting_an_unsupported_field_fails_strict_validation():
    for unsupported, value in (
        ("numb_aparam", 3),
        ("resnet_dt", False),
        ("intensive", True),
        ("distinguish_types", False),
    ):
        payload = {
            "type": "group_property",
            "property_name": "y",
            unsupported: value,
        }
        normalized = _fitting_net_schema().normalize_value(payload)
        try:
            _fitting_net_schema().check_value(normalized, strict=True)
        except ArgumentKeyError:
            pass
        else:
            raise AssertionError(f"expected {unsupported!r} to fail strict check")


def test_group_reduce_key_was_rejected_by_the_reused_property_schema():
    """Regression pin: the bug this fixes. Before adding ``group_reduce`` to
    ``fitting_group_property()``, the same strict-mode check that accepts it
    above raised ``ArgumentKeyError`` -- confirmed here against the plain
    ``property`` schema, which is exactly what ``fitting_group_property()``
    used to return unmodified (plus the activation-default override).
    """
    from deepmd.utils.argcheck import (
        fitting_property,
    )

    old_schema = Argument("fitting_net", dict, fitting_property())
    value = {"property_name": "y", "group_reduce": "sum"}
    normalized = old_schema.normalize_value(value)
    try:
        old_schema.check_value(normalized, strict=True)
    except ArgumentKeyError:
        pass
    else:
        raise AssertionError(
            "expected ArgumentKeyError: the plain property schema has no "
            "group_reduce field"
        )
