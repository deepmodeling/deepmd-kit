# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.utils.argcheck import (
    loss_args,
)


def test_energy_hessian_loss_type_is_energy_alias() -> None:
    """The legacy Hessian loss type must normalize to the energy schema."""
    canonical = loss_args().normalize_value(
        {
            "type": "ener",
            "start_pref_h": 2.0,
            "limit_pref_h": 1.0,
        }
    )
    legacy = loss_args().normalize_value(
        {
            "type": "ener_hess",
            "start_pref_h": 2.0,
            "limit_pref_h": 1.0,
        }
    )

    loss_args().check_value(legacy, strict=True)
    assert legacy == canonical
    assert legacy["type"] == "ener"
