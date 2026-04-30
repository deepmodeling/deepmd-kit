# SPDX-License-Identifier: LGPL-3.0-or-later
"""Schema-drift regression test for ``_has_message_passing``.

``_has_message_passing`` (in ``deepmd/pt_expt/utils/serialization.py``)
gates whether the dual-artifact ``.pt2`` is produced for GNN models —
specifically, whether the with-comm AOTInductor module is compiled and
nested inside the archive. The detection relies on a chain of attribute
lookups:

* ``model.atomic_model.descriptor``
* ``descriptor.has_message_passing()``
* For repflows/repformers: ``block.use_loc_mapping``

A rename of any of these (refactor in the dpmodel descriptor layer, a
new GNN block name, etc.) silently disables the with-comm artifact and
multi-rank LAMMPS users get a single-artifact .pt2 that crashes on the
first ghost exchange — with no test failure to flag the breakage.

This test pins the contract: assert ``_has_message_passing`` returns
the documented value for each baseline configuration.
"""

from __future__ import (
    annotations,
)

import copy

import pytest

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.pt_expt.utils.serialization import (
    _has_message_passing,
)


def _se_e2_a_config() -> dict:
    """Non-GNN descriptor — must report False."""
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "sel": [20, 20],
            "neuron": [2, 4],
            "axis_neuron": 2,
            "type_one_side": True,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [4, 4],
            "resnet_dt": True,
            "precision": "float64",
            "seed": 1,
        },
    }


def _dpa1_config() -> dict:
    """DPA1 (se_atten) — non-GNN; must report False."""
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_atten",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "sel": 20,
            "neuron": [2, 4],
            "axis_neuron": 2,
            "attn": 5,
            "attn_layer": 1,
            "type_one_side": True,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [4, 4],
            "resnet_dt": True,
            "precision": "float64",
            "seed": 1,
        },
    }


def _dpa3_config(use_loc_mapping: bool) -> dict:
    """DPA3 (repflows). use_loc_mapping=False -> True, True -> False."""
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa3",
            "repflow": {
                "n_dim": 8,
                "e_dim": 6,
                "a_dim": 4,
                "nlayers": 1,
                "e_rcut": 4.0,
                "e_rcut_smth": 0.5,
                "e_sel": 8,
                "a_rcut": 3.5,
                "a_rcut_smth": 0.5,
                "a_sel": 4,
                "axis_neuron": 4,
                "update_angle": False,
            },
            "use_loc_mapping": use_loc_mapping,
        },
        "fitting_net": {"neuron": [16, 16], "seed": 1},
    }


def _dpa2_config() -> dict:
    """DPA2 (repformer) — GNN; repformer has no use_loc_mapping knob,
    so always reports True.
    """
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa2",
            "repinit": {
                "rcut": 6.0,
                "rcut_smth": 2.0,
                "nsel": 20,
                "neuron": [2, 4],
                "axis_neuron": 4,
                "tebd_dim": 8,
                "tebd_input_mode": "concat",
                "set_davg_zero": True,
                "type_one_side": True,
                "use_three_body": False,
            },
            "repformer": {
                "rcut": 3.0,
                "rcut_smth": 1.5,
                "nsel": 10,
                "nlayers": 1,
                "g1_dim": 8,
                "g2_dim": 5,
                "axis_neuron": 4,
                "update_g1_has_conv": True,
                "update_g1_has_drrd": True,
                "update_g1_has_grrg": True,
                "update_g2_has_attn": True,
                "attn1_hidden": 8,
                "attn1_nhead": 2,
                "attn2_hidden": 5,
                "attn2_nhead": 1,
                "update_style": "res_avg",
                "set_davg_zero": True,
            },
            "concat_output_tebd": True,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [4, 4],
            "resnet_dt": True,
            "seed": 1,
        },
    }


@pytest.mark.parametrize(
    "config_factory,expected",
    [
        (_se_e2_a_config, False),
        (_dpa1_config, False),
        (lambda: _dpa3_config(use_loc_mapping=True), False),
        (lambda: _dpa3_config(use_loc_mapping=False), True),
        (_dpa2_config, True),
    ],
    ids=[
        "se_e2_a-non-gnn",
        "dpa1-non-gnn",
        "dpa3-use-loc-mapping-true",
        "dpa3-use-loc-mapping-false",
        "dpa2-repformer",
    ],
)
def test_has_message_passing_matches_descriptor_kind(config_factory, expected) -> None:
    """``_has_message_passing`` must report the documented value for
    each baseline descriptor configuration.

    A False positive (non-GNN reported as GNN) wastes compile time on
    a useless with-comm artifact. A False negative (GNN with
    use_loc_mapping=False reported as non-GNN) is worse: multi-rank
    LAMMPS gets a single-artifact .pt2 and crashes on the first ghost
    exchange. This test pins both directions.
    """
    config = config_factory()
    model = get_model(copy.deepcopy(config))
    assert _has_message_passing(model) is expected


def test_has_message_passing_no_descriptor_returns_false() -> None:
    """Models without a single ``atomic_model.descriptor`` (e.g. linear
    / ZBL / frozen) must report False — the function defends against
    AttributeError and treats the model as local.
    """

    class _StubAtomicModel:
        # Intentionally no ``descriptor`` attribute.
        pass

    class _StubModel:
        atomic_model = _StubAtomicModel()

    assert _has_message_passing(_StubModel()) is False


def test_has_message_passing_descriptor_without_query_returns_false() -> None:
    """If the descriptor exists but lacks ``has_message_passing``, the
    function must report False rather than raise.
    """

    class _StubDescriptor:
        # Intentionally no ``has_message_passing`` method.
        pass

    class _StubAtomicModel:
        descriptor = _StubDescriptor()

    class _StubModel:
        atomic_model = _StubAtomicModel()

    assert _has_message_passing(_StubModel()) is False
