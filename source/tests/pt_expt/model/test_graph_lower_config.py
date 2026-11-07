# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pre-construction graph capability agrees with runtime model routing."""

import pytest

from deepmd.pt_expt.model.graph_lower import (
    model_config_skips_neighbor_stat,
    model_uses_graph_lower,
)
from deepmd.pt_expt.model.get_model import get_model


@pytest.mark.parametrize("descriptor_type", ["dpa4", "dpa4c"])
def test_graph_config_matches_constructed_model(descriptor_type):
    descriptor = {
        "type": descriptor_type,
        "rcut": 3.0,
        "channels": 8,
        "n_radial": 4,
        "lmax": 2,
        "seed": 1,
    }
    if descriptor_type == "dpa4":
        descriptor.update(sel=20, n_blocks=1, mmax=1)
    config = {
        "type_map": ["H"],
        "descriptor": descriptor,
        "fitting_net": {"neuron": [8]},
    }
    assert model_config_skips_neighbor_stat(config)
    assert model_uses_graph_lower(get_model(config))


@pytest.mark.parametrize(
    "model,expected",
    [
        ({"descriptor": {"type": "dpa1", "sel": "auto:1.2"}}, False),
        (
            {"descriptor": {"type": "dpa1", "sel": 8}, "fitting_net": {"type": "dos"}},
            False,
        ),
        (
            {"descriptor": {"type": "dpa1", "sel": 8}, "spin": {"use_spin": [True]}},
            False,
        ),
        ({"descriptor": {"type": "dpa4"}, "spin": {"scheme": "native"}}, True),
        ({"descriptor": {"type": "dpa4c"}}, True),
        ({"type": "dpa4"}, True),
        ({"descriptor": {"type": "dpa1", "sel": 8}}, False),
        ({"descriptor": {"type": "dpa2"}}, False),
        ({"descriptor": {"type": "dpa4", "sel": "auto"}}, True),
        ({"descriptor": {"type": "dpa4"}, "fitting_net": {"type": "dos"}}, False),
        ({"type": "frozen"}, False),
        ({"descriptor": {"type": "dpa1", "sel": 8}, "use_srtab": "table.txt"}, False),
        (
            {
                "type": "linear_ener",
                "models": [
                    {"descriptor": {"type": "dpa4"}},
                    {"type": "inner_potential"},
                ],
            },
            True,
        ),
        (
            {
                "type": "linear_ener",
                "models": [{"descriptor": {"type": "dpa4"}}, {"type": "pairtab"}],
            },
            False,
        ),
        ({"type": "linear_ener", "models": []}, False),
    ],
)
def test_graph_config_conservative_boundaries(model, expected):
    assert model_config_skips_neighbor_stat(model) is expected


@pytest.mark.parametrize("sel", ["auto", "auto:1.5", 32, [16, 16]])
def test_stat_free_selection_uses_schema_default_without_mutation(sel):
    from deepmd.pt_expt.model.graph_lower import resolve_stat_free_selections
    from deepmd.utils.argcheck import DEFAULT_DPA4_SEL

    config = {"descriptor": {"type": "dpa4", "sel": sel}}
    actual = resolve_stat_free_selections(config)
    assert actual["descriptor"]["sel"] == (
        DEFAULT_DPA4_SEL if isinstance(sel, str) else sel
    )
    assert config["descriptor"]["sel"] == sel


@pytest.mark.parametrize("sel", ["invalid", "auto:bad", "auto:1:2"])
def test_stat_free_selection_rejects_invalid_auto(sel):
    from deepmd.pt_expt.model.graph_lower import resolve_stat_free_selections

    with pytest.raises((ValueError, RuntimeError)):
        resolve_stat_free_selections({"descriptor": {"type": "dpa4", "sel": sel}})
