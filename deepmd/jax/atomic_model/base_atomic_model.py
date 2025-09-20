# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.jax.common import (
    ArrayAPIVariable,
    to_jax_array,
)
from deepmd.jax.utils.exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)


def base_atomic_model_set_attr(name, value):
    if name in {"out_bias", "out_std"}:
        value = to_jax_array(value)
        if value is not None:
            value = ArrayAPIVariable(value)
    elif name == "pair_excl" and value is not None:
        value = PairExcludeMask(value.ntypes, value.exclude_types)
    elif name == "atom_excl" and value is not None:
        value = AtomExcludeMask(value.ntypes, value.exclude_types)
    return value
