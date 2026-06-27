# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
from deepmd.dpmodel.descriptor.nep import (
    DescrptNep as DescrptNepDP,
)
from deepmd.dpmodel.descriptor.nep import (
    NepEmbeddingCoeff as NepEmbeddingCoeffDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    register_dpmodel_mapping,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


@flax_module
class NepEmbeddingCoeff(NepEmbeddingCoeffDP):
    """JAX wrapper storing the NEP expansion coefficients as a flax parameter."""

    _jax_skip_auto_convert_attrs: ClassVar[set[str]] = {"coeff"}

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "coeff":
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIParam(value) if self.trainable else ArrayAPIVariable(value)
        return super().__setattr__(name, value)


register_dpmodel_mapping(
    NepEmbeddingCoeffDP,
    lambda v: NepEmbeddingCoeff.deserialize(v.serialize()),
)


@BaseDescriptor.register("nep")
@flax_module
class DescrptNep(DescrptNepDP):
    """JAX (flax) wrapper around the array-API NEP descriptor.

    The coefficient tables, exclusion mask, and statistic arrays are converted
    to their JAX counterparts automatically by ``flax_module`` (the side-effect
    imports register the required converters); no bespoke attribute handling is
    required at the descriptor level.
    """

    pass
