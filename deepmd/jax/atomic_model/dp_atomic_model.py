# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel as DPAtomicModelDP
from deepmd.jax.atomic_model.base_atomic_model import (
    base_atomic_model_set_attr,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


class DPAtomicModel(DPAtomicModelDP):
    base_descriptor_cls = BaseDescriptor
    """The base descriptor class."""
    base_fitting_cls = BaseFitting
    """The base fitting class."""

    def __setattr__(self, name: str, value: Any) -> None:
        value = base_atomic_model_set_attr(name, value)
        return super().__setattr__(name, value)
