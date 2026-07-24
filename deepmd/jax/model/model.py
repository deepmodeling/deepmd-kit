# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_zbl_model import (
    DPZBLModel,
)

_model_factory = BackendModelFactory(
    descriptor_base=BaseDescriptor,
    fitting_base=BaseFitting,
    model_base=BaseModel,
    backend_name="JAX",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
)
get_standard_model = _model_factory.get_standard_model
get_zbl_model = _model_factory.get_zbl_model
get_model = _model_factory.get_model
