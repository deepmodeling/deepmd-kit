# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_zbl_model import (
    DPZBLModel,
)
from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.dpmodel.model.model_factory import (
    get_spin_model as get_spin_model_from_factory,
)
from deepmd.dpmodel.model.spin_model import (
    SpinModel,
)

_model_factory = BackendModelFactory(
    descriptor_base=BaseDescriptor,
    fitting_base=BaseFitting,
    model_base=BaseModel,
    backend_name="DP",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
)
get_standard_model = _model_factory.get_standard_model
get_zbl_model = _model_factory.get_zbl_model


def get_spin_model(data: dict) -> SpinModel:
    """Get a spin model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    return get_spin_model_from_factory(
        data,
        standard_model_factory=get_standard_model,
        spin_model=SpinModel,
    )


def get_model(data: dict) -> BaseModel:
    """Get a model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    return _model_factory.get_model(
        data,
        spin_model_factory=get_spin_model,
    )
