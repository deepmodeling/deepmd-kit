# Create a model in PyTorch

If you'd like to create a new model that isn't covered by the existing DeePMD-kit library, but reuse DeePMD-kit's other efficient modules such as data processing, trainner, etc, you may want to read this section.

To incorporate your custom model you'll need to:

1. Register and implement new components (e.g. descriptor) in a Python file.
2. Register new arguments for user inputs.
3. Package new codes into a Python package.
4. Test new models.

## Design a new component

With DeePMD-kit v3, we have expanded support to include two additional backends alongside TensorFlow: the PyTorch backend and the framework-independent backend (dpmodel). The PyTorch backend adopts a highly modularized design to provide flexibility and extensibility. It ensures a consistent experience for both training and inference, aligning with the TensorFlow backend.

The framework-independent backend is implemented in pure NumPy, serving as a reference backend to ensure consistency in tests. Its design pattern closely parallels that of the PyTorch backend.

### New descriptors

When creating a new descriptor, it is essential to inherit from both the {py:class}`deepmd.pt.model.descriptor.base_descriptor.BaseDescriptor` class and the {py:class}`torch.nn.Module` class. Abstract methods, including {py:class}`deepmd.pt.model.descriptor.base_descriptor.BaseDescriptor.forward`, must be implemented, while others remain optional. It is crucial to adhere to the original method arguments without any modifications. Once the implementation is complete, the next step involves registering the component with a designated key:

```py
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("some_descrpt")
class SomeDescript(BaseDescriptor, torch.nn.Module):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass

    def get_rcut(self) -> float:
        pass

    def get_nnei(self) -> int:
        pass

    def get_ntypes(self) -> int:
        pass

    def get_dim_out(self) -> int:
        pass

    def get_dim_emb(self) -> int:
        pass

    def mixed_types(self) -> bool:
        pass

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        pass

    def serialize(self) -> dict:
        pass

    def deserialize(cls, data: dict) -> "SomeDescript":
        pass

    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        pass
```

The serialize and deserialize methods are important for cross-backend model conversion.

### New fitting nets

In many instances, there is no requirement to create a new fitting net. For fitting user-defined scalar properties, the {py:class}`deepmd.pt.model.task.ener.InvarFitting` class can be utilized. However, if there is a need for a new fitting net, one should inherit from both the {py:class}`deepmd.pt.model.task.base_fitting.BaseFitting` class and the {py:class}`torch.nn.Module` class. Alternatively, for a more straightforward approach, inheritance from the {py:class}`deepmd.pt.model.task.fitting.GeneralFitting` class is also an option.

```py
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.dpmodel import (
    FittingOutputDef,
    fitting_check_output,
)


@GeneralFitting.register("some_fitting")
@fitting_check_output
class SomeFittingNet(GeneralFitting):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        pass

    def output_def(self) -> FittingOutputDef:
        pass
```

### New models

The PyTorch backend's model architecture is meticulously structured with multiple layers of abstraction, ensuring a high degree of flexibility. Typically, the process commences with an atomic model responsible for atom-wise property calculations. This atomic model inherits from both the {py:class}`deepmd.pt.model.atomic_model.base_atomic_model.BaseAtomicModel` class and the {py:class}`torch.nn.Module` class.

Subsequently, the `AtomicModel` is encapsulated using the `make_model(AtomicModel)` function, which leverages the `deepmd.pt.model.model.make_model.make_model` function. The purpose of the `make_model` wrapper is to facilitate the translation between atomic property predictions and the extended property predictions and differentiation , e.g. the reduction of atomic energy contribution and the autodiff for calculating the forces and virial. The developers usually need to implement an `AtomicModel` not a `Model`.

```py
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)


class SomeAtomicModel(BaseAtomicModel, torch.nn.Module):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass

    def forward_atomic(self):
        pass
```

## Register new arguments

To let someone uses your new component in their input file, you need to create a new method that returns some `Argument` of your new component, and then register new arguments. For example, the code below

```py
from typing import List

from dargs import Argument
from deepmd.utils.argcheck import descrpt_args_plugin


@descrpt_args_plugin.register("some_descrpt")
def descrpt_some_args() -> List[Argument]:
    return [
        Argument("arg1", bool, optional=False, doc="balabala"),
        Argument("arg2", float, optional=True, default=6.0, doc="haha"),
    ]
```

allows one to use your new descriptor as below:

```json
"descriptor" :{
    "type": "some_descrpt",
    "arg1": true,
    "arg2": 6.0
}
```

The arguments here should be consistent with the class arguments of your new component.

## Unit tests

When transferring features from another backend to the PyTorch backend, it is essential to include a regression test in `/source/tests/consistent` to validate the consistency of the PyTorch backend with other backends. Presently, the regression tests cover self-consistency and cross-backend consistency between TensorFlow, PyTorch, and DP (Numpy) through the serialization/deserialization technique.

During the development of new components within the PyTorch backend, it is necessary to provide a DP (Numpy) implementation and incorporate corresponding regression tests. For PyTorch components, developers are also required to include a unit test using `torch.jit`.
