# Create a model

If you'd like to create a new model that isn't covered by the existing DeePMD-kit library, but reuse DeePMD-kit's other efficient modules such as data processing, trainner, etc, you may want to read this section.

To incorporate your custom model you'll need to:
1. Register and implement new components (e.g. descriptor) in a Python file. You may also want to register new TensorFlow OPs if necessary.
2. Register new arguments for user inputs.
3. Package new codes into a Python package.
4. Test new models.

## Design a new component

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}
When creating a new component, take descriptor as the example, you should inherit from {py:class}`deepmd.tf.descriptor.descriptor.Descriptor` class and override several methods. Abstract methods such as {py:class}`deepmd.tf.descriptor.descriptor.Descriptor.build` must be implemented and others are not. You should keep arguments of these methods unchanged.

After implementation, you need to register the component with a key:
```py
from deepmd.tf.descriptor import Descriptor


@Descriptor.register("some_descrpt")
class SomeDescript(Descriptor):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass
```
:::

:::{tab-item} PyTorch {{ pytorch_icon }}
The PyTorch backend follows a meticulously structured inheritance pattern. When creating a new descriptor, it is essential to inherit from both the {py:class}`deepmd.pt.model.descriptor.base_descriptor.BaseDescriptor` class and the {py:class}`torch.nn.Module` class. Abstract methods, including {py:class}`deepmd.pt.model.descriptor.base_descriptor.BaseDescriptor.fwd`, must be implemented, while others remain optional. It is crucial to adhere to the original method arguments without any modifications. Once the implementation is complete, the next step involves registering the component with a designated key:

```py
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("some_descrpt")
class SomeDescript(BaseDescriptor, torch.nn.Module):
    def __init__(self, arg1: bool, arg2: float) -> None:
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

    @classmethod
    def deserialize(cls, data: dict) -> "SomeDescript":
        pass
```

The serialize and deserialize methods are important for cross-backend model conversion.


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

The model architecture within the PyTorch backend is structured with multiple layers of abstraction to provide a high degree of flexibility. Generally, the process begins with an atomic model responsible for handling atom-wise property calculations. This atomic model should inherit from the {py:class}`deepmd.pt.model.atomic_model.base_atomic_model.BaseAtomicModel` class and the {py:class}`torch.nn.Module` class.

Subsequently, the `AtomicModel` is encapsulated using the `make_model(AtomicModel)` function, employing the `deepmd.pt.model.model.make_model.make_model` function. The purpose of the `make_model` wrapper is to facilitate the translation between the original system and the extended system.

Finally, the entire model is wrapped within a `DPModel`, which must inherit from the {py:class}`deepmd.pt.model.model.model.BaseModel` class and include the aforementioned `make_model(AtomicModel)`. The user directly interacts with a wrapper built on top of the `DPModel` to seamlessly handle result translations.

```py
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.make_model import (
    make_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)


class SomeAtomicModel(BaseAtomicModel, torch.nn.Module):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass

    def forward_atomic(self):
        pass


@BaseModel.register("some_model")
class SomeDPModel(make_model(SomeAtomicModel), BaseModel):
    pass


class SomeModel(SomeDPModel):
    pass
```

:::
:::{tab-item} DPModel {{ dpmodel_icon }}

The DPModel backend is implemented using pure NumPy, serving as a reference backend for maintaining consistency in tests. The design pattern closely mirrors that of the PyTorch backend, and developers can refer to the PyTorch development guide for detailed instructions.

:::
::::
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

## Package new codes

You may use `setuptools` to package new codes into a new Python package. It's crucial to add your new component to `entry_points['deepmd']` in `setup.py`:

```py
entry_points = (
    {
        "deepmd": [
            "some_descrpt=deepmd_some_descrtpt:SomeDescript",
        ],
    },
)
```

where `deepmd_some_descrtpt` is the module of your codes. It is equivalent to `from deepmd_some_descrtpt import SomeDescript`.

If you place `SomeDescript` and `descrpt_some_args` into different modules, you are also expected to add `descrpt_some_args` to `entry_points`.

After you install your new package, you can now use `dp train` to run your new model.
