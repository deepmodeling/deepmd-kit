# Create a model in TensorFlow {{ tensorflow_icon }}

If you'd like to create a new model that isn't covered by the existing DeePMD-kit library, but reuse DeePMD-kit's other efficient modules such as data processing, trainer, etc, you may want to read this section.

To incorporate your custom model you'll need to:

1. Register and implement new components (e.g. descriptor) in a Python file. You may also want to register new TensorFlow OPs if necessary.
2. Register new arguments for user inputs.
3. Package new codes into a Python package.
4. Test new models.

## Design a new component

When creating a new component, take descriptor as the example, one should inherit from the {py:class}`deepmd.tf.descriptor.descriptor.Descriptor` class and override several methods. Abstract methods such as {py:class}`deepmd.tf.descriptor.descriptor.Descriptor.build` must be implemented and others are not. You should keep arguments of these methods unchanged.

After implementation, you need to register the component with a key:

```py
from deepmd.tf.descriptor import Descriptor


@Descriptor.register("some_descrpt")
class SomeDescript(Descriptor):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass
```

## Register new arguments

To let someone uses your new component in their input file, you need to create a new method that returns some `Argument` of your new component, and then register new arguments. For example, the code below

```py
from typing import List

from dargs import Argument
from deepmd.utils.argcheck import descrpt_args_plugin


@descrpt_args_plugin.register("some_descrpt")
def descrpt_some_args() -> list[Argument]:
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

You may package new codes into a new Python package if you don't want to contribute it to the main DeePMD-kit repository.
It's crucial to add your new component to `project.entry-points."deepmd"` in `pyproject.toml`:

```toml
[project.entry-points."deepmd"]
some_descrpt = "deepmd_some_descrtpt:SomeDescript"
```

where `deepmd_some_descrtpt` is the module of your codes. It is equivalent to `from deepmd_some_descrtpt import SomeDescript`.

If you place `SomeDescript` and `descrpt_some_args` into different modules, you are also expected to add `descrpt_some_args` to `entry_points`.

After you install your new package, you can now use `dp train` to run your new model.

### Package customized C++ OPs

You may need to use customized TensorFlow C++ OPs in the new model.
Follow [TensorFlow documentation](https://www.tensorflow.org/guide/create_op) to create one library.

When using your customized C++ OPs in the Python interface, use {py:meth}`tf.load_op_library` to load the OP library in the module defined in `entry_points`.

When using your customized C++ OPs in the C++ library, define the environment variable {envvar}`DP_PLUGIN_PATH` to load the OP library.
