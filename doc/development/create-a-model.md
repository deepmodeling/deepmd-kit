# Create a model

If you'd like to create a new model that isn't covered by the existing DeePMD-kit library, but reuse DeePMD-kit's other efficient module such as data processing, trainner, etc, you may want to read this section.

To incorporate your custom model you'll need to:
1. Register and implement new components (e.g. descriptor) in a Python file. You may also want to regiester new TensorFlow OPs if necessary.
2. Register new arguments for user inputs.
3. Package new codes into a Python package.
4. Test new models.

## Design a new component

When creating a new component, take descriptor as the example, you should inherit {py:class}`deepmd.descriptor.descriptor.Descriptor` class and override several methods. Abstract methods such as {py:class}`deepmd.descriptor.descriptor.Descriptor.build` must be implemented and others are not. You should keep arguments of these methods unchanged.

After implementation, you need to register the component with a key:
```py
from deepmd.descriptor import Descriptor

@Descriptor.register("some_descrpt")
class SomeDescript(Descriptor):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass
```

## Register new arguments

To let some one uses your new component in their input file, you need to create a new methods that returns some `Argument` of your new component, and then register new arguments. For example, the code below

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

The arguments here should be consistent with the class arguments of your new componenet.

## Package new codes

You may use `setuptools` to package new codes into a new Python package. It's crirical to add your new component to `entry_points['deepmd']` in `setup.py`:

```py
    entry_points={
        'deepmd': [
            'some_descrpt=deepmd_some_descrtpt:SomeDescript',
        ],
    },
```

where `deepmd_some_descrtpt` is the module of your codes. It is equivalent to `from deepmd_some_descrtpt import SomeDescript`.

If you place `SomeDescript` and `descrpt_some_args` into different modules, you are also expected to add `descrpt_some_args` to `entry_points`.

After you install your new package, you can now use `dp train` to run your new model.
