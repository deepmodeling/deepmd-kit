# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import numpy as np
import tensorflow as tf

from deepmd._vendors import ndtensorflow as xp
from deepmd.tf2.common import (
    unwrap_value,
)
from deepmd.tf2.make_model import (
    model_call_from_call_lower,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.utils._dpmodel import (
    format_nlist,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _default_jit_compile() -> bool:
    return _env_flag("DP_JIT")


class _ExportConstantArray:
    """Array-like export constant that traces as ``tf.constant``."""

    __array_priority__ = 2

    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value)

    @property
    def dtype(self) -> tf.DType:
        return tf.as_dtype(self._value.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape

    @property
    def ndim(self) -> int:
        return self._value.ndim

    @property
    def size(self) -> int:
        return self._value.size

    @property
    def T(self) -> Any:
        return self._array().T

    @property
    def mT(self) -> Any:
        return self._array().mT

    def _array(self) -> Any:
        return xp.asarray(tf.constant(self._value))

    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any:
        del api_version
        return xp

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        return self._value.astype(dtype) if dtype is not None else self._value

    def __tf_tensor__(
        self,
        dtype: tf.DType | None = None,
        name: str | None = None,
    ) -> tf.Tensor:
        return tf.constant(self._value, dtype=dtype, name=name)

    def __getitem__(self, key: Any, /) -> Any:
        return self._array()[key]

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self) -> Any:
        return iter(self._array())

    def __bool__(self, /) -> bool:
        return bool(self._value.item())

    def __complex__(self, /) -> complex:
        return complex(self._value.item())

    def __float__(self, /) -> float:
        return float(self._value.item())

    def __index__(self, /) -> int:
        return int(self._value.item())

    def __int__(self, /) -> int:
        return int(self._value.item())

    def __repr__(self) -> str:
        return f"_ExportConstantArray({self._value!r})"

    def astype(
        self,
        dtype: tf.DType,
        /,
        *,
        copy: bool = True,
        device: str | None = None,
    ) -> Any:
        return self._array().astype(dtype, copy=copy, device=device)

    def reshape(self, *shape: Any, copy: bool | None = None) -> Any:
        return self._array().reshape(*shape, copy=copy)

    def ravel(self) -> Any:
        return self._array().ravel()

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Any:
        return self._array().squeeze(axis=axis)

    def to_device(
        self,
        device: str,
        /,
        *,
        stream: int | Any | None = None,
    ) -> Any:
        return self._array().to_device(device, stream=stream)

    def unwrap(self) -> tf.Tensor:
        return tf.constant(self._value)


def _export_binary_forward(name: str) -> Callable[[Any, Any], Any]:
    def method(self: _ExportConstantArray, other: Any, /) -> Any:
        return getattr(xp, name)(self._array(), other)

    return method


def _export_binary_reflected(name: str) -> Callable[[Any, Any], Any]:
    def method(self: _ExportConstantArray, other: Any, /) -> Any:
        return getattr(xp, name)(other, self._array())

    return method


def _export_unary(name: str) -> Callable[[Any], Any]:
    def method(self: _ExportConstantArray) -> Any:
        return getattr(xp, name)(self._array())

    return method


_ExportConstantArray.__add__ = _export_binary_forward("add")  # type: ignore[attr-defined]
_ExportConstantArray.__radd__ = _export_binary_reflected("add")  # type: ignore[attr-defined]
_ExportConstantArray.__and__ = _export_binary_forward("bitwise_and")  # type: ignore[attr-defined]
_ExportConstantArray.__rand__ = _export_binary_reflected("bitwise_and")  # type: ignore[attr-defined]
_ExportConstantArray.__floordiv__ = _export_binary_forward("floor_divide")  # type: ignore[attr-defined]
_ExportConstantArray.__rfloordiv__ = _export_binary_reflected("floor_divide")  # type: ignore[attr-defined]
_ExportConstantArray.__ge__ = _export_binary_forward("greater_equal")  # type: ignore[attr-defined]
_ExportConstantArray.__le__ = _export_binary_reflected("greater_equal")  # type: ignore[attr-defined]
_ExportConstantArray.__gt__ = _export_binary_forward("greater")  # type: ignore[attr-defined]
_ExportConstantArray.__lt__ = _export_binary_reflected("greater")  # type: ignore[attr-defined]
_ExportConstantArray.__lshift__ = _export_binary_forward("bitwise_left_shift")  # type: ignore[attr-defined]
_ExportConstantArray.__rlshift__ = _export_binary_reflected("bitwise_left_shift")  # type: ignore[attr-defined]
_ExportConstantArray.__matmul__ = _export_binary_forward("matmul")  # type: ignore[attr-defined]
_ExportConstantArray.__rmatmul__ = _export_binary_reflected("matmul")  # type: ignore[attr-defined]
_ExportConstantArray.__mod__ = _export_binary_forward("remainder")  # type: ignore[attr-defined]
_ExportConstantArray.__rmod__ = _export_binary_reflected("remainder")  # type: ignore[attr-defined]
_ExportConstantArray.__mul__ = _export_binary_forward("multiply")  # type: ignore[attr-defined]
_ExportConstantArray.__rmul__ = _export_binary_reflected("multiply")  # type: ignore[attr-defined]
_ExportConstantArray.__or__ = _export_binary_forward("bitwise_or")  # type: ignore[attr-defined]
_ExportConstantArray.__ror__ = _export_binary_reflected("bitwise_or")  # type: ignore[attr-defined]
_ExportConstantArray.__pow__ = _export_binary_forward("pow")  # type: ignore[attr-defined]
_ExportConstantArray.__rpow__ = _export_binary_reflected("pow")  # type: ignore[attr-defined]
_ExportConstantArray.__rshift__ = _export_binary_forward("bitwise_right_shift")  # type: ignore[attr-defined]
_ExportConstantArray.__rrshift__ = _export_binary_reflected("bitwise_right_shift")  # type: ignore[attr-defined]
_ExportConstantArray.__sub__ = _export_binary_forward("subtract")  # type: ignore[attr-defined]
_ExportConstantArray.__rsub__ = _export_binary_reflected("subtract")  # type: ignore[attr-defined]
_ExportConstantArray.__truediv__ = _export_binary_forward("divide")  # type: ignore[attr-defined]
_ExportConstantArray.__rtruediv__ = _export_binary_reflected("divide")  # type: ignore[attr-defined]
_ExportConstantArray.__xor__ = _export_binary_forward("bitwise_xor")  # type: ignore[attr-defined]
_ExportConstantArray.__rxor__ = _export_binary_reflected("bitwise_xor")  # type: ignore[attr-defined]
_ExportConstantArray.__eq__ = _export_binary_forward("equal")  # type: ignore[attr-defined]
_ExportConstantArray.__ne__ = _export_binary_forward("not_equal")  # type: ignore[attr-defined]
_ExportConstantArray.__abs__ = _export_unary("abs")  # type: ignore[attr-defined]
_ExportConstantArray.__invert__ = _export_unary("bitwise_invert")  # type: ignore[attr-defined]
_ExportConstantArray.__neg__ = _export_unary("negative")  # type: ignore[attr-defined]
_ExportConstantArray.__pos__ = _export_unary("positive")  # type: ignore[attr-defined]


def _as_export_constant(value: Any) -> _ExportConstantArray:
    if isinstance(value, xp.Array):
        value = value.unwrap()
    if isinstance(value, tf.Variable):
        value = value.numpy()
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    return _ExportConstantArray(value)


def _freeze_tf2_constants_for_export(value: Any, seen: set[int]) -> Any:
    """Freeze TensorFlow state inside deepmd objects to export constants."""
    if isinstance(value, (xp.Array, tf.Variable, tf.Tensor)):
        return _as_export_constant(value)
    if isinstance(value, list):
        for ii, item in enumerate(value):
            value[ii] = _freeze_tf2_constants_for_export(item, seen)
        return value
    if isinstance(value, dict):
        for kk, item in list(value.items()):
            value[kk] = _freeze_tf2_constants_for_export(item, seen)
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_tf2_constants_for_export(item, seen) for item in value)
    if not hasattr(value, "__dict__"):
        return value
    if not type(value).__module__.startswith("deepmd."):
        return value

    oid = id(value)
    if oid in seen:
        return value
    seen.add(oid)
    for name, item in list(value.__dict__.items()):
        frozen = _freeze_tf2_constants_for_export(item, seen)
        if frozen is not item:
            object.__setattr__(value, name, frozen)
    return value


def _freeze_tf2_variables_for_export(model: tf.Module) -> None:
    """Freeze TF2 trainable state to constants for C++ SavedModel inference."""
    _freeze_tf2_constants_for_export(model, set())


def deserialize_to_file(
    model_file: str, data: dict, *, jit_compile: bool | None = None
) -> None:
    """Deserialize the dictionary to a TensorFlow SavedModel."""
    if not model_file.endswith(".savedmodeltf"):
        raise ValueError("TF2 backend only supports the .savedmodeltf extension")
    return deserialize_to_savedmodel(model_file, data, jit_compile=jit_compile)


def deserialize_to_savedmodel(
    model_file: str, data: dict, *, jit_compile: bool | None = None
) -> None:
    """Deserialize the dictionary to a TensorFlow SavedModel directory."""
    if jit_compile is None:
        jit_compile = _default_jit_compile()

    # Import model registrations before deserializing the dpmodel payload.
    import deepmd.tf2.model.model  # noqa: F401

    model = BaseModel.deserialize(data["model"])
    _freeze_tf2_variables_for_export(model)
    model_def_script = data["model_def_script"]

    tf_model = tf.Module()

    def call_lower_with_fixed_do_atomic_virial(
        do_atomic_virial: bool,
    ) -> Callable:
        def call_lower(
            coord: tf.Tensor,
            atype: tf.Tensor,
            nlist: tf.Tensor,
            mapping: tf.Tensor,
            fparam: tf.Tensor,
            aparam: tf.Tensor,
        ) -> dict[str, tf.Tensor]:
            return unwrap_value(
                model.call_common_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=do_atomic_virial,
                )
            )

        return call_lower

    @tf.function(
        autograph=True,
        jit_compile=jit_compile,
        input_signature=[
            tf.TensorSpec([None, None, 3], tf.float64),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, None], tf.int64),
            tf.TensorSpec([None, None], tf.int64),
            tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
            tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
        ],
    )
    def call_lower_without_atomic_virial(
        coord: tf.Tensor,
        atype: tf.Tensor,
        nlist: tf.Tensor,
        mapping: tf.Tensor,
        fparam: tf.Tensor,
        aparam: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        nlist = format_nlist(coord, nlist, model.get_nnei(), model.get_rcut())
        return call_lower_with_fixed_do_atomic_virial(False)(
            coord, atype, nlist, mapping, fparam, aparam
        )

    tf_model.call_lower = call_lower_without_atomic_virial

    @tf.function(
        autograph=True,
        jit_compile=jit_compile,
        input_signature=[
            tf.TensorSpec([None, None, 3], tf.float64),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, None], tf.int64),
            tf.TensorSpec([None, None], tf.int64),
            tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
            tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
        ],
    )
    def call_lower_with_atomic_virial(
        coord: tf.Tensor,
        atype: tf.Tensor,
        nlist: tf.Tensor,
        mapping: tf.Tensor,
        fparam: tf.Tensor,
        aparam: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        nlist = format_nlist(coord, nlist, model.get_nnei(), model.get_rcut())
        return call_lower_with_fixed_do_atomic_virial(True)(
            coord, atype, nlist, mapping, fparam, aparam
        )

    tf_model.call_lower_atomic_virial = call_lower_with_atomic_virial

    def make_call_whether_do_atomic_virial(do_atomic_virial: bool) -> Callable:
        call_lower = (
            call_lower_with_atomic_virial
            if do_atomic_virial
            else call_lower_without_atomic_virial
        )

        def call(
            coord: tf.Tensor,
            atype: tf.Tensor,
            box: tf.Tensor,
            fparam: tf.Tensor,
            aparam: tf.Tensor,
        ) -> dict[str, tf.Tensor]:
            return unwrap_value(
                model_call_from_call_lower(
                    call_lower=call_lower,
                    rcut=model.get_rcut(),
                    sel=model.get_sel(),
                    mixed_types=model.mixed_types(),
                    model_output_def=model.model_output_def(),
                    coord=coord,
                    atype=atype,
                    box=box,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                )
            )

        return call

    @tf.function(
        autograph=True,
        input_signature=[
            tf.TensorSpec([None, None, 3], tf.float64),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, None], tf.float64),
            tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
            tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
        ],
    )
    def call_with_atomic_virial(
        coord: tf.Tensor,
        atype: tf.Tensor,
        box: tf.Tensor,
        fparam: tf.Tensor,
        aparam: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return make_call_whether_do_atomic_virial(True)(
            coord, atype, box, fparam, aparam
        )

    tf_model.call_atomic_virial = call_with_atomic_virial

    @tf.function(
        autograph=True,
        input_signature=[
            tf.TensorSpec([None, None, 3], tf.float64),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, None], tf.float64),
            tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
            tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
        ],
    )
    def call_without_atomic_virial(
        coord: tf.Tensor,
        atype: tf.Tensor,
        box: tf.Tensor,
        fparam: tf.Tensor,
        aparam: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return make_call_whether_do_atomic_virial(False)(
            coord, atype, box, fparam, aparam
        )

    tf_model.call = call_without_atomic_virial

    @tf.function
    def get_type_map() -> tf.Tensor:
        return tf.constant(model.get_type_map(), dtype=tf.string)

    tf_model.get_type_map = get_type_map

    @tf.function
    def get_rcut() -> tf.Tensor:
        return tf.constant(model.get_rcut(), dtype=tf.double)

    tf_model.get_rcut = get_rcut

    @tf.function
    def get_dim_fparam() -> tf.Tensor:
        return tf.constant(model.get_dim_fparam(), dtype=tf.int64)

    tf_model.get_dim_fparam = get_dim_fparam

    @tf.function
    def get_dim_aparam() -> tf.Tensor:
        return tf.constant(model.get_dim_aparam(), dtype=tf.int64)

    tf_model.get_dim_aparam = get_dim_aparam

    @tf.function
    def get_sel_type() -> tf.Tensor:
        return tf.constant(model.get_sel_type(), dtype=tf.int64)

    tf_model.get_sel_type = get_sel_type

    @tf.function
    def is_aparam_nall() -> tf.Tensor:
        return tf.constant(model.is_aparam_nall(), dtype=tf.bool)

    tf_model.is_aparam_nall = is_aparam_nall

    @tf.function
    def model_output_type() -> tf.Tensor:
        return tf.constant(model.model_output_type(), dtype=tf.string)

    tf_model.model_output_type = model_output_type

    @tf.function
    def mixed_types() -> tf.Tensor:
        return tf.constant(model.mixed_types(), dtype=tf.bool)

    tf_model.mixed_types = mixed_types

    if model.get_min_nbor_dist() is not None:

        @tf.function
        def get_min_nbor_dist() -> tf.Tensor:
            return tf.constant(model.get_min_nbor_dist(), dtype=tf.double)

        tf_model.get_min_nbor_dist = get_min_nbor_dist

    @tf.function
    def get_sel() -> tf.Tensor:
        return tf.constant(model.get_sel(), dtype=tf.int64)

    tf_model.get_sel = get_sel

    @tf.function
    def get_model_def_script() -> tf.Tensor:
        return tf.constant(
            json.dumps(model_def_script, separators=(",", ":")), dtype=tf.string
        )

    tf_model.get_model_def_script = get_model_def_script

    @tf.function
    def has_message_passing() -> tf.Tensor:
        return tf.constant(model.has_message_passing(), dtype=tf.bool)

    tf_model.has_message_passing = has_message_passing
    tf_model.do_message_passing = has_message_passing

    @tf.function
    def has_default_fparam() -> tf.Tensor:
        return tf.constant(model.has_default_fparam(), dtype=tf.bool)

    tf_model.has_default_fparam = has_default_fparam

    @tf.function
    def get_default_fparam() -> tf.Tensor:
        default_fparam = model.get_default_fparam()
        if default_fparam is None:
            return tf.constant([], dtype=tf.double)
        return tf.constant(default_fparam, dtype=tf.double)

    tf_model.get_default_fparam = get_default_fparam

    # property models: persist the output name/dimension/intensiveness so the
    # evaluator can dispatch to DeepProperty and reshape the output.
    if hasattr(model, "get_var_name"):

        @tf.function
        def get_var_name() -> tf.Tensor:
            return tf.constant(model.get_var_name(), dtype=tf.string)

        tf_model.get_var_name = get_var_name

        @tf.function
        def get_task_dim() -> tf.Tensor:
            return tf.constant(model.get_task_dim(), dtype=tf.int64)

        tf_model.get_task_dim = get_task_dim

        @tf.function
        def get_intensive() -> tf.Tensor:
            return tf.constant(model.get_intensive(), dtype=tf.bool)

        tf_model.get_intensive = get_intensive

    tf.saved_model.save(
        tf_model,
        model_file,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
    )


def serialize_from_file(model_file: str) -> dict:
    """Serialize a TF2 SavedModel to a dictionary.

    SavedModel does not currently carry enough structured variable metadata to
    round-trip back to the DeePMD dictionary format.
    """
    raise ValueError(f"TF2 backend cannot serialize {model_file!r} to a model dict")
