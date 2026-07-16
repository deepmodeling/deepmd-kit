# SPDX-License-Identifier: LGPL-3.0-or-later

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from functools import (
    wraps,
)
from importlib import (
    import_module,
)
from typing import (
    Any,
    TypeVar,
)

import numpy as np
import tensorflow as tf

from deepmd._vendors import ndtensorflow as xp
from deepmd.dpmodel.common import (
    NativeOP,
)


def to_tensorflow_array(array: Any | None) -> Any:
    """Convert an object to an ndtensorflow Array.

    Parameters
    ----------
    array
        The object to convert.

    Returns
    -------
    ndtensorflow.Array
        The TensorFlow-backed array.
    """
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        return xp.asarray(tf.convert_to_tensor(array))
    return xp.asarray(array)


def to_tf_tensor(array: Any | None) -> tf.Tensor | None:
    """Unwrap a TensorFlow-backed Array to a TensorFlow tensor."""
    if array is None:
        return None
    if isinstance(array, xp.Array):
        return array.unwrap()
    if isinstance(array, tf.Tensor):
        return array
    return tf.convert_to_tensor(array)


def wrap_tensor(tensor: Any | None) -> Any | None:
    """Wrap a TensorFlow tensor as an ndtensorflow Array."""
    if tensor is None:
        return None
    return xp.asarray(tensor)


def wrap_value(value: Any) -> Any:
    """Recursively wrap TensorFlow tensors as ndtensorflow Arrays."""
    if isinstance(value, dict):
        return {kk: wrap_value(vv) for kk, vv in value.items()}
    if isinstance(value, tuple):
        return tuple(wrap_value(vv) for vv in value)
    if isinstance(value, list):
        return [wrap_value(vv) for vv in value]
    return wrap_tensor(value)


def unwrap_value(value: Any) -> Any:
    """Recursively unwrap ndtensorflow Arrays for TensorFlow SavedModel returns."""
    if isinstance(value, xp.Array):
        return value.unwrap()
    if isinstance(value, dict):
        return {kk: unwrap_value(vv) for kk, vv in value.items()}
    if isinstance(value, tuple):
        return tuple(unwrap_value(vv) for vv in value)
    if isinstance(value, list):
        return [unwrap_value(vv) for vv in value]
    return value


_PACKAGE_ROOT = __name__.rsplit(".", 1)[0]
_DPMODEL_TO_TF2: dict[type[Any], Callable[[Any], Any]] = {}
_AUTO_WRAPPED_CLASSES: dict[type[NativeOP], type[Any]] = {}
_REGISTRATIONS_READY = False
_REGISTRATIONS_IN_PROGRESS = False
_REGISTRATION_MODULES = (
    f"{_PACKAGE_ROOT}.utils.network",
    f"{_PACKAGE_ROOT}.utils.exclude_mask",
    f"{_PACKAGE_ROOT}.utils.type_embed",
    f"{_PACKAGE_ROOT}.descriptor.dpa1",
    f"{_PACKAGE_ROOT}.descriptor.se_atten_v2",
    f"{_PACKAGE_ROOT}.descriptor.se_e2_a",
    f"{_PACKAGE_ROOT}.descriptor.se_e2_r",
    f"{_PACKAGE_ROOT}.descriptor.se_t",
    f"{_PACKAGE_ROOT}.descriptor.se_t_tebd",
    f"{_PACKAGE_ROOT}.descriptor.repformers",
    f"{_PACKAGE_ROOT}.descriptor.dpa2",
    f"{_PACKAGE_ROOT}.descriptor.repflows",
    f"{_PACKAGE_ROOT}.descriptor.dpa3",
    f"{_PACKAGE_ROOT}.descriptor.dpa4",
    f"{_PACKAGE_ROOT}.descriptor.hybrid",
    f"{_PACKAGE_ROOT}.fitting",
    f"{_PACKAGE_ROOT}.atomic_model.dp_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.energy_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.dipole_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.dos_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.polar_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.property_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.pairtab_atomic_model",
    f"{_PACKAGE_ROOT}.atomic_model.linear_atomic_model",
    f"{_PACKAGE_ROOT}.model",
)


class TF2List(list):
    def append(self, item: Any) -> None:
        return super().append(convert_tf2_value(item))

    def extend(self, items: list[Any]) -> None:
        return super().extend(convert_tf2_value(item) for item in items)

    def insert(self, index: int, item: Any) -> None:
        return super().insert(index, convert_tf2_value(item))

    def __setitem__(self, index: Any, item: Any) -> None:
        if isinstance(index, slice):
            item = [convert_tf2_value(ii) for ii in item]
        else:
            item = convert_tf2_value(item)
        return super().__setitem__(index, item)


def register_dpmodel_mapping(
    dpmodel_cls: type[Any], converter: Callable[[Any], Any]
) -> None:
    """Register how to convert a dpmodel object to its tf2 wrapper."""
    _DPMODEL_TO_TF2[dpmodel_cls] = converter


def _looks_like_dpmodel_class(cls: type[Any]) -> bool:
    module = cls.__module__
    return module == "deepmd.dpmodel" or module.startswith("deepmd.dpmodel.")


def _looks_like_dpmodel_object(value: Any) -> bool:
    return _looks_like_dpmodel_class(type(value))


def _looks_like_tf2_object(value: Any) -> bool:
    module = type(value).__module__
    return module == _PACKAGE_ROOT or module.startswith(f"{_PACKAGE_ROOT}.")


def _ensure_registrations() -> None:
    global _REGISTRATIONS_IN_PROGRESS, _REGISTRATIONS_READY

    if _REGISTRATIONS_READY or _REGISTRATIONS_IN_PROGRESS:
        return

    _REGISTRATIONS_IN_PROGRESS = True
    try:
        for module in _REGISTRATION_MODULES:
            import_module(module)
        _REGISTRATIONS_READY = True
    finally:
        _REGISTRATIONS_IN_PROGRESS = False


def try_convert_module(value: Any) -> Any | None:
    """Convert a registered dpmodel object to its tf2 wrapper."""
    if _looks_like_tf2_object(value):
        return None
    converter = _DPMODEL_TO_TF2.get(type(value))
    if converter is not None:
        return converter(value)
    if _looks_like_dpmodel_object(value):
        _ensure_registrations()
        converter = _DPMODEL_TO_TF2.get(type(value))
        if converter is not None:
            return converter(value)
    if isinstance(value, NativeOP):
        return _auto_wrap_native_op(value)
    return None


def _auto_wrap_native_op(value: NativeOP) -> Any:
    cls = type(value)
    if cls not in _AUTO_WRAPPED_CLASSES:
        wrapped_cls = type(
            cls.__name__,
            (cls,),
            {
                "__module__": __name__,
                "__qualname__": cls.__qualname__,
            },
        )
        _AUTO_WRAPPED_CLASSES[cls] = tf2_module(wrapped_cls)
    wrapped_cls = _AUTO_WRAPPED_CLASSES[cls]
    if not (hasattr(value, "serialize") and hasattr(wrapped_cls, "deserialize")):
        raise TypeError(
            f"Cannot auto-wrap {cls.__name__}: "
            "it must implement serialize()/deserialize() or be explicitly "
            "registered via register_dpmodel_mapping()."
        )
    return wrapped_cls.deserialize(value.serialize())


def _try_convert_list(value: list[Any], *, keep_converting: bool = False) -> list[Any]:
    converted = TF2List() if keep_converting else []
    changed = keep_converting
    for item in value:
        converted_item = convert_tf2_value(item)
        converted.append(converted_item)
        changed = changed or converted_item is not item
    return converted if changed else value


def convert_tf2_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_tensorflow_array(value)

    if isinstance(value, list):
        return _try_convert_list(value)

    converted = try_convert_module(value)
    if converted is not None:
        return converted

    return value


def tf2_setattr(obj: Any, name: str, value: Any) -> Any:
    if name in getattr(obj, "_tf2_skip_auto_convert_attrs", ()):
        return value

    if isinstance(value, list) and name in getattr(obj, "_tf2_data_list_attrs", ()):
        return _try_convert_list(value, keep_converting=True)

    return convert_tf2_value(value)


T = TypeVar("T")


def tf2_module(module: type[T]) -> type[T]:
    """Wrap a dpmodel subclass as a TensorFlow ``tf.Module``."""

    @wraps(module, updated=())
    class TF2Module(module, tf.Module):  # type: ignore[misc, valid-type]
        @staticmethod
        def _tf2_array_variable_storage_name(name: str) -> str:
            return f"_tf2_{name}_variable"

        @staticmethod
        def _tf2_array_variable_list_storage_name(name: str) -> str:
            return f"_tf2_{name}_variables"

        def _tf2_array_variable_attr_names(self) -> set[str]:
            return set(getattr(self, "_tf2_array_variable_attrs", ()))

        def _tf2_array_variable_list_attr_names(self) -> set[str]:
            return set(getattr(self, "_tf2_array_variable_list_attrs", ()))

        def _set_tf2_array_variable(self, name: str, value: Any) -> None:
            storage_name = self._tf2_array_variable_storage_name(name)
            trainable_by_name = object.__getattribute__(self, "__dict__").get(
                "_tf2_array_variable_trainable", {}
            )
            if value is None:
                tf.Module.__setattr__(self, storage_name, None)
                object.__getattribute__(self, "__dict__").pop(name, None)
                return
            tensor = to_tf_tensor(value)
            variable = tf.Variable(
                tensor,
                trainable=bool(
                    trainable_by_name.get(
                        name,
                        getattr(self, "trainable", True),
                    )
                ),
                name=name,
            )
            tf.Module.__setattr__(self, storage_name, variable)
            # The variable-backed accessor owns this value now. Keeping the
            # original eager tensor in the public slot doubles parameter RAM.
            object.__getattribute__(self, "__dict__").pop(name, None)

        def _set_tf2_array_variable_list(self, name: str, value: Any) -> None:
            storage_name = self._tf2_array_variable_list_storage_name(name)
            trainable_by_name = object.__getattribute__(self, "__dict__").get(
                "_tf2_array_variable_list_trainable", {}
            )
            variables = []
            for idx, item in enumerate(value):
                tensor = to_tf_tensor(item)
                variables.append(
                    tf.Variable(
                        tensor,
                        trainable=bool(
                            trainable_by_name.get(
                                name,
                                getattr(self, "trainable", True),
                            )
                        ),
                        name=f"{name}_{idx}",
                    )
                )
            tf.Module.__setattr__(self, storage_name, variables)
            object.__getattribute__(self, "__dict__").pop(name, None)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            tf.Module.__init__(self)
            super().__init__(*args, **kwargs)
            for name in list(self.__dict__):
                value = self.__dict__[name]
                if isinstance(value, list):
                    converted = _try_convert_list(
                        value,
                        keep_converting=name
                        in getattr(self, "_tf2_data_list_attrs", ()),
                    )
                    if converted is not value:
                        setattr(self, name, converted)
            self._refresh_tf2_trackable_lists()

        def _refresh_tf2_trackable_lists(self) -> None:
            """Rebuild trackable list containers after backend conversion."""
            seen: set[int] = set()

            def visit(value: Any) -> None:
                if value is None or isinstance(value, (str, bytes, int, float, bool)):
                    return
                if isinstance(value, (np.ndarray, tf.Tensor, tf.Variable, xp.Array)):
                    return
                value_id = id(value)
                if value_id in seen:
                    return
                seen.add(value_id)

                if isinstance(value, Mapping):
                    for item in value.values():
                        visit(item)
                    return
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for item in value:
                        visit(item)
                    return

                try:
                    value_dict = object.__getattribute__(value, "__dict__")
                except AttributeError:
                    return

                for attr_name, attr_value in list(value_dict.items()):
                    if attr_name.startswith("_"):
                        continue
                    if not isinstance(attr_value, list):
                        continue
                    if any(isinstance(item, tf.Module) for item in attr_value):
                        setattr(value, attr_name, list(attr_value))

                try:
                    value_dict = object.__getattribute__(value, "__dict__")
                except AttributeError:
                    return
                for attr_name, attr_value in list(value_dict.items()):
                    if attr_name.startswith("_"):
                        continue
                    visit(attr_value)

            visit(self)

        def __getattribute__(self, name: str) -> Any:
            if not name.startswith("_tf2_"):
                try:
                    array_attrs = object.__getattribute__(
                        self, "_tf2_array_variable_attrs"
                    )
                except AttributeError:
                    array_attrs = ()
                if name in array_attrs:
                    storage_name = object.__getattribute__(
                        self,
                        "_tf2_array_variable_storage_name",
                    )(name)
                    variable = object.__getattribute__(self, storage_name)
                    return None if variable is None else to_tensorflow_array(variable)

                try:
                    list_attrs = object.__getattribute__(
                        self, "_tf2_array_variable_list_attrs"
                    )
                except AttributeError:
                    list_attrs = ()
                if name in list_attrs:
                    storage_name = object.__getattribute__(
                        self,
                        "_tf2_array_variable_list_storage_name",
                    )(name)
                    variables = object.__getattribute__(self, storage_name)
                    return [to_tensorflow_array(var) for var in variables]
            return super().__getattribute__(name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in self._tf2_array_variable_attr_names():
                self._set_tf2_array_variable(name, value)
                return None
            if name in self._tf2_array_variable_list_attr_names():
                self._set_tf2_array_variable_list(name, value)
                return None
            value = tf2_setattr(self, name, value)
            return super().__setattr__(name, value)

    original_deserialize = getattr(module, "deserialize", None)
    if original_deserialize is not None:

        @classmethod
        def deserialize(cls: type[Any], data: Any) -> Any:
            deserialize_func = getattr(original_deserialize, "__func__", None)
            if deserialize_func is None:
                obj = original_deserialize(data)
            else:
                obj = deserialize_func(cls, data)
            refresh = getattr(obj, "_refresh_tf2_trackable_lists", None)
            if callable(refresh):
                refresh()
            return obj

        TF2Module.deserialize = deserialize

    if hasattr(TF2Module, "deserialize"):
        for base in module.__bases__:
            if base in (object, NativeOP):
                continue
            if (
                _looks_like_dpmodel_class(base)
                and hasattr(base, "serialize")
                and base not in _DPMODEL_TO_TF2
            ):

                def _converter(v: Any, _cls: type[Any] = TF2Module) -> Any:
                    return _cls.deserialize(v.serialize())

                _DPMODEL_TO_TF2[base] = _converter

    return TF2Module
