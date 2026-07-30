# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from functools import (
    wraps,
)
from importlib import (
    import_module,
)
from threading import (
    Condition,
    get_ident,
)
from typing import (
    Any,
    TypeVar,
    overload,
)

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.jax.env import (
    flax_version,
    jax,
    jnp,
    nnx,
)


class ArrayAPIVariable(nnx.Variable):
    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.value.__array__(*args, **kwargs)

    def __array_namespace__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__array_namespace__(*args, **kwargs)

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack_device__(*args, **kwargs)


@overload
def to_jax_array(array: np.ndarray) -> jnp.ndarray: ...


@overload
def to_jax_array(array: None) -> None: ...


def to_jax_array(array: np.ndarray | None) -> jnp.ndarray | None:
    """Convert a numpy array to a JAX array.

    Parameters
    ----------
    array : np.ndarray
        The numpy array to convert.

    Returns
    -------
    jnp.ndarray
        The JAX tensor.
    """
    if array is None:
        return None
    return jnp.array(array)


_DPMODEL_TO_JAX: dict[type[Any], Callable[[Any], Any]] = {}
_AUTO_WRAPPED_CLASSES: dict[type[NativeOP], type[Any]] = {}
_FLAX_0_12 = Version("0.12.0")
_REGISTRATIONS_READY = False
_REGISTRATIONS_IN_PROGRESS = False
_REGISTRATIONS_OWNER: int | None = None
_REGISTRATIONS_COND = Condition()
_REGISTRATION_MODULES = (
    "deepmd.jax.utils.network",
    "deepmd.jax.utils.exclude_mask",
    "deepmd.jax.utils.type_embed",
    "deepmd.jax.descriptor",
    "deepmd.jax.fitting",
    "deepmd.jax.atomic_model.dp_atomic_model",
    "deepmd.jax.atomic_model.energy_atomic_model",
    "deepmd.jax.atomic_model.dipole_atomic_model",
    "deepmd.jax.atomic_model.dos_atomic_model",
    "deepmd.jax.atomic_model.polar_atomic_model",
    "deepmd.jax.atomic_model.property_atomic_model",
    "deepmd.jax.atomic_model.pairtab_atomic_model",
    "deepmd.jax.atomic_model.linear_atomic_model",
    "deepmd.jax.model",
)


def register_dpmodel_mapping(
    dpmodel_cls: type[Any], converter: Callable[[Any], Any]
) -> None:
    """Register how to convert a dpmodel object to its JAX wrapper."""
    _DPMODEL_TO_JAX[dpmodel_cls] = converter


def _looks_like_dpmodel_object(value: Any) -> bool:
    module = type(value).__module__
    return module == "deepmd.dpmodel" or module.startswith("deepmd.dpmodel.")


def _ensure_registrations() -> None:
    global _REGISTRATIONS_IN_PROGRESS, _REGISTRATIONS_OWNER, _REGISTRATIONS_READY

    current_thread = get_ident()
    with _REGISTRATIONS_COND:
        if _REGISTRATIONS_READY:
            return
        while _REGISTRATIONS_IN_PROGRESS:
            if _REGISTRATIONS_OWNER == current_thread:
                return
            _REGISTRATIONS_COND.wait()
            if _REGISTRATIONS_READY:
                return
        _REGISTRATIONS_IN_PROGRESS = True
        _REGISTRATIONS_OWNER = current_thread

    success = False
    try:
        for module in _REGISTRATION_MODULES:
            import_module(module)
        success = True
    finally:
        with _REGISTRATIONS_COND:
            _REGISTRATIONS_READY = success
            _REGISTRATIONS_IN_PROGRESS = False
            _REGISTRATIONS_OWNER = None
            _REGISTRATIONS_COND.notify_all()


def try_convert_module(value: Any) -> Any | None:
    """Convert a registered dpmodel object to its JAX wrapper."""
    converter = _DPMODEL_TO_JAX.get(type(value))
    if converter is not None:
        return converter(value)
    if _looks_like_dpmodel_object(value):
        _ensure_registrations()
        converter = _DPMODEL_TO_JAX.get(type(value))
        if converter is not None:
            return converter(value)
    if isinstance(value, NativeOP):
        return _auto_wrap_native_op(value)
    return None


def _auto_wrap_native_op(value: NativeOP) -> Any:
    cls = type(value)
    if cls not in _AUTO_WRAPPED_CLASSES:
        _AUTO_WRAPPED_CLASSES[cls] = flax_module(cls)
    wrapped_cls = _AUTO_WRAPPED_CLASSES[cls]
    if not (hasattr(value, "serialize") and hasattr(wrapped_cls, "deserialize")):
        raise TypeError(
            f"Cannot auto-wrap {cls.__name__}: "
            "it must implement serialize()/deserialize() or be explicitly "
            "registered via register_dpmodel_mapping()."
        )
    return wrapped_cls.deserialize(value.serialize())


def _use_nnx_list() -> bool:
    return Version(flax_version) >= _FLAX_0_12 and hasattr(nnx, "List")


def _wrap_list(value: list[Any]) -> Any:
    if _use_nnx_list():
        return nnx.List([nnx.data(item) for item in value])
    return value


def _try_convert_list(value: list[Any]) -> Any | None:
    if not value:
        return None

    converted = []
    changed = False
    for item in value:
        if isinstance(item, np.ndarray):
            converted.append(ArrayAPIVariable(to_jax_array(item)))
            changed = True
        elif isinstance(item, (nnx.Module, nnx.Variable)):
            converted.append(item)
            changed = True
        elif item is None:
            converted.append(item)
        else:
            module = try_convert_module(item)
            if module is None:
                return None
            converted.append(module)
            changed = True

    if not changed:
        return None
    return _wrap_list(converted)


def dpmodel_setattr(obj: nnx.Module, name: str, value: Any) -> tuple[bool, Any]:
    """Common ``__setattr__`` conversion for Flax wrappers around dpmodel objects."""
    if name in getattr(obj, "_jax_skip_auto_convert_attrs", ()):
        return False, value

    current = vars(obj).get(name)
    if isinstance(current, nnx.Variable) and isinstance(value, (np.ndarray, jax.Array)):
        if isinstance(value, np.ndarray):
            value = to_jax_array(value)
        if Version(flax_version) >= _FLAX_0_12:
            current.set_value(value)
        else:
            current.value = value
        return True, current

    if (
        isinstance(value, list)
        and name in getattr(obj, "_jax_data_list_attrs", ())
        and _use_nnx_list()
    ):
        return False, _try_convert_list(value) or _wrap_list(value)

    if isinstance(value, np.ndarray):
        return False, ArrayAPIVariable(to_jax_array(value))

    if isinstance(value, list):
        converted_list = _try_convert_list(value)
        if converted_list is not None:
            return False, converted_list

    if not isinstance(value, nnx.Module):
        converted = try_convert_module(value)
        if converted is not None:
            return False, converted

    return False, value


T = TypeVar("T")


def flax_module(
    module: type[T],
) -> type[T]:  # runtime: actually returns type[T & nnx.Module]
    """Convert a NativeOP to a Flax module.

    Parameters
    ----------
    module : type[NativeOP]
        The NativeOP to convert.

    Returns
    -------
    type[flax.nnx.Module]
        The Flax module.

    Examples
    --------
    >>> @flax_module
    ... class MyModule(NativeOP):
    ...     pass
    """
    metas = set()
    if not issubclass(type(nnx.Module), type(module)):
        metas.add(type(module))
    if not issubclass(type(module), type(nnx.Module)):
        metas.add(type(nnx.Module))

    class MixedMetaClass(*metas):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return type(nnx.Module).__call__(self, *args, **kwargs)

    @wraps(module, updated=())
    class FlaxModule(module, nnx.Module, metaclass=MixedMetaClass):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            return super().__init_subclass__(**kwargs)

        def __setattr__(self, name: str, value: Any) -> None:
            handled, value = dpmodel_setattr(self, name, value)
            if not handled:
                try:
                    return super().__setattr__(name, value)
                except ValueError as err:
                    msg = str(err)
                    if (
                        Version(flax_version) >= _FLAX_0_12
                        and "Cannot assign data value" in msg
                        and "to static attribute" in msg
                    ):
                        return super().__setattr__(name, nnx.data(value))
                    raise
            return None

    if hasattr(FlaxModule, "deserialize"):
        for base in module.__bases__:
            if base in (object, NativeOP, nnx.Module):
                continue
            if issubclass(base, nnx.Module):
                continue
            if hasattr(base, "serialize") and base not in _DPMODEL_TO_JAX:

                def _converter(v: Any, _cls: type[Any] = FlaxModule) -> Any:
                    return _cls.deserialize(v.serialize())

                _DPMODEL_TO_JAX[base] = _converter

    return FlaxModule
