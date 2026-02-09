# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common utilities for the pt_expt backend.

This module provides the core infrastructure for automatically wrapping dpmodel
classes (array_api_compat-based) as PyTorch modules. The key insight is to
detect attributes by their **value type** rather than by hard-coded names:

- numpy arrays → torch buffers (persistent state like statistics, masks)
- dpmodel objects → pt_expt torch.nn.Module wrappers (via registry lookup)
- None values → clear existing buffers

This eliminates the need to manually enumerate attribute names in each wrapper's
__setattr__ method, making the codebase more maintainable when dpmodel adds
new attributes.
"""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
    overload,
)

import numpy as np
import torch

from deepmd.dpmodel.common import (
    NativeOP,
)

# ---------------------------------------------------------------------------
# dpmodel → pt_expt converter registry
# ---------------------------------------------------------------------------
_DPMODEL_TO_PT_EXPT: dict[type[NativeOP], Callable[[NativeOP], torch.nn.Module]] = {}
"""Registry mapping dpmodel classes to their pt_expt converter functions.

This registry is populated at module import time via `register_dpmodel_mapping`
calls in each pt_expt wrapper module (e.g., exclude_mask.py, network.py). When
dpmodel_setattr encounters a dpmodel object, it looks up the object's type in
this registry to find the appropriate converter.

Examples of registered mappings:
- AtomExcludeMaskDP → lambda v: AtomExcludeMask(v.ntypes, exclude_types=...)
- NetworkCollectionDP → lambda v: NetworkCollection.deserialize(v.serialize())
"""


def register_dpmodel_mapping(
    dpmodel_cls: type[NativeOP], converter: Callable[[NativeOP], torch.nn.Module]
) -> None:
    """Register a converter that turns a dpmodel instance into a pt_expt Module.

    This function is called at module import time by each pt_expt wrapper to
    register how dpmodel objects should be converted when they're assigned as
    attributes. The converter is a callable that takes a dpmodel instance and
    returns the corresponding pt_expt torch.nn.Module wrapper.

    Parameters
    ----------
    dpmodel_cls : type[NativeOP]
        The dpmodel class to register (e.g., AtomExcludeMaskDP, NetworkCollectionDP).
        This is the key used for lookup in dpmodel_setattr.
    converter : Callable[[NativeOP], torch.nn.Module]
        A callable that converts a dpmodel instance to a pt_expt module.
        Common patterns:
        - Reconstruct from constructor args: lambda v: PtExptClass(v.ntypes, ...)
        - Round-trip via serialization: lambda v: PtExptClass.deserialize(v.serialize())

    Notes
    -----
    This function must be called AFTER the pt_expt wrapper class is defined but
    BEFORE dpmodel_setattr might encounter instances of dpmodel_cls. In practice,
    this means calling it immediately after the wrapper class definition at module
    import time.

    Examples
    --------
    >>> register_dpmodel_mapping(
    ...     AtomExcludeMaskDP,
    ...     lambda v: AtomExcludeMask(
    ...         v.ntypes, exclude_types=list(v.get_exclude_types())
    ...     ),
    ... )
    """
    _DPMODEL_TO_PT_EXPT[dpmodel_cls] = converter


def try_convert_module(value: Any) -> torch.nn.Module | None:
    """Convert a dpmodel object to its pt_expt wrapper if a converter is registered.

    This function looks up the exact type of *value* in the _DPMODEL_TO_PT_EXPT
    registry. If a converter is found, it invokes it to produce a torch.nn.Module
    wrapper; otherwise it returns None.

    Parameters
    ----------
    value : Any
        The value to potentially convert. Typically a dpmodel object like
        AtomExcludeMaskDP or NetworkCollectionDP.

    Returns
    -------
    torch.nn.Module or None
        The converted pt_expt module if a converter is registered for value's
        type, otherwise None.

    Notes
    -----
    This function uses exact type matching (not isinstance checks) to ensure
    predictable behavior. Each dpmodel class must be explicitly registered via
    register_dpmodel_mapping.

    The function is called by dpmodel_setattr when it encounters an object that
    might be a dpmodel instance. If conversion succeeds, the caller should use
    the converted module instead of the original value.
    """
    converter = _DPMODEL_TO_PT_EXPT.get(type(value))
    if converter is not None:
        return converter(value)
    return None


def dpmodel_setattr(obj: torch.nn.Module, name: str, value: Any) -> tuple[bool, Any]:
    """Common __setattr__ logic for pt_expt wrappers around dpmodel classes.

    This function implements automatic attribute detection by value type, eliminating
    the need to hard-code attribute names in each wrapper's __setattr__ method. It
    handles three cases:

    1. **numpy arrays → torch buffers**: Persistent state like statistics (davg, dstd)
       or masks that should be saved in state_dict and moved with .to(device).
    2. **None values → clear buffers**: Setting an existing buffer to None.
    3. **dpmodel objects → pt_expt modules**: Nested dpmodel objects like
       AtomExcludeMaskDP or NetworkCollectionDP are converted to their pt_expt
       wrappers via the registry.

    Parameters
    ----------
    obj : torch.nn.Module
        The pt_expt wrapper object whose attribute is being set. Must be a
        torch.nn.Module (caller should verify this).
    name : str
        The attribute name being set.
    value : Any
        The value being assigned. This function inspects the type to determine
        how to handle it.

    Returns
    -------
    handled : bool
        True if the attribute has been fully set (caller should NOT call
        super().__setattr__). False if the caller should forward the (possibly
        converted) value to super().__setattr__(name, value).
    value : Any
        The value to use. May be converted (e.g., dpmodel object → pt_expt module)
        or unchanged (e.g., scalar, list, or unregistered object).

    Notes
    -----
    **Why this design is safe:**

    - In dpmodel, all persistent arrays use `self.xxx = np.array(...)`. Scalars
      use `.item()`, lists use `.tolist()`. So `isinstance(value, np.ndarray)`
      reliably identifies buffer-worthy attributes.
    - torch.Tensor values assigned to existing buffers fall through to
      torch.nn.Module.__setattr__, which correctly updates them.
    - dpmodel objects are identified by registry lookup (exact type match), so
      only explicitly registered types are converted.
    - The function checks `"_buffers" in obj.__dict__` to ensure the object has
      been initialized as a torch.nn.Module before attempting buffer operations.

    **Circular import resolution:**

    The function uses a deferred import `from deepmd.pt_expt.utils import env`
    inside the function body. This breaks the circular dependency chain:
    common.py → utils/__init__.py → exclude_mask.py → common.py. The import is
    cached by Python after the first call, so there's no performance penalty.

    **Usage pattern:**

    Typical wrapper classes use this three-line pattern:

    >>> class MyWrapper(MyDPModel, torch.nn.Module):
    ...     def __setattr__(self, name, value):
    ...         handled, value = dpmodel_setattr(self, name, value)
    ...         if not handled:
    ...             super().__setattr__(name, value)

    Examples
    --------
    >>> # Case 1: numpy array → buffer
    >>> obj.davg = np.array([1.0, 2.0])  # becomes torch.Tensor buffer
    >>>
    >>> # Case 2: clear buffer
    >>> obj.davg = None  # sets buffer to None
    >>>
    >>> # Case 3: dpmodel object → pt_expt module
    >>> obj.emask = AtomExcludeMaskDP(...)  # becomes AtomExcludeMask module
    """
    from deepmd.pt_expt.utils import env  # deferred - avoids circular import

    # numpy array → torch buffer
    if isinstance(value, np.ndarray) and "_buffers" in obj.__dict__:
        tensor = torch.as_tensor(value, device=env.DEVICE)
        if name in obj._buffers:
            obj._buffers[name] = tensor
            return True, tensor
        obj.register_buffer(name, tensor)
        return True, tensor

    # clear an existing buffer to None
    if value is None and "_buffers" in obj.__dict__ and name in obj._buffers:
        obj._buffers[name] = None
        return True, None

    # dpmodel object → pt_expt module
    if "_modules" in obj.__dict__:
        converted = try_convert_module(value)
        if converted is not None:
            return False, converted
        # Note: Some NativeOP objects (like EnvMat) don't need conversion and can
        # be used directly. If a NativeOP truly needs conversion but isn't registered,
        # it will fail at runtime when the object is actually used.

    return False, value


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
@overload
def to_torch_array(array: np.ndarray) -> torch.Tensor: ...


@overload
def to_torch_array(array: None) -> None: ...


@overload
def to_torch_array(array: torch.Tensor) -> torch.Tensor: ...


def to_torch_array(array: Any) -> torch.Tensor | None:
    """Convert input to a torch tensor on the pt_expt device.

    This utility function handles conversion from various array-like types (numpy
    arrays, torch tensors on different devices, etc.) to torch tensors on the
    pt_expt backend's configured device.

    Parameters
    ----------
    array : Any
        The input to convert. Can be:
        - None (returns None)
        - torch.Tensor (moves to pt_expt device)
        - numpy array or array-like (converts to torch.Tensor on pt_expt device)

    Returns
    -------
    torch.Tensor or None
        The input as a torch tensor on the pt_expt device (env.DEVICE), or None
        if the input was None.

    Notes
    -----
    This function uses the same deferred import pattern as dpmodel_setattr to
    avoid circular dependencies. The env module determines the target device
    (typically CPU for pt_expt).

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> tensor = to_torch_array(arr)
    >>> tensor.device
    device(type='cpu')  # or whatever env.DEVICE is set to
    """
    from deepmd.pt_expt.utils import env  # deferred - avoids circular import

    if array is None:
        return None
    if torch.is_tensor(array):
        return array.to(device=env.DEVICE)
    return torch.as_tensor(array, device=env.DEVICE)
