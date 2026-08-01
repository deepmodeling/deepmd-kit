# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities shared by PyTorch checkpoint backends."""

from collections.abc import (
    Mapping,
)
from typing import (
    Any,
)


def detect_pt_checkpoint_backend(checkpoint: Any) -> str | None:
    """Detect the parameter dialect of a raw PyTorch checkpoint.

    Parameters
    ----------
    checkpoint : Any
        A checkpoint payload or its unwrapped model state dictionary.

    Returns
    -------
    str or None
        ``"pt-expt"`` or ``"pt"`` when the parameter names identify one
        backend unambiguously, otherwise ``None``.
    """
    state_dict = checkpoint
    if isinstance(state_dict, Mapping) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, Mapping):
        return None

    keys = tuple(key for key in state_dict if isinstance(key, str))

    # Weight names are decisive. pt_expt DPA4 also contains ordinary
    # torch-native ``.bias`` parameters, so bias names cannot override a
    # clear ``.w`` versus ``.matrix`` distinction.
    has_pt_expt_weight = any(key.endswith(".w") for key in keys)
    has_pt_weight = any(key.endswith(".matrix") for key in keys)
    if has_pt_expt_weight or has_pt_weight:
        if has_pt_expt_weight == has_pt_weight:
            return None
        return "pt-expt" if has_pt_expt_weight else "pt"

    # A lone ``.b`` is specific to pt_expt's NativeLayer. A lone ``.bias``
    # is not specific to pt because pt_expt models can contain torch-native
    # modules with that suffix, so it remains deliberately unclassified.
    has_pt_expt_bias = any(key.endswith(".b") for key in keys)
    has_pt_bias = any(key.endswith(".bias") for key in keys)
    if has_pt_expt_bias and not has_pt_bias:
        return "pt-expt"
    return None
