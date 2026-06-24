# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single chokepoint for all ``deepmd`` internal API and ``torch`` calls.

Every import from ``deepmd.pt.*``, ``deepmd.utils.model_branch_dict``, or
``torch`` that is needed by the rest of ``dpa_adapt`` must go through
this module.  No other file in ``dpa_adapt`` may import those packages directly.

All functions that load ``torch`` or ``deepmd.pt`` keep the import inside the
function body so that importing this module is cheap.
"""

from __future__ import (
    annotations,
)

import logging
from typing import (
    Any,
)

# ``get_model_dict`` is backend-agnostic and lightweight — safe at module level.
from deepmd.utils.model_branch_dict import get_model_dict as _get_model_dict

_LOG = logging.getLogger("dpa_adapt")


def resolve_dp_command() -> str:
    """Return the ``dp`` executable associated with the current Python env."""
    import os as _os
    import shutil as _shutil
    import sys as _sys
    import sysconfig as _sysconfig
    from pathlib import Path as _Path

    exe_name = "dp.exe" if _os.name == "nt" else "dp"
    scripts_dir = _sysconfig.get_path("scripts")
    candidates = [
        _Path(_sys.executable).parent / exe_name,
    ]
    if scripts_dir:
        candidates.append(_Path(scripts_dir) / exe_name)
    for candidate in candidates:
        if candidate.is_file():
            return _os.fspath(candidate)

    found = _shutil.which("dp")
    if found:
        return found

    return "dp"


# ---------------------------------------------------------------------------
# torch I/O
# ---------------------------------------------------------------------------


def _is_url_or_name(path: str) -> bool:
    """Return True if *path* looks like a URL or a built-in model name rather
    than a local file path.
    """
    import os as _os

    return not _os.path.exists(path)


def resolve_pretrained_path(pretrained: str, cache_dir: str | None = None) -> str:
    """Resolve *pretrained* to a local file path, downloading if necessary.

    If *pretrained* is a local checkpoint path, it is returned unchanged.  This
    includes non-existing path-like values so callers can raise their own
    context-specific ``not found`` errors or tests can monkeypatch checkpoint
    loading.  Bare names (e.g. ``"DPA-3.1-3M"``) are resolved via
    :func:`deepmd.pretrained.download.resolve_model_path`.
    """
    import os as _os
    from pathlib import Path as _Path

    pretrained = _os.fspath(pretrained)

    if _os.path.isfile(pretrained):
        return pretrained

    p = _Path(pretrained)
    is_path_like = (
        p.is_absolute()
        or any(sep and sep in pretrained for sep in (_os.sep, _os.altsep))
        or p.suffix.lower() in {".pt", ".pth"}
    )
    if is_path_like:
        return pretrained

    from deepmd.pretrained.download import resolve_model_path as _download

    path = _download(pretrained, cache_dir=cache_dir)
    _LOG.info("Resolved pretrained model: %s", path)
    return _os.fspath(path)


def load_torch_file(path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint or frozen bundle.

    Always uses ``weights_only=False`` because deepmd checkpoints carry
    ``_extra_state`` (non-tensor metadata) and dpa_adapt frozen bundles
    carry ``sklearn`` pipeline objects.
    """
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except RuntimeError as exc:
        if "Invalid magic number" not in str(exc):
            raise
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model_from_config(input_param: dict[str, Any]):
    """Build a (non-JIT) DPA model from an input-parameter dict.

    Returns a ``ModelWrapper`` whose inner model is accessible as
    ``wrapper.model["Default"]``.
    """
    from deepmd.pt.model.model import (
        get_model,
    )
    from deepmd.pt.train.wrapper import (
        ModelWrapper,
    )

    model = get_model(input_param)
    return ModelWrapper(model)


# ---------------------------------------------------------------------------
# multi-task branch helpers
# ---------------------------------------------------------------------------


def resolve_model_branch(model_dict: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Resolve multi-task model-branch aliases.

    Returns ``(alias_dict, model_dict)`` — the same tuple shape as the
    upstream ``get_model_dict``.
    """
    return _get_model_dict(model_dict)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def get_torch_device() -> Any:
    """Return ``torch.device("cuda")`` if a GPU is available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# descriptor extraction (the fragile chain)
# ---------------------------------------------------------------------------


class _DescriptorExtraction:
    """Thin wrapper around a loaded model that runs a *single* forward pass
    with ``eval_descriptor_hook`` enabled and returns per-atom descriptors.

    This is the lowest-level building block.  Callers (like
    ``DPAFineTuner._extract_features``) are responsible for pooling,
    batching, and tensor creation.
    """

    def __init__(self, wrapper) -> None:
        inner = wrapper.model["Default"]
        self._inner_model = inner
        self._atomic_model = inner.atomic_model

    def _enable_hook(self) -> None:
        self._atomic_model.set_eval_descriptor_hook(True)

    def _disable_hook(self) -> None:
        self._atomic_model.set_eval_descriptor_hook(False)

    def _clear_accumulator(self) -> None:
        self._atomic_model.eval_descriptor_list.clear()

    def _run_forward(self, coord, atype, box):
        """Run ``forward_common`` and return per-atom descriptors (detached).

        Parameters
        ----------
        coord : torch.Tensor
            (n_frames, n_atoms*3), float64, requires_grad.
        atype : torch.Tensor
            (n_frames, n_atoms), int64.
        box : torch.Tensor
            (n_frames, 9), float64.

        Returns
        -------
        torch.Tensor
            (n_frames, n_atoms, feat_dim), detached.
        """
        if not coord.requires_grad:
            raise RuntimeError(
                "forward_common requires coord to have requires_grad=True"
            )
        self._clear_accumulator()
        self._inner_model.forward_common(coord, atype, box)
        return self._atomic_model.eval_descriptor().detach()
