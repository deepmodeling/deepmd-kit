# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training utility functions."""

from __future__ import (
    annotations,
)

import os
from contextlib import (
    contextmanager,
)
from math import (
    ceil,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
    )


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    stable: bool = True,
) -> torch.Tensor:
    """
    Clip gradients in place so their global L2 norm does not exceed ``max_norm``.

    The norm is computed and applied on device and returned without a host
    synchronization. Non-finite gradients are not reported here; see
    :class:`NonFiniteGradGuard`.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        Parameters whose gradients are clipped in place.
    max_norm : float
        Maximum allowed global L2 norm of the gradients.
    stable : bool, optional
        Norm reduction strategy. ``True`` scales the gradients by their largest
        magnitude before the float64 reduction, keeping the norm finite for an
        arbitrarily large but finite gradient of any dtype. ``False`` uses the
        native reduction, which propagates ``DTensor`` sharding under FSDP2 but is
        not overflow-safe.

    Returns
    -------
    torch.Tensor
        The global gradient norm before clipping, or a CPU zero when no gradient
        is present.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros((), dtype=torch.float64, device="cpu")
    grads = [p.grad for p in params]

    # === Step 1. Global L2 norm ===
    if stable:
        # Normalize by the largest magnitude before the float64 reduction; the
        # factor cancels in the product and keeps the sum of squares finite for
        # any dtype. The ``foreach`` ops fuse the per-parameter passes.
        scale = torch.stack(torch._foreach_norm(grads, float("inf"))).max()
        scale = torch.where(scale > 0, scale, scale.new_ones(()))
        scaled = torch._foreach_norm(torch._foreach_div(grads, scale), 2.0)
        total_norm = scale.double() * torch.linalg.vector_norm(
            torch.stack(scaled).double()
        )
    else:
        total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=False)

    # === Step 2. Rescale gradients by the clamped coefficient ===
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)
    return total_norm


class NonFiniteGradGuard:
    """
    Detect non-finite gradient norms without a per-step host synchronization.

    :meth:`update` accumulates the non-finite condition on device; the result is
    read back only by :meth:`raise_if_nonfinite`. Calling the check before each
    checkpoint keeps a diverged interval from being written while leaving the
    training step free of host reads.
    """

    def __init__(self) -> None:
        self._nonfinite: torch.Tensor | None = None

    def update(self, total_norm: torch.Tensor) -> None:
        """
        Accumulate whether ``total_norm`` is non-finite.

        Parameters
        ----------
        total_norm : torch.Tensor
            The gradient norm returned by :func:`clip_grad_norm_`.
        """
        nonfinite = ~torch.isfinite(total_norm)
        if self._nonfinite is not None:
            nonfinite |= self._nonfinite.to(nonfinite.device)
        self._nonfinite = nonfinite

    def raise_if_nonfinite(
        self,
        named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]],
    ) -> None:
        """
        Raise if any norm accumulated since the previous call was non-finite.

        On failure the current gradient state is reported via
        :func:`raise_nonfinite_gradient_norm`.

        Parameters
        ----------
        named_parameters : Callable[[], Iterable[tuple[str, torch.nn.Parameter]]]
            Accessor for the model's named parameters, consulted only when raising.

        Raises
        ------
        RuntimeError
            If a non-finite gradient norm was recorded.
        """
        if self._nonfinite is None:
            return
        diverged = bool(self._nonfinite)
        self._nonfinite = None
        if diverged:
            raise_nonfinite_gradient_norm(named_parameters())


def raise_nonfinite_gradient_norm(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    """
    Raise a ``RuntimeError`` reporting the current non-finite gradients.

    Parameters whose current gradient is non-finite are listed by name and shape.
    When every current individual gradient is finite, the accumulated guard flag
    came from an earlier step in the checkpoint interval or from the norm
    reduction.

    Parameters
    ----------
    named_parameters : Iterable[tuple[str, torch.nn.Parameter]]
        The model's named parameters.

    Raises
    ------
    RuntimeError
        Always; this is the divergence-reporting path.
    """
    bad_params = []
    for name, param in named_parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm()
        if not torch.isfinite(grad_norm):
            bad_params.append(
                f"  {name}: grad_norm={grad_norm}, shape={list(param.shape)}"
            )
    detail = (
        "\n".join(bad_params)
        if bad_params
        else (
            "  (all current individual gradients are finite; non-finite norm was "
            "recorded earlier in the checkpoint interval or in the norm reduction)"
        )
    )
    raise RuntimeError(
        "Non-finite gradient norm; training has diverged.\n"
        f"Parameters with non-finite gradients:\n{detail}"
    )


@contextmanager
def scoped_env_defaults(defaults: dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set missing environment variables and restore them afterward."""
    previous = {key: os.environ.get(key) for key in defaults}
    try:
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def latest_checkpoint_path(prefix: str, step_label: int, save_dir: Path | None) -> Path:
    """
    Resolve the on-disk path of a periodic checkpoint file.

    Parameters
    ----------
    prefix : str
        The checkpoint prefix, e.g. ``model.ckpt`` or its EMA counterpart.
    step_label : int
        The training step encoded into the filename.
    save_dir : Path or None
        The configured checkpoint directory. When ``None`` the file follows
        ``prefix`` relative to the working directory.

    Returns
    -------
    Path
        ``save_dir/<prefix name>-<step>.pt`` when ``save_dir`` is set, otherwise
        ``<prefix>-<step>.pt`` relative to the working directory.
    """
    directory = save_dir if save_dir is not None else Path(prefix).parent
    return directory / f"{Path(prefix).name}-{step_label}.pt"


def resolve_best_checkpoint_dir(
    validating_params: dict[str, Any], save_ckpt: str
) -> Path:
    """
    Resolve the directory for full-validation best checkpoints.

    Parameters
    ----------
    validating_params : dict
        The ``validating`` section of the training configuration.
    save_ckpt : str
        The regular checkpoint prefix from ``training.save_ckpt``.

    Returns
    -------
    Path
        ``validating.save_best_dir`` when set, otherwise the directory derived
        from ``save_ckpt``.
    """
    save_best_dir = validating_params.get("save_best_dir")
    if save_best_dir:
        return Path(save_best_dir)
    return Path(save_ckpt).parent


def resolve_keep_ckpt_count(
    ckpt_keep_ratio: float | None, num_steps: int, save_freq: int
) -> int | None:
    """
    Convert a checkpoint-retention ratio into a sliding-window keep count.

    Periodic checkpoints are saved every ``save_freq`` steps, so a run of
    ``num_steps`` produces about ``num_steps // save_freq`` of them. Keeping the
    most recent ``ceil(ratio * total)`` of those is equivalent to retaining the
    final ``ratio`` fraction of the run by step, without the caller computing
    the count by hand.

    Parameters
    ----------
    ckpt_keep_ratio : float or None
        The fraction of the training run, by step, whose periodic checkpoints
        are retained. ``None`` leaves the keep count unchanged.
    num_steps : int
        The total number of training steps, already resolved (including when
        derived from ``numb_epoch``).
    save_freq : int
        The checkpoint saving frequency in steps.

    Returns
    -------
    int or None
        The number of most recent checkpoints to keep (at least one), or
        ``None`` when ``ckpt_keep_ratio`` is not set.
    """
    if ckpt_keep_ratio is None:
        return None
    total_periodic_ckpts = max(1, num_steps // save_freq)
    return max(1, ceil(ckpt_keep_ratio * total_periodic_ckpts))
