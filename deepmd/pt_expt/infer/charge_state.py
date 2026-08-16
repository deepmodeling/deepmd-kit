# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host-side handling of the frame charge and spin-multiplicity condition.

A charge-conditioned model reaches its condition by one of two routes. An
uncompressed model reads it as an ordinary forward input, while a compressed
one folds it into frozen tables at export time and serves a different condition
by rebuilding them. Both routes start from the same user request, so this
module owns the step that turns that request into validated charge states, and
the rebuild that the folded route performs.

What counts as an acceptable value is a property of the loaded model, not of
this layer. A model that embeds the condition by indexing tables declares the
row range of each value, and a condition outside those ranges is rejected here
because neither the gather nor the compiled kernel bounds-checks the index. A
model that embeds the condition continuously declares no ranges, and only the
width of a request is constrained.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from torch._inductor.package import (
        AOTICompiledModel,
    )

#: Ranges are declared per value, so a request is described by its position.
_VALUE_LABELS = ("first", "second", "third", "fourth")


def _check_against_ranges(
    states: np.ndarray, table_ranges: Sequence[Sequence[int]]
) -> None:
    """Reject any state that addresses no row of the declared tables."""
    if states.shape[1] != len(table_ranges):
        raise ValueError(
            f"charge_spin states are {states.shape[1]} values wide, but the "
            f"model indexes {len(table_ranges)} tables."
        )
    # A non-finite value is not a row index either, so it is reported as the
    # integrality failure rather than reaching the range comparison, whose
    # message would have to render it.
    integral = np.isfinite(states) & (states == np.floor(states))
    for column, (low, high) in enumerate(table_ranges):
        label = _VALUE_LABELS[column] if column < len(_VALUE_LABELS) else str(column)
        values = states[:, column]
        offending = values[~integral[:, column]]
        if offending.size:
            raise ValueError(
                f"The {label} charge_spin value indexes a table row and must "
                f"be an integer, got {offending[0]}"
            )
        outside = values[(values < low) | (values >= high)]
        if outside.size:
            raise ValueError(
                f"The {label} charge_spin value must lie in [{low}, {high}), "
                f"got {outside[0]:.0f}"
            )


def charge_states(
    charge_spin: Any,
    width: int,
    table_ranges: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Read a requested condition as validated charge states.

    Parameters
    ----------
    charge_spin : Any
        Requested condition, of any shape holding a whole number of states.
    width : int
        Number of values in one charge state.
    table_ranges : Sequence[Sequence[int]], optional
        Half-open row range of each value, as the model declares it. ``None``
        leaves the values unconstrained beyond the width.

    Returns
    -------
    np.ndarray
        The requested states with shape ``(n, width)``.

    Raises
    ------
    ValueError
        If the request does not hold at least one whole state, or names a
        state that no declared table row answers.
    """
    values = np.asarray(charge_spin, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size % width:
        raise ValueError(
            f"charge_spin carries {values.size} values, which is not a positive "
            f"whole number of {width}-wide charge states."
        )
    states = values.reshape(-1, width)
    if table_ranges is not None:
        _check_against_ranges(states, table_ranges)
    return states


def charge_states_per_frame(
    charge_spin: Any,
    nframes: int,
    dim_chg_spin: int,
    table_ranges: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Read a requested condition as one validated state per frame.

    Parameters
    ----------
    charge_spin : Any
        Requested condition, holding one state for each frame.
    nframes : int
        Number of frames the forward covers.
    dim_chg_spin : int
        Number of values in one charge state.
    table_ranges : Sequence[Sequence[int]], optional
        Half-open row range of each value, as the model declares it.

    Returns
    -------
    np.ndarray
        The requested states with shape ``(nframes, dim_chg_spin)``.

    Raises
    ------
    ValueError
        If the request does not hold exactly one valid state per frame.
    """
    states = charge_states(charge_spin, dim_chg_spin, table_ranges)
    if states.shape[0] != nframes:
        raise ValueError(
            f"charge_spin must hold one charge state per frame: expected "
            f"{nframes} states of width {dim_chg_spin}, got {states.shape[0]}."
        )
    return states


def single_charge_state(
    charge_spin: Any,
    width: int,
    table_ranges: Sequence[Sequence[int]] | None = None,
) -> tuple[float, ...]:
    """Reduce a requested condition to the one state a folded snapshot serves.

    A folded condition lives in tables that are shared by the whole snapshot,
    so it is a property of the loaded model rather than of a frame. A request
    that names one state per frame is honoured only when every frame names the
    same one.

    Parameters
    ----------
    charge_spin : Any
        Requested condition, of any shape holding a whole number of states.
    width : int
        Number of values in one charge state.
    table_ranges : Sequence[Sequence[int]], optional
        Half-open row range of each value, as the model declares it.

    Returns
    -------
    tuple[float, ...]
        The requested state, with ``width`` values.

    Raises
    ------
    ValueError
        If the request does not hold at least one valid state, or holds
        several states that are not all equal.
    """
    states = charge_states(charge_spin, width, table_ranges)
    if not bool((states == states[0]).all()):
        raise ValueError(
            "This model folds one charge state into its frozen tables and "
            "therefore serves a single state at a time, but charge_spin names "
            f"{states.shape[0]} states that are not all equal."
        )
    return tuple(states[0].tolist())


class ChargeStateFold:
    """The rebuild of the frozen tables that carry a compressed charge state.

    A compressed charge-conditioned descriptor evaluates its frame condition
    once, when the model is frozen, into a handful of tables. Those tables
    reach the compiled lower as module constants, so serving a different
    condition means rebuilding them and writing them over those constants
    rather than re-evaluating the condition on every step. The archive
    therefore ships a second compiled artifact that performs the rebuild,
    together with the name of the constant each of its outputs replaces.

    Every lower lifts its constants independently, so the names hold only for
    the lower they were resolved against at freeze time. Only a compressed
    DPA4C descriptor folds a charge state, and that family never carries
    message passing across ranks, so an archive with a fold holds exactly one
    lower and the question of a second set of names does not arise.

    Parameters
    ----------
    model_file : str
        Path to the ``.pt2`` archive.
    metadata : dict[str, Any]
        Parsed archive metadata.
    target : AOTICompiledModel
        The lower whose constants carry the condition.

    Attributes
    ----------
    width : int
        Number of values in a charge state this fold accepts.

    Raises
    ------
    ValueError
        If the archive declares a fold it cannot supply, or names no width
        for a charge state.
    """

    def __init__(
        self,
        model_file: str,
        metadata: dict[str, Any],
        target: AOTICompiledModel,
    ) -> None:
        import tempfile
        import zipfile

        from torch._inductor import (
            aoti_load_package,
        )

        from deepmd.pt_expt.utils.serialization import (
            PT2_EXTRA_PREFIX,
        )

        self._constants = [str(name) for name in metadata["charge_state_constants"]]
        # The lower reads no condition, so what the model accepts is the state
        # the snapshot was frozen against, which is also the layout the rebuild
        # consumes.
        default_chg_spin = metadata.get("default_chg_spin")
        if not default_chg_spin:
            raise ValueError(
                f"'{model_file}' ships a charge-state fold but names no "
                "default_chg_spin, so the width of a charge state is unknown."
            )
        self.width = len(default_chg_spin)

        entry = PT2_EXTRA_PREFIX + "charge_state.pt2"
        with zipfile.ZipFile(model_file, "r") as zf:
            if entry not in zf.namelist():
                raise ValueError(
                    f"Invalid .pt2 file '{model_file}': it declares "
                    f"charge_state_constants but carries no '{entry}', so it "
                    "cannot serve a runtime charge state."
                )
            archive = zf.read(entry)
        # ``aoti_load_package`` reads a path, so the nested archive is extracted
        # to a temporary file that this object owns and releases with itself.
        self._archive = tempfile.NamedTemporaryFile(suffix=".pt2")
        self._archive.write(archive)
        self._archive.flush()
        self._runner = aoti_load_package(self._archive.name)
        self._target = target
        self._applied: tuple[float, ...] | None = None

    @classmethod
    def load(
        cls,
        model_file: str,
        metadata: dict[str, Any],
        target: AOTICompiledModel,
    ) -> ChargeStateFold | None:
        """Load the rebuild an archive declares, if it declares one.

        The constant-name field is the archive's claim that the rebuild ships
        with it, so an archive that declares the names and cannot supply the
        rebuild is malformed and fails here rather than degrading silently.

        Parameters
        ----------
        model_file : str
            Path to the ``.pt2`` archive.
        metadata : dict[str, Any]
            Parsed archive metadata.
        target : AOTICompiledModel
            The lower whose constants carry the condition.

        Returns
        -------
        ChargeStateFold or None
            The fold, or ``None`` when the archive declares none.
        """
        if "charge_state_constants" not in metadata:
            return None
        return cls(model_file, metadata, target)

    def apply(self, charge_spin: tuple[float, ...]) -> None:
        """Rebuild the tables for a condition and write them over the constants.

        Rebuilding is skipped when the condition already applies, so a run that
        evaluates many frames at one condition pays for it once.

        Applying a condition overwrites loaded module state and is therefore
        not safe to interleave with a forward pass.

        Parameters
        ----------
        charge_spin : tuple[float, ...]
            The condition, with :attr:`width` values.

        Raises
        ------
        RuntimeError
            If the rebuild returns a different number of tables than the
            archive names constants.
        """
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        if charge_spin == self._applied:
            return
        # The rebuild consumes the condition in the (1, width) float32 layout
        # the inference lower would receive.
        tables = self._runner(
            torch.tensor([charge_spin], dtype=torch.float32, device=DEVICE)
        )
        if len(tables) != len(self._constants):
            raise RuntimeError(
                f"The charge-state fold returned {len(tables)} tables but the "
                f"archive names {len(self._constants)} constants; it cannot "
                "serve a runtime charge state."
            )
        # An unnamed output belongs to a mechanism this model has disabled and
        # has no constant to reach.
        self._target.load_constants(
            {name: table for name, table in zip(self._constants, tables) if name},
            check_full_update=False,
        )
        self._applied = charge_spin
