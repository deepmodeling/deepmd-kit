# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

from deepmd.infer.deep_tensor import (
    DeepTensor,
)

if TYPE_CHECKING:
    import numpy as np


class DeepPolar(DeepTensor):
    """Deep polar model.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    @property
    def output_tensor_name(self) -> str:
        return "polar"


class DeepGlobalPolar(DeepTensor):
    @property
    def output_tensor_name(self) -> str:
        return "global_polar"

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: list[int] | np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        mixed_type: bool = False,
        **kwargs: dict,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types : list[int] or np.ndarray
            The atom types
            The list should contain natoms ints
        atomic
            If True (default), return the atomic tensor
            Otherwise return the global tensor
        fparam
            Not used in this model
        aparam
            Not used in this model
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        tensor
            The returned tensor
            If atomic == False then of size nframes x output_dim
            else of size nframes x natoms x output_dim
        """
        return super().eval(
            coords,
            cells,
            atom_types,
            atomic=atomic,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
            **kwargs,
        )
