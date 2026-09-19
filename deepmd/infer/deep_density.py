# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)

from .deep_eval import (
    DeepEval,
)


class DeepDensity(DeepEval):
    """Charge density evaluated on grid points.

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
    def output_def(self) -> ModelOutputDef:
        """Get the output definition of this model.

        The output definition is identical to that of the density
        fitting net of the model: the density is predicted on grid
        points, and is neither reducible nor differentiable.
        """
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        "density",
                        [1],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                    ),
                ]
            )
        )

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: list[int] | np.ndarray,
        grid: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        mixed_type: bool = False,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Evaluate the density on grid points.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the atoms, in shape (nframes, natoms, 3).
        cells : np.ndarray
            The cell vectors of the system, in shape (nframes, 9). If the system
            is not periodic, set it to None.
        atom_types : list[int] or np.ndarray
            The types of the atoms. If mixed_type is False, the shape is (natoms,);
            otherwise, the shape is (nframes, natoms).
        grid : np.ndarray
            The coordinates of the grid points, in shape (nframes, ngrid, 3).
        fparam : np.ndarray, optional
            The frame parameters, by default None.
        aparam : np.ndarray, optional
            The atomic parameters, by default None.
        mixed_type : bool, optional
            Whether the atom_types is mixed type, by default False.
        **kwargs : dict[str, Any]
            Keyword arguments.

        Returns
        -------
        density
            The density on the grid points, in shape (nframes, ngrid).
        """
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            _,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        results = self.deep_eval.eval(
            coords,
            cells,
            atom_types,
            False,
            fparam=fparam,
            aparam=aparam,
            grid=self._standard_grid(grid, nframes),
            **kwargs,
        )
        return results["density"].reshape(nframes, -1)

    @staticmethod
    def _standard_grid(grid: np.ndarray, nframes: int) -> np.ndarray:
        """Normalise the grid coordinates to ``(nframes, ngrid, 3)``.

        The auto batcher slices every argument with ``ndim > 1`` along axis
        0, so a natural single-frame ``(ngrid, 3)`` input would silently be
        truncated to one grid point; carry the frame dimension explicitly.
        """
        arr = np.asarray(grid)
        if arr.ndim == 2:
            if nframes != 1:
                raise ValueError(
                    f"grid of shape {arr.shape} is ambiguous for {nframes} "
                    "frames; pass grid with shape (nframes, ngrid, 3)"
                )
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"grid must have shape (nframes, ngrid, 3), got {arr.shape}"
            )
        if arr.shape[0] != nframes:
            raise ValueError(f"grid has {arr.shape[0]} frames but coord has {nframes}")
        return arr


__all__ = ["DeepDensity"]
