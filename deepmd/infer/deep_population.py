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


class DeepPopulation(DeepEval):
    """Deep Population model.

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
        """Get the output definition of this model."""
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        "population",
                        [2],
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
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        mixed_type: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate the atomic charge population.

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
        atomic : bool, optional
            Kept for interface compatibility. Population is always returned per atom.
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
        atomic_population
            The atomic charge population in shape (nframes, natoms, 2), where the
            last dimension stores the (alpha, beta) spin channel populations.
        """
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        results = self.deep_eval.eval(
            coords,
            cells,
            atom_types,
            atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )

        atomic_population = results["population"].reshape(nframes, natoms, 2)

        return (atomic_population,)


__all__ = ["DeepPopulation"]
