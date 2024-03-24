# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)


class DeepTensor(DeepEval):
    """Deep Tensor Model.

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

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[List[int], np.ndarray],
        atomic: bool = True,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
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
        efield
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
        if atomic:
            return results[self.output_tensor_name].reshape(nframes, natoms, -1)
        else:
            return results[f"{self.output_tensor_name}_redu"].reshape(nframes, -1)

    def eval_full(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
        **kwargs: dict,
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate the model with interface similar to the energy model.
        Will return global tensor, component-wise force and virial
        and optionally atomic tensor and atomic virial.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Whether to calculate atomic tensor and virial
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
            The global tensor.
            shape: [nframes x nout]
        force
            The component-wise force (negative derivative) on each atom.
            shape: [nframes x nout x natoms x 3]
        virial
            The component-wise virial of the tensor.
            shape: [nframes x nout x 9]
        atom_tensor
            The atomic tensor. Only returned when atomic == True
            shape: [nframes x natoms x nout]
        atom_virial
            The atomic virial. Only returned when atomic == True
            shape: [nframes x nout x natoms x 9]
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

        energy = results[f"{self.output_tensor_name}_redu"].reshape(nframes, -1)
        force = results[f"{self.output_tensor_name}_derv_r"].reshape(
            nframes, -1, natoms, 3
        )
        virial = results[f"{self.output_tensor_name}_derv_c_redu"].reshape(
            nframes, -1, 9
        )
        if atomic:
            atomic_energy = results[self.output_tensor_name].reshape(
                nframes, natoms, -1
            )
            atomic_virial = results[f"{self.output_tensor_name}_derv_c"].reshape(
                nframes, -1, natoms, 9
            )
            return (
                energy,
                force,
                virial,
                atomic_energy,
                atomic_virial,
            )
        else:
            return (
                energy,
                force,
                virial,
            )

    @property
    @abstractmethod
    def output_tensor_name(self) -> str:
        """The name of the tensor."""

    @property
    def output_def(self) -> ModelOutputDef:
        """Get the output definition of this model."""
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        self.output_tensor_name,
                        shape=[-1],
                        reduciable=True,
                        r_differentiable=True,
                        c_differentiable=True,
                        atomic=True,
                    ),
                ]
            )
        )
