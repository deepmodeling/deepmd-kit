# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd_utils.utils.batch_size import (
    AutoBatchSize,
)

from .backend import (
    DPBackend,
    detect_backend,
)


class DeepPot(ABC):
    """Potential energy model.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    """

    @abstractmethod
    def __init__(
        self,
        model_file,
        *args,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list=None,
        **kwargs,
    ) -> None:
        pass

    def __new__(cls, model_file: str, *args, **kwargs):
        if cls is DeepPot:
            backend = detect_backend(model_file)
            if backend == DPBackend.TensorFlow:
                from deepmd.infer.deep_pot import DeepPot as DeepPotTF

                return super().__new__(DeepPotTF)
            elif backend == DPBackend.PyTorch:
                from deepmd_pt.infer.deep_eval import DeepPot as DeepPotPT

                return super().__new__(DeepPotPT)
            else:
                raise NotImplementedError("Unsupported backend: " + str(backend))
        return super().__new__(cls)

    @abstractmethod
    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate energy, force, and virial. If atomic is True,
        also return atomic energy and atomic virial.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the atoms, in shape (nframes, natoms, 3).
        cells : np.ndarray
            The cell vectors of the system, in shape (nframes, 9). If the system
            is not periodic, set it to None.
        atom_types : List[int]
            The types of the atoms. If mixed_type is False, the shape is (natoms,);
            otherwise, the shape is (nframes, natoms).
        atomic : bool, optional
            Whether to return atomic energy and atomic virial, by default False.
        fparam : np.ndarray, optional
            The frame parameters, by default None.
        aparam : np.ndarray, optional
            The atomic parameters, by default None.
        efield : np.ndarray, optional
            The electric field, by default None.
        mixed_type : bool, optional
            Whether the system contains mixed atom types, by default False.

        Returns
        -------
        energy
            The energy of the system, in shape (nframes,).
        force
            The force of the system, in shape (nframes, natoms, 3).
        virial
            The virial of the system, in shape (nframes, 9).
        atomic_energy
            The atomic energy of the system, in shape (nframes, natoms). Only returned
            when atomic is True.
        atomic_virial
            The atomic virial of the system, in shape (nframes, natoms, 9). Only returned
            when atomic is True.
        """
        # This method has been used by:
        # documentation python.md
        # dp model_devi: +fparam, +aparam, +mixed_type
        # dp test: +atomic, +fparam, +aparam, +efield, +mixed_type
        # finetune: +mixed_type
        # dpdata
        # ase

    @abstractmethod
    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""

    @abstractmethod
    def get_type_map(self) -> List[str]:
        """Get the type map (element name of the atom types) of this model."""


__all__ = ["DeepPot"]
