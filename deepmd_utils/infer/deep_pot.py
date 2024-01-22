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

from deepmd_utils.utils import (
    AutoBatchSize,
)

from .backend import (
    Backend,
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
        ...

    def __new__(cls, model_file: str, *args, **kwargs):
        if cls is DeepPot:
            backend = detect_backend(model_file)
            if backend == Backend.TensorFlow:
                from deepmd.infer.deep_pot import DeepPot as DeepPotTF

                return super().__new__(DeepPotTF)
            elif backend == Backend.PyTorch:
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
        """Evaluate the model."""
        # This method has been used by:
        # documentation python.md
        # dp model_devi: +fparam, +aparam, +mixed_type
        # dp test: +atomic, +fparam, +aparam, +efield, +mixed_type
        # finetune: +mixed_type
        # dpdata
        # ase


__all__ = ["DeepPot"]
