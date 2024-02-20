# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.infer.deep_tensor import (
    DeepTensor,
)


class DeepDipole(DeepTensor):
    """Deep dipole model.

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
        return "dipole"
