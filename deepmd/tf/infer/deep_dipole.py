# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.tf.infer.deep_tensor import (
    DeepTensor,
)

__all__ = [
    "DeepDipole",
]


class DeepDipoleOld(DeepTensor):
    # used for DipoleChargeModifier only
    """Constructor.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    input_map : dict, optional
        The input map for tf.import_graph_def. Only work with default tf graph
    neighbor_list : ase.neighborlist.NeighborList, optional
        The neighbor list object. If None, then build the native neighbor list.

    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False,
        input_map: Optional[dict] = None,
        neighbor_list=None,
    ) -> None:
        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # output tensor
                "t_tensor": "o_dipole:0",
            },
            **self.tensors,
        )

        DeepTensor.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
