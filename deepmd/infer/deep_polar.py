from deepmd.infer.deep_tensor import DeepTensor
import numpy as np

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pathlib import Path


class DeepPolar(DeepTensor):
    """Constructor.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation

    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self, model_file: "Path", load_prefix: str = "load", default_tf_graph: bool = False
    ) -> None:

        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # output tensor
                "t_tensor": "o_polar:0",
            },
            **self.tensors
        )

        DeepTensor.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
        )

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")


class DeepGlobalPolar(DeepTensor):
    """Constructor.

    Parameters
    ----------
    model_file : str
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    """

    def __init__(
        self, model_file: str, load_prefix: str = "load", default_tf_graph: bool = False
    ) -> None:

        self.tensors.update(
            {
                "t_sel_type": "model_attr/sel_type:0",
                # output tensor
                "t_tensor": "o_global_polar:0",
            }
        )

        DeepTensor.__init__(
            self,
            model_file,
            9,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
        )

    def eval(
        self,
        coords: np.array,
        cells: np.array,
        atom_types: List[int],
        atomic: bool = True,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
        efield: Optional[np.array] = None,
    ) -> np.array:
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
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            Not used in this model
        aparam
            Not used in this model
        efield
            Not used in this model

        Returns
        -------
        tensor
            The returned tensor
            If atomic == False then of size nframes x variable_dof
            else of size nframes x natoms x variable_dof
        """
        return DeepTensor.eval(self, coords, cells, atom_types, atomic=False)

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
