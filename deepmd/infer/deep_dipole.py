from deepmd.infer.deep_tensor import DeepTensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class DeepDipole(DeepTensor):
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
                "t_tensor": "o_dipole:0",
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
