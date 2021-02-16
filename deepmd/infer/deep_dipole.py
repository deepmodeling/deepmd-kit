from deepmd.infer.deep_eval import DeepTensor


class DeepDipole(DeepTensor):
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
                "t_tensor": "o_dipole:0",
            }
        )

        DeepTensor.__init__(
            self,
            model_file,
            3,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
        )

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
