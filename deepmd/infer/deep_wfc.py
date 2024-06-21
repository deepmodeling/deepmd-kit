# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.infer.deep_tensor import (
    OldDeepTensor,
)


class DeepWFC(OldDeepTensor):
    """Deep WFC model.

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
        return "wfc"

    @property
    def output_def(self) -> ModelOutputDef:
        """Get the output definition of this model."""
        # no reducible or differentiable output is defined
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        self.output_tensor_name,
                        shape=[-1],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                        atomic=True,
                    ),
                ]
            )
        )
