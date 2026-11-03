# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)

from .invar_fitting import (
    InvarFitting,
)


@fitting_check_output
class LatentChargeFitting(InvarFitting):
    r"""Per-atom latent charges predicted from the invariant descriptor.

    The network maps the descriptor of an atom, together with the frame
    parameters, atomic parameters and case embedding of the energy fitting it
    accompanies, to ``dim_out`` latent charge channels in units of the
    elementary charge :math:`e`. The charges are an intermediate field of a
    long-range module rather than a model output: the module constrains and
    pairs them, and the potential energy is the only quantity they are
    trained through.

    The constructor arguments are those of :class:`InvarFitting` with the
    output variable fixed to ``latent_charge``.
    """

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "latent_charge"
        return data
