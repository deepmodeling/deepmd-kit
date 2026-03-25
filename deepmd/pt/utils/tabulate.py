# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch-specific DPTabulate wrapper.

Inherits the numpy math from ``deepmd.utils.tabulate_math.DPTabulate``
and adds torch-specific ``_convert_numpy_to_tensor`` and
``_get_descrpt_type`` (isinstance checks against PT descriptor classes).
"""

from typing import (
    Any,
)

import torch

import deepmd
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.tabulate_math import DPTabulate as DPTabulateBase
from deepmd.utils.tabulate_math import (  # noqa: F401 — re-export for test compatibility
    unaggregated_dy2_dx,
    unaggregated_dy2_dx_s,
    unaggregated_dy_dx,
    unaggregated_dy_dx_s,
)


class DPTabulate(DPTabulateBase):
    r"""PyTorch tabulation wrapper.

    Accepts a PT ``ActivationFn`` module and delegates all math to the
    numpy base class. Only overrides tensor conversion and descriptor
    type detection.

    Parameters
    ----------
    descrpt
        Descriptor of the original model.
    neuron
        Number of neurons in each hidden layer of the embedding net.
    type_one_side
        Try to build N_types tables.
    exclude_types
        Excluded type pairs.
    activation_fn
        The activation function name or PT ``ActivationFn`` module.
    """

    def __init__(
        self,
        descrpt: Any,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] | None = None,
        activation_fn: str | ActivationFn = "tanh",
    ) -> None:
        exclude_types = [] if exclude_types is None else exclude_types
        activation_fn_name = (
            activation_fn.activation
            if isinstance(activation_fn, ActivationFn)
            else str(activation_fn)
        )
        super().__init__(
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            activation_fn_name=activation_fn_name,
        )

    def _get_descrpt_type(self) -> str:
        """Detect descriptor type via isinstance checks against PT classes."""
        if isinstance(
            self.descrpt,
            (
                deepmd.pt.model.descriptor.DescrptDPA1,
                deepmd.pt.model.descriptor.DescrptDPA2,
            ),
        ):
            return "Atten"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
            return "A"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
            return "R"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            return "T"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeTTebd):
            return "T_TEBD"
        raise RuntimeError(f"Unsupported descriptor {self.descrpt}")

    def _convert_numpy_to_tensor(self) -> None:
        """Convert self.data from np.ndarray to torch.Tensor."""
        self._convert_numpy_float_to_int()
        for ii in self.data:
            self.data[ii] = torch.tensor(self.data[ii], device=env.DEVICE)  # pylint: disable=no-explicit-dtype
