# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPTabulate for the pt_expt backend.

Inherits the numpy math from ``deepmd.utils.tabulate_math.DPTabulate``
and overrides ``_convert_numpy_to_tensor`` for torch tensor conversion
and ``_get_descrpt_type`` for serialization-based type detection.
No dependency on the pt backend.
"""

from typing import (
    Any,
)

from deepmd.utils.tabulate_math import DPTabulate as DPTabulateBase


class DPTabulate(DPTabulateBase):
    """Tabulation helper for pt_expt descriptors.

    The descriptor passed to this class must serialize to a dict with
    ``@variables`` (davg/dstd) and ``embeddings`` at the top level.
    For DPA2, pass the **repinit block** (not the full DPA2 descriptor)
    so that ``serialize()`` returns the correct structure directly.

    Parameters
    ----------
    descrpt
        The pt_expt descriptor (or sub-block) instance.
    neuron
        Number of neurons in each hidden layer of the embedding net.
    type_one_side
        Whether to use one-side type embedding.
    exclude_types
        Excluded type pairs.
    activation_fn_name
        Name of the activation function (e.g. "tanh", "gelu").
    """

    def __init__(
        self,
        descrpt: Any,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__(
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            activation_fn_name=activation_fn_name,
        )

    def _get_descrpt_type(self) -> str:
        """Determine descriptor type from serialized data."""
        data = self.descrpt.serialize()
        type_str = data.get("type", "")
        type_map = {
            "se_e2_a": "A",
            "se_r": "R",
            "se_e3": "T",
            "se_e3_tebd": "T_TEBD",
            "dpa1": "Atten",
            "se_atten_v2": "Atten",
        }
        descrpt_type = type_map.get(type_str)
        if descrpt_type is None:
            raise RuntimeError(f"Unsupported descriptor type: {type_str}")
        return descrpt_type

    def _convert_numpy_to_tensor(self) -> None:
        """Convert self.data from np.ndarray to torch.Tensor."""
        import torch

        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        self._convert_numpy_float_to_int()
        for ii in self.data:
            self.data[ii] = torch.tensor(self.data[ii], device=DEVICE)  # pylint: disable=no-explicit-dtype
