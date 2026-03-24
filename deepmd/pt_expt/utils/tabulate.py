# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPTabulate for the pt_expt backend.

Subclasses the pt backend's DPTabulate, overriding _get_descrpt_type() to
detect descriptor types via serialized data rather than isinstance checks
against pt-specific classes.
"""

from typing import (
    Any,
)

from deepmd.pt.utils.tabulate import DPTabulate as DPTabulatePT
from deepmd.pt.utils.utils import (
    ActivationFn,
)


class DPTabulate(DPTabulatePT):
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
    activation_fn
        The activation function used in the embedding net.
    """

    def __init__(
        self,
        descrpt: Any,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn: ActivationFn = ActivationFn("tanh"),
    ) -> None:
        # Parent's __init__ uses `deepmd.pt.model.descriptor.DescrptDPA2` via
        # lazy attribute access (`import deepmd` + `deepmd.pt.model.descriptor`).
        # Ensure the submodule is imported so the attribute chain resolves.
        import deepmd.pt.model.descriptor  # noqa: F401

        super().__init__(descrpt, neuron, type_one_side, exclude_types, activation_fn)

    def _get_descrpt_type(self) -> str:
        """Determine descriptor type from serialized data.

        Instead of isinstance checks against pt classes, use the "type" key
        from the serialized descriptor dict.
        """
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
