# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPTabulate for the pt_expt backend.

Subclasses the pt backend's DPTabulate, overriding _get_descrpt_type() to
detect descriptor types via serialized data rather than isinstance checks
against pt-specific classes. Also overrides __init__ to handle the DPA2
repinit_variable extraction without isinstance.
"""

from typing import (
    Any,
)

from deepmd.pt.utils.tabulate import DPTabulate as DPTabulatePT
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.tabulate import (
    BaseTabulate,
)


class DPTabulate(DPTabulatePT):
    """Tabulation helper for pt_expt descriptors.

    Parameters
    ----------
    descrpt
        The pt_expt descriptor instance.
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
        # Call BaseTabulate.__init__ directly (skip DPTabulatePT.__init__)
        # to avoid the isinstance(descrpt, DescrptDPA2) check.
        BaseTabulate.__init__(
            self,
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            True,
        )
        self.descrpt_type = self._get_descrpt_type()

        supported_descrpt_type = ("Atten", "A", "T", "T_TEBD", "R")
        if self.descrpt_type in supported_descrpt_type:
            self.sel_a = self.descrpt.get_sel()
            self.rcut = self.descrpt.get_rcut()
            self.rcut_smth = self.descrpt.get_rcut_smth()
        else:
            raise RuntimeError("Unsupported descriptor")

        activation_map = {
            "tanh": 1,
            "gelu": 2,
            "gelu_tf": 2,
            "relu": 3,
            "relu6": 4,
            "softplus": 5,
            "sigmoid": 6,
            "silu": 7,
        }
        activation = activation_fn.activation
        if activation in activation_map:
            self.functype = activation_map[activation]
        else:
            raise RuntimeError("Unknown activation function type!")

        self.activation_fn = activation_fn
        serialized = self.descrpt.serialize()
        # For DPA2, use repinit_variable (detected by presence of key)
        if "repinit_variable" in serialized:
            serialized = serialized["repinit_variable"]
        self.davg = serialized["@variables"]["davg"]
        self.dstd = serialized["@variables"]["dstd"]
        self.embedding_net_nodes = serialized["embeddings"]["networks"]

        self.ntypes = self.descrpt.get_ntypes()

        self.layer_size = self._get_layer_size()
        self.table_size = self._get_table_size()

        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        self.data_type = self._get_data_type()
        self.last_layer_size = self._get_last_layer_size()

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
            "dpa2": "Atten",
        }

        descrpt_type = type_map.get(type_str)
        if descrpt_type is None:
            raise RuntimeError(f"Unsupported descriptor type: {type_str}")
        return descrpt_type
