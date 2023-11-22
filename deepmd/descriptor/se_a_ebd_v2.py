# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

from deepmd.utils.spin import (
    Spin,
)

from .descriptor import (
    Descriptor,
)
from deepmd.env import (
    tf,
)
import numpy as np
from deepmd.utils.compress import (
    get_extra_side_embedding_net_variable,
    get_two_side_type_embedding,
    make_data,
)
from deepmd.utils.graph import (
    get_attention_layer_variables_from_graph_def,
    get_pattern_nodes_from_graph_def,
    get_tensor_by_name_from_graph,
    get_tensor_by_type,
)
from .se_a import (
    DescrptSeA,
)

log = logging.getLogger(__name__)


@Descriptor.register("se_a_tpe_v2")
@Descriptor.register("se_a_ebd_v2")
class DescrptSeAEbdV2(DescrptSeA):
    r"""A compressible se_a_ebd model.

    This model is a warpper for DescriptorSeA, which set stripped_type_embedding=True.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        multi_task: bool = False,
        spin: Optional[Spin] = None,
        **kwargs,
    ) -> None:
        DescrptSeA.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            neuron=neuron,
            axis_neuron=axis_neuron,
            resnet_dt=resnet_dt,
            trainable=trainable,
            seed=seed,
            type_one_side=type_one_side,
            exclude_types=exclude_types,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            uniform_seed=uniform_seed,
            multi_task=multi_task,
            spin=spin,
            stripped_type_embedding=True,
            **kwargs,
        )
    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        super().init_variables(graph=graph, graph_def=graph_def, suffix=suffix)
        self.extra_embedding_net_variables = {}
        if self.type_one_side:
            extra_suffix = "_one_side_ebd"
        else:
            extra_suffix = "_two_side_ebd"
        for i in range(1, self.layer_size + 1):
            matrix_pattern = f"filter_type_all{suffix}/matrix_{i}{extra_suffix}"
            self.extra_embedding_net_variables[
                matrix_pattern
            ] = self._get_two_embed_variables(graph_def, matrix_pattern)
            bias_pattern = f"filter_type_all{suffix}/bias_{i}{extra_suffix}"
            self.extra_embedding_net_variables[
                bias_pattern
            ] = self._get_two_embed_variables(graph_def, bias_pattern)

    def _get_two_embed_variables(self, graph_def, pattern: str):
        node = get_pattern_nodes_from_graph_def(graph_def, pattern)[pattern]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(
                node.tensor_content,
                dtype=tf.as_dtype(node.dtype).as_numpy_dtype,
            )
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        return np.reshape(tensor_value, tensor_shape)
