# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging
from typing import (
    Callable,
    List,
    Optional,
)

from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)

from deepmd.utils.argcheck_nvnmd import (
    nvnmd_args,
)
from deepmd.utils.plugin import (
    Plugin,
)

log = logging.getLogger(__name__)


# TODO: import from a module outside tf/pt
ACTIVATION_FN_DICT = {
    "relu": None,
    "relu6": None,
    "softplus": None,
    "sigmoid": None,
    "tanh": None,
    "gelu": None,
    "gelu_tf": None,
    "None": None,
    "none": None,
}
# TODO: import from a module outside tf/pt
PRECISION_DICT = {
    "default": None,
    "float16": None,
    "float32": None,
    "float64": None,
    "bfloat16": None,
}

doc_only_tf_supported = "(Supported Backend: TensorFlow) "
doc_only_pt_supported = "(Supported Backend: PyTorch) "


def list_to_doc(xx):
    items = []
    for ii in xx:
        if len(items) == 0:
            items.append(f'"{ii}"')
        else:
            items.append(f', "{ii}"')
    items.append(".")
    return "".join(items)


def make_link(content, ref_key):
    return (
        f"`{content} <{ref_key}_>`_"
        if not dargs.RAW_ANCHOR
        else f"`{content} <#{ref_key}>`_"
    )


def type_embedding_args():
    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_seed = "Random seed for parameter initialization"
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_precision = f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_trainable = "If the parameters in the embedding net are trainable"

    return [
        Argument("neuron", List[int], optional=True, default=[8], doc=doc_neuron),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, default=None, doc=doc_seed),
    ]


def spin_args():
    doc_use_spin = "Whether to use atomic spin model for each atom type"
    doc_spin_norm = "The magnitude of atomic spin for each atom type with spin"
    doc_virtual_len = "The distance between virtual atom representing spin and its corresponding real atom for each atom type with spin"

    return [
        Argument("use_spin", List[bool], doc=doc_use_spin),
        Argument("spin_norm", List[float], doc=doc_spin_norm),
        Argument("virtual_len", List[float], doc=doc_virtual_len),
    ]


#  --- Descriptor configurations: --- #


class ArgsPlugin:
    def __init__(self) -> None:
        self.__plugin = Plugin()

    def register(
        self, name: str, alias: Optional[List[str]] = None, doc: str = ""
    ) -> Callable[[], List[Argument]]:
        """Register a descriptor argument plugin.

        Parameters
        ----------
        name : str
            the name of a descriptor
        alias : List[str], optional
            the list of aliases of this descriptor

        Returns
        -------
        Callable[[], List[Argument]]
            the registered descriptor argument method

        Examples
        --------
        >>> some_plugin = ArgsPlugin()
        >>> @some_plugin.register("some_descrpt")
            def descrpt_some_descrpt_args():
                return []
        """
        # convert alias to hashed item
        if isinstance(alias, list):
            alias = tuple(alias)
        return self.__plugin.register((name, alias, doc))

    def get_all_argument(self, exclude_hybrid: bool = False) -> List[Argument]:
        """Get all arguments.

        Parameters
        ----------
        exclude_hybrid : bool
            exclude hybrid descriptor to prevent circular calls

        Returns
        -------
        List[Argument]
            all arguments
        """
        arguments = []
        for (name, alias, doc), metd in self.__plugin.plugins.items():
            if exclude_hybrid and name == "hybrid":
                continue
            arguments.append(
                Argument(name=name, dtype=dict, sub_fields=metd(), alias=alias, doc=doc)
            )
        return arguments


descrpt_args_plugin = ArgsPlugin()


@descrpt_args_plugin.register("loc_frame", doc=doc_only_tf_supported)
def descrpt_local_frame_args():
    doc_sel_a = "A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_a[i]` gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor."
    doc_sel_r = "A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_r[i]` gives the selected number of type-i neighbors. Only relative distance of the neighbors are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius."
    doc_rcut = "The cut-off radius. The default value is 6.0"
    doc_axis_rule = "A list of integers. The length should be 6 times of the number of types. \n\n\
- axis_rule[i*6+0]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
- axis_rule[i*6+1]: type of the atom defining the first axis of type-i atom.\n\n\
- axis_rule[i*6+2]: index of the axis atom defining the first axis. Note that the neighbors with the same class and type are sorted according to their relative distance.\n\n\
- axis_rule[i*6+3]: class of the atom defining the second axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
- axis_rule[i*6+4]: type of the atom defining the second axis of type-i atom.\n\n\
- axis_rule[i*6+5]: index of the axis atom defining the second axis. Note that the neighbors with the same class and type are sorted according to their relative distance."

    return [
        Argument("sel_a", List[int], optional=False, doc=doc_sel_a),
        Argument("sel_r", List[int], optional=False, doc=doc_sel_r),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("axis_rule", List[int], optional=False, doc=doc_axis_rule),
    ]


@descrpt_args_plugin.register("se_e2_a", alias=["se_a"])
def descrpt_se_a_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
    doc_rcut = "The cut-off radius."
    doc_rcut_smth = "Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`"
    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_axis_neuron = "Size of the submatrix of G (embedding matrix)."
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_type_one_side = r"If true, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters."
    doc_precision = f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_trainable = "If the parameters in the embedding net is trainable"
    doc_seed = "Random seed for parameter initialization"
    doc_exclude_types = "The excluded pairs of types which have no interaction with each other. For example, `[[0, 1]]` means no interaction between type 0 and type 1."
    doc_set_davg_zero = "Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used"

    return [
        Argument("sel", [List[int], str], optional=True, default="auto", doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument(
            "neuron", List[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        Argument(
            "axis_neuron",
            int,
            optional=True,
            default=4,
            alias=["n_axis_neuron"],
            doc=doc_axis_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument(
            "type_one_side", bool, optional=True, default=False, doc=doc_type_one_side
        ),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "exclude_types",
            List[List[int]],
            optional=True,
            default=[],
            doc=doc_exclude_types,
        ),
        Argument(
            "set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero
        ),
    ]


@descrpt_args_plugin.register(
    "se_e3", alias=["se_at", "se_a_3be", "se_t"], doc=doc_only_tf_supported
)
def descrpt_se_t_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
    doc_rcut = "The cut-off radius."
    doc_rcut_smth = "Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`"
    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_trainable = "If the parameters in the embedding net are trainable"
    doc_seed = "Random seed for parameter initialization"
    doc_set_davg_zero = "Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used"

    return [
        Argument("sel", [List[int], str], optional=True, default="auto", doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument(
            "neuron", List[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero
        ),
    ]


@descrpt_args_plugin.register("se_a_tpe", alias=["se_a_ebd"], doc=doc_only_tf_supported)
def descrpt_se_a_tpe_args():
    doc_type_nchanl = "number of channels for type embedding"
    doc_type_nlayer = "number of hidden layers of type embedding net"
    doc_numb_aparam = "dimension of atomic parameter. if set to a value > 0, the atomic parameters are embedded."

    return [
        *descrpt_se_a_args(),
        Argument("type_nchanl", int, optional=True, default=4, doc=doc_type_nchanl),
        Argument("type_nlayer", int, optional=True, default=2, doc=doc_type_nlayer),
        Argument("numb_aparam", int, optional=True, default=0, doc=doc_numb_aparam),
    ]


@descrpt_args_plugin.register("se_e2_r", alias=["se_r"])
def descrpt_se_r_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
    doc_rcut = "The cut-off radius."
    doc_rcut_smth = "Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`"
    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_type_one_side = r"If true, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters."
    doc_precision = f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_trainable = "If the parameters in the embedding net are trainable"
    doc_seed = "Random seed for parameter initialization"
    doc_exclude_types = "The excluded pairs of types which have no interaction with each other. For example, `[[0, 1]]` means no interaction between type 0 and type 1."
    doc_set_davg_zero = "Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used"

    return [
        Argument("sel", [List[int], str], optional=True, default="auto", doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument(
            "neuron", List[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument(
            "type_one_side", bool, optional=True, default=False, doc=doc_type_one_side
        ),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "exclude_types",
            List[List[int]],
            optional=True,
            default=[],
            doc=doc_exclude_types,
        ),
        Argument(
            "set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero
        ),
    ]


@descrpt_args_plugin.register("hybrid")
def descrpt_hybrid_args():
    doc_list = "A list of descriptor definitions"

    return [
        Argument(
            "list",
            list,
            optional=False,
            doc=doc_list,
            repeat=True,
            sub_fields=[],
            sub_variants=[descrpt_variant_type_args(exclude_hybrid=True)],
            fold_subdoc=True,
        )
    ]


def descrpt_se_atten_common_args():
    doc_sel = 'This parameter set the number of selected neighbors. Note that this parameter is a little different from that in other descriptors. Instead of separating each type of atoms, only the summation matters. And this number is highly related with the efficiency, thus one should not make it too large. Usually 200 or less is enough, far away from the GPU limitation 4096. It can be:\n\n\
    - `int`. The maximum number of neighbor atoms to be considered. We recommend it to be less than 200. \n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. Only the summation of `sel[i]` matters, and it is recommended to be less than 200.\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
    doc_rcut = "The cut-off radius."
    doc_rcut_smth = "Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`"
    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_axis_neuron = "Size of the submatrix of G (embedding matrix)."
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = (
        doc_only_tf_supported + 'Whether to use a "Timestep" in the skip connection'
    )
    doc_type_one_side = (
        doc_only_tf_supported
        + r"If true, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters."
    )
    doc_precision = (
        doc_only_tf_supported
        + f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    )
    doc_trainable = (
        doc_only_tf_supported + "If the parameters in the embedding net is trainable"
    )
    doc_seed = "Random seed for parameter initialization"
    doc_exclude_types = (
        doc_only_tf_supported
        + "The excluded pairs of types which have no interaction with each other. For example, `[[0, 1]]` means no interaction between type 0 and type 1."
    )
    doc_attn = "The length of hidden vectors in attention layers"
    doc_attn_layer = "The number of attention layers. Note that model compression of `se_atten` is only enabled when attn_layer==0 and stripped_type_embedding is True"
    doc_attn_dotr = "Whether to do dot product with the normalized relative coordinates"
    doc_attn_mask = "Whether to do mask on the diagonal in the attention matrix"

    return [
        Argument(
            "sel", [int, List[int], str], optional=True, default="auto", doc=doc_sel
        ),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument(
            "neuron", List[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        Argument(
            "axis_neuron",
            int,
            optional=True,
            default=4,
            alias=["n_axis_neuron"],
            doc=doc_axis_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument(
            "type_one_side", bool, optional=True, default=False, doc=doc_type_one_side
        ),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "exclude_types",
            List[List[int]],
            optional=True,
            default=[],
            doc=doc_exclude_types,
        ),
        Argument("attn", int, optional=True, default=128, doc=doc_attn),
        Argument("attn_layer", int, optional=True, default=2, doc=doc_attn_layer),
        Argument("attn_dotr", bool, optional=True, default=True, doc=doc_attn_dotr),
        Argument("attn_mask", bool, optional=True, default=False, doc=doc_attn_mask),
    ]


@descrpt_args_plugin.register("se_atten", alias=["dpa1"])
def descrpt_se_atten_args():
    doc_stripped_type_embedding = "Whether to strip the type embedding into a separated embedding network. Setting it to `False` will fall back to the previous version of `se_atten` which is non-compressible."
    doc_smooth_type_embdding = "When using stripped type embedding, whether to dot smooth factor on the network output of type embedding to keep the network smooth, instead of setting `set_davg_zero` to be True."
    doc_set_davg_zero = "Set the normalization average to zero. This option should be set when `se_atten` descriptor or `atom_ener` in the energy fitting is used"
    doc_tebd_dim = "The dimension of atom type embedding."
    doc_temperature = "The scaling factor of normalization in calculations of attention weights, which is used to scale the matmul(Q, K)."
    doc_scaling_factor = (
        "The scaling factor of normalization in calculations of attention weights, which is used to scale the matmul(Q, K). "
        "If `temperature` is None, the scaling of attention weights is (N_hidden_dim * scaling_factor)**0.5. "
        "Else, the scaling of attention weights is setting to `temperature`."
    )
    doc_normalize = (
        "Whether to normalize the hidden vectors during attention calculation."
    )
    doc_concat_output_tebd = (
        "Whether to concat type embedding at the output of the descriptor."
    )
    doc_deprecated = "This feature will be removed in a future release."

    return [
        *descrpt_se_atten_common_args(),
        Argument(
            "stripped_type_embedding",
            bool,
            optional=True,
            default=False,
            doc=doc_only_tf_supported + doc_stripped_type_embedding,
        ),
        Argument(
            "smooth_type_embdding",
            bool,
            optional=True,
            default=False,
            doc=doc_only_tf_supported + doc_smooth_type_embdding,
        ),
        Argument(
            "set_davg_zero", bool, optional=True, default=True, doc=doc_set_davg_zero
        ),
        # pt only
        Argument(
            "tebd_dim",
            int,
            optional=True,
            default=8,
            doc=doc_only_pt_supported + doc_tebd_dim,
        ),
        Argument(
            "tebd_input_mode",
            str,
            optional=True,
            default="concat",
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "post_ln",
            bool,
            optional=True,
            default=True,
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "ffn",
            bool,
            optional=True,
            default=False,
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "ffn_embed_dim",
            int,
            optional=True,
            default=1024,
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "scaling_factor",
            float,
            optional=True,
            default=1.0,
            doc=doc_only_pt_supported + doc_scaling_factor,
        ),
        Argument(
            "head_num",
            int,
            optional=True,
            default=1,
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "normalize",
            bool,
            optional=True,
            default=True,
            doc=doc_only_pt_supported + doc_normalize,
        ),
        Argument(
            "temperature",
            float,
            optional=True,
            doc=doc_only_pt_supported + doc_temperature,
        ),
        Argument(
            "return_rot",
            bool,
            optional=True,
            default=False,
            doc=doc_only_pt_supported + doc_deprecated,
        ),
        Argument(
            "concat_output_tebd",
            bool,
            optional=True,
            default=True,
            doc=doc_only_pt_supported + doc_concat_output_tebd,
        ),
    ]


@descrpt_args_plugin.register("se_atten_v2", doc=doc_only_tf_supported)
def descrpt_se_atten_v2_args():
    doc_set_davg_zero = "Set the normalization average to zero. This option should be set when `se_atten` descriptor or `atom_ener` in the energy fitting is used"

    return [
        *descrpt_se_atten_common_args(),
        Argument(
            "set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero
        ),
    ]


@descrpt_args_plugin.register("dpa2", doc=doc_only_pt_supported)
def descrpt_dpa2_args():
    # Generate by GitHub Copilot
    doc_repinit_rcut = "The cut-off radius of the repinit block"
    doc_repinit_rcut_smth = "From this position the inverse distance smoothly decays to 0 at the cut-off. Use in the repinit block."
    doc_repinit_nsel = "Maximally possible number of neighbors for repinit block."
    doc_repformer_rcut = "The cut-off radius of the repformer block"
    doc_repformer_rcut_smth = "From this position the inverse distance smoothly decays to 0 at the cut-off. Use in the repformer block."
    doc_repformer_nsel = "Maximally possible number of neighbors for repformer block."
    doc_tebd_dim = "The dimension of atom type embedding"
    doc_concat_output_tebd = (
        "Whether to concat type embedding at the output of the descriptor."
    )
    doc_repinit_neuron = "repinit block: the number of neurons in the embedding net."
    doc_repinit_axis_neuron = (
        "repinit block: the number of dimension of split in the symmetrization op."
    )
    doc_repinit_activation = (
        "repinit block: the activation function in the embedding net"
    )
    doc_repformer_nlayers = "repformers block: the number of repformer layers"
    doc_repformer_g1_dim = "repformers block: the dimension of single-atom rep"
    doc_repformer_g2_dim = "repformers block: the dimension of invariant pair-atom rep"
    doc_repformer_axis_dim = (
        "repformers block: the number of dimension of split in the symmetrization ops."
    )
    doc_repformer_do_bn_mode = "repformers block: do batch norm in the repformer layers"
    doc_repformer_bn_momentum = "repformers block: moment in the batch normalization"
    doc_repformer_update_g1_has_conv = (
        "repformers block: update the g1 rep with convolution term"
    )
    doc_repformer_update_g1_has_drrd = (
        "repformers block: update the g1 rep with the drrd term"
    )
    doc_repformer_update_g1_has_grrg = (
        "repformers block: update the g1 rep with the grrg term"
    )
    doc_repformer_update_g1_has_attn = (
        "repformers block: update the g1 rep with the localized self-attention"
    )
    doc_repformer_update_g2_has_g1g1 = (
        "repformers block: update the g2 rep with the g1xg1 term"
    )
    doc_repformer_update_g2_has_attn = (
        "repformers block: update the g2 rep with the gated self-attention"
    )
    doc_repformer_update_h2 = "repformers block: update the h2 rep"
    doc_repformer_attn1_hidden = (
        "repformers block: the hidden dimension of localized self-attention"
    )
    doc_repformer_attn1_nhead = (
        "repformers block: the number of heads in localized self-attention"
    )
    doc_repformer_attn2_hidden = (
        "repformers block: the hidden dimension of gated self-attention"
    )
    doc_repformer_attn2_nhead = (
        "repformers block: the number of heads in gated self-attention"
    )
    doc_repformer_attn2_has_gate = (
        "repformers block: has gate in the gated self-attention"
    )
    doc_repformer_activation = "repformers block: the activation function in the MLPs."
    doc_repformer_update_style = "repformers block: style of update a rep. can be res_avg or res_incr. res_avg updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n) res_incr updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)"
    doc_repformer_set_davg_zero = "repformers block: set the avg to zero in statistics"
    doc_repformer_add_type_ebd_to_seq = (
        "repformers block: concatenate the type embedding at the output"
    )
    return [
        Argument("repinit_rcut", float, doc=doc_repinit_rcut),
        Argument("repinit_rcut_smth", float, doc=doc_repinit_rcut_smth),
        Argument("repinit_nsel", int, doc=doc_repinit_nsel),
        Argument("repformer_rcut", float, doc=doc_repformer_rcut),
        Argument("repformer_rcut_smth", float, doc=doc_repformer_rcut_smth),
        Argument("repformer_nsel", int, doc=doc_repformer_nsel),
        Argument("tebd_dim", int, optional=True, default=8, doc=doc_tebd_dim),
        Argument(
            "concat_output_tebd",
            bool,
            optional=True,
            default=True,
            doc=doc_concat_output_tebd,
        ),
        Argument(
            "repinit_neuron",
            list,
            optional=True,
            default=[25, 50, 100],
            doc=doc_repinit_neuron,
        ),
        Argument(
            "repinit_axis_neuron",
            int,
            optional=True,
            default=16,
            doc=doc_repinit_axis_neuron,
        ),
        Argument("repinit_set_davg_zero", bool, optional=True, default=True),
        Argument(
            "repinit_activation",
            str,
            optional=True,
            default="tanh",
            doc=doc_repinit_activation,
        ),
        Argument(
            "repformer_nlayers",
            int,
            optional=True,
            default=3,
            doc=doc_repformer_nlayers,
        ),
        Argument(
            "repformer_g1_dim",
            int,
            optional=True,
            default=128,
            doc=doc_repformer_g1_dim,
        ),
        Argument(
            "repformer_g2_dim", int, optional=True, default=16, doc=doc_repformer_g2_dim
        ),
        Argument(
            "repformer_axis_dim",
            int,
            optional=True,
            default=4,
            doc=doc_repformer_axis_dim,
        ),
        Argument(
            "repformer_do_bn_mode",
            str,
            optional=True,
            default="no",
            doc=doc_repformer_do_bn_mode,
        ),
        Argument(
            "repformer_bn_momentum",
            float,
            optional=True,
            default=0.1,
            doc=doc_repformer_bn_momentum,
        ),
        Argument(
            "repformer_update_g1_has_conv",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g1_has_conv,
        ),
        Argument(
            "repformer_update_g1_has_drrd",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g1_has_drrd,
        ),
        Argument(
            "repformer_update_g1_has_grrg",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g1_has_grrg,
        ),
        Argument(
            "repformer_update_g1_has_attn",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g1_has_attn,
        ),
        Argument(
            "repformer_update_g2_has_g1g1",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g2_has_g1g1,
        ),
        Argument(
            "repformer_update_g2_has_attn",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_update_g2_has_attn,
        ),
        Argument(
            "repformer_update_h2",
            bool,
            optional=True,
            default=False,
            doc=doc_repformer_update_h2,
        ),
        Argument(
            "repformer_attn1_hidden",
            int,
            optional=True,
            default=64,
            doc=doc_repformer_attn1_hidden,
        ),
        Argument(
            "repformer_attn1_nhead",
            int,
            optional=True,
            default=4,
            doc=doc_repformer_attn1_nhead,
        ),
        Argument(
            "repformer_attn2_hidden",
            int,
            optional=True,
            default=16,
            doc=doc_repformer_attn2_hidden,
        ),
        Argument(
            "repformer_attn2_nhead",
            int,
            optional=True,
            default=4,
            doc=doc_repformer_attn2_nhead,
        ),
        Argument(
            "repformer_attn2_has_gate",
            bool,
            optional=True,
            default=False,
            doc=doc_repformer_attn2_has_gate,
        ),
        Argument(
            "repformer_activation",
            str,
            optional=True,
            default="tanh",
            doc=doc_repformer_activation,
        ),
        Argument(
            "repformer_update_style",
            str,
            optional=True,
            default="res_avg",
            doc=doc_repformer_update_style,
        ),
        Argument(
            "repformer_set_davg_zero",
            bool,
            optional=True,
            default=True,
            doc=doc_repformer_set_davg_zero,
        ),
        Argument(
            "repformer_add_type_ebd_to_seq",
            bool,
            optional=True,
            default=False,
            doc=doc_repformer_add_type_ebd_to_seq,
        ),
    ]


@descrpt_args_plugin.register(
    "se_a_ebd_v2", alias=["se_a_tpe_v2"], doc=doc_only_tf_supported
)
def descrpt_se_a_ebd_v2_args():
    return descrpt_se_a_args()


@descrpt_args_plugin.register("se_a_mask", doc=doc_only_tf_supported)
def descrpt_se_a_mask_args():
    doc_sel = 'This parameter sets the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'

    doc_neuron = "Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built."
    doc_axis_neuron = "Size of the submatrix of G (embedding matrix)."
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_type_one_side = r"If true, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters."
    doc_exclude_types = "The excluded pairs of types which have no interaction with each other. For example, `[[0, 1]]` means no interaction between type 0 and type 1."
    doc_precision = f"The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_trainable = "If the parameters in the embedding net is trainable"
    doc_seed = "Random seed for parameter initialization"

    return [
        Argument("sel", [List[int], str], optional=True, default="auto", doc=doc_sel),
        Argument(
            "neuron", List[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        Argument(
            "axis_neuron",
            int,
            optional=True,
            default=4,
            alias=["n_axis_neuron"],
            doc=doc_axis_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument(
            "type_one_side", bool, optional=True, default=False, doc=doc_type_one_side
        ),
        Argument(
            "exclude_types",
            List[List[int]],
            optional=True,
            default=[],
            doc=doc_exclude_types,
        ),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
    ]


def descrpt_variant_type_args(exclude_hybrid: bool = False) -> Variant:
    link_lf = make_link("loc_frame", "model/descriptor[loc_frame]")
    link_se_e2_a = make_link("se_e2_a", "model/descriptor[se_e2_a]")
    link_se_e2_r = make_link("se_e2_r", "model/descriptor[se_e2_r]")
    link_se_e3 = make_link("se_e3", "model/descriptor[se_e3]")
    link_se_a_tpe = make_link("se_a_tpe", "model/descriptor[se_a_tpe]")
    link_hybrid = make_link("hybrid", "model/descriptor[hybrid]")
    link_se_atten = make_link("se_atten", "model/descriptor[se_atten]")
    link_se_atten_v2 = make_link("se_atten_v2", "model/descriptor[se_atten_v2]")
    doc_descrpt_type = "The type of the descritpor. See explanation below. \n\n\
- `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.\n\n\
- `se_e2_a`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor.\n\n\
- `se_e2_r`: Used by the smooth edition of Deep Potential. Only the distance between atoms is used to construct the descriptor.\n\n\
- `se_e3`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Three-body embedding will be used by this descriptor.\n\n\
- `se_a_tpe`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Type embedding will be used by this descriptor.\n\n\
- `se_atten`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Attention mechanism will be used by this descriptor.\n\n\
- `se_atten_v2`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Attention mechanism with new modifications will be used by this descriptor.\n\n\
- `se_a_mask`: Used by the smooth edition of Deep Potential. It can accept a variable number of atoms in a frame (Non-PBC system). *aparam* are required as an indicator matrix for the real/virtual sign of input atoms. \n\n\
- `hybrid`: Concatenate of a list of descriptors as a new descriptor."

    return Variant(
        "type",
        descrpt_args_plugin.get_all_argument(exclude_hybrid=exclude_hybrid),
        doc=doc_descrpt_type,
    )


#  --- Fitting net configurations: --- #
fitting_args_plugin = ArgsPlugin()


@fitting_args_plugin.register("ener")
def fitting_ener():
    doc_numb_fparam = "The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams."
    doc_numb_aparam = "The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams."
    doc_neuron = "The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built."
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_precision = f"The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_trainable = f"Whether the parameters in the fitting net are trainable. This option can be\n\n\
- bool: True if all parameters of the fitting net are trainable, False otherwise.\n\n\
- list of bool{doc_only_tf_supported}: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of this list should be equal to len(`neuron`)+1."
    doc_rcond = "The condition number used to determine the inital energy shift for each type of atoms. See `rcond` in :py:meth:`numpy.linalg.lstsq` for more details."
    doc_seed = "Random seed for parameter initialization of the fitting net"
    doc_atom_ener = "Specify the atomic energy in vacuum for each type"
    doc_layer_name = (
        "The name of the each layer. The length of this list should be equal to n_neuron + 1. "
        "If two layers, either in the same fitting or different fittings, "
        "have the same name, they will share the same neural network parameters. "
        "The shape of these layers should be the same. "
        "If null is given for a layer, parameters will not be shared."
    )
    doc_use_aparam_as_mask = (
        "Whether to use the aparam as a mask in input."
        "If True, the aparam will not be used in fitting net for embedding."
        "When descrpt is se_a_mask, the aparam will be used as a mask to indicate the input atom is real/virtual. And use_aparam_as_mask should be set to True."
    )

    return [
        Argument("numb_fparam", int, optional=True, default=0, doc=doc_numb_fparam),
        Argument("numb_aparam", int, optional=True, default=0, doc=doc_numb_aparam),
        Argument(
            "neuron",
            List[int],
            optional=True,
            default=[120, 120, 120],
            alias=["n_neuron"],
            doc=doc_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument(
            "trainable",
            [List[bool], bool],
            optional=True,
            default=True,
            doc=doc_trainable,
        ),
        Argument(
            "rcond", [float, type(None)], optional=True, default=None, doc=doc_rcond
        ),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "atom_ener",
            List[Optional[float]],
            optional=True,
            default=[],
            doc=doc_atom_ener,
        ),
        Argument("layer_name", List[str], optional=True, doc=doc_layer_name),
        Argument(
            "use_aparam_as_mask",
            bool,
            optional=True,
            default=False,
            doc=doc_use_aparam_as_mask,
        ),
    ]


@fitting_args_plugin.register("dos", doc=doc_only_tf_supported)
def fitting_dos():
    doc_numb_fparam = "The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams."
    doc_numb_aparam = "The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams."
    doc_neuron = "The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built."
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_precision = f"The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_trainable = "Whether the parameters in the fitting net are trainable. This option can be\n\n\
- bool: True if all parameters of the fitting net are trainable, False otherwise.\n\n\
- list of bool: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of tihs list should be equal to len(`neuron`)+1."
    doc_rcond = "The condition number used to determine the inital energy shift for each type of atoms. See `rcond` in :py:meth:`numpy.linalg.lstsq` for more details."
    doc_seed = "Random seed for parameter initialization of the fitting net"
    doc_numb_dos = (
        "The number of gridpoints on which the DOS is evaluated (NEDOS in VASP)"
    )

    return [
        Argument("numb_fparam", int, optional=True, default=0, doc=doc_numb_fparam),
        Argument("numb_aparam", int, optional=True, default=0, doc=doc_numb_aparam),
        Argument(
            "neuron", List[int], optional=True, default=[120, 120, 120], doc=doc_neuron
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("precision", str, optional=True, default="float64", doc=doc_precision),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument(
            "trainable",
            [List[bool], bool],
            optional=True,
            default=True,
            doc=doc_trainable,
        ),
        Argument(
            "rcond", [float, type(None)], optional=True, default=None, doc=doc_rcond
        ),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("numb_dos", int, optional=True, default=300, doc=doc_numb_dos),
    ]


@fitting_args_plugin.register("polar")
def fitting_polar():
    doc_neuron = "The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built."
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f"The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_scale = "The output of the fitting net (polarizability matrix) will be scaled by ``scale``"
    # doc_diag_shift = 'The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.'
    doc_fit_diag = "Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix."
    doc_sel_type = "The atom types for which the atomic polarizability will be provided. If not set, all types will be selected."
    doc_seed = "Random seed for parameter initialization of the fitting net"

    # YWolfeee: user can decide whether to use shift diag
    doc_shift_diag = "Whether to shift the diagonal of polar, which is beneficial to training. Default is true."

    return [
        Argument(
            "neuron",
            List[int],
            optional=True,
            default=[120, 120, 120],
            alias=["n_neuron"],
            doc=doc_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument("fit_diag", bool, optional=True, default=True, doc=doc_fit_diag),
        Argument(
            "scale", [List[float], float], optional=True, default=1.0, doc=doc_scale
        ),
        # Argument("diag_shift", [list,float], optional = True, default = 0.0, doc = doc_diag_shift),
        Argument("shift_diag", bool, optional=True, default=True, doc=doc_shift_diag),
        Argument(
            "sel_type",
            [List[int], int, None],
            optional=True,
            alias=["pol_type"],
            doc=doc_sel_type + doc_only_tf_supported,
        ),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
    ]


# def fitting_global_polar():
#    return fitting_polar()


@fitting_args_plugin.register("dipole")
def fitting_dipole():
    doc_neuron = "The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built."
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())} Note that "gelu" denotes the custom operator version, and "gelu_tf" denotes the TF standard version. If you set "None" or "none" here, no activation function will be used.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f"The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())} Default follows the interface precision."
    doc_sel_type = "The atom types for which the atomic dipole will be provided. If not set, all types will be selected."
    doc_seed = "Random seed for parameter initialization of the fitting net"
    return [
        Argument(
            "neuron",
            List[int],
            optional=True,
            default=[120, 120, 120],
            alias=["n_neuron"],
            doc=doc_neuron,
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc=doc_activation_function,
        ),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default="default", doc=doc_precision),
        Argument(
            "sel_type",
            [List[int], int, None],
            optional=True,
            alias=["dipole_type"],
            doc=doc_sel_type + doc_only_tf_supported,
        ),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
    ]


#   YWolfeee: Delete global polar mode, merge it into polar mode and use loss setting to support.
def fitting_variant_type_args():
    doc_descrpt_type = "The type of the fitting. See explanation below. \n\n\
- `ener`: Fit an energy model (potential energy surface).\n\n\
- `dos` : Fit a density of states model. The total density of states / site-projected density of states labels should be provided by `dos.npy` or `atom_dos.npy` in each data system. The file has number of frames lines and number of energy grid columns (times number of atoms in `atom_dos.npy`). See `loss` parameter. \n\n\
- `dipole`: Fit an atomic dipole model. Global dipole labels or atomic dipole labels for all the selected atoms (see `sel_type`) should be provided by `dipole.npy` in each data system. The file either has number of frames lines and 3 times of number of selected atoms columns, or has number of frames lines and 3 columns. See `loss` parameter.\n\n\
- `polar`: Fit an atomic polarizability model. Global polarizazbility labels or atomic polarizability labels for all the selected atoms (see `sel_type`) should be provided by `polarizability.npy` in each data system. The file eith has number of frames lines and 9 times of number of selected atoms columns, or has number of frames lines and 9 columns. See `loss` parameter.\n\n"

    return Variant(
        "type",
        fitting_args_plugin.get_all_argument(),
        optional=True,
        default_tag="ener",
        doc=doc_descrpt_type,
    )


#  --- Modifier configurations: --- #
def modifier_dipole_charge():
    doc_model_name = "The name of the frozen dipole model file."
    doc_model_charge_map = f"The charge of the WFCC. The list length should be the same as the {make_link('sel_type', 'model/fitting_net[dipole]/sel_type')}. "
    doc_sys_charge_map = f"The charge of real atoms. The list length should be the same as the {make_link('type_map', 'model/type_map')}"
    doc_ewald_h = "The grid spacing of the FFT grid. Unit is A"
    doc_ewald_beta = f"The splitting parameter of Ewald sum. Unit is A^{-1}"

    return [
        Argument("model_name", str, optional=False, doc=doc_model_name),
        Argument(
            "model_charge_map", List[float], optional=False, doc=doc_model_charge_map
        ),
        Argument("sys_charge_map", List[float], optional=False, doc=doc_sys_charge_map),
        Argument("ewald_beta", float, optional=True, default=0.4, doc=doc_ewald_beta),
        Argument("ewald_h", float, optional=True, default=1.0, doc=doc_ewald_h),
    ]


def modifier_variant_type_args():
    doc_modifier_type = "The type of modifier. See explanation below.\n\n\
-`dipole_charge`: Use WFCC to model the electronic structure of the system. Correct the long-range interaction"
    return Variant(
        "type",
        [
            Argument("dipole_charge", dict, modifier_dipole_charge()),
        ],
        optional=False,
        doc=doc_modifier_type,
    )


#  --- model compression configurations: --- #
def model_compression():
    doc_model_file = "The input model file, which will be compressed by the DeePMD-kit."
    doc_table_config = "The arguments of model compression, including extrapolate(scale of model extrapolation), stride(uniform stride of tabulation's first and second table), and frequency(frequency of tabulation overflow check)."
    doc_min_nbor_dist = (
        "The nearest distance between neighbor atoms saved in the frozen model."
    )

    return [
        Argument("model_file", str, optional=False, doc=doc_model_file),
        Argument("table_config", List[float], optional=False, doc=doc_table_config),
        Argument("min_nbor_dist", float, optional=False, doc=doc_min_nbor_dist),
    ]


#  --- model compression configurations: --- #
def model_compression_type_args():
    doc_compress_type = "The type of model compression, which should be consistent with the descriptor type."

    return Variant(
        "type",
        [Argument("se_e2_a", dict, model_compression(), alias=["se_a"])],
        optional=True,
        default_tag="se_e2_a",
        doc=doc_compress_type,
    )


def model_args(exclude_hybrid=False):
    doc_type_map = "A list of strings. Give the name to each type of atoms. It is noted that the number of atom type of training system must be less than 128 in a GPU environment. If not given, type.raw in each system should use the same type indexes, and type_map.raw will take no effect."
    doc_data_stat_nbatch = "The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics."
    doc_data_stat_protect = "Protect parameter for atomic energy regression."
    doc_data_bias_nsample = "The number of training samples in a system to compute and change the energy bias."
    doc_type_embedding = "The type embedding."
    doc_modifier = "The modifier of model output."
    doc_use_srtab = "The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly."
    doc_smin_alpha = "The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided."
    doc_sw_rmin = "The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided."
    doc_sw_rmax = "The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided."
    doc_srtab_add_bias = "Whether add energy bias from the statistics of the data to short-range tabulated atomic energy. It only takes effect when `use_srtab` is provided."
    doc_compress_config = "Model compression configurations"
    doc_spin = "The settings for systems with spin."
    doc_atom_exclude_types = "Exclude the atomic contribution of the listed atom types"
    doc_pair_exclude_types = "The atom pairs of the listed types are not treated to be neighbors, i.e. they do not see each other."

    hybrid_models = []
    if not exclude_hybrid:
        hybrid_models.extend(
            [
                pairwise_dprc(),
                linear_ener_model_args(),
            ]
        )
    return Argument(
        "model",
        dict,
        [
            Argument("type_map", List[str], optional=True, doc=doc_type_map),
            Argument(
                "data_stat_nbatch",
                int,
                optional=True,
                default=10,
                doc=doc_data_stat_nbatch,
            ),
            Argument(
                "data_stat_protect",
                float,
                optional=True,
                default=1e-2,
                doc=doc_data_stat_protect,
            ),
            Argument(
                "data_bias_nsample",
                int,
                optional=True,
                default=10,
                doc=doc_data_bias_nsample,
            ),
            Argument(
                "use_srtab",
                str,
                optional=True,
                doc=doc_only_tf_supported + doc_use_srtab,
            ),
            Argument(
                "smin_alpha",
                float,
                optional=True,
                doc=doc_only_tf_supported + doc_smin_alpha,
            ),
            Argument(
                "sw_rmin", float, optional=True, doc=doc_only_tf_supported + doc_sw_rmin
            ),
            Argument(
                "sw_rmax", float, optional=True, doc=doc_only_tf_supported + doc_sw_rmax
            ),
            Argument(
                "pair_exclude_types",
                list,
                optional=True,
                default=[],
                doc=doc_only_pt_supported + doc_pair_exclude_types,
            ),
            Argument(
                "atom_exclude_types",
                list,
                optional=True,
                default=[],
                doc=doc_only_pt_supported + doc_atom_exclude_types,
            ),
            Argument(
                "srtab_add_bias",
                bool,
                optional=True,
                default=True,
                doc=doc_only_tf_supported + doc_srtab_add_bias,
            ),
            Argument(
                "type_embedding",
                dict,
                type_embedding_args(),
                [],
                optional=True,
                doc=doc_only_tf_supported + doc_type_embedding,
            ),
            Argument(
                "modifier",
                dict,
                [],
                [modifier_variant_type_args()],
                optional=True,
                doc=doc_only_tf_supported + doc_modifier,
            ),
            Argument(
                "compress",
                dict,
                [],
                [model_compression_type_args()],
                optional=True,
                doc=doc_only_tf_supported + doc_compress_config,
                fold_subdoc=True,
            ),
            Argument("spin", dict, spin_args(), [], optional=True, doc=doc_spin),
        ],
        [
            Variant(
                "type",
                [
                    standard_model_args(),
                    multi_model_args(),
                    frozen_model_args(),
                    pairtab_model_args(),
                    *hybrid_models,
                ],
                optional=True,
                default_tag="standard",
            ),
        ],
    )


def standard_model_args() -> Argument:
    doc_descrpt = "The descriptor of atomic environment."
    doc_fitting = "The fitting of physical properties."

    ca = Argument(
        "standard",
        dict,
        [
            Argument(
                "descriptor", dict, [], [descrpt_variant_type_args()], doc=doc_descrpt
            ),
            Argument(
                "fitting_net",
                dict,
                [],
                [fitting_variant_type_args()],
                doc=doc_fitting,
            ),
        ],
        doc="Stardard model, which contains a descriptor and a fitting.",
    )
    return ca


def multi_model_args() -> Argument:
    doc_descrpt = "The descriptor of atomic environment. See model[standard]/descriptor for details."
    doc_fitting_net_dict = "The dictionary of multiple fitting nets in multi-task mode. Each fitting_net_dict[fitting_key] is the single definition of fitting of physical properties with user-defined name `fitting_key`."

    ca = Argument(
        "multi",
        dict,
        [
            Argument(
                "descriptor",
                dict,
                [],
                [descrpt_variant_type_args()],
                doc=doc_descrpt,
                fold_subdoc=True,
            ),
            Argument("fitting_net_dict", dict, doc=doc_fitting_net_dict),
        ],
        doc=doc_only_tf_supported + "Multiple-task model.",
    )
    return ca


def pairwise_dprc() -> Argument:
    qm_model_args = model_args(exclude_hybrid=True)
    qm_model_args.name = "qm_model"
    qm_model_args.fold_subdoc = True
    qmmm_model_args = model_args(exclude_hybrid=True)
    qmmm_model_args.name = "qmmm_model"
    qmmm_model_args.fold_subdoc = True
    ca = Argument(
        "pairwise_dprc",
        dict,
        [
            qm_model_args,
            qmmm_model_args,
        ],
        doc=doc_only_tf_supported,
    )
    return ca


def frozen_model_args() -> Argument:
    doc_model_file = "Path to the frozen model file."
    ca = Argument(
        "frozen",
        dict,
        [
            Argument("model_file", str, optional=False, doc=doc_model_file),
        ],
        doc=doc_only_tf_supported,
    )
    return ca


def pairtab_model_args() -> Argument:
    doc_tab_file = "Path to the tabulation file."
    doc_rcut = "The cut-off radius."
    doc_sel = 'This parameter set the number of selected neighbors. Note that this parameter is a little different from that in other descriptors. Instead of separating each type of atoms, only the summation matters. And this number is highly related with the efficiency, thus one should not make it too large. Usually 200 or less is enough, far away from the GPU limitation 4096. It can be:\n\n\
    - `int`. The maximum number of neighbor atoms to be considered. We recommend it to be less than 200. \n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. Only the summation of `sel[i]` matters, and it is recommended to be less than 200.\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
    ca = Argument(
        "pairtab",
        dict,
        [
            Argument("tab_file", str, optional=False, doc=doc_tab_file),
            Argument("rcut", float, optional=False, doc=doc_rcut),
            Argument("sel", [int, List[int], str], optional=False, doc=doc_sel),
        ],
        doc=doc_only_tf_supported + "Pairwise tabulation energy model.",
    )
    return ca


def linear_ener_model_args() -> Argument:
    doc_weights = (
        "If the type is list of float, a list of weights for each model. "
        'If "mean", the weights are set to be 1 / len(models). '
        'If "sum", the weights are set to be 1.'
    )
    models_args = model_args(exclude_hybrid=True)
    models_args.name = "models"
    models_args.fold_subdoc = True
    models_args.set_dtype(list)
    models_args.set_repeat(True)
    models_args.doc = "The sub-models."
    ca = Argument(
        "linear_ener",
        dict,
        [
            models_args,
            Argument(
                "weights",
                [list, str],
                optional=False,
                doc=doc_weights,
            ),
        ],
        doc=doc_only_tf_supported,
    )
    return ca


#  --- Learning rate configurations: --- #
def learning_rate_exp():
    doc_start_lr = "The learning rate at the start of the training."
    doc_stop_lr = "The desired learning rate at the end of the training."
    doc_decay_steps = (
        "The learning rate is decaying every this number of training steps."
    )

    args = [
        Argument("start_lr", float, optional=True, default=1e-3, doc=doc_start_lr),
        Argument("stop_lr", float, optional=True, default=1e-8, doc=doc_stop_lr),
        Argument("decay_steps", int, optional=True, default=5000, doc=doc_decay_steps),
    ]
    return args


def learning_rate_variant_type_args():
    doc_lr = "The type of the learning rate."

    return Variant(
        "type",
        [Argument("exp", dict, learning_rate_exp())],
        optional=True,
        default_tag="exp",
        doc=doc_lr,
    )


def learning_rate_args():
    doc_scale_by_worker = "When parallel training or batch size scaled, how to alter learning rate. Valid values are `linear`(default), `sqrt` or `none`."
    doc_lr = "The definitio of learning rate"
    return Argument(
        "learning_rate",
        dict,
        [
            Argument(
                "scale_by_worker",
                str,
                optional=True,
                default="linear",
                doc=doc_scale_by_worker,
            )
        ],
        [learning_rate_variant_type_args()],
        optional=True,
        doc=doc_lr,
    )


def learning_rate_dict_args():
    doc_learning_rate_dict = (
        "The dictionary of definitions of learning rates in multi-task mode. "
        "Each learning_rate_dict[fitting_key], with user-defined name `fitting_key` in `model/fitting_net_dict`, is the single definition of learning rate.\n"
    )
    ca = Argument(
        "learning_rate_dict", dict, [], [], optional=True, doc=doc_learning_rate_dict
    )
    return ca


#  --- Loss configurations: --- #
def start_pref(item, label=None, abbr=None):
    if label is None:
        label = item
    if abbr is None:
        abbr = item
    return f"The prefactor of {item} loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the {label} label should be provided by file {label}.npy in each data system. If both start_pref_{abbr} and limit_pref_{abbr} are set to 0, then the {item} will be ignored."


def limit_pref(item):
    return f"The prefactor of {item} loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity."


loss_args_plugin = ArgsPlugin()


@loss_args_plugin.register("ener")
def loss_ener():
    doc_start_pref_e = start_pref("energy", abbr="e")
    doc_limit_pref_e = limit_pref("energy")
    doc_start_pref_f = start_pref("force", abbr="f")
    doc_limit_pref_f = limit_pref("force")
    doc_start_pref_v = start_pref("virial", abbr="v")
    doc_limit_pref_v = limit_pref("virial")
    doc_start_pref_ae = start_pref("atomic energy", label="atom_ener", abbr="ae")
    doc_limit_pref_ae = limit_pref("atomic energy")
    doc_start_pref_pf = start_pref(
        "atomic prefactor force", label="atom_pref", abbr="pf"
    )
    doc_limit_pref_pf = limit_pref("atomic prefactor force")
    doc_start_pref_gf = start_pref("generalized force", label="drdq", abbr="gf")
    doc_limit_pref_gf = limit_pref("generalized force")
    doc_numb_generalized_coord = "The dimension of generalized coordinates. Required when generalized force loss is used."
    doc_relative_f = "If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label."
    doc_enable_atom_ener_coeff = "If true, the energy will be computed as \\sum_i c_i E_i. c_i should be provided by file atom_ener_coeff.npy in each data system, otherwise it's 1."
    return [
        Argument(
            "start_pref_e",
            [float, int],
            optional=True,
            default=0.02,
            doc=doc_start_pref_e,
        ),
        Argument(
            "limit_pref_e",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_limit_pref_e,
        ),
        Argument(
            "start_pref_f",
            [float, int],
            optional=True,
            default=1000,
            doc=doc_start_pref_f,
        ),
        Argument(
            "limit_pref_f",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_limit_pref_f,
        ),
        Argument(
            "start_pref_v",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_v,
        ),
        Argument(
            "limit_pref_v",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_v,
        ),
        Argument(
            "start_pref_ae",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_ae,
        ),
        Argument(
            "limit_pref_ae",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_ae,
        ),
        Argument(
            "start_pref_pf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_pf,
        ),
        Argument(
            "limit_pref_pf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_pf,
        ),
        Argument("relative_f", [float, None], optional=True, doc=doc_relative_f),
        Argument(
            "enable_atom_ener_coeff",
            [bool],
            optional=True,
            default=False,
            doc=doc_enable_atom_ener_coeff,
        ),
        Argument(
            "start_pref_gf",
            float,
            optional=True,
            default=0.0,
            doc=doc_start_pref_gf,
        ),
        Argument(
            "limit_pref_gf",
            float,
            optional=True,
            default=0.0,
            doc=doc_limit_pref_gf,
        ),
        Argument(
            "numb_generalized_coord",
            int,
            optional=True,
            default=0,
            doc=doc_numb_generalized_coord,
        ),
    ]


@loss_args_plugin.register("ener_spin")
def loss_ener_spin():
    doc_start_pref_e = start_pref("energy")
    doc_limit_pref_e = limit_pref("energy")
    doc_start_pref_fr = start_pref("force_real_atom")
    doc_limit_pref_fr = limit_pref("force_real_atom")
    doc_start_pref_fm = start_pref("force_magnetic")
    doc_limit_pref_fm = limit_pref("force_magnetic")
    doc_start_pref_v = start_pref("virial")
    doc_limit_pref_v = limit_pref("virial")
    doc_start_pref_ae = start_pref("atom_ener")
    doc_limit_pref_ae = limit_pref("atom_ener")
    doc_start_pref_pf = start_pref("atom_pref")
    doc_limit_pref_pf = limit_pref("atom_pref")
    doc_relative_f = "If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label."
    doc_enable_atom_ener_coeff = r"If true, the energy will be computed as \sum_i c_i E_i. c_i should be provided by file atom_ener_coeff.npy in each data system, otherwise it's 1."
    return [
        Argument(
            "start_pref_e",
            [float, int],
            optional=True,
            default=0.02,
            doc=doc_start_pref_e,
        ),
        Argument(
            "limit_pref_e",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_limit_pref_e,
        ),
        Argument(
            "start_pref_fr",
            [float, int],
            optional=True,
            default=1000,
            doc=doc_start_pref_fr,
        ),
        Argument(
            "limit_pref_fr",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_limit_pref_fr,
        ),
        Argument(
            "start_pref_fm",
            [float, int],
            optional=True,
            default=10000,
            doc=doc_start_pref_fm,
        ),
        Argument(
            "limit_pref_fm",
            [float, int],
            optional=True,
            default=10.0,
            doc=doc_limit_pref_fm,
        ),
        Argument(
            "start_pref_v",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_v,
        ),
        Argument(
            "limit_pref_v",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_v,
        ),
        Argument(
            "start_pref_ae",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_ae,
        ),
        Argument(
            "limit_pref_ae",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_ae,
        ),
        Argument(
            "start_pref_pf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_pf,
        ),
        Argument(
            "limit_pref_pf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_pf,
        ),
        Argument("relative_f", [float, None], optional=True, doc=doc_relative_f),
        Argument(
            "enable_atom_ener_coeff",
            [bool],
            optional=True,
            default=False,
            doc=doc_enable_atom_ener_coeff,
        ),
    ]


@loss_args_plugin.register("dos", doc=doc_only_tf_supported)
def loss_dos():
    doc_start_pref_dos = start_pref("Density of State (DOS)")
    doc_limit_pref_dos = limit_pref("Density of State (DOS)")
    doc_start_pref_cdf = start_pref(
        "Cumulative Distribution Function (cumulative intergral of DOS)"
    )
    doc_limit_pref_cdf = limit_pref(
        "Cumulative Distribution Function (cumulative intergral of DOS)"
    )
    doc_start_pref_ados = start_pref("atomic DOS (site-projected DOS)")
    doc_limit_pref_ados = limit_pref("atomic DOS (site-projected DOS)")
    doc_start_pref_acdf = start_pref("Cumulative integral of atomic DOS")
    doc_limit_pref_acdf = limit_pref("Cumulative integral of atomic DOS")
    return [
        Argument(
            "start_pref_dos",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_dos,
        ),
        Argument(
            "limit_pref_dos",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_dos,
        ),
        Argument(
            "start_pref_cdf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_cdf,
        ),
        Argument(
            "limit_pref_cdf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_cdf,
        ),
        Argument(
            "start_pref_ados",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_start_pref_ados,
        ),
        Argument(
            "limit_pref_ados",
            [float, int],
            optional=True,
            default=1.00,
            doc=doc_limit_pref_ados,
        ),
        Argument(
            "start_pref_acdf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_start_pref_acdf,
        ),
        Argument(
            "limit_pref_acdf",
            [float, int],
            optional=True,
            default=0.00,
            doc=doc_limit_pref_acdf,
        ),
    ]


# YWolfeee: Modified to support tensor type of loss args.
@loss_args_plugin.register("tensor", doc=doc_only_tf_supported)
def loss_tensor():
    # doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If only `pref` is provided or both are not provided, training will be global mode, i.e. the shape of 'polarizability.npy` or `dipole.npy` should be #frams x [9 or 3]."
    # doc_local_weight =  "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If only `pref_atomic` is provided, training will be atomic mode, i.e. the shape of `polarizability.npy` or `dipole.npy` should be #frames x ([9 or 3] x #selected atoms). If both `pref` and `pref_atomic` are provided, training will be combined mode, and atomic label should be provided as well."
    doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to global label, i.e. 'polarizability.npy` or `dipole.npy`, whose shape should be #frames x [9 or 3]. If it's larger than 0.0, this npy should be included."
    doc_local_weight = "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to atomic label, i.e. `atomic_polarizability.npy` or `atomic_dipole.npy`, whose shape should be #frames x ([9 or 3] x #selected atoms). If it's larger than 0.0, this npy should be included. Both `pref` and `pref_atomic` should be provided, and either can be set to 0.0."
    return [
        Argument(
            "pref", [float, int], optional=False, default=None, doc=doc_global_weight
        ),
        Argument(
            "pref_atomic",
            [float, int],
            optional=False,
            default=None,
            doc=doc_local_weight,
        ),
    ]


def loss_variant_type_args():
    doc_loss = "The type of the loss. When the fitting type is `ener`, the loss type should be set to `ener` or left unset. When the fitting type is `dipole` or `polar`, the loss type should be set to `tensor`."

    return Variant(
        "type",
        loss_args_plugin.get_all_argument(),
        optional=True,
        default_tag="ener",
        doc=doc_loss,
    )


def loss_args():
    doc_loss = "The definition of loss function. The loss type should be set to `tensor`, `ener` or left unset."
    ca = Argument(
        "loss", dict, [], [loss_variant_type_args()], optional=True, doc=doc_loss
    )
    return ca


def loss_dict_args():
    doc_loss_dict = (
        "The dictionary of definitions of multiple loss functions in multi-task mode. "
        "Each loss_dict[fitting_key], with user-defined name `fitting_key` in `model/fitting_net_dict`, is the single definition of loss function, whose type should be set to `tensor`, `ener` or left unset.\n"
    )
    ca = Argument("loss_dict", dict, [], [], optional=True, doc=doc_loss_dict)
    return ca


#  --- Training configurations: --- #
def training_data_args():  # ! added by Ziyao: new specification style for data systems.
    link_sys = make_link("systems", "training/training_data/systems")
    doc_systems = (
        "The data systems for training. "
        "This key can be provided with a list that specifies the systems, or be provided with a string "
        "by which the prefix of all systems are given and the list of the systems is automatically generated."
    )
    doc_set_prefix = f"The prefix of the sets in the {link_sys}."
    doc_batch_size = f'This key can be \n\n\
- list: the length of which is the same as the {link_sys}. The batch size of each system is given by the elements of the list.\n\n\
- int: all {link_sys} use the same batch size.\n\n\
- string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.\n\n\
- string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.\n\n\
- string "mixed:N": the batch data will be sampled from all systems and merged into a mixed system with the batch size N. Only support the se_atten descriptor.\n\n\
If MPI is used, the value should be considered as the batch size per task.'
    doc_auto_prob_style = 'Determine the probability of systems automatically. The method is assigned by this key and can be\n\n\
- "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()\n\n\
- "prob_sys_size" : the probability of a system is proportional to the number of batches in the system\n\n\
- "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.'
    doc_sys_probs = (
        "A list of float if specified. "
        "Should be of the same length as `systems`, "
        "specifying the probability of each system."
    )

    args = [
        Argument(
            "systems", [List[str], str], optional=False, default=".", doc=doc_systems
        ),
        Argument("set_prefix", str, optional=True, default="set", doc=doc_set_prefix),
        Argument(
            "batch_size",
            [List[int], int, str],
            optional=True,
            default="auto",
            doc=doc_batch_size,
        ),
        Argument(
            "auto_prob",
            str,
            optional=True,
            default="prob_sys_size",
            doc=doc_auto_prob_style,
            alias=[
                "auto_prob_style",
            ],
        ),
        Argument(
            "sys_probs",
            List[float],
            optional=True,
            default=None,
            doc=doc_sys_probs,
            alias=["sys_weights"],
        ),
    ]

    doc_training_data = "Configurations of training data."
    return Argument(
        "training_data",
        dict,
        optional=True,
        sub_fields=args,
        sub_variants=[],
        doc=doc_training_data,
    )


def validation_data_args():  # ! added by Ziyao: new specification style for data systems.
    link_sys = make_link("systems", "training/validation_data/systems")
    doc_systems = (
        "The data systems for validation. "
        "This key can be provided with a list that specifies the systems, or be provided with a string "
        "by which the prefix of all systems are given and the list of the systems is automatically generated."
    )
    doc_set_prefix = f"The prefix of the sets in the {link_sys}."
    doc_batch_size = f'This key can be \n\n\
- list: the length of which is the same as the {link_sys}. The batch size of each system is given by the elements of the list.\n\n\
- int: all {link_sys} use the same batch size.\n\n\
- string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.\n\n\
- string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.'
    doc_auto_prob_style = 'Determine the probability of systems automatically. The method is assigned by this key and can be\n\n\
- "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()\n\n\
- "prob_sys_size" : the probability of a system is proportional to the number of batches in the system\n\n\
- "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.'
    doc_sys_probs = (
        "A list of float if specified. "
        "Should be of the same length as `systems`, "
        "specifying the probability of each system."
    )
    doc_numb_btch = "An integer that specifies the number of batches to be sampled for each validation period."

    args = [
        Argument(
            "systems", [List[str], str], optional=False, default=".", doc=doc_systems
        ),
        Argument("set_prefix", str, optional=True, default="set", doc=doc_set_prefix),
        Argument(
            "batch_size",
            [List[int], int, str],
            optional=True,
            default="auto",
            doc=doc_batch_size,
        ),
        Argument(
            "auto_prob",
            str,
            optional=True,
            default="prob_sys_size",
            doc=doc_auto_prob_style,
            alias=[
                "auto_prob_style",
            ],
        ),
        Argument(
            "sys_probs",
            List[float],
            optional=True,
            default=None,
            doc=doc_sys_probs,
            alias=["sys_weights"],
        ),
        Argument(
            "numb_btch",
            int,
            optional=True,
            default=1,
            doc=doc_numb_btch,
            alias=[
                "numb_batch",
            ],
        ),
    ]

    doc_validation_data = (
        "Configurations of validation data. Similar to that of training data, "
        "except that a `numb_btch` argument may be configured"
    )
    return Argument(
        "validation_data",
        dict,
        optional=True,
        default=None,
        sub_fields=args,
        sub_variants=[],
        doc=doc_validation_data,
    )


def mixed_precision_args():  # ! added by Denghui.
    doc_output_prec = 'The precision for mixed precision params. " \
        "The trainable variables precision during the mixed precision training process, " \
        "supported options are float32 only currently.'
    doc_compute_prec = 'The precision for mixed precision compute. " \
        "The compute precision during the mixed precision training process, "" \
        "supported options are float16 and bfloat16 currently.'

    args = [
        Argument(
            "output_prec", str, optional=True, default="float32", doc=doc_output_prec
        ),
        Argument(
            "compute_prec", str, optional=False, default="float16", doc=doc_compute_prec
        ),
    ]

    doc_mixed_precision = "Configurations of mixed precision."
    return Argument(
        "mixed_precision",
        dict,
        optional=True,
        sub_fields=args,
        sub_variants=[],
        doc=doc_mixed_precision,
    )


def training_args():  # ! modified by Ziyao: data configuration isolated.
    doc_numb_steps = "Number of training batch. Each training uses one batch of data."
    doc_seed = "The random seed for getting frames from the training data set."
    doc_disp_file = "The file for printing learning curve."
    doc_disp_freq = "The frequency of printing learning curve."
    doc_save_freq = "The frequency of saving check point."
    doc_save_ckpt = "The path prefix of saving check point files."
    doc_disp_training = "Displaying verbose information during training."
    doc_time_training = "Timing durining training."
    doc_profiling = "Profiling during training."
    doc_profiling_file = "Output file for profiling."
    doc_enable_profiler = "Enable TensorFlow Profiler (available in TensorFlow 2.3) or PyTorch Profiler to analyze performance. The log will be saved to `tensorboard_log_dir`."
    doc_tensorboard = "Enable tensorboard"
    doc_tensorboard_log_dir = "The log directory of tensorboard outputs"
    doc_tensorboard_freq = "The frequency of writing tensorboard events."
    doc_data_dict = (
        "The dictionary of multi DataSystems in multi-task mode. "
        "Each data_dict[fitting_key], with user-defined name `fitting_key` in `model/fitting_net_dict`, "
        "contains training data and optional validation data definitions."
    )
    doc_fitting_weight = (
        "Each fitting_weight[fitting_key], with user-defined name `fitting_key` in `model/fitting_net_dict`, "
        "is the training weight of fitting net `fitting_key`. "
        "Fitting nets with higher weights will be selected with higher probabilities to be trained in one step. "
        "Weights will be normalized and minus ones will be ignored. "
        "If not set, each fitting net will be equally selected when training."
    )
    doc_warmup_steps = (
        "The number of steps for learning rate warmup. During warmup, "
        "the learning rate begins at zero and progressively increases linearly to `start_lr`, "
        "rather than starting directly from `start_lr`"
    )
    doc_gradient_max_norm = (
        "Clips the gradient norm to a maximum value. "
        "If the gradient norm exceeds this value, it will be clipped to this limit. "
        "No gradient clipping will occur if set to 0."
    )
    doc_stat_file = (
        "The file path for saving the data statistics results. "
        "If set, the results will be saved and directly loaded during the next training session, "
        "avoiding the need to recalculate the statistics"
    )
    doc_opt_type = "The type of optimizer to use."
    doc_kf_blocksize = "The blocksize for the Kalman filter."

    arg_training_data = training_data_args()
    arg_validation_data = validation_data_args()
    mixed_precision_data = mixed_precision_args()

    args = [
        arg_training_data,
        arg_validation_data,
        mixed_precision_data,
        Argument(
            "numb_steps", int, optional=False, doc=doc_numb_steps, alias=["stop_batch"]
        ),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument(
            "disp_file", str, optional=True, default="lcurve.out", doc=doc_disp_file
        ),
        Argument("disp_freq", int, optional=True, default=1000, doc=doc_disp_freq),
        Argument("save_freq", int, optional=True, default=1000, doc=doc_save_freq),
        Argument(
            "save_ckpt", str, optional=True, default="model.ckpt", doc=doc_save_ckpt
        ),
        Argument(
            "disp_training", bool, optional=True, default=True, doc=doc_disp_training
        ),
        Argument(
            "time_training", bool, optional=True, default=True, doc=doc_time_training
        ),
        Argument(
            "profiling",
            bool,
            optional=True,
            default=False,
            doc=doc_only_tf_supported + doc_profiling,
        ),
        Argument(
            "profiling_file",
            str,
            optional=True,
            default="timeline.json",
            doc=doc_only_tf_supported + doc_profiling_file,
        ),
        Argument(
            "enable_profiler",
            bool,
            optional=True,
            default=False,
            doc=doc_enable_profiler,
        ),
        Argument(
            "tensorboard", bool, optional=True, default=False, doc=doc_tensorboard
        ),
        Argument(
            "tensorboard_log_dir",
            str,
            optional=True,
            default="log",
            doc=doc_tensorboard_log_dir,
        ),
        Argument(
            "tensorboard_freq", int, optional=True, default=1, doc=doc_tensorboard_freq
        ),
        Argument("data_dict", dict, optional=True, doc=doc_data_dict),
        Argument("fitting_weight", dict, optional=True, doc=doc_fitting_weight),
        Argument(
            "warmup_steps",
            int,
            optional=True,
            doc=doc_only_pt_supported + doc_warmup_steps,
        ),
        Argument(
            "gradient_max_norm",
            float,
            optional=True,
            doc=doc_only_pt_supported + doc_gradient_max_norm,
        ),
        Argument(
            "stat_file", str, optional=True, doc=doc_only_pt_supported + doc_stat_file
        ),
    ]
    variants = [
        Variant(
            "opt_type",
            choices=[
                Argument("Adam", dict, [], [], optional=True),
                Argument(
                    "LKF",
                    dict,
                    [
                        Argument(
                            "kf_blocksize",
                            int,
                            optional=True,
                            doc=doc_only_pt_supported + doc_kf_blocksize,
                        ),
                    ],
                    [],
                    optional=True,
                ),
            ],
            optional=True,
            default_tag="Adam",
            doc=doc_only_pt_supported + doc_opt_type,
        )
    ]

    doc_training = "The training options."
    return Argument("training", dict, args, variants, doc=doc_training)


def make_index(keys):
    ret = []
    for ii in keys:
        ret.append(make_link(ii, ii))
    return ", ".join(ret)


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    ptr = []
    for ii in gen_args():
        ptr.append(ii.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    # ptr.insert(0, make_index(key_words))

    return "\n\n".join(ptr)


def gen_json(**kwargs):
    return json.dumps(
        tuple(gen_args()),
        cls=ArgumentEncoder,
    )


def gen_args(**kwargs) -> List[Argument]:
    return [
        model_args(),
        learning_rate_args(),
        learning_rate_dict_args(),
        loss_args(),
        loss_dict_args(),
        training_args(),
        nvnmd_args(),
    ]


def normalize_multi_task(data):
    # single-task or multi-task mode
    if data["model"].get("type", "standard") not in ("standard", "multi"):
        return data
    single_fitting_net = "fitting_net" in data["model"].keys()
    single_training_data = "training_data" in data["training"].keys()
    single_valid_data = "validation_data" in data["training"].keys()
    single_loss = "loss" in data.keys()
    single_learning_rate = "learning_rate" in data.keys()
    multi_fitting_net = "fitting_net_dict" in data["model"].keys()
    multi_training_data = "data_dict" in data["training"].keys()
    multi_loss = "loss_dict" in data.keys()
    multi_fitting_weight = "fitting_weight" in data["training"].keys()
    multi_learning_rate = "learning_rate_dict" in data.keys()
    assert (single_fitting_net == single_training_data) and (
        multi_fitting_net == multi_training_data
    ), (
        "In single-task mode, 'model/fitting_net' and 'training/training_data' must be defined at the same time! "
        "While in multi-task mode, 'model/fitting_net_dict', 'training/data_dict' "
        "must be defined at the same time! Please check your input script. "
    )
    assert not (single_fitting_net and multi_fitting_net), (
        "Single-task mode and multi-task mode can not be performed together. "
        "Please check your input script and choose just one format! "
    )
    assert (
        single_fitting_net or multi_fitting_net
    ), "Please define your fitting net and training data! "
    if multi_fitting_net:
        assert not single_valid_data, (
            "In multi-task mode, 'training/validation_data' should not appear "
            "outside 'training/data_dict'! Please check your input script."
        )
        assert (
            not single_loss
        ), "In multi-task mode, please use 'model/loss_dict' in stead of 'model/loss'! "
        assert (
            "type_map" in data["model"]
        ), "In multi-task mode, 'model/type_map' must be defined! "
        data["model"]["type"] = "multi"
        data["model"]["fitting_net_dict"] = normalize_fitting_net_dict(
            data["model"]["fitting_net_dict"]
        )
        data["training"]["data_dict"] = normalize_data_dict(
            data["training"]["data_dict"]
        )
        data["loss_dict"] = (
            normalize_loss_dict(
                data["model"]["fitting_net_dict"].keys(), data["loss_dict"]
            )
            if multi_loss
            else {}
        )
        if multi_learning_rate:
            data["learning_rate_dict"] = normalize_learning_rate_dict(
                data["model"]["fitting_net_dict"].keys(), data["learning_rate_dict"]
            )
        elif single_learning_rate:
            data[
                "learning_rate_dict"
            ] = normalize_learning_rate_dict_with_single_learning_rate(
                data["model"]["fitting_net_dict"].keys(), data["learning_rate"]
            )
        fitting_weight = (
            data["training"]["fitting_weight"] if multi_fitting_weight else None
        )
        data["training"]["fitting_weight"] = normalize_fitting_weight(
            data["model"]["fitting_net_dict"].keys(),
            data["training"]["data_dict"].keys(),
            fitting_weight=fitting_weight,
        )
    else:
        assert not multi_loss, "In single-task mode, please use 'model/loss' in stead of 'model/loss_dict'! "
        assert not multi_learning_rate, "In single-task mode, please use 'model/learning_rate' in stead of 'model/learning_rate_dict'! "
    return data


def normalize_fitting_net_dict(fitting_net_dict):
    new_dict = {}
    base = Argument("base", dict, [], [fitting_variant_type_args()], doc="")
    for fitting_key_item in fitting_net_dict:
        data = base.normalize_value(
            fitting_net_dict[fitting_key_item], trim_pattern="_*"
        )
        base.check_value(data, strict=True)
        new_dict[fitting_key_item] = data
    return new_dict


def normalize_data_dict(data_dict):
    new_dict = {}
    base = Argument(
        "base", dict, [training_data_args(), validation_data_args()], [], doc=""
    )
    for data_system_key_item in data_dict:
        data = base.normalize_value(data_dict[data_system_key_item], trim_pattern="_*")
        base.check_value(data, strict=True)
        new_dict[data_system_key_item] = data
    return new_dict


def normalize_loss_dict(fitting_keys, loss_dict):
    # check the loss dict
    failed_loss_keys = [item for item in loss_dict if item not in fitting_keys]
    assert (
        not failed_loss_keys
    ), "Loss dict key(s) {} not have corresponding fitting keys in {}! ".format(
        str(failed_loss_keys), str(list(fitting_keys))
    )
    new_dict = {}
    base = Argument("base", dict, [], [loss_variant_type_args()], doc="")
    for item in loss_dict:
        data = base.normalize_value(loss_dict[item], trim_pattern="_*")
        base.check_value(data, strict=True)
        new_dict[item] = data
    return new_dict


def normalize_learning_rate_dict(fitting_keys, learning_rate_dict):
    # check the learning_rate dict
    failed_learning_rate_keys = [
        item for item in learning_rate_dict if item not in fitting_keys
    ]
    assert not failed_learning_rate_keys, "Learning rate dict key(s) {} not have corresponding fitting keys in {}! ".format(
        str(failed_learning_rate_keys), str(list(fitting_keys))
    )
    new_dict = {}
    base = Argument("base", dict, [], [learning_rate_variant_type_args()], doc="")
    for item in learning_rate_dict:
        data = base.normalize_value(learning_rate_dict[item], trim_pattern="_*")
        base.check_value(data, strict=True)
        new_dict[item] = data
    return new_dict


def normalize_learning_rate_dict_with_single_learning_rate(fitting_keys, learning_rate):
    new_dict = {}
    base = Argument("base", dict, [], [learning_rate_variant_type_args()], doc="")
    data = base.normalize_value(learning_rate, trim_pattern="_*")
    base.check_value(data, strict=True)
    for fitting_key in fitting_keys:
        new_dict[fitting_key] = data
    return new_dict


def normalize_fitting_weight(fitting_keys, data_keys, fitting_weight=None):
    # check the mapping
    failed_data_keys = [item for item in data_keys if item not in fitting_keys]
    assert (
        not failed_data_keys
    ), "Data dict key(s) {} not have corresponding fitting keys in {}! ".format(
        str(failed_data_keys), str(list(fitting_keys))
    )
    empty_fitting_keys = []
    valid_fitting_keys = []
    for item in fitting_keys:
        if item not in data_keys:
            empty_fitting_keys.append(item)
        else:
            valid_fitting_keys.append(item)
    if empty_fitting_keys:
        log.warning(
            "Fitting net(s) {} have no data and will not be used in training.".format(
                str(empty_fitting_keys)
            )
        )
    num_pair = len(valid_fitting_keys)
    assert num_pair > 0, "No valid training data systems for fitting nets!"

    # check and normalize the fitting weight
    new_weight = {}
    if fitting_weight is None:
        equal_weight = 1.0 / num_pair
        for item in fitting_keys:
            new_weight[item] = equal_weight if item in valid_fitting_keys else 0.0
    else:
        failed_weight_keys = [
            item for item in fitting_weight if item not in fitting_keys
        ]
        assert not failed_weight_keys, "Fitting weight key(s) {} not have corresponding fitting keys in {}! ".format(
            str(failed_weight_keys), str(list(fitting_keys))
        )
        sum_prob = 0.0
        for item in fitting_keys:
            if item in valid_fitting_keys:
                if (
                    item in fitting_weight
                    and isinstance(fitting_weight[item], (int, float))
                    and fitting_weight[item] > 0.0
                ):
                    sum_prob += fitting_weight[item]
                    new_weight[item] = fitting_weight[item]
                else:
                    valid_fitting_keys.remove(item)
                    log.warning(
                        f"Fitting net '{item}' has zero or invalid weight "
                        "and will not be used in training."
                    )
                    new_weight[item] = 0.0
            else:
                new_weight[item] = 0.0
        assert sum_prob > 0.0, "No valid training weight for fitting nets!"
        # normalize
        for item in new_weight:
            new_weight[item] /= sum_prob
    return new_weight


def normalize(data):
    data = normalize_multi_task(data)

    base = Argument("base", dict, gen_args())
    data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data


if __name__ == "__main__":
    gen_doc()
