# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    NetworkCollection,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa1 import (
    DescrptDPA1,
    NeighborGatedAttention,
)


@BaseDescriptor.register("se_atten_v2")
class DescrptSeAttenV2(DescrptDPA1):
    r"""Attention-based descriptor (version 2) which uses stripped type embedding.

    This descriptor inherits from :class:`DescrptDPA1` and uses the same attention-based
    mechanism, but with `tebd_input_mode="strip"` by default. The descriptor
    :math:`\mathcal{D}^i \in \mathbb{R}^{M \times M_{<}}` is computed as:

    .. math::
        \mathcal{D}^i = \frac{1}{N_c^2}(\hat{\mathcal{G}}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \hat{\mathcal{G}}^i_<,

    where :math:`\hat{\mathcal{G}}^i` is the embedding matrix after self-attention layers,
    and :math:`\mathcal{R}^i` is the coordinate matrix (see :class:`DescrptDPA1` for details).

    The key difference from DPA-1 is that the type embedding is processed by a separate
    embedding network and combined multiplicatively with the radial embedding:

    .. math::
        \mathcal{G}^i = \mathcal{N}_r(s(r)) \odot \mathcal{N}_t(\mathcal{T}) + \mathcal{N}_r(s(r)),

    where :math:`\mathcal{N}_r` is the radial embedding network, :math:`\mathcal{N}_t` is
    the type embedding network, and :math:`\odot` denotes element-wise multiplication.

    Parameters
    ----------
    rcut: float
            The cut-off radius :math:`r_c`
    rcut_smth: float
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
    ntypes : int
            Number of element types
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron: int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    tebd_dim: int
            Dimension of the type embedding
    resnet_dt: bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable: bool
            If the weights of this descriptors are trainable.
    trainable_ln: bool
            Whether to use trainable shift and scale weights in layer normalization.
    ln_eps: float, Optional
            The epsilon value for layer normalization.
    type_one_side: bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
    attn: int
            Hidden dimension of the attention vectors
    attn_layer: int
            Number of attention layers
    attn_dotr: bool
            If dot the angular gate to the attention weights
    attn_mask: bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version. True option is not implemented.)
            If mask the diagonal of attention weights
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    set_davg_zero: bool
            Set the shift of embedding net input to zero.
    activation_function: str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision: str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    scaling_factor: float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
    normalize: bool
            Whether to normalize the hidden vectors in attention weights calculation.
    temperature: float
            If not None, the scaling of attention weights is `temperature` itself.
    concat_output_tebd: bool
            Whether to concat type embedding at the output of the descriptor.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    seed : int, Optional
            Random seed for initializing the network parameters.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int] | int,
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: float | None = None,
        trainable_ln: bool = True,
        ln_eps: float | None = 1e-5,
        concat_output_tebd: bool = True,
        spin: Any | None = None,
        stripped_type_embedding: bool | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
        # consistent with argcheck, not used though
        seed: int | list[int] | None = None,
    ) -> None:
        DescrptDPA1.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode="strip",
            resnet_dt=resnet_dt,
            trainable=trainable,
            type_one_side=type_one_side,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            exclude_types=exclude_types,
            env_protection=env_protection,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            smooth_type_embedding=True,
            concat_output_tebd=concat_output_tebd,
            spin=spin,
            stripped_type_embedding=stripped_type_embedding,
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            # consistent with argcheck, not used though
            seed=seed,
        )

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        obj = self.se_atten
        data = {
            "@class": "Descriptor",
            "type": "se_atten_v2",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "tebd_dim": obj.tebd_dim,
            "set_davg_zero": obj.set_davg_zero,
            "attn": obj.attn,
            "attn_layer": obj.attn_layer,
            "attn_dotr": obj.attn_dotr,
            "attn_mask": False,
            "activation_function": obj.activation_function,
            "resnet_dt": obj.resnet_dt,
            "scaling_factor": obj.scaling_factor,
            "normalize": obj.normalize,
            "temperature": obj.temperature,
            "trainable_ln": obj.trainable_ln,
            "ln_eps": obj.ln_eps,
            "type_one_side": obj.type_one_side,
            "concat_output_tebd": self.concat_output_tebd,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "embeddings_strip": obj.embeddings_strip.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
            ## to be updated when the options are supported.
            "trainable": self.trainable,
            "spin": None,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeAttenV2":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers")
        data.pop("env_mat")
        embeddings_strip = data.pop("embeddings_strip")
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
        obj = cls(**data)

        obj.se_atten["davg"] = variables["davg"]
        obj.se_atten["dstd"] = variables["dstd"]
        obj.se_atten.embeddings = NetworkCollection.deserialize(embeddings)
        obj.se_atten.embeddings_strip = NetworkCollection.deserialize(embeddings_strip)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj
