# SPDX-License-Identifier: LGPL-3.0-or-later
import re
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
)
from deepmd.dpmodel.utils.type_embed import (
    get_econf_tebd,
)
from deepmd.tf.common import (
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    TYPE_EMBEDDING_PATTERN,
    tf,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.utils.graph import (
    get_type_embedding_net_variables_from_graph_def,
)
from deepmd.tf.utils.network import (
    embedding_net,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def embed_atom_type(
    ntypes: int,
    natoms: tf.Tensor,
    type_embedding: tf.Tensor,
):
    """Make the embedded type for the atoms in system.
    The atoms are assumed to be sorted according to the type,
    thus their types are described by a `tf.Tensor` natoms, see explanation below.

    Parameters
    ----------
    ntypes:
        Number of types.
    natoms:
        The number of atoms. This tensor has the length of Ntypes + 2
        natoms[0]: number of local atoms
        natoms[1]: total number of atoms held by this processor
        natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
    type_embedding:
        The type embedding.
        It has the shape of [ntypes, embedding_dim]

    Returns
    -------
    atom_embedding
        The embedded type of each atom.
        It has the shape of [numb_atoms, embedding_dim]
    """
    te_out_dim = type_embedding.get_shape().as_list()[-1]
    atype = []
    for ii in range(ntypes):
        atype.append(tf.tile([ii], [natoms[2 + ii]]))
    atype = tf.concat(atype, axis=0)
    atm_embed = tf.nn.embedding_lookup(
        type_embedding, tf.cast(atype, dtype=tf.int32)
    )  # (nf*natom)*nchnl
    atm_embed = tf.reshape(atm_embed, [-1, te_out_dim])
    return atm_embed


class TypeEmbedNet:
    r"""Type embedding network.

    Parameters
    ----------
    ntypes : int
        Number of atom types
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    padding
            Concat the zero padding to the output, as the default embedding of empty type.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        neuron: List[int],
        resnet_dt: bool = False,
        activation_function: Union[str, None] = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[int] = None,
        uniform_seed: bool = False,
        padding: bool = False,
        use_econf_tebd: bool = False,
        type_map: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Constructor."""
        self.ntypes = ntypes
        self.neuron = neuron
        self.seed = seed
        self.filter_resnet_dt = resnet_dt
        self.filter_precision = get_precision(precision)
        self.filter_activation_fn_name = str(activation_function)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.trainable = trainable
        self.uniform_seed = uniform_seed
        self.type_embedding_net_variables = None
        self.padding = padding
        self.use_econf_tebd = use_econf_tebd
        self.type_map = type_map
        if self.use_econf_tebd:
            self.econf_tebd, _ = get_econf_tebd(self.type_map, precision=precision)
        self.model_type = None

    def build(
        self,
        ntypes: int,
        reuse=None,
        suffix="",
    ):
        """Build the computational graph for the descriptor.

        Parameters
        ----------
        ntypes
            Number of atom types.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        embedded_types
            The computational graph for embedded types
        """
        assert ntypes == self.ntypes
        if not self.use_econf_tebd:
            types = tf.convert_to_tensor(list(range(ntypes)), dtype=tf.int32)
            ebd_type = tf.cast(
                tf.one_hot(tf.cast(types, dtype=tf.int32), int(ntypes)),
                self.filter_precision,
            )
        else:
            ebd_type = tf.cast(
                tf.convert_to_tensor(self.econf_tebd),
                self.filter_precision,
            )
        ebd_type = tf.reshape(ebd_type, [ntypes, -1])
        name = "type_embed_net" + suffix
        if (
            nvnmd_cfg.enable
            and (nvnmd_cfg.version == 1)
            and (nvnmd_cfg.restore_descriptor or nvnmd_cfg.restore_fitting_net)
        ):
            self.type_embedding_net_variables = nvnmd_cfg.get_dp_init_weights()
        with tf.variable_scope(name, reuse=reuse):
            ebd_type = embedding_net(
                ebd_type,
                self.neuron,
                activation_fn=self.filter_activation_fn,
                precision=self.filter_precision,
                resnet_dt=self.filter_resnet_dt,
                seed=self.seed,
                trainable=self.trainable,
                initial_variables=self.type_embedding_net_variables,
                uniform_seed=self.uniform_seed,
            )
        ebd_type = tf.reshape(ebd_type, [-1, self.neuron[-1]])  # ntypes * neuron[-1]
        if self.padding:
            last_type = tf.cast(tf.zeros([1, self.neuron[-1]]), self.filter_precision)
            ebd_type = tf.concat([ebd_type, last_type], 0)  # (ntypes + 1) * neuron[-1]
        self.ebd_type = tf.identity(ebd_type, name="t_typeebd" + suffix)
        return self.ebd_type

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix="",
        model_type="original_model",
    ) -> None:
        """Init the type embedding net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix
            Name suffix to identify this descriptor
        model_type
            Indicator of whether this model is a compressed model
        """
        self.model_type = model_type
        self.type_embedding_net_variables = (
            get_type_embedding_net_variables_from_graph_def(graph_def, suffix=suffix)
        )

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        Model
            The deserialized model
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data_cls = data.pop("@class")
        assert data_cls == "TypeEmbedNet", f"Invalid class {data_cls}"

        embedding_net = EmbeddingNet.deserialize(data.pop("embedding"))
        embedding_net_variables = {}
        for layer_idx, layer in enumerate(embedding_net.layers):
            embedding_net_variables[
                f"type_embed_net{suffix}/matrix_{layer_idx + 1}"
            ] = layer.w
            embedding_net_variables[f"type_embed_net{suffix}/bias_{layer_idx + 1}"] = (
                layer.b
            )
            if layer.idt is not None:
                embedding_net_variables[
                    f"type_embed_net{suffix}/idt_{layer_idx + 1}"
                ] = layer.idt.reshape(1, -1)
            else:
                # prevent keyError
                embedding_net_variables[
                    f"type_embed_net{suffix}/idt_{layer_idx + 1}"
                ] = 0.0

        type_embedding_net = cls(**data)
        type_embedding_net.type_embedding_net_variables = embedding_net_variables
        return type_embedding_net

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Parameters
        ----------
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The serialized data
        """
        if suffix != "":
            type_embedding_pattern = (
                TYPE_EMBEDDING_PATTERN.replace("/(idt)", suffix + "/(idt)")
                .replace("/(bias)", suffix + "/(bias)")
                .replace("/(matrix)", suffix + "/(matrix)")
            )
        else:
            type_embedding_pattern = TYPE_EMBEDDING_PATTERN
        assert self.type_embedding_net_variables is not None
        embed_input_dim = self.ntypes
        if self.use_econf_tebd:
            from deepmd.utils.econf_embd import (
                ECONF_DIM,
            )

            embed_input_dim = ECONF_DIM
        embedding_net = EmbeddingNet(
            in_dim=embed_input_dim,
            neuron=self.neuron,
            activation_function=self.filter_activation_fn_name,
            resnet_dt=self.filter_resnet_dt,
            precision=self.filter_precision.name,
        )
        for key, value in self.type_embedding_net_variables.items():
            m = re.search(type_embedding_pattern, key)
            m = [mm for mm in m.groups() if mm is not None]
            layer_idx = int(m[1]) - 1
            weight_name = m[0]
            if weight_name == "idt":
                value = value.ravel()
            embedding_net[layer_idx][weight_name] = value

        return {
            "@class": "TypeEmbedNet",
            "@version": 1,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "resnet_dt": self.filter_resnet_dt,
            "precision": self.filter_precision.name,
            "activation_function": self.filter_activation_fn_name,
            "trainable": self.trainable,
            "padding": self.padding,
            "use_econf_tebd": self.use_econf_tebd,
            "type_map": self.type_map,
            "embedding": embedding_net.serialize(),
        }
