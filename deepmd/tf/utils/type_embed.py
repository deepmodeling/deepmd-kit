# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.tf.common import (
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
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
    """

    def __init__(
        self,
        neuron: List[int] = [],
        resnet_dt: bool = False,
        activation_function: Union[str, None] = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[int] = None,
        uniform_seed: bool = False,
        padding: bool = False,
        **kwargs,
    ) -> None:
        """Constructor."""
        self.neuron = neuron
        self.seed = seed
        self.filter_resnet_dt = resnet_dt
        self.filter_precision = get_precision(precision)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.trainable = trainable
        self.uniform_seed = uniform_seed
        self.type_embedding_net_variables = None
        self.padding = padding
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
        types = tf.convert_to_tensor(list(range(ntypes)), dtype=tf.int32)
        ebd_type = tf.cast(
            tf.one_hot(tf.cast(types, dtype=tf.int32), int(ntypes)),
            self.filter_precision,
        )
        ebd_type = tf.reshape(ebd_type, [-1, ntypes])
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
