# SPDX-License-Identifier: LGPL-3.0-or-later
import re
from typing import (
    List,
    Set,
    Tuple,
)

from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.tf.env import (
    EMBEDDING_NET_PATTERN,
    tf,
)
from deepmd.tf.utils.graph import (
    get_embedding_net_variables_from_graph_def,
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)

from .descriptor import (
    Descriptor,
)


class DescrptSe(Descriptor):
    """A base class for smooth version of descriptors.

    Notes
    -----
    All of these descriptors have an environmental matrix and an
    embedding network (:meth:`deepmd.tf.utils.network.embedding_net`), so
    they can share some similiar methods without defining them twice.

    Attributes
    ----------
    embedding_net_variables : dict
        initial embedding network variables
    descrpt_reshape : tf.Tensor
        the reshaped descriptor
    descrpt_deriv : tf.Tensor
        the descriptor derivative
    rij : tf.Tensor
        distances between two atoms
    nlist : tf.Tensor
        the neighbor list

    """

    def _identity_tensors(self, suffix: str = "") -> None:
        """Identify tensors which are expected to be stored and restored.

        Notes
        -----
        These tensors will be indentitied:
            self.descrpt_reshape : o_rmat
            self.descrpt_deriv : o_rmat_deriv
            self.rij : o_rij
            self.nlist : o_nlist
        Thus, this method should be called during building the descriptor and
        after these tensors are initialized.

        Parameters
        ----------
        suffix : str
            The suffix of the scope
        """
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name="o_rmat" + suffix)
        self.descrpt_deriv = tf.identity(
            self.descrpt_deriv, name="o_rmat_deriv" + suffix
        )
        self.rij = tf.identity(self.rij, name="o_rij" + suffix)
        self.nlist = tf.identity(self.nlist, name="o_nlist" + suffix)

    def get_tensor_names(self, suffix: str = "") -> Tuple[str]:
        """Get names of tensors.

        Parameters
        ----------
        suffix : str
            The suffix of the scope

        Returns
        -------
        Tuple[str]
            Names of tensors
        """
        return (
            f"o_rmat{suffix}:0",
            f"o_rmat_deriv{suffix}:0",
            f"o_rij{suffix}:0",
            f"o_nlist{suffix}:0",
        )

    def pass_tensors_from_frz_model(
        self,
        descrpt_reshape: tf.Tensor,
        descrpt_deriv: tf.Tensor,
        rij: tf.Tensor,
        nlist: tf.Tensor,
    ):
        """Pass the descrpt_reshape tensor as well as descrpt_deriv tensor from the frz graph_def.

        Parameters
        ----------
        descrpt_reshape
            The passed descrpt_reshape tensor
        descrpt_deriv
            The passed descrpt_deriv tensor
        rij
            The passed rij tensor
        nlist
            The passed nlist tensor
        """
        self.rij = rij
        self.nlist = nlist
        self.descrpt_deriv = descrpt_deriv
        self.descrpt_reshape = descrpt_reshape

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope
        """
        self.embedding_net_variables = get_embedding_net_variables_from_graph_def(
            graph_def, suffix=suffix
        )
        self.davg = get_tensor_by_name_from_graph(
            graph, "descrpt_attr%s/t_avg" % suffix
        )
        self.dstd = get_tensor_by_name_from_graph(
            graph, "descrpt_attr%s/t_std" % suffix
        )

    @property
    def precision(self) -> tf.DType:
        """Precision of filter network."""
        return self.filter_precision

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        # default behavior is to update sel which is a list
        local_jdata_cpy = local_jdata.copy()
        return UpdateSel().update_one_sel(global_jdata, local_jdata_cpy, False)

    def serialize_network(
        self,
        ntypes: int,
        ndim: int,
        in_dim: int,
        neuron: List[int],
        activation_function: str,
        resnet_dt: bool,
        variables: dict,
        excluded_types: Set[Tuple[int, int]] = set(),
        suffix: str = "",
    ) -> dict:
        """Serialize network.

        Parameters
        ----------
        ntypes : int
            The number of types
        ndim : int
            The dimension of elements
        in_dim : int
            The input dimension
        neuron : List[int]
            The neuron list
        activation_function : str
            The activation function
        resnet_dt : bool
            Whether to use resnet
        variables : dict
            The input variables
        excluded_types : Set[Tuple[int, int]], optional
            The excluded types
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The converted network data
        """
        embeddings = NetworkCollection(
            ntypes=ntypes,
            ndim=ndim,
            network_type="embedding_network",
        )
        if ndim == 2:
            for type_i, type_j in excluded_types:
                # initialize an empty network for the excluded types
                embeddings[(type_i, type_j)] = EmbeddingNet(
                    in_dim=in_dim,
                    neuron=neuron,
                    activation_function=activation_function,
                    resnet_dt=resnet_dt,
                    precision=self.precision.name,
                )
                embeddings[(type_j, type_i)] = EmbeddingNet(
                    in_dim=in_dim,
                    neuron=neuron,
                    activation_function=activation_function,
                    resnet_dt=resnet_dt,
                    precision=self.precision.name,
                )
                embeddings[(type_i, type_j)].clear()
                embeddings[(type_j, type_i)].clear()

        if suffix != "":
            embedding_net_pattern = (
                EMBEDDING_NET_PATTERN.replace("/(idt)", suffix + "/(idt)")
                .replace("/(bias)", suffix + "/(bias)")
                .replace("/(matrix)", suffix + "/(matrix)")
            )
        else:
            embedding_net_pattern = EMBEDDING_NET_PATTERN
        for key, value in variables.items():
            m = re.search(embedding_net_pattern, key)
            m = [mm for mm in m.groups() if mm is not None]
            typei = m[0]
            typej = "_".join(m[3:]) if len(m[3:]) else "all"
            layer_idx = int(m[2]) - 1
            weight_name = m[1]
            if ndim == 0:
                network_idx = ()
            elif ndim == 1:
                network_idx = (int(typej),)
            elif ndim == 2:
                network_idx = (int(typei), int(typej))
            else:
                raise ValueError(f"Invalid ndim: {ndim}")
            if embeddings[network_idx] is None:
                # initialize the network if it is not initialized
                embeddings[network_idx] = EmbeddingNet(
                    in_dim=in_dim,
                    neuron=neuron,
                    activation_function=activation_function,
                    resnet_dt=resnet_dt,
                    precision=self.precision.name,
                )
            assert embeddings[network_idx] is not None
            if weight_name == "idt":
                value = value.ravel()
            embeddings[network_idx][layer_idx][weight_name] = value
        return embeddings.serialize()

    @classmethod
    def deserialize_network(cls, data: dict, suffix: str = "") -> dict:
        """Deserialize network.

        Parameters
        ----------
        data : dict
            The input network data
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        variables : dict
            The input variables
        """
        embedding_net_variables = {}
        embeddings = NetworkCollection.deserialize(data)
        for ii in range(embeddings.ntypes**embeddings.ndim):
            net_idx = []
            rest_ii = ii
            for _ in range(embeddings.ndim):
                net_idx.append(rest_ii % embeddings.ntypes)
                rest_ii //= embeddings.ntypes
            net_idx = tuple(net_idx)
            if embeddings.ndim in (0, 1):
                key0 = "all"
                key1 = f"_{ii}"
            elif embeddings.ndim == 2:
                key0 = f"{net_idx[0]}"
                key1 = f"_{net_idx[1]}"
            else:
                raise ValueError(f"Invalid ndim: {embeddings.ndim}")
            network = embeddings[net_idx]
            assert network is not None
            for layer_idx, layer in enumerate(network.layers):
                embedding_net_variables[
                    f"filter_type_{key0}{suffix}/matrix_{layer_idx + 1}{key1}"
                ] = layer.w
                embedding_net_variables[
                    f"filter_type_{key0}{suffix}/bias_{layer_idx + 1}{key1}"
                ] = layer.b
                if layer.idt is not None:
                    embedding_net_variables[
                        f"filter_type_{key0}{suffix}/idt_{layer_idx + 1}{key1}"
                    ] = layer.idt.reshape(1, -1)
                else:
                    # prevent keyError
                    embedding_net_variables[
                        f"filter_type_{key0}{suffix}/idt_{layer_idx + 1}{key1}"
                    ] = 0.0
        return embedding_net_variables
