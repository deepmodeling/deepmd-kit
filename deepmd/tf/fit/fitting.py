# SPDX-License-Identifier: LGPL-3.0-or-later
import re
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
    List,
)

from deepmd.dpmodel.utils.network import (
    FittingNet,
    NetworkCollection,
)
from deepmd.tf.env import (
    FITTING_NET_PATTERN,
    tf,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.utils import (
    Plugin,
    PluginVariant,
)


class Fitting(PluginVariant):
    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable:
        """Register a Fitting plugin.

        Parameters
        ----------
        key : str
            the key of a Fitting

        Returns
        -------
        Fitting
            the registered Fitting

        Examples
        --------
        >>> @Fitting.register("some_fitting")
            class SomeFitting(Fitting):
                pass
        """
        return Fitting.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            try:
                fitting_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of fitting should be set by `type`")
            if fitting_type in Fitting.__plugins.plugins:
                cls = Fitting.__plugins.plugins[fitting_type]
            else:
                raise RuntimeError("Unknown descriptor type: " + fitting_type)
        return super().__new__(cls)

    @property
    def precision(self) -> tf.DType:
        """Precision of fitting network."""
        return self.fitting_precision

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the fitting net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope

        Notes
        -----
        This method is called by others when the fitting supported initialization from the given variables.
        """
        raise NotImplementedError(
            "Fitting %s doesn't support initialization from the given variables!"
            % type(self).__name__
        )

    @abstractmethod
    def get_loss(self, loss: dict, lr) -> Loss:
        """Get the loss function.

        Parameters
        ----------
        loss : dict
            the loss dict
        lr : LearningRateExp
            the learning rate

        Returns
        -------
        Loss
            the loss function
        """

    def serialize_network(
        self,
        ntypes: int,
        ndim: int,
        in_dim: int,
        neuron: List[int],
        activation_function: str,
        resnet_dt: bool,
        variables: dict,
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
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The converted network data
        """
        fittings = NetworkCollection(
            ntypes=ntypes,
            ndim=ndim,
            network_type="fitting_network",
        )
        if suffix != "":
            fitting_net_pattern = (
                FITTING_NET_PATTERN.replace("/(idt)", suffix + "/(idt)")
                .replace("/(bias)", suffix + "/(bias)")
                .replace("/(matrix)", suffix + "/(matrix)")
            )
        else:
            fitting_net_pattern = FITTING_NET_PATTERN
        for key, value in variables.items():
            m = re.search(fitting_net_pattern, key)
            m = [mm for mm in m.groups() if mm is not None]
            layer_idx = int(m[0]) if m[0] != "final" else len(neuron)
            weight_name = m[-1]
            if ndim == 0:
                network_idx = ()
            elif ndim == 1:
                network_idx = (int(m[1]),)
            else:
                raise ValueError(f"Invalid ndim: {ndim}")
            if fittings[network_idx] is None:
                # initialize the network if it is not initialized
                fittings[network_idx] = FittingNet(
                    in_dim=in_dim,
                    out_dim=1,
                    neuron=neuron,
                    activation_function=activation_function,
                    resnet_dt=resnet_dt,
                    precision=self.precision.name,
                    bias_out=True,
                )
            assert fittings[network_idx] is not None
            if weight_name == "idt":
                value = value.ravel()
            fittings[network_idx][layer_idx][weight_name] = value
        return fittings.serialize()

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
            if embeddings.ndim == 0:
                key = ""
            elif embeddings.ndim == 1:
                key = "_type_" + key[5:]
            else:
                raise ValueError(f"Invalid ndim: {embeddings.ndim}")
            network = embeddings[net_idx]
            assert network is not None
            for layer_idx, layer in enumerate(network.layers):
                if layer_idx == len(network.layers) - 1:
                    layer_name = "final_layer"
                else:
                    layer_name = f"layer_{layer_idx}"
                embedding_net_variables[f"{layer_name}{key}{suffix}/matrix"] = layer.w
                embedding_net_variables[f"{layer_name}{key}{suffix}/bias"] = layer.b
                if layer.idt is not None:
                    embedding_net_variables[
                        f"{layer_name}{key}{suffix}/idt"
                    ] = layer.idt.reshape(1, -1)
                else:
                    # prevent keyError
                    embedding_net_variables[f"{layer_name}{key}{suffix}/idt_"] = 0.0
        return embedding_net_variables
