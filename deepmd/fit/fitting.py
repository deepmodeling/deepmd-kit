# SPDX-License-Identifier: LGPL-3.0-or-later
import re
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
    DefaultDict,
)

from deepmd.env import (
    FITTING_NET_PATTERN,
    tf,
)
from deepmd.loss.loss import (
    Loss,
)
from deepmd.utils import (
    Plugin,
    PluginVariant,
)
from deepmd_utils.model_format import (
    NativeNet,
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

    @classmethod
    def get_class_by_input(cls, input: dict):
        try:
            fitting_type = input["type"]
        except KeyError:
            raise KeyError("the type of fitting should be set by `type`")
        if fitting_type in Fitting.__plugins.plugins:
            return Fitting.__plugins.plugins[fitting_type]
        else:
            raise RuntimeError("Unknown descriptor type: " + fitting_type)

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            cls = Fitting.get_class_by_input(kwargs)
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

    @classmethod
    def deserialize(cls, data: dict):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        if cls is Fitting:
            return Fitting.get_class_by_input(data).deserialize(data)
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    def serialize(self) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        raise NotImplementedError("Not implemented in class %s" % self.__name__)

    def to_dp_variables(self, variables: dict) -> dict:
        """Convert the variables to deepmd format.

        Parameters
        ----------
        variables : dict
            The input variables

        Returns
        -------
        dict
            The converted variables
        """
        networks = DefaultDict(NativeNet)
        for key, value in variables.items():
            m = re.search(FITTING_NET_PATTERN, key)
            m = [mm for mm in m.groups() if mm is not None]
            # type_{m[1]}/layer_{m[0]}/{m[-1]}
            atom_type = m[1] if len(m) >= 3 else "all"
            layer_idx = int(m[0]) if m[0] != "final" else len(self.n_neuron)
            weight_name = m[-1]
            networks[f"type_{atom_type}"][layer_idx][weight_name] = value
        return {key: value.serialize() for key, value in networks.items()}

    @classmethod
    def from_dp_variables(cls, variables: dict) -> dict:
        """Convert the variables from deepmd format.

        Parameters
        ----------
        variables : dict
            The input variables

        Returns
        -------
        dict
            The converted variables
        """
        embedding_net_variables = {}
        for key, value in variables.items():
            if key[5:] == "all":
                key = ""
            else:
                key = "_type_" + key[5:]
            network = NativeNet.deserialize(value)
            for layer_idx, layer in enumerate(network.layers):
                if layer_idx == len(network.layers) - 1:
                    layer_name = "final_layer"
                else:
                    layer_name = f"layer_{layer_idx}"
                embedding_net_variables[f"{layer_name}{key}/matrix"] = layer.w
                embedding_net_variables[f"{layer_name}{key}/bias"] = layer.b
                if layer.idt is not None:
                    embedding_net_variables[f"{layer_name}{key}/idt"] = layer.idt
        return embedding_net_variables
