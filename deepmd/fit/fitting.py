# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
)

from deepmd.env import (
    tf,
)
from deepmd.loss.loss import (
    Loss,
)
from deepmd.utils import (
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
            f"Fitting {type(self).__name__} doesn't support initialization from the given variables!"
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
