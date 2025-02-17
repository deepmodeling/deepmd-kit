# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABCMeta,
    abstractmethod,
)

import numpy as np

from deepmd.tf.env import (
    tf,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class Loss(metaclass=ABCMeta):
    """The abstract class for the loss function."""

    @abstractmethod
    def build(
        self,
        learning_rate: tf.Tensor,
        natoms: tf.Tensor,
        model_dict: dict[str, tf.Tensor],
        label_dict: dict[str, tf.Tensor],
        suffix: str,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Build the loss function graph.

        Parameters
        ----------
        learning_rate : tf.Tensor
            learning rate
        natoms : tf.Tensor
            number of atoms
        model_dict : dict[str, tf.Tensor]
            A dictionary that maps model keys to tensors
        label_dict : dict[str, tf.Tensor]
            A dictionary that maps label keys to tensors
        suffix : str
            suffix

        Returns
        -------
        tf.Tensor
            the total squared loss
        dict[str, tf.Tensor]
            A dictionary that maps loss keys to more loss tensors
        """

    @abstractmethod
    def eval(
        self,
        sess: tf.Session,
        feed_dict: dict[tf.placeholder, tf.Tensor],
        natoms: tf.Tensor,
    ) -> dict:
        """Eval the loss function.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow session
        feed_dict : dict[tf.placeholder, tf.Tensor]
            A dictionary that maps graph elements to values
        natoms : tf.Tensor
            number of atoms

        Returns
        -------
        dict
            A dictionary that maps keys to values. It
            should contain key `natoms`
        """

    @staticmethod
    def display_if_exist(loss: tf.Tensor, find_property: float) -> tf.Tensor:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : tf.Tensor
            the loss tensor
        find_property : float
            whether the property is found
        """
        return tf.cond(
            tf.cast(find_property, tf.bool),
            lambda: loss,
            lambda: tf.cast(np.nan, dtype=loss.dtype),
        )

    @property
    @abstractmethod
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the loss module.

        Parameters
        ----------
        suffix : str
            The suffix of the loss module

        Returns
        -------
        dict
            The serialized loss module
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict, suffix: str = "") -> "Loss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module
        suffix : str
            The suffix of the loss module

        Returns
        -------
        Loss
            The deserialized loss module
        """
        raise NotImplementedError

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """No actual effect.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope
        """
        pass
