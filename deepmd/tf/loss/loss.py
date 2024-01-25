# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    Dict,
    Tuple,
)

import numpy as np

from deepmd.tf.env import (
    tf,
)


class Loss(metaclass=ABCMeta):
    """The abstract class for the loss function."""

    @abstractmethod
    def build(
        self,
        learning_rate: tf.Tensor,
        natoms: tf.Tensor,
        model_dict: Dict[str, tf.Tensor],
        label_dict: Dict[str, tf.Tensor],
        suffix: str,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
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
        feed_dict: Dict[tf.placeholder, tf.Tensor],
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
