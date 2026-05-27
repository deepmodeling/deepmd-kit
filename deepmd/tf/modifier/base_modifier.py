# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.tf.infer import (
        DeepEval,
    )

from deepmd.dpmodel.modifier.base_modifier import (
    make_base_modifier,
)
from deepmd.tf.infer import (
    DeepPot,
)


class BaseModifier(DeepPot, make_base_modifier()):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct a basic model for different tasks."""
        DeepPot.__init__(self, *args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_params_from_frozen_model(model: "DeepEval") -> dict:
        """Extract the modifier parameters from a model.

        This method should extract the necessary parameters from a model
        to create an instance of this modifier.

        Parameters
        ----------
        model
            The model from which to extract parameters

        Returns
        -------
        dict
            The modifier parameters
        """
        pass
