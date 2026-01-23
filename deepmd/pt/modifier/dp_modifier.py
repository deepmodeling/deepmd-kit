# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.model.model import (
    BaseModel,
)
from deepmd.pt.modifier.base_modifier import (
    BaseModifier,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.serialization import (
    serialize_from_file,
)


class DPModifier(BaseModifier):
    def __init__(
        self,
        dp_model: torch.nn.Module | None = None,
        dp_model_file_name: str | None = None,
        use_cache: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__(use_cache=use_cache)

        if dp_model_file_name is None and dp_model is None:
            raise AttributeError("`model_name` or `model` should be specified.")
        if dp_model_file_name is not None and dp_model is not None:
            raise AttributeError(
                "`model_name` and `model` cannot be used simultaneously."
            )

        if dp_model is not None:
            self._model = dp_model.to(env.DEVICE)
        if dp_model_file_name is not None:
            data = serialize_from_file(dp_model_file_name)
            model_type = data["model"]["type"]
            if model_type != "standard":
                raise ValueError(
                    f"DPModifier only support standard model. Unsupported model type: {model_type}"
                )
            self._model = (
                BaseModel.get_class_by_type(data["model"]["fitting"]["type"])
                .deserialize(data["model"])
                .to(env.DEVICE)
            )
        self._model.eval()
        # use jit model for inference
        self.model = torch.jit.script(self._model)

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        dd = BaseModifier.serialize(self)
        dd.update(
            {
                "dp_model": self._model.serialize(),
            }
        )
        return dd

    @classmethod
    def get_modifier(cls, modifier_params: dict) -> "DPModifier":
        """Get the modifier by the parameters.

        By default, all the parameters are directly passed to the constructor.
        If not, override this method.

        Parameters
        ----------
        modifier_params : dict
            The modifier parameters

        Returns
        -------
        BaseModifier
            The modifier
        """
        modifier_params = modifier_params.copy()
        modifier_params.pop("type", None)
        # convert model_name str
        model_name = modifier_params.pop("model_name", None)
        if model_name is not None:
            modifier_params["dp_model"] = None
            modifier_params["dp_model_file_name"] = model_name
        modifier = cls(**modifier_params)
        return modifier
