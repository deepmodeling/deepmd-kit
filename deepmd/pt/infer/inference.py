# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.env import (
    DEVICE,
    JIT,
)

if torch.__version__.startswith("2"):
    import torch._dynamo
log = logging.getLogger(__name__)


class Tester:
    def __init__(
        self,
        model_ckpt,
        head=None,
    ) -> None:
        """Construct a DeePMD tester.

        Args:
        - config: The Dict-like configuration with training options.
        """
        # Model
        state_dict = torch.load(model_ckpt, map_location=DEVICE, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = state_dict["_extra_state"]["model_params"]
        self.multi_task = "model_dict" in model_params
        if self.multi_task:
            assert head is not None, "Head must be specified in multitask mode!"
            self.head = head
            assert head in model_params["model_dict"], (
                f"Specified head {head} not found in model {model_ckpt}! "
                f"Available ones are {list(model_params['model_dict'].keys())}."
            )
            model_params = model_params["model_dict"][head]
            state_dict_head = {"_extra_state": state_dict["_extra_state"]}
            for item in state_dict:
                if f"model.{head}." in item:
                    state_dict_head[
                        item.replace(f"model.{head}.", "model.Default.")
                    ] = state_dict[item].clone()
            state_dict = state_dict_head

        model_params.pop(
            "hessian_mode", None
        )  # wrapper Hessian to Energy model due to JIT limit
        self.model_params = deepcopy(model_params)
        self.model = get_model(model_params).to(DEVICE)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model)  # inference only
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)
        self.wrapper.load_state_dict(state_dict)
