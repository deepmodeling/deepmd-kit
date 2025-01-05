# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)

import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.train.wrapper import (
    ModelWrapper,
)
from deepmd.pd.utils.env import (
    DEVICE,
    JIT,
)

log = logging.getLogger(__name__)


class Tester:
    def __init__(
        self,
        model_ckpt,
        head=None,
    ):
        """Construct a DeePMD tester.

        Args:
        - config: The Dict-like configuration with training options.
        """
        # Model
        state_dict = paddle.load(model_ckpt)
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

        self.model_params = deepcopy(model_params)
        self.model = get_model(model_params).to(DEVICE)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model)  # inference only
        if JIT:
            raise NotImplementedError
            # self.wrapper = paddle.jit.to_static(self.wrapper)
        self.wrapper.set_state_dict(state_dict)
