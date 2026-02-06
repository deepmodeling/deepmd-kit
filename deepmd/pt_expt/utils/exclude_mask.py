# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.pt_expt.utils import (
    env,
)


class AtomExcludeMask(AtomExcludeMaskDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        AtomExcludeMaskDP.__init__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "type_mask" and "_buffers" in self.__dict__:
            value = None if value is None else torch.as_tensor(value, device=env.DEVICE)
            if name in self._buffers:
                self._buffers[name] = value
                return
            self.register_buffer(name, value)
            return
        return super().__setattr__(name, value)


class PairExcludeMask(PairExcludeMaskDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        PairExcludeMaskDP.__init__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "type_mask" and "_buffers" in self.__dict__:
            value = None if value is None else torch.as_tensor(value, device=env.DEVICE)
            if name in self._buffers:
                self._buffers[name] = value
                return
            self.register_buffer(name, value)
            return
        return super().__setattr__(name, value)
