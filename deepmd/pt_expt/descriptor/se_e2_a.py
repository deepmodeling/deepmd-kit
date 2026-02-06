# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt_expt.utils.network import (
    NetworkCollection,
)

torch = importlib.import_module("torch")


@BaseDescriptor.register("se_e2_a_expt")
@BaseDescriptor.register("se_a_expt")
class DescrptSeA(DescrptSeADP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        DescrptSeADP.__init__(self, *args, **kwargs)
        self._convert_state()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"davg", "dstd"} and "_buffers" in self.__dict__:
            tensor = (
                None if value is None else torch.as_tensor(value, device=env.DEVICE)
            )
            if name in self._buffers:
                self._buffers[name] = tensor
                return
            return super().__setattr__(name, tensor)
        if name == "embeddings" and "_modules" in self.__dict__:
            if value is not None and not isinstance(value, torch.nn.Module):
                if hasattr(value, "serialize"):
                    value = NetworkCollection.deserialize(value.serialize())
                elif isinstance(value, dict):
                    value = NetworkCollection.deserialize(value)
            return super().__setattr__(name, value)
        if name == "emask" and "_modules" in self.__dict__:
            if value is not None and not isinstance(value, torch.nn.Module):
                value = PairExcludeMask(
                    self.ntypes, exclude_types=list(value.get_exclude_types())
                )
            return super().__setattr__(name, value)
        return super().__setattr__(name, value)

    def _convert_state(self) -> None:
        if self.davg is not None:
            davg = torch.as_tensor(self.davg, device=env.DEVICE)
            if "davg" in self._buffers:
                self._buffers["davg"] = davg
            else:
                if hasattr(self, "davg"):
                    delattr(self, "davg")
                self.register_buffer("davg", davg)
        if self.dstd is not None:
            dstd = torch.as_tensor(self.dstd, device=env.DEVICE)
            if "dstd" in self._buffers:
                self._buffers["dstd"] = dstd
            else:
                if hasattr(self, "dstd"):
                    delattr(self, "dstd")
                self.register_buffer("dstd", dstd)
        if self.embeddings is not None:
            self.embeddings = NetworkCollection.deserialize(self.embeddings.serialize())
        if self.emask is not None:
            self.emask = PairExcludeMask(
                self.ntypes, exclude_types=list(self.emask.get_exclude_types())
            )

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        type_embedding: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del extended_atype_embd, type_embedding
        descrpt, rot_mat, g2, h2, sw = self.call(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        return descrpt, rot_mat, g2, h2, sw
