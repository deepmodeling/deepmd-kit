# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils import BaseSpin as DPBaseSpin
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)


class Spin(DPBaseSpin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_scale_mask = to_torch_tensor(
            self.virtual_scale * self.use_spin
        ).view([-1])
        self.spin_mask = to_torch_tensor(self.spin_mask)

    def get_virtual_scale_mask(self):
        return self.virtual_scale_mask

    def get_spin_mask(self):
        return self.spin_mask

    def serialize(
        self,
    ) -> dict:
        return {
            "use_spin": self.use_spin,
            "virtual_scale": self.virtual_scale,
        }

    @classmethod
    def deserialize(
        cls,
        data: dict,
    ) -> "Spin":
        return cls(**data)
