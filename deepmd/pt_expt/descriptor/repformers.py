# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt wrapper around dpmodel ``DescrptBlockRepformers``.

Mirrors ``deepmd/pt_expt/descriptor/repflows.py``: overrides
``_exchange_ghosts`` so the per-layer ghost exchange uses the opaque
``deepmd_export::border_op`` when a ``comm_dict`` is provided.
"""

from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers as DescrptBlockRepformersDP,
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class DescrptBlockRepformers(DescrptBlockRepformersDP):
    """pt_expt wrapper for the Repformers descriptor block."""

    def _exchange_ghosts(
        self,
        g1: torch.Tensor,
        mapping_tiled: torch.Tensor | None,
        comm_dict: dict | None,
        nall: int,
        nloc: int,
    ) -> torch.Tensor:
        if comm_dict is None:
            return super()._exchange_ghosts(
                g1,
                mapping_tiled,
                comm_dict,
                nall,
                nloc,
            )

        has_spin = "has_spin" in comm_dict
        if has_spin:
            real_nloc, real_nall = nloc // 2, nall // 2
            real_pad = real_nall - real_nloc
            g1_real, g1_virt = torch.split(g1, [real_nloc, real_nloc], dim=1)
            mix = torch.cat([g1_real, g1_virt], dim=2)
            padded = torch.nn.functional.pad(
                mix.squeeze(0),
                (0, 0, 0, real_pad),
                value=0.0,
            )
        else:
            padded = torch.nn.functional.pad(
                g1.squeeze(0),
                (0, 0, 0, nall - nloc),
                value=0.0,
            )

        exchanged = torch.ops.deepmd_export.border_op(
            comm_dict["send_list"],
            comm_dict["send_proc"],
            comm_dict["recv_proc"],
            comm_dict["send_num"],
            comm_dict["recv_num"],
            padded,
            comm_dict["communicator"],
            comm_dict["nlocal"],
            comm_dict["nghost"],
        ).unsqueeze(0)

        if has_spin:
            ng1 = g1.shape[-1]
            real_ext, virt_ext = torch.split(exchanged, [ng1, ng1], dim=2)
            return concat_switch_virtual(real_ext, virt_ext, real_nloc)
        return exchanged


register_dpmodel_mapping(
    DescrptBlockRepformersDP,
    lambda v: DescrptBlockRepformers.deserialize(v.serialize()),
)
