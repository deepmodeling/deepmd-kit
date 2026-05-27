# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt wrapper around dpmodel ``DescrptBlockRepflows``.

The wrapper overrides ``_exchange_ghosts`` so that, when running under
LAMMPS multi-rank with a non-None ``comm_dict``, each layer of the
RepFlow message-passing block exchanges ghost-atom embeddings via the
opaque ``deepmd_export::border_op`` wrapper (registered in
``deepmd/pt_expt/utils/comm.py``). This survives ``torch.export`` and
AOTInductor packaging.

When ``comm_dict is None`` (single-rank inference / training), the
default array-api ``_exchange_ghosts`` from the dpmodel block is used —
zero behavioural change.
"""

from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.descriptor.repflows import (
    DescrptBlockRepflows as DescrptBlockRepflowsDP,
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class DescrptBlockRepflows(DescrptBlockRepflowsDP):
    """pt_expt wrapper for the RepFlow descriptor block."""

    def _exchange_ghosts(
        self,
        node_ebd: torch.Tensor,
        mapping_tiled: torch.Tensor | None,
        comm_dict: dict | None,
        nall: int,
        nloc: int,
    ) -> torch.Tensor:
        if comm_dict is None:
            return super()._exchange_ghosts(
                node_ebd,
                mapping_tiled,
                comm_dict,
                nall,
                nloc,
            )
        # Pt's parallel branch (repflows.py:580-587) requires the
        # extended-region pathway (use_loc_mapping=False).  The
        # local-mapping codepath skips the per-layer ghost exchange
        # entirely, so combining it with comm_dict is contradictory.
        # Surface this as a clear error rather than producing silently
        # wrong results.
        if getattr(self, "use_loc_mapping", False):
            raise RuntimeError(
                "DescrptBlockRepflows._exchange_ghosts: comm_dict is "
                "set but use_loc_mapping=True. Multi-rank parallel "
                "inference requires use_loc_mapping=False so per-layer "
                "ghost exchange is meaningful."
            )
        # The squeeze(0) / unsqueeze(0) dance below assumes a single
        # frame.  LAMMPS always feeds nb=1 in production; refuse loudly
        # if a Python caller batches frames so the mismatch surfaces
        # here rather than as a malformed border_op tensor downstream.
        if node_ebd.shape[0] != 1:
            raise RuntimeError(
                "DescrptBlockRepflows._exchange_ghosts: comm_dict path "
                "only supports nf=1 (got nf="
                f"{node_ebd.shape[0]}). Multi-frame batching with "
                "comm_dict is not supported."
            )

        has_spin = "has_spin" in comm_dict
        if has_spin:
            real_nloc, real_nall = nloc // 2, nall // 2
            real_pad = real_nall - real_nloc
            node_real, node_virt = torch.split(
                node_ebd,
                [real_nloc, real_nloc],
                dim=1,
            )
            # combine real + virtual along feature dim, then pad to nall.
            mix = torch.cat([node_real, node_virt], dim=2)
            padded = torch.nn.functional.pad(
                mix.squeeze(0),
                (0, 0, 0, real_pad),
                value=0.0,
            )
        else:
            padded = torch.nn.functional.pad(
                node_ebd.squeeze(0),
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
            n_dim = node_ebd.shape[-1]
            real_ext, virt_ext = torch.split(exchanged, [n_dim, n_dim], dim=2)
            return concat_switch_virtual(real_ext, virt_ext, real_nloc)
        return exchanged


# Register the converter so dpmodel's auto-wrap path picks up our pt_expt
# subclass instead of the generic _auto_wrap_native_op fallback. Without
# this, the override above would never fire.
register_dpmodel_mapping(
    DescrptBlockRepflowsDP,
    lambda v: DescrptBlockRepflows.deserialize(v.serialize()),
)
