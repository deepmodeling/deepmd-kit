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
        # The squeeze(0) / unsqueeze(0) dance below assumes a single
        # frame.  LAMMPS always feeds nb=1 in production; refuse loudly
        # if a Python caller batches frames so the mismatch surfaces
        # here rather than as a malformed border_op tensor downstream.
        if g1.shape[0] != 1:
            raise RuntimeError(
                "DescrptBlockRepformers._exchange_ghosts: comm_dict path "
                "only supports nf=1 (got nf="
                f"{g1.shape[0]}). Multi-frame batching with comm_dict is "
                "not supported."
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

    def _exchange_ghosts_graph(
        self,
        g1: torch.Tensor,
        comm_dict: dict | None,
        n_total: int,
    ) -> torch.Tensor:
        """Graph-path per-layer halo refresh via ``border_op``.

        Flat single-frame counterpart of :meth:`_exchange_ghosts`: ``g1`` is
        already ``(N, ng1)`` with ``N == nlocal + nghost`` (owned prefix
        first), so no squeeze/pad/unsqueeze dance is needed -- ``border_op``
        overwrites the halo rows ``[nlocal, N)`` in place with the owner
        rows and returns the same tensor.  Identity without ``comm_dict``
        (ghost-free Python graphs / extended single-process graphs).  Spin
        models never route the graph lower (``disable_graph_lower``), so a
        ``has_spin`` comm_dict reaching here is a programming error.

        Parameters
        ----------
        g1
            Flat node-wise atomic invariant rep, with shape [n_total, ng1].
        comm_dict
            MPI communication metadata (``send_list``, ``send_proc``,
            ``recv_proc``, ``send_num``, ``recv_num``, ``communicator``,
            ``nlocal``, ``nghost``); ``None`` for single-process graphs.
        n_total
            Total number of nodes (``nlocal + nghost``).

        Returns
        -------
        g1 : torch.Tensor
            The node channel with halo rows refreshed, with shape
            [n_total, ng1].

        Raises
        ------
        NotImplementedError
            If ``comm_dict`` carries ``has_spin``.
        """
        if comm_dict is None:
            return super()._exchange_ghosts_graph(g1, comm_dict, n_total)
        if "has_spin" in comm_dict:
            raise NotImplementedError(
                "spin models do not route the graph lower (disable_graph_lower)"
            )
        return torch.ops.deepmd_export.border_op(
            comm_dict["send_list"],
            comm_dict["send_proc"],
            comm_dict["recv_proc"],
            comm_dict["send_num"],
            comm_dict["recv_num"],
            g1,
            comm_dict["communicator"],
            comm_dict["nlocal"],
            comm_dict["nghost"],
        )


register_dpmodel_mapping(
    DescrptBlockRepformersDP,
    lambda v: DescrptBlockRepformers.deserialize(v.serialize()),
)
