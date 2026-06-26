# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import pytest

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    edge_env_mat,
    from_dense_quartet,
)


class TestEdgeEnvMat(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.rcut, self.rcut_smth = 4.0, 0.5
        self.nf, self.nloc, self.nnei = 1, 4, 6
        self.ext_coord = rng.normal(size=(self.nf, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)
        nlist = -np.ones((self.nf, self.nloc, self.nnei), dtype=np.int64)
        for i in range(self.nloc):
            ns = [j for j in range(self.nloc) if j != i][: self.nnei]
            nlist[0, i, : len(ns)] = ns
        self.nlist = nlist
        self.mapping = np.arange(self.nloc, dtype=np.int64)[None]
        self.nt = 2
        self.davg = rng.normal(size=(self.nt, 4))
        self.dstd = np.abs(rng.normal(size=(self.nt, 4))) + 0.5

    def test_matches_envmat_slice(self) -> None:
        davg_dense = np.broadcast_to(
            self.davg[:, None, :], (self.nt, self.nnei, 4)
        ).copy()
        dstd_dense = np.broadcast_to(
            self.dstd[:, None, :], (self.nt, self.nnei, 4)
        ).copy()
        dmat, _, _ = EnvMat(self.rcut, self.rcut_smth).call(
            self.ext_coord, self.atype, self.nlist, davg_dense, dstd_dense
        )

        ng = from_dense_quartet(self.ext_coord, self.nlist, self.mapping)
        center_type = self.atype.reshape(-1)[ng.edge_index[1]]
        em = edge_env_mat(
            ng.edge_vec, center_type, self.davg, self.dstd, self.rcut, self.rcut_smth
        )

        ei = ng.edge_index[:, ng.edge_mask]
        for k in range(ei.shape[1]):
            src, dst = int(ei[0, k]), int(ei[1, k])
            slot = list(self.nlist[0, dst]).index(src)
            np.testing.assert_allclose(
                em[k], dmat[0, dst, slot], rtol=1e-12, atol=1e-12
            )

    def test_slot_broadcast_stats(self) -> None:
        """After compute_input_stats, DescrptBlockSeAtten stats must be
        slot-uniform: mean[:, k, :] == mean[:, 0, :] for all slots k.
        This property is what allows edge_env_mat to use (ntypes, 4) stats
        instead of (ntypes, nnei, 4) stats.
        """
        from deepmd.dpmodel.descriptor import (
            DescrptDPA1,
        )

        rng = np.random.default_rng(42)
        nloc = 6
        nf = 3
        rcut = 4.0
        rcut_smth = 0.5
        ntypes = 2
        sel = [6, 6]

        coord = rng.normal(size=(nf, nloc, 3)).astype(np.float64)
        # scale so atoms are within rcut of each other
        coord = coord * 1.2
        atype = np.array([[0, 1, 0, 1, 0, 1]] * nf, dtype=np.int64)
        # non-periodic: box=None
        data = [
            {
                "coord": coord,
                "atype": atype,
                "box": None,
            }
        ]

        dpa1 = DescrptDPA1(rcut, rcut_smth, sel, ntypes=ntypes)
        dpa1.compute_input_stats(data)
        block = dpa1.se_atten

        nnei = block.nnei
        for k in range(1, nnei):
            np.testing.assert_allclose(
                block.mean[:, 0, :],
                block.mean[:, k, :],
                rtol=0,
                atol=0,
                err_msg=f"mean slot {k} != slot 0",
            )
            np.testing.assert_allclose(
                block.stddev[:, 0, :],
                block.stddev[:, k, :],
                rtol=0,
                atol=0,
                err_msg=f"stddev slot {k} != slot 0",
            )


# ── Protection parity (Task 6) ────────────────────────────────────────────────


@pytest.mark.parametrize("protection", [0.0, 1e-2])  # env-mat protection offset
def test_edge_env_mat_protection_parity(protection):
    """edge_env_mat(protection=p, edge_mask=...) must match EnvMat(protection=p).call slice."""
    rng = np.random.default_rng(7)
    rcut, rcut_smth = 4.0, 0.5
    nf, nloc, nnei = 1, 4, 6
    nt = 2

    ext_coord = rng.normal(size=(nf, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    # Build nlist with at most 3 valid neighbors per atom; slots 3-5 are padding (-1).
    nlist = -np.ones((nf, nloc, nnei), dtype=np.int64)
    for i in range(nloc):
        ns = [j for j in range(nloc) if j != i][:nnei]
        nlist[0, i, : len(ns)] = ns
    mapping = np.arange(nloc, dtype=np.int64)[None]

    davg = rng.normal(size=(nt, 4))
    dstd = np.abs(rng.normal(size=(nt, 4))) + 0.5

    # ── dense reference (EnvMat.call) ──────────────────────────────────────
    davg_dense = np.broadcast_to(davg[:, None, :], (nt, nnei, 4)).copy()
    dstd_dense = np.broadcast_to(dstd[:, None, :], (nt, nnei, 4)).copy()
    dmat, _, _ = EnvMat(rcut, rcut_smth, protection=protection).call(
        ext_coord, atype, nlist, davg_dense, dstd_dense
    )

    # ── graph path (edge_env_mat with edge_mask) ───────────────────────────
    ng = from_dense_quartet(ext_coord, nlist, mapping)
    center_type = atype.reshape(-1)[ng.edge_index[1]]
    em = edge_env_mat(
        ng.edge_vec,
        center_type,
        davg,
        dstd,
        rcut,
        rcut_smth,
        protection=protection,
        edge_mask=ng.edge_mask,
    )

    # Compare valid edges only, matched to their dense (frame, dst, slot) position.
    ei = ng.edge_index[:, ng.edge_mask]
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        slot = list(nlist[0, dst]).index(src)
        np.testing.assert_allclose(
            em[ng.edge_mask][k],
            dmat[0, dst, slot],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"protection={protection}, edge {k} (src={src}, dst={dst}, slot={slot})",
        )
