# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parts of the nv path that run without a GPU.

Performance work on the nv builder is CUDA-bound and stays behind the opt-in
suite (test_nv_graph_builder.py), but two things are checkable anywhere
nvalchemiops imports. The decode (``nv_matrix_to_ijs``) is pure torch index
arithmetic, so its regression-prone parts (``// max_neighbors``, ``% nloc``,
frame isolation, slot-validity mask, and the lift back onto a padded batch)
are pinned here with synthetic inputs. The builder's agreement with the dense
reference builder is pinned here too, since the search itself is device
agnostic even though it is only worth running on a GPU.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.pt.utils.nv_nlist import (
    is_nv_available,
)
from deepmd.pt_expt.utils.nv_graph_builder import (
    nv_matrix_to_ijs,
)


def _edge_set(i, j, s, f):
    return {
        (int(f[e]), int(i[e]), int(j[e]), tuple(int(x) for x in s[e]))
        for e in range(i.shape[0])
    }


class TestNvMatrixDecode:
    def test_two_frames_hand_checked(self) -> None:
        """nf=2, nloc=3, max_neighbors=2; edges and shifts checked by hand."""
        nloc = 3
        # flattened centers 0..5; frame 0 = atoms 0-2, frame 1 = atoms 3-5.
        # matrix[dst, slot] = src (flattened); only the first num_neighbors
        # slots are valid, the rest is stale garbage that MUST be ignored.
        neighbor_matrix = torch.tensor(
            [
                [1, 2],  # center 0: neighbors 1, 2
                [0, 9],  # center 1: neighbor 0 (slot 1 = garbage)
                [0, 9],  # center 2: neighbor 0 (slot 1 = garbage)
                [4, 9],  # center 3 (frame 1, local 0): neighbor 4 (local 1)
                [3, 9],  # center 4 (frame 1, local 1): neighbor 3 (local 0)
                [9, 9],  # center 5: no neighbors (all garbage)
            ],
            dtype=torch.int32,
        )
        num_neighbors = torch.tensor([2, 1, 1, 1, 1, 0], dtype=torch.int32)
        shifts = torch.zeros((6, 2, 3), dtype=torch.int32)
        shifts[0, 1] = torch.tensor([1, 0, -1], dtype=torch.int32)  # edge 0->2

        i, j, s, f = nv_matrix_to_ijs(neighbor_matrix, num_neighbors, shifts, nloc)

        assert i.dtype == j.dtype == s.dtype == f.dtype == torch.int64
        assert _edge_set(i, j, s, f) == {
            (0, 0, 1, (0, 0, 0)),
            (0, 0, 2, (1, 0, -1)),
            (0, 1, 0, (0, 0, 0)),
            (0, 2, 0, (0, 0, 0)),
            (1, 0, 1, (0, 0, 0)),  # frame 1: local indices via % nloc
            (1, 1, 0, (0, 0, 0)),
        }

    def test_empty_no_neighbors(self) -> None:
        """All-zero num_neighbors yields zero edges (no garbage leaks)."""
        neighbor_matrix = torch.full((4, 3), 7, dtype=torch.int32)
        num_neighbors = torch.zeros((4,), dtype=torch.int32)
        shifts = torch.zeros((4, 3, 3), dtype=torch.int32)
        i, j, s, f = nv_matrix_to_ijs(neighbor_matrix, num_neighbors, shifts, 2)
        assert i.shape == (0,) and j.shape == (0,)
        assert s.shape == (0, 3) and f.shape == (0,)

    def test_search_over_real_atoms_only(self) -> None:
        """A search restricted to the real atoms still reports batch indices.

        A mixed-nloc batch is padded to a rectangular width, and the search is
        given only the real slots, so its output is indexed on those. The decode
        must lift both endpoints back onto the batch, or the frame and local
        indices it derives would be meaningless.

        Batch: nf=2, nloc=3, real atoms at flat positions 0, 1, 3, 4, 5 -- that
        is, frame 0 holds two atoms and frame 1 holds three.
        """
        node_index = torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64)
        # Search indices 0..4 address those five atoms in order.
        neighbor_matrix = torch.tensor(
            [
                [1, 9],  # search 0 (batch 0, frame 0 local 0) -> search 1
                [0, 9],  # search 1 (batch 1, frame 0 local 1) -> search 0
                [3, 4],  # search 2 (batch 3, frame 1 local 0) -> search 3, 4
                [2, 9],  # search 3 (batch 4, frame 1 local 1) -> search 2
                [2, 9],  # search 4 (batch 5, frame 1 local 2) -> search 2
            ],
            dtype=torch.int32,
        )
        num_neighbors = torch.tensor([1, 1, 2, 1, 1], dtype=torch.int32)
        shifts = torch.zeros((5, 2, 3), dtype=torch.int32)

        i, j, s, f = nv_matrix_to_ijs(
            neighbor_matrix, num_neighbors, shifts, 3, node_index=node_index
        )
        assert _edge_set(i, j, s, f) == {
            (0, 0, 1, (0, 0, 0)),
            (0, 1, 0, (0, 0, 0)),
            (1, 0, 1, (0, 0, 0)),
            (1, 0, 2, (0, 0, 0)),
            (1, 1, 0, (0, 0, 0)),
            (1, 2, 0, (0, 0, 0)),
        }

    def test_identity_node_index_matches_the_unrestricted_decode(self) -> None:
        """Searching every slot is the case where the two indexings coincide."""
        rng = np.random.default_rng(3)
        nf, nloc, mn = 2, 4, 3
        total = nf * nloc
        num = torch.from_numpy(rng.integers(0, mn + 1, size=total)).to(torch.int32)
        mat = torch.zeros((total, mn), dtype=torch.int32)
        for dst in range(total):
            frame = dst // nloc
            mat[dst] = torch.from_numpy(
                rng.integers(frame * nloc, (frame + 1) * nloc, size=mn)
            ).to(torch.int32)
        shf = torch.from_numpy(rng.integers(-1, 2, size=(total, mn, 3))).to(torch.int32)

        plain = nv_matrix_to_ijs(mat, num, shf, nloc)
        lifted = nv_matrix_to_ijs(
            mat, num, shf, nloc, node_index=torch.arange(total, dtype=torch.int64)
        )
        for got, expected in zip(lifted, plain, strict=True):
            torch.testing.assert_close(got, expected)

    def test_random_vs_oracle(self) -> None:
        """Random matrices match a brute-force python oracle."""
        rng = np.random.default_rng(11)
        nf, nloc, mn = 3, 4, 5
        total = nf * nloc
        num = rng.integers(0, mn + 1, size=total)
        mat = np.zeros((total, mn), dtype=np.int64)
        shf = rng.integers(-2, 3, size=(total, mn, 3))
        oracle = set()
        for dst in range(total):
            frame = dst // nloc
            for slot in range(mn):
                # batch isolation: valid srcs share the center's frame
                src = int(rng.integers(frame * nloc, (frame + 1) * nloc))
                mat[dst, slot] = src
                if slot < num[dst]:
                    oracle.add(
                        (
                            frame,
                            dst % nloc,
                            src % nloc,
                            tuple(int(x) for x in shf[dst, slot]),
                        )
                    )
        i, j, s, f = nv_matrix_to_ijs(
            torch.from_numpy(mat).to(torch.int32),
            torch.from_numpy(num).to(torch.int32),
            torch.from_numpy(shf).to(torch.int32),
            nloc,
        )
        assert _edge_set(i, j, s, f) == oracle


def _graph_edges(graph) -> set:
    """Edges as (src, dst, rounded edge_vec), so two builders can be compared."""
    keep = graph.edge_mask
    return {
        (int(s), int(d), *(round(float(x), 8) for x in v))
        for s, d, v in zip(
            graph.edge_index[0][keep],
            graph.edge_index[1][keep],
            graph.edge_vec[keep],
            strict=True,
        )
    }


@pytest.mark.skipif(not is_nv_available(), reason="nvalchemi-toolkit-ops not installed")
class TestNvBuilderOnPaddedBatches:
    """The nv builder withholds phantom atoms from the search itself.

    Its cost grows with the square of the frame width, so a mixed-nloc batch
    hands it the real atoms alone and lifts the resulting indices back onto the
    padded batch. Both halves of that have to be exact.
    """

    @staticmethod
    def _batch(nlocs, boxlen=9.0, seed=0):
        rng = np.random.default_rng(seed)
        width = max(nlocs)
        coord = np.zeros((len(nlocs), width, 3))
        atype = np.full((len(nlocs), width), -1, dtype=np.int64)
        for frame, nloc in enumerate(nlocs):
            coord[frame, :nloc] = rng.uniform(0.0, boxlen, (nloc, 3))
            atype[frame, :nloc] = rng.integers(0, 2, nloc)
        return (
            torch.tensor(coord),
            torch.tensor(atype),
            torch.tensor(np.tile(np.eye(3)[None] * boxlen, (len(nlocs), 1, 1))),
        )

    @pytest.mark.parametrize(
        ("nlocs", "boxlen"),
        [((6, 6, 6), 9.0), ((4, 7, 3), 9.0), ((5, 9, 2), 6.0)],
        ids=["uniform", "mixed-nloc", "mixed-nloc-dense-box"],
    )
    def test_matches_the_dense_builder(self, nlocs, boxlen) -> None:
        from deepmd.pt_expt.utils.nv_graph_builder import (
            build_neighbor_graph_nv,
        )

        coord, atype, box = self._batch(nlocs, boxlen)
        dense = build_neighbor_graph(coord, atype, box, 4.0)
        nv = build_neighbor_graph_nv(coord, atype, box, 4.0)
        assert _graph_edges(nv) == _graph_edges(dense)
        np.testing.assert_array_equal(nv.n_node.numpy(), dense.n_node.numpy())

    def test_search_enumerates_the_real_atoms_only(self, monkeypatch) -> None:
        """Padding must not reach the search, only its output indexing."""
        from deepmd.pt_expt.utils import (
            nv_graph_builder,
        )

        nlocs = (4, 7, 3)
        coord, atype, box = self._batch(nlocs)
        searched: list[int] = []
        original = nv_graph_builder.nv_search_matrix

        def spy(*args, **kwargs):
            result = original(*args, **kwargs)
            searched.append(int(result[2].shape[0]))
            return result

        monkeypatch.setattr(nv_graph_builder, "nv_search_matrix", spy)
        nv_graph_builder.build_neighbor_graph_nv(coord, atype, box, 4.0)
        assert searched == [sum(nlocs)], (
            f"the search saw {searched} atoms; the batch holds {sum(nlocs)} real "
            f"atoms padded to {len(nlocs) * max(nlocs)} slots"
        )
