# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU unit tests for the nv dense-matrix -> (i, j, S) decode.

The GPU ``neighbor_list`` search in ``build_neighbor_graph_nv`` is CUDA-only
and stays behind the opt-in CUDA suite (test_nv_graph_builder.py); the decode
(``nv_matrix_to_ijs``) is pure torch index arithmetic, so its regression-prone
parts (``// max_neighbors``, ``% nloc``, frame isolation, slot-validity mask)
are pinned here on the default CI with synthetic inputs.
"""

import numpy as np
import torch

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
