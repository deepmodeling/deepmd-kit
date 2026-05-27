# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-backend consistency tests for compute_output_stats."""

from collections import (
    defaultdict,
)

import numpy as np
import pytest

from deepmd.dpmodel.utils.stat import (
    _compute_output_stats_atomic as compute_output_stats_atomic_dp,
)
from deepmd.dpmodel.utils.stat import (
    _compute_output_stats_global as compute_output_stats_global_dp,
)
from deepmd.dpmodel.utils.stat import compute_output_stats as compute_output_stats_dp

from ..common import (
    INSTALLED_PD,
    INSTALLED_PT,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.utils.stat import (
        _compute_output_stats_atomic as compute_output_stats_atomic_pt,
    )
    from deepmd.pt.utils.stat import (
        _compute_output_stats_global as compute_output_stats_global_pt,
    )
    from deepmd.pt.utils.stat import compute_output_stats as compute_output_stats_pt
    from deepmd.pt.utils.utils import to_numpy_array as to_numpy_array_pt
if INSTALLED_PD:
    import paddle

    from deepmd.pd.utils.stat import (
        _compute_output_stats_atomic as compute_output_stats_atomic_pd,
    )
    from deepmd.pd.utils.stat import (
        _compute_output_stats_global as compute_output_stats_global_pd,
    )
    from deepmd.pd.utils.stat import compute_output_stats as compute_output_stats_pd
    from deepmd.pd.utils.utils import to_numpy_array as to_numpy_array_pd

NTYPES = 2
NFRAMES = 2
NLOC = 4


def _make_data(
    has_global: bool,
    has_atomic: bool,
    mixed_type: bool,
    exclude_types: list[int],
) -> tuple[list[dict], dict, dict]:
    """Build stat data with numpy arrays and precomputed indices.

    Returns
    -------
    sampled : list[dict]
        Data with numpy arrays.
    global_sampled_idx : dict
        Precomputed indices for global labels.
    atomic_sampled_idx : dict
        Precomputed indices for atomic labels.
    """
    rng = np.random.default_rng(42)

    # atype: [nframes, nloc]
    atype = np.array([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=np.int64)

    # natoms: [nframes, 2+ntypes] = [nloc_total, nloc_real, count_type0, count_type1]
    natoms = np.array([[4, 4, 2, 2], [4, 4, 2, 2]], dtype=np.int64)

    if mixed_type:
        # For mixed type, atype may have different counts per frame,
        # but natoms is padded uniformly. real_natoms_vec has actual counts.
        atype = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int64)
        real_natoms_vec = np.array([[4, 4, 2, 2], [4, 4, 1, 3]], dtype=np.int64)

    # Atomic labels: [nframes, nloc, 1]
    atom_energy = rng.normal(size=(NFRAMES, NLOC, 1))
    # Global labels: sum of atom_energy per frame -> [nframes, 1]
    energy = atom_energy.sum(axis=1)

    keys = ["energy"]

    # Build a single system dict (both frames in one system)
    system_np: dict = {
        "atype": atype,
        "natoms": natoms.copy(),
    }
    if mixed_type:
        system_np["real_natoms_vec"] = real_natoms_vec.copy()

    if has_global:
        system_np["energy"] = energy
        system_np["find_energy"] = np.float32(1.0)
    if has_atomic:
        system_np["atom_energy"] = atom_energy
        system_np["find_atom_energy"] = np.float32(1.0)
    if exclude_types:
        system_np["atom_exclude_types"] = exclude_types

    sampled = [system_np]

    # Precompute indices (same logic used by all backends' compute_output_stats)
    atomic_sampled_idx: dict = defaultdict(list)
    global_sampled_idx: dict = defaultdict(list)
    for kk in keys:
        for idx, s in enumerate(sampled):
            if ("find_atom_" + kk) in s and s["find_atom_" + kk] > 0.0:
                atomic_sampled_idx[kk].append(idx)
            if ("find_" + kk) in s and s["find_" + kk] > 0.0:
                global_sampled_idx[kk].append(idx)

    return sampled, global_sampled_idx, atomic_sampled_idx


def _np_to_torch(sampled: list[dict]) -> list[dict]:
    """Convert numpy sampled data to torch tensors."""
    result = []
    for d in sampled:
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v.copy())
            elif isinstance(v, np.float32):
                out[k] = v
            else:
                out[k] = v
        result.append(out)
    return result


def _np_to_paddle(sampled: list[dict]) -> list[dict]:
    """Convert numpy sampled data to paddle tensors."""
    result = []
    for d in sampled:
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = paddle.to_tensor(v.copy())
            elif isinstance(v, np.float32):
                out[k] = v
            else:
                out[k] = v
        result.append(out)
    return result


@pytest.mark.skipif(not INSTALLED_PT, reason="PyTorch is not installed")
class TestComputeOutputStatConsistencyPT:
    """Cross-backend consistency tests for compute_output_stats_global/atomic: dp vs pt."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_global(self, mixed_type, exclude_types) -> None:
        """compute_output_stats_global dp vs pt."""
        sampled, global_idx, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_global_dp(
            sampled, NTYPES, keys, global_sampled_idx=global_idx
        )
        pt_bias, pt_std = compute_output_stats_global_pt(
            sampled_pt, NTYPES, keys, global_sampled_idx=global_idx
        )

        for kk in keys:
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pt_bias[kk], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pt_std[kk], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_atomic(self, mixed_type, exclude_types) -> None:
        """compute_output_stats_atomic dp vs pt."""
        sampled, _, atomic_idx = _make_data(
            has_global=False,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_atomic_dp(
            sampled, NTYPES, keys, atomic_sampled_idx=atomic_idx
        )
        pt_bias, pt_std = compute_output_stats_atomic_pt(
            sampled_pt, NTYPES, keys, atomic_sampled_idx=atomic_idx
        )

        for kk in keys:
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pt_bias[kk], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pt_std[kk], rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not INSTALLED_PT, reason="PyTorch is not installed")
class TestComputeOutputStatFullConsistencyPT:
    """Cross-backend consistency tests for the top-level compute_output_stats: dp vs pt."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_global_only(self, mixed_type, exclude_types) -> None:
        """Global labels only through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pt_bias, pt_std = compute_output_stats_pt(sampled_pt, NTYPES, keys)

        for kk in keys:
            pt_bias_np = to_numpy_array_pt(pt_bias[kk])
            pt_std_np = to_numpy_array_pt(pt_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pt_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pt_std_np, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_atomic_only(self, mixed_type, exclude_types) -> None:
        """Atomic labels only through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=False,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pt_bias, pt_std = compute_output_stats_pt(sampled_pt, NTYPES, keys)

        for kk in keys:
            pt_bias_np = to_numpy_array_pt(pt_bias[kk])
            pt_std_np = to_numpy_array_pt(pt_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pt_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pt_std_np, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_both_global_and_atomic(self, mixed_type, exclude_types) -> None:
        """Both global and atomic labels through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pt_bias, pt_std = compute_output_stats_pt(sampled_pt, NTYPES, keys)

        for kk in keys:
            pt_bias_np = to_numpy_array_pt(pt_bias[kk])
            pt_std_np = to_numpy_array_pt(pt_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pt_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pt_std_np, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not INSTALLED_PD, reason="Paddle is not installed")
class TestComputeOutputStatConsistencyPD:
    """Cross-backend consistency tests for compute_output_stats_global/atomic: dp vs pd."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_global(self, mixed_type, exclude_types) -> None:
        """compute_output_stats_global dp vs pd."""
        sampled, global_idx, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_global_dp(
            sampled, NTYPES, keys, global_sampled_idx=global_idx
        )
        pd_bias, pd_std = compute_output_stats_global_pd(
            sampled_pd, NTYPES, keys, global_sampled_idx=global_idx
        )

        for kk in keys:
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pd_bias[kk], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pd_std[kk], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_atomic(self, mixed_type, exclude_types) -> None:
        """compute_output_stats_atomic dp vs pd."""
        sampled, _, atomic_idx = _make_data(
            has_global=False,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_atomic_dp(
            sampled, NTYPES, keys, atomic_sampled_idx=atomic_idx
        )
        pd_bias, pd_std = compute_output_stats_atomic_pd(
            sampled_pd, NTYPES, keys, atomic_sampled_idx=atomic_idx
        )

        for kk in keys:
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pd_bias[kk], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pd_std[kk], rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not INSTALLED_PD, reason="Paddle is not installed")
class TestComputeOutputStatFullConsistencyPD:
    """Cross-backend consistency tests for the top-level compute_output_stats: dp vs pd."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_global_only(self, mixed_type, exclude_types) -> None:
        """Global labels only through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pd_bias, pd_std = compute_output_stats_pd(sampled_pd, NTYPES, keys)

        for kk in keys:
            pd_bias_np = to_numpy_array_pd(pd_bias[kk])
            pd_std_np = to_numpy_array_pd(pd_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pd_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pd_std_np, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_atomic_only(self, mixed_type, exclude_types) -> None:
        """Atomic labels only through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=False,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pd_bias, pd_std = compute_output_stats_pd(sampled_pd, NTYPES, keys)

        for kk in keys:
            pd_bias_np = to_numpy_array_pd(pd_bias[kk])
            pd_std_np = to_numpy_array_pd(pd_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pd_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pd_std_np, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    @pytest.mark.parametrize("exclude_types", [[], [1]])  # atom_exclude_types
    def test_both_global_and_atomic(self, mixed_type, exclude_types) -> None:
        """Both global and atomic labels through full compute_output_stats."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=exclude_types,
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]

        dp_bias, dp_std = compute_output_stats_dp(sampled, NTYPES, keys)
        pd_bias, pd_std = compute_output_stats_pd(sampled_pd, NTYPES, keys)

        for kk in keys:
            pd_bias_np = to_numpy_array_pd(pd_bias[kk])
            pd_std_np = to_numpy_array_pd(pd_std[kk])
            assert dp_bias[kk].shape[0] == NTYPES
            np.testing.assert_allclose(dp_bias[kk], pd_bias_np, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(dp_std[kk], pd_std_np, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not INSTALLED_PT, reason="PyTorch is not installed")
class TestComputeOutputStatNoMutationPT:
    """Verify that stat functions do not mutate input sampled data (pt)."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    def test_global_no_mutation(self, mixed_type) -> None:
        """compute_output_stats_global must not mutate input with exclude_types."""
        sampled, global_idx, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=[1],
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]
        natoms_key = "real_natoms_vec" if mixed_type else "natoms"

        dp_natoms_before = sampled[0][natoms_key].copy()
        pt_natoms_before = sampled_pt[0][natoms_key].clone()

        compute_output_stats_global_dp(
            sampled, NTYPES, keys, global_sampled_idx=global_idx
        )
        compute_output_stats_global_pt(
            sampled_pt, NTYPES, keys, global_sampled_idx=global_idx
        )

        np.testing.assert_array_equal(sampled[0][natoms_key], dp_natoms_before)
        np.testing.assert_array_equal(
            sampled_pt[0][natoms_key].numpy(), pt_natoms_before.numpy()
        )

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    def test_full_no_mutation(self, mixed_type) -> None:
        """compute_output_stats must not mutate input with exclude_types."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=[1],
        )
        sampled_pt = _np_to_torch(sampled)
        keys = ["energy"]
        natoms_key = "real_natoms_vec" if mixed_type else "natoms"

        dp_natoms_before = sampled[0][natoms_key].copy()
        pt_natoms_before = sampled_pt[0][natoms_key].clone()

        compute_output_stats_dp(sampled, NTYPES, keys)
        compute_output_stats_pt(sampled_pt, NTYPES, keys)

        np.testing.assert_array_equal(sampled[0][natoms_key], dp_natoms_before)
        np.testing.assert_array_equal(
            sampled_pt[0][natoms_key].numpy(), pt_natoms_before.numpy()
        )


@pytest.mark.skipif(not INSTALLED_PD, reason="Paddle is not installed")
class TestComputeOutputStatNoMutationPD:
    """Verify that stat functions do not mutate input sampled data (pd)."""

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    def test_global_no_mutation(self, mixed_type) -> None:
        """compute_output_stats_global must not mutate input with exclude_types."""
        sampled, global_idx, _ = _make_data(
            has_global=True,
            has_atomic=False,
            mixed_type=mixed_type,
            exclude_types=[1],
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]
        natoms_key = "real_natoms_vec" if mixed_type else "natoms"

        dp_natoms_before = sampled[0][natoms_key].copy()
        pd_natoms_before = sampled_pd[0][natoms_key].numpy().copy()

        compute_output_stats_global_dp(
            sampled, NTYPES, keys, global_sampled_idx=global_idx
        )
        compute_output_stats_global_pd(
            sampled_pd, NTYPES, keys, global_sampled_idx=global_idx
        )

        np.testing.assert_array_equal(sampled[0][natoms_key], dp_natoms_before)
        np.testing.assert_array_equal(
            sampled_pd[0][natoms_key].numpy(), pd_natoms_before
        )

    @pytest.mark.parametrize("mixed_type", [False, True])  # mixed_type
    def test_full_no_mutation(self, mixed_type) -> None:
        """compute_output_stats must not mutate input with exclude_types."""
        sampled, _, _ = _make_data(
            has_global=True,
            has_atomic=True,
            mixed_type=mixed_type,
            exclude_types=[1],
        )
        sampled_pd = _np_to_paddle(sampled)
        keys = ["energy"]
        natoms_key = "real_natoms_vec" if mixed_type else "natoms"

        dp_natoms_before = sampled[0][natoms_key].copy()
        pd_natoms_before = sampled_pd[0][natoms_key].numpy().copy()

        compute_output_stats_dp(sampled, NTYPES, keys)
        compute_output_stats_pd(sampled_pd, NTYPES, keys)

        np.testing.assert_array_equal(sampled[0][natoms_key], dp_natoms_before)
        np.testing.assert_array_equal(
            sampled_pd[0][natoms_key].numpy(), pd_natoms_before
        )
