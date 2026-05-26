# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for probability-weighted stat merging in descriptor share_params."""

from typing import (
    ClassVar,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
    merge_env_stat,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.descriptor.dpa2 import (
    DescrptDPA2,
)
from deepmd.pt_expt.descriptor.dpa3 import (
    DescrptDPA3,
)
from deepmd.pt_expt.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pt_expt.descriptor.se_t import (
    DescrptSeT,
)
from deepmd.pt_expt.descriptor.se_t_tebd import (
    DescrptSeTTebd,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)

from ...seed import (
    GLOBAL_SEED,
)


def _make_stats(ntypes: int, last_dim: int, rng: np.random.Generator) -> dict:
    """Create synthetic StatItem stats for an env mat descriptor.

    The stats dict has keys "r_{i}" and optionally "a_{i}" for each type,
    matching the EnvMatStatSe convention.
    """
    stats = {}
    for ti in range(ntypes):
        # Use moderate values to avoid zero-division
        n = rng.uniform(100, 500)
        s = rng.uniform(-10, 10)
        sq = s**2 / n + rng.uniform(0.01, 1.0)  # ensure variance > 0
        stats[f"r_{ti}"] = StatItem(number=n, sum=s, squared_sum=sq * n)
        if last_dim == 4:
            n_a = rng.uniform(100, 500)
            s_a = rng.uniform(-10, 10)
            sq_a = s_a**2 / n_a + rng.uniform(0.01, 1.0)
            stats[f"a_{ti}"] = StatItem(number=n_a, sum=s_a, squared_sum=sq_a * n_a)
    return stats


def _compute_expected_buffers(descriptor, merged_stats, last_dim):
    """Compute expected mean/stddev from merged stats using EnvMatStatSe."""
    env_stat = EnvMatStatSe(descriptor)
    env_stat.stats = merged_stats
    mean, stddev = env_stat()
    return mean, stddev


def _merge_stats(base_stats, link_stats, model_prob):
    """Manually merge stats dicts."""
    merged = {}
    for kk in base_stats:
        merged[kk] = base_stats[kk] + link_stats[kk] * model_prob
    return merged


class TestStatMergeSeA:
    """Test stat merging for se_e2_a descriptor."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.ntypes = 2
        self.nnei = sum(self.sel)
        self.last_dim = 4

    def _make_descriptor(self, seed):
        return DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=seed).to(
            self.device
        )

    @pytest.mark.parametrize("model_prob", [0.6, 1.0, 0.1])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged davg/dstd match manually computed values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.stats = base_stats
        dd_link.stats = link_stats

        # Set initial davg/dstd on base
        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)

        # Compute expected
        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base, merged_stats, self.last_dim
        )

        # share_params with stat merging
        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        # Verify buffers match expected
        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.dstd.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

        # Verify stats updated for chaining
        for kk in merged_stats:
            assert abs(dd_base.stats[kk].number - merged_stats[kk].number) < 1e-10
            assert abs(dd_base.stats[kk].sum - merged_stats[kk].sum) < 1e-10

    def test_buffers_aliased(self) -> None:
        """After share_params, link buffers should be aliased to base."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.5, resume=False)

        for key in dd_base._buffers:
            assert dd_link._buffers[key] is dd_base._buffers[key], (
                f"Buffer {key} not aliased"
            )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging and preserve original buffers."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.stats = base_stats
        dd_link.stats = link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)

        original_davg = dd_base.davg.clone()
        original_dstd = dd_base.dstd.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        # Buffers should be unchanged
        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            dd_base.dstd.detach().cpu().numpy(),
            original_dstd.detach().cpu().numpy(),
        )

    def test_none_stats_skips_merge(self) -> None:
        """When stats is None, merging should be silently skipped."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        # stats is not set (default None)
        assert getattr(dd_base, "stats", None) is None

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)
        original_davg = dd_base.davg.clone()

        # Should not raise
        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=False)

        # davg should be unchanged (merge was skipped)
        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )


class TestStatMergeSeR:
    """Test stat merging for se_r descriptor."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 100)
        self.ntypes = 2
        self.nnei = sum(self.sel)
        self.last_dim = 1

    def _make_descriptor(self, seed):
        return DescrptSeR(self.rcut, self.rcut_smth, self.sel, seed=seed).to(
            self.device
        )

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged davg/dstd match manually computed values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.stats = base_stats
        dd_link.stats = link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.dstd.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.stats = base_stats
        dd_link.stats = link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)
        original_davg = dd_base.davg.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )


class TestStatMergeSeT:
    """Test stat merging for se_t descriptor."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 200)
        self.ntypes = 2
        self.nnei = sum(self.sel)
        self.last_dim = 4

    def _make_descriptor(self, seed):
        return DescrptSeT(self.rcut, self.rcut_smth, self.sel, seed=seed).to(
            self.device
        )

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged davg/dstd match manually computed values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.stats = base_stats
        dd_link.stats = link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.dstd.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.stats = base_stats
        dd_link.stats = link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dstd0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.davg = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd_base.dstd = torch.tensor(dstd0, dtype=torch.float64, device=self.device)
        original_davg = dd_base.davg.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )


class TestStatMergeDPA1:
    """Test stat merging for DPA1 descriptor (se_atten block has mean/stddev)."""

    rcut = 2.2
    rcut_smth = 0.4
    sel = 7

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 300)
        self.ntypes = 2
        self.nnei = self.sel
        self.last_dim = 4

    def _make_descriptor(self, seed):
        return DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.ntypes,
            seed=seed,
        ).to(self.device)

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged mean/stddev on se_atten block match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.se_atten.stats = base_stats
        dd_link.se_atten.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.se_atten.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.se_atten.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.se_atten, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.se_atten.mean.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.se_atten.stddev.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging on se_atten block."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.se_atten.stats = base_stats
        dd_link.se_atten.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.se_atten.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.se_atten.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.se_atten.mean.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.se_atten.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )

    def test_level1_no_merge(self) -> None:
        """Level 1 shares type_embedding only, no stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.se_atten.stats = base_stats
        dd_link.se_atten.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dd_base.se_atten.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.se_atten.mean.clone()

        dd_link.share_params(dd_base, shared_level=1, model_prob=0.6, resume=False)

        # type_embedding shared
        assert dd_link._modules["type_embedding"] is dd_base._modules["type_embedding"]
        # se_atten NOT shared
        assert dd_link._modules["se_atten"] is not dd_base._modules["se_atten"]
        # stats unchanged
        np.testing.assert_allclose(
            dd_base.se_atten.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )


class TestStatMergeDPA2:
    """Test stat merging for DPA2 descriptor (repinit and repformers blocks)."""

    rcut = 2.2
    rcut_smth = 0.4
    sel_mix: ClassVar = [7]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 350)
        self.ntypes = 2
        self.nnei = sum(self.sel_mix)
        self.last_dim = 4

    def _make_descriptor(self, seed):
        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode="strip",
            set_davg_zero=False,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=self.nnei // 2,
            nlayers=3,
            g1_dim=20,
            g2_dim=10,
            axis_neuron=4,
            update_g1_has_conv=True,
            update_g1_has_drrd=False,
            update_g1_has_grrg=False,
            update_g1_has_attn=False,
            update_g2_has_g1g1=False,
            update_g2_has_attn=True,
            update_h2=False,
            attn1_hidden=20,
            attn1_nhead=2,
            attn2_hidden=10,
            attn2_nhead=2,
            attn2_has_gate=True,
            update_style="res_residual",
            set_davg_zero=False,
        )
        dd = DescrptDPA2(
            self.ntypes,
            repinit=repinit,
            repformer=repformer,
            smooth=True,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            seed=seed,
        ).to(self.device)
        return dd

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge_repinit(self, model_prob) -> None:
        """Verify merged mean/stddev on repinit block match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.repinit.stats = base_stats
        dd_link.repinit.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.repinit.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.repinit.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.repinit, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.repinit.mean.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.repinit.stddev.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge_repformers(self, model_prob) -> None:
        """Verify merged mean/stddev on repformers block match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        nnei_repformers = self.nnei // 2
        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.repformers.stats = base_stats
        dd_link.repformers.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, nnei_repformers, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, nnei_repformers, self.last_dim))
        )
        dd_base.repformers.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.repformers.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.repformers, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.repformers.mean.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.repformers.stddev.detach().cpu().numpy(),
            expected_stddev,
            rtol=1e-10,
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging on all blocks."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats_ri = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats_ri = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repinit.stats = base_stats_ri
        dd_link.repinit.stats = link_stats_ri

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dd_base.repinit.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.repinit.mean.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.repinit.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )

    def test_level1_no_merge(self) -> None:
        """Level 1 shares type_embedding only, no stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repinit.stats = base_stats
        dd_link.repinit.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        dd_base.repinit.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.repinit.mean.clone()

        dd_link.share_params(dd_base, shared_level=1, model_prob=0.6, resume=False)

        # type_embedding shared
        assert dd_link._modules["type_embedding"] is dd_base._modules["type_embedding"]
        # repinit NOT shared
        assert dd_link._modules["repinit"] is not dd_base._modules["repinit"]
        # stats unchanged
        np.testing.assert_allclose(
            dd_base.repinit.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )


class TestStatMergeDPA3:
    """Test stat merging for DPA3 descriptor (repflows block has mean/stddev)."""

    rcut = 2.2
    rcut_smth = 0.4
    sel = 7

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 400)
        self.ntypes = 2
        self.nnei = self.sel
        self.last_dim = 4

    def _make_descriptor(self, seed, fix_stat_std=0.0):
        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=self.sel,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=self.sel - 1,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            smooth_edge_update=True,
            fix_stat_std=fix_stat_std,
        )
        dd = DescrptDPA3(
            self.ntypes,
            repflow=repflow,
            seed=seed,
        ).to(self.device)
        # Override set_davg_zero for testing (default True in repflows)
        if fix_stat_std == 0.0:
            dd.repflows.set_davg_zero = False
        return dd

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged mean/stddev on repflows block match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.repflows.stats = base_stats
        dd_link.repflows.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.repflows.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.repflows.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.repflows, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.repflows.mean.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.repflows.stddev.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    def test_default_config_skips_merge(self) -> None:
        """Default DPA3 has set_davg_zero=True and set_stddev_constant=True, so merge is no-op."""
        dd_base = self._make_descriptor(GLOBAL_SEED, fix_stat_std=0.3)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1, fix_stat_std=0.3)
        # Restore defaults
        dd_base.repflows.set_davg_zero = True
        dd_link.repflows.set_davg_zero = True

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repflows.stats = base_stats
        dd_link.repflows.stats = link_stats

        original_mean = dd_base.repflows.mean.clone()
        original_stddev = dd_base.repflows.stddev.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=False)

        # Buffers should be unchanged
        np.testing.assert_allclose(
            dd_base.repflows.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            dd_base.repflows.stddev.detach().cpu().numpy(),
            original_stddev.detach().cpu().numpy(),
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repflows.stats = base_stats
        dd_link.repflows.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.repflows.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.repflows.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.repflows.mean.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.repflows.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )


class TestStatMergeSeTTebd:
    """Test stat merging for se_t_tebd descriptor (se_ttebd block has mean/stddev)."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 500)
        self.ntypes = 2
        self.nnei = sum(self.sel)
        self.last_dim = 4

    def _make_descriptor(self, seed):
        dd = DescrptSeTTebd(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.ntypes,
            set_davg_zero=False,
            seed=seed,
        ).to(self.device)
        return dd

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge(self, model_prob) -> None:
        """Verify merged mean/stddev on se_ttebd block match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.se_ttebd.stats = base_stats
        dd_link.se_ttebd.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.se_ttebd.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.se_ttebd.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.se_ttebd, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.se_ttebd.mean.detach().cpu().numpy(), expected_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            dd_base.se_ttebd.stddev.detach().cpu().numpy(), expected_stddev, rtol=1e-10
        )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.se_ttebd.stats = base_stats
        dd_link.se_ttebd.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, self.nnei, self.last_dim))
        )
        dd_base.se_ttebd.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.se_ttebd.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.se_ttebd.mean.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.se_ttebd.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )


class TestStatMergeDPA2ThreeBody:
    """Test stat merging for DPA2 descriptor with use_three_body=True."""

    rcut = 2.2
    rcut_smth = 0.4
    sel_mix: ClassVar = [7]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 600)
        self.ntypes = 2
        self.nnei = sum(self.sel_mix)
        self.last_dim = 4
        self.three_body_sel = 5
        self.three_body_rcut = self.rcut
        self.three_body_rcut_smth = self.rcut_smth

    def _make_descriptor(self, seed):
        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode="strip",
            set_davg_zero=False,
            use_three_body=True,
            three_body_sel=self.three_body_sel,
            three_body_rcut=self.three_body_rcut,
            three_body_rcut_smth=self.three_body_rcut_smth,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=self.nnei // 2,
            nlayers=3,
            g1_dim=20,
            g2_dim=10,
            axis_neuron=4,
            update_g1_has_conv=True,
            update_g1_has_drrd=False,
            update_g1_has_grrg=False,
            update_g1_has_attn=False,
            update_g2_has_g1g1=False,
            update_g2_has_attn=True,
            update_h2=False,
            attn1_hidden=20,
            attn1_nhead=2,
            attn2_hidden=10,
            attn2_nhead=2,
            attn2_has_gate=True,
            update_style="res_residual",
            set_davg_zero=False,
        )
        dd = DescrptDPA2(
            self.ntypes,
            repinit=repinit,
            repformer=repformer,
            smooth=True,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            seed=seed,
        ).to(self.device)
        return dd

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge_three_body(self, model_prob) -> None:
        """Verify merged mean/stddev on repinit_three_body block."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        assert dd_base.use_three_body
        assert dd_base.repinit_three_body is not None

        # repinit_three_body is a DescrptBlockSeTTebd with mean/stddev
        nnei_3b = self.three_body_sel
        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.repinit_three_body.stats = base_stats
        dd_link.repinit_three_body.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, nnei_3b, self.last_dim))
        stddev0 = 0.1 + np.abs(
            self.rng.normal(size=(self.ntypes, nnei_3b, self.last_dim))
        )
        dd_base.repinit_three_body.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        dd_base.repinit_three_body.stddev = torch.tensor(
            stddev0, dtype=torch.float64, device=self.device
        )

        merged_stats = _merge_stats(base_stats, link_stats, model_prob)
        expected_mean, expected_stddev = _compute_expected_buffers(
            dd_base.repinit_three_body, merged_stats, self.last_dim
        )

        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        np.testing.assert_allclose(
            dd_base.repinit_three_body.mean.detach().cpu().numpy(),
            expected_mean,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            dd_base.repinit_three_body.stddev.detach().cpu().numpy(),
            expected_stddev,
            rtol=1e-10,
        )

    def test_three_body_aliased(self) -> None:
        """After share_params, repinit_three_body modules should be aliased."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repinit_three_body.stats = base_stats
        dd_link.repinit_three_body.stats = link_stats

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=False)

        assert (
            dd_link._modules["repinit_three_body"]
            is dd_base._modules["repinit_three_body"]
        )

    def test_resume_skips_three_body_merge(self) -> None:
        """resume=True should skip stat merging on three-body block."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        nnei_3b = self.three_body_sel
        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.repinit_three_body.stats = base_stats
        dd_link.repinit_three_body.stats = link_stats

        mean0 = self.rng.normal(size=(self.ntypes, nnei_3b, self.last_dim))
        dd_base.repinit_three_body.mean = torch.tensor(
            mean0, dtype=torch.float64, device=self.device
        )
        original_mean = dd_base.repinit_three_body.mean.clone()

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.repinit_three_body.mean.detach().cpu().numpy(),
            original_mean.detach().cpu().numpy(),
        )


class TestStatMergeHybrid:
    """Test stat merging for hybrid descriptor (passes model_prob to sub-descriptors)."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 700)
        self.ntypes = 2
        self.nnei = sum(self.sel)

    def _make_descriptor(self, seed):
        """Create a hybrid descriptor with se_e2_a (last_dim=4) + se_r (last_dim=1)."""
        dd = DescrptHybrid(
            list=[
                DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=seed),
                DescrptSeR(self.rcut, self.rcut_smth, self.sel, seed=seed + 10),
            ],
        ).to(self.device)
        return dd

    @pytest.mark.parametrize("model_prob", [0.6, 1.0])  # probability weight
    def test_stat_merge_sub_descriptors(self, model_prob) -> None:
        """Verify merged davg/dstd on each sub-descriptor match expected values."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        # SeA sub-descriptor (last_dim=4)
        sea_base_stats = _make_stats(self.ntypes, 4, self.rng)
        sea_link_stats = _make_stats(self.ntypes, 4, self.rng)
        dd_base.descrpt_list[0].stats = sea_base_stats
        dd_link.descrpt_list[0].stats = sea_link_stats

        davg0_sea = self.rng.normal(size=(self.ntypes, self.nnei, 4))
        dstd0_sea = 0.1 + np.abs(self.rng.normal(size=(self.ntypes, self.nnei, 4)))
        dd_base.descrpt_list[0].davg = torch.tensor(
            davg0_sea, dtype=torch.float64, device=self.device
        )
        dd_base.descrpt_list[0].dstd = torch.tensor(
            dstd0_sea, dtype=torch.float64, device=self.device
        )

        # SeR sub-descriptor (last_dim=1)
        ser_base_stats = _make_stats(self.ntypes, 1, self.rng)
        ser_link_stats = _make_stats(self.ntypes, 1, self.rng)
        dd_base.descrpt_list[1].stats = ser_base_stats
        dd_link.descrpt_list[1].stats = ser_link_stats

        davg0_ser = self.rng.normal(size=(self.ntypes, self.nnei, 1))
        dstd0_ser = 0.1 + np.abs(self.rng.normal(size=(self.ntypes, self.nnei, 1)))
        dd_base.descrpt_list[1].davg = torch.tensor(
            davg0_ser, dtype=torch.float64, device=self.device
        )
        dd_base.descrpt_list[1].dstd = torch.tensor(
            dstd0_ser, dtype=torch.float64, device=self.device
        )

        # Compute expected for SeA
        merged_sea = _merge_stats(sea_base_stats, sea_link_stats, model_prob)
        exp_mean_sea, exp_std_sea = _compute_expected_buffers(
            dd_base.descrpt_list[0], merged_sea, 4
        )

        # Compute expected for SeR
        merged_ser = _merge_stats(ser_base_stats, ser_link_stats, model_prob)
        exp_mean_ser, exp_std_ser = _compute_expected_buffers(
            dd_base.descrpt_list[1], merged_ser, 1
        )

        # share_params on hybrid passes model_prob to each sub-descriptor
        dd_link.share_params(
            dd_base, shared_level=0, model_prob=model_prob, resume=False
        )

        # Verify SeA sub-descriptor buffers
        np.testing.assert_allclose(
            dd_base.descrpt_list[0].davg.detach().cpu().numpy(),
            exp_mean_sea,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            dd_base.descrpt_list[0].dstd.detach().cpu().numpy(),
            exp_std_sea,
            rtol=1e-10,
        )

        # Verify SeR sub-descriptor buffers
        np.testing.assert_allclose(
            dd_base.descrpt_list[1].davg.detach().cpu().numpy(),
            exp_mean_ser,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            dd_base.descrpt_list[1].dstd.detach().cpu().numpy(),
            exp_std_ser,
            rtol=1e-10,
        )

    def test_sub_descriptors_aliased(self) -> None:
        """After share_params, sub-descriptor modules should be aliased."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        # Populate stats to avoid None-guard early return
        for i in range(2):
            last_dim = 4 if i == 0 else 1
            dd_base.descrpt_list[i].stats = _make_stats(self.ntypes, last_dim, self.rng)
            dd_link.descrpt_list[i].stats = _make_stats(self.ntypes, last_dim, self.rng)

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=False)

        for i in range(2):
            assert (
                dd_link.descrpt_list[i].davg.data_ptr()
                == dd_base.descrpt_list[i].davg.data_ptr()
            )

    def test_resume_skips_merge(self) -> None:
        """resume=True should skip stat merging on all sub-descriptors."""
        dd_base = self._make_descriptor(GLOBAL_SEED)
        dd_link = self._make_descriptor(GLOBAL_SEED + 1)

        sea_base_stats = _make_stats(self.ntypes, 4, self.rng)
        sea_link_stats = _make_stats(self.ntypes, 4, self.rng)
        dd_base.descrpt_list[0].stats = sea_base_stats
        dd_link.descrpt_list[0].stats = sea_link_stats

        davg0 = self.rng.normal(size=(self.ntypes, self.nnei, 4))
        dd_base.descrpt_list[0].davg = torch.tensor(
            davg0, dtype=torch.float64, device=self.device
        )
        original_davg = dd_base.descrpt_list[0].davg.clone()

        # Need stats on all sub-descriptors to avoid None guard
        dd_base.descrpt_list[1].stats = _make_stats(self.ntypes, 1, self.rng)
        dd_link.descrpt_list[1].stats = _make_stats(self.ntypes, 1, self.rng)

        dd_link.share_params(dd_base, shared_level=0, model_prob=0.6, resume=True)

        np.testing.assert_allclose(
            dd_base.descrpt_list[0].davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )


class TestMergeEnvStatUnit:
    """Unit tests for the merge_env_stat function directly."""

    rcut = 2.2
    rcut_smth = 0.4
    sel: ClassVar = [5, 2]

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.rng = np.random.default_rng(GLOBAL_SEED + 600)
        self.ntypes = 2
        self.nnei = sum(self.sel)
        self.last_dim = 4

    def test_merge_produces_correct_stats(self) -> None:
        """merge_env_stat should compute merged = base + link * model_prob."""
        dd_base = DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED).to(
            self.device
        )

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.stats = base_stats

        dd_link = DescrptSeA(
            self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED + 1
        ).to(self.device)
        dd_link.stats = link_stats

        model_prob = 0.3
        merge_env_stat(dd_base, dd_link, model_prob)

        for kk in base_stats:
            expected = base_stats[kk] + link_stats[kk] * model_prob
            assert abs(dd_base.stats[kk].number - expected.number) < 1e-10
            assert abs(dd_base.stats[kk].sum - expected.sum) < 1e-10
            assert abs(dd_base.stats[kk].squared_sum - expected.squared_sum) < 1e-10

    def test_chaining_three_models(self) -> None:
        """Merging stats from 3 models should accumulate correctly."""
        dd_base = DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED).to(
            self.device
        )
        dd_link1 = DescrptSeA(
            self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED + 1
        ).to(self.device)
        dd_link2 = DescrptSeA(
            self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED + 2
        ).to(self.device)

        stats_base = _make_stats(self.ntypes, self.last_dim, self.rng)
        stats_link1 = _make_stats(self.ntypes, self.last_dim, self.rng)
        stats_link2 = _make_stats(self.ntypes, self.last_dim, self.rng)

        dd_base.stats = stats_base
        dd_link1.stats = stats_link1
        dd_link2.stats = stats_link2

        prob1, prob2 = 0.5, 0.3

        merge_env_stat(dd_base, dd_link1, prob1)
        merge_env_stat(dd_base, dd_link2, prob2)

        for kk in stats_base:
            expected = (
                stats_base[kk] + stats_link1[kk] * prob1 + stats_link2[kk] * prob2
            )
            assert abs(dd_base.stats[kk].number - expected.number) < 1e-10
            assert abs(dd_base.stats[kk].sum - expected.sum) < 1e-10

    def test_set_davg_zero_respected(self) -> None:
        """When set_davg_zero=True, davg should remain zero after merging."""
        dd_base = DescrptSeA(
            self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED, set_davg_zero=True
        ).to(self.device)
        dd_link = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            seed=GLOBAL_SEED + 1,
            set_davg_zero=True,
        ).to(self.device)

        base_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        link_stats = _make_stats(self.ntypes, self.last_dim, self.rng)
        dd_base.stats = base_stats
        dd_link.stats = link_stats

        original_davg = dd_base.davg.clone()
        merge_env_stat(dd_base, dd_link, 0.6)

        # davg should stay zero
        np.testing.assert_allclose(
            dd_base.davg.detach().cpu().numpy(),
            original_davg.detach().cpu().numpy(),
        )
        # but dstd should be updated
        assert dd_base.stats is not base_stats  # stats dict replaced
