# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests: dpmodel ``dpa4_nn`` modules vs the reference pt ``sezm_nn`` modules.

This file is extended task-by-task as the DPA4 port progresses. Index-table and
numeric-helper parity is exact (``assert_array_equal`` / tight rtol). Module-level
weight-copy parity tests (build the pt module in float64, copy its ``state_dict``
into the dpmodel module via ``pt_state_to_numpy``, compare forwards with
``assert_parity``) are added by the tasks that port each module.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4_nn import (
    indexing as dp_indexing,
)
from deepmd.dpmodel.descriptor.dpa4_nn import (
    utils as dp_utils,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    indexing as pt_indexing,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    utils as pt_utils,
)


def pt_state_to_numpy(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def assert_parity(a, t, rtol=1e-12, atol=1e-14):
    np.testing.assert_allclose(
        np.asarray(a), t.detach().cpu().numpy(), rtol=rtol, atol=atol
    )


CPU = torch.device("cpu")


class TestIndexingParity:
    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    def test_get_so3_dim_of_lmax(self, lmax) -> None:
        assert dp_indexing.get_so3_dim_of_lmax(lmax) == pt_indexing.get_so3_dim_of_lmax(
            lmax
        )

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    def test_map_degree_idx(self, lmax) -> None:
        res = dp_indexing.map_degree_idx(lmax)
        ref = pt_indexing.map_degree_idx(lmax, device=CPU)
        assert res.dtype == np.int64
        np.testing.assert_array_equal(res, ref.numpy())

    @pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4])  # incl. lmax=0 empty branch
    def test_build_gie_zonal_index(self, lmax) -> None:
        res = dp_indexing.build_gie_zonal_index(lmax)
        ref = pt_indexing.build_gie_zonal_index(lmax, device=CPU)
        assert len(res) == len(ref) == 3
        for r, t in zip(res, ref, strict=True):
            assert r.dtype == np.int64
            np.testing.assert_array_equal(r, t.numpy())

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    def test_so3_packed_index(self, lmax) -> None:
        for degree in range(lmax + 1):
            for m in range(-degree, degree + 1):
                assert dp_indexing.so3_packed_index(
                    degree, m
                ) == pt_indexing.so3_packed_index(degree, m)

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_build_l_major_index(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        res = dp_indexing.build_l_major_index(lmax, mmax)
        ref = pt_indexing.build_l_major_index(lmax, mmax, device=CPU)
        assert res.dtype == np.int64
        np.testing.assert_array_equal(res, ref.numpy())

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_build_m_major_index(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        res = dp_indexing.build_m_major_index(lmax, mmax)
        ref = pt_indexing.build_m_major_index(lmax, mmax, device=CPU)
        assert res.dtype == np.int64
        np.testing.assert_array_equal(res, ref.numpy())

    def test_m_major_index_literal(self) -> None:
        # layout contract anchor, cross-checked with sezm_nn docs:
        # lmax=2, mmax=1: m=0 block (l=0..2), then m=-1, then m=+1
        np.testing.assert_array_equal(
            dp_indexing.build_m_major_index(2, 1), [0, 2, 6, 1, 5, 3, 7]
        )

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_build_m_major_l_index(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        res = dp_indexing.build_m_major_l_index(lmax, mmax)
        ref = pt_indexing.build_m_major_l_index(lmax, mmax, device=CPU)
        assert res.dtype == np.int64
        np.testing.assert_array_equal(res, ref.numpy())

    @pytest.mark.parametrize(
        "builder",
        ["build_l_major_index", "build_m_major_index", "build_m_major_l_index"],
    )  # index builder under test
    @pytest.mark.parametrize(
        "lmax,mmax", [(-1, 0), (1, -1), (1, 2)]
    )  # lmax<0, mmax<0, mmax>lmax error branches
    def test_index_builder_errors(self, builder, lmax, mmax) -> None:
        with pytest.raises(ValueError):
            getattr(dp_indexing, builder)(lmax, mmax)
        with pytest.raises(ValueError):
            getattr(pt_indexing, builder)(lmax, mmax, device=CPU)

    @pytest.mark.parametrize(
        "lmax", [1, 2, 3, 4]
    )  # max degree; lmax==mmax hits the all-ones branch
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_build_rotate_inv_rescale(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        degree_index_np = dp_indexing.build_m_major_l_index(lmax, mmax)
        degree_index_pt = pt_indexing.build_m_major_l_index(lmax, mmax, device=CPU)
        res = dp_indexing.build_rotate_inv_rescale(
            lmax, mmax, degree_index_np, dtype=np.float64
        )
        ref = pt_indexing.build_rotate_inv_rescale(
            lmax, mmax, degree_index_pt, device=CPU, dtype=torch.float64
        )
        assert res.dtype == np.float64
        np.testing.assert_allclose(res, ref.numpy(), rtol=1e-15, atol=0.0)

    @pytest.mark.parametrize(
        "lmax,mmax", [(-1, 0), (1, -1), (1, 2)]
    )  # lmax<0, mmax<0, mmax>lmax error branches
    def test_build_rotate_inv_rescale_errors(self, lmax, mmax) -> None:
        degree_index = np.zeros(1, dtype=np.int64)
        with pytest.raises(ValueError):
            dp_indexing.build_rotate_inv_rescale(lmax, mmax, degree_index)

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_project_D_to_m(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        rng = np.random.default_rng(2026)
        nfull = dp_indexing.get_so3_dim_of_lmax(4)
        d_full_np = rng.normal(size=(5, nfull, nfull))
        d_full_pt = torch.from_numpy(d_full_np)
        idx_np = dp_indexing.build_m_major_index(lmax, mmax)
        idx_pt = pt_indexing.build_m_major_index(lmax, mmax, device=CPU)
        ebed = dp_indexing.get_so3_dim_of_lmax(lmax)
        # cache=None branch
        res = dp_indexing.project_D_to_m(d_full_np, idx_np, ebed, None, lmax, mmax)
        ref = pt_indexing.project_D_to_m(d_full_pt, idx_pt, ebed, None, lmax, mmax)
        assert res.shape == (5, idx_np.shape[0], ebed)
        np.testing.assert_array_equal(np.asarray(res), ref.numpy())
        # cache branch: miss then hit (returned object identical)
        cache: dict = {}
        first = dp_indexing.project_D_to_m(d_full_np, idx_np, ebed, cache, lmax, mmax)
        second = dp_indexing.project_D_to_m(d_full_np, idx_np, ebed, cache, lmax, mmax)
        assert second is first
        np.testing.assert_array_equal(np.asarray(first), ref.numpy())

    @pytest.mark.parametrize("lmax", [1, 2, 3, 4])  # max spherical harmonic degree
    @pytest.mark.parametrize("mmax", [1, 2])  # max order |m|
    def test_project_Dt_from_m(self, lmax, mmax) -> None:
        if mmax > lmax:
            pytest.skip("mmax must be <= lmax")
        rng = np.random.default_rng(2027)
        nfull = dp_indexing.get_so3_dim_of_lmax(4)
        dt_full_np = rng.normal(size=(5, nfull, nfull))
        dt_full_pt = torch.from_numpy(dt_full_np)
        idx_np = dp_indexing.build_m_major_index(lmax, mmax)
        idx_pt = pt_indexing.build_m_major_index(lmax, mmax, device=CPU)
        ebed = dp_indexing.get_so3_dim_of_lmax(lmax)
        # cache=None branch
        res = dp_indexing.project_Dt_from_m(dt_full_np, idx_np, ebed, None, lmax, mmax)
        ref = pt_indexing.project_Dt_from_m(dt_full_pt, idx_pt, ebed, None, lmax, mmax)
        assert res.shape == (5, ebed, idx_np.shape[0])
        np.testing.assert_array_equal(np.asarray(res), ref.numpy())
        # cache branch: miss then hit (returned object identical)
        cache: dict = {}
        first = dp_indexing.project_Dt_from_m(
            dt_full_np, idx_np, ebed, cache, lmax, mmax
        )
        second = dp_indexing.project_Dt_from_m(
            dt_full_np, idx_np, ebed, cache, lmax, mmax
        )
        assert second is first
        np.testing.assert_array_equal(np.asarray(first), ref.numpy())

    def test_project_works_on_torch_tensors(self) -> None:
        # dpmodel project_* are array-API: must accept torch tensors at runtime
        lmax, mmax = 2, 1
        rng = np.random.default_rng(2028)
        nfull = dp_indexing.get_so3_dim_of_lmax(4)
        d_full_np = rng.normal(size=(5, nfull, nfull))
        d_full_pt = torch.from_numpy(d_full_np)
        idx_np = dp_indexing.build_m_major_index(lmax, mmax)
        ebed = dp_indexing.get_so3_dim_of_lmax(lmax)
        res = dp_indexing.project_D_to_m(d_full_pt, idx_np, ebed, None, lmax, mmax)
        assert isinstance(res, torch.Tensor)
        ref = dp_indexing.project_D_to_m(d_full_np, idx_np, ebed, None, lmax, mmax)
        np.testing.assert_array_equal(res.numpy(), np.asarray(ref))
        rest = dp_indexing.project_Dt_from_m(d_full_pt, idx_np, ebed, None, lmax, mmax)
        assert isinstance(rest, torch.Tensor)
        reft = dp_indexing.project_Dt_from_m(d_full_np, idx_np, ebed, None, lmax, mmax)
        np.testing.assert_array_equal(rest.numpy(), np.asarray(reft))


class TestUtilsParity:
    @pytest.mark.parametrize("dtype", ["float64", "float32"])  # input precision
    def test_safe_norm(self, dtype) -> None:
        rng = np.random.default_rng(1234)
        x = rng.normal(size=(8, 3)).astype(getattr(np, dtype))
        # include zero vectors: exercises the eps regularization path
        x[2, :] = 0.0
        x[5, :] = 0.0
        res = dp_utils.safe_norm(x)
        ref = pt_utils.safe_norm(torch.from_numpy(x))
        assert res.shape == (8, 1)
        np.testing.assert_allclose(res, ref.numpy(), rtol=1e-15, atol=0.0)

    def test_safe_norm_all_zero(self) -> None:
        # pure eps branch: norm of zero vector equals eps exactly
        x = np.zeros((4, 3), dtype=np.float64)
        res = dp_utils.safe_norm(x, eps=1e-7)
        ref = pt_utils.safe_norm(torch.from_numpy(x), eps=1e-7)
        np.testing.assert_allclose(res, ref.numpy(), rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(np.asarray(res), 1e-7, rtol=1e-15)

    def test_safe_norm_float16_promotion(self) -> None:
        # fp16 input: both implementations compute in fp32, cast back to fp16
        rng = np.random.default_rng(4321)
        x = rng.normal(size=(8, 3)).astype(np.float16)
        x[3, :] = 0.0
        res = dp_utils.safe_norm(x)
        ref = pt_utils.safe_norm(torch.from_numpy(x))
        assert np.asarray(res).dtype == np.float16
        np.testing.assert_array_equal(np.asarray(res), ref.numpy())

    def test_safe_norm_torch_input(self) -> None:
        # dpmodel safe_norm is array-API: must accept torch tensors
        rng = np.random.default_rng(999)
        x = rng.normal(size=(8, 3))
        x[0, :] = 0.0
        res = dp_utils.safe_norm(torch.from_numpy(x))
        assert isinstance(res, torch.Tensor)
        ref = pt_utils.safe_norm(torch.from_numpy(x))
        np.testing.assert_allclose(res.numpy(), ref.numpy(), rtol=1e-15, atol=0.0)

    def test_attn_res_modes(self) -> None:
        assert dp_utils.ATTN_RES_MODES == pt_utils.ATTN_RES_MODES

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (np.float16, np.float32),  # promoted branch
            (np.float32, np.float32),  # unchanged branch
            (np.float64, np.float64),  # unchanged branch
        ],
    )  # (input dtype, expected promoted dtype)
    def test_get_promoted_dtype(self, in_dtype, out_dtype) -> None:
        assert np.dtype(dp_utils.get_promoted_dtype(np.dtype(in_dtype))) == np.dtype(
            out_dtype
        )

    def test_get_promoted_dtype_bfloat16(self) -> None:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        assert np.dtype(
            dp_utils.get_promoted_dtype(np.dtype(ml_dtypes.bfloat16))
        ) == np.dtype(np.float32)

    def test_init_trunc_normal_fan_in_out(self) -> None:
        fan_out, fan_in = 256, 128
        w = np.empty((fan_out, fan_in), dtype=np.float64)
        dp_utils.init_trunc_normal_fan_in_out(w, seed=7)
        std = 1.0 / np.sqrt(fan_in + fan_out)
        # truncation bound respected
        assert np.abs(w).max() <= 3.0 * std
        # statistics close to the (truncated) normal target
        assert abs(w.mean()) < 5.0 * std / np.sqrt(w.size)
        assert 0.8 * std < w.std() < 1.05 * std
        # reproducible for identical seed
        w2 = np.empty_like(w)
        dp_utils.init_trunc_normal_fan_in_out(w2, seed=7)
        np.testing.assert_array_equal(w, w2)
        # scale parameter rescales std
        w3 = np.empty_like(w)
        dp_utils.init_trunc_normal_fan_in_out(w3, seed=7, scale=2.0)
        assert np.abs(w3).max() <= 6.0 * std
        assert w3.std() > w.std()

    def test_init_trunc_normal_fan_in_out_errors(self) -> None:
        with pytest.raises(ValueError):
            dp_utils.init_trunc_normal_fan_in_out(
                np.empty((2, 3, 4), dtype=np.float64), seed=0
            )
        with pytest.raises(ValueError):
            dp_utils.init_trunc_normal_fan_in_out(
                np.empty((2, 3), dtype=np.float64), seed=0, scale=0.0
            )


class TestRadialParity:
    rcut = 6.0

    def _r_grid(self) -> np.ndarray:
        # r=0 (sinc/envelope zero-distance branch), r=rcut (envelope boundary),
        # r>rcut (envelope-zero branch), plus a dense inside/outside sweep
        return np.concatenate(
            [[0.0, self.rcut, self.rcut + 0.5], np.linspace(0.05, 6.5, 200)]
        )[:, None]

    @pytest.mark.parametrize("exponent", [5, 7])  # envelope polynomial exponent
    def test_envelope(self, exponent) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            C3CutoffEnvelope as DPEnvelope,
        )
        from deepmd.pt.model.descriptor.sezm_nn.radial import (
            C3CutoffEnvelope as PTEnvelope,
        )

        pt_mod = PTEnvelope(rcut=self.rcut, exponent=exponent, dtype=torch.float64)
        dp_mod = DPEnvelope(rcut=self.rcut, exponent=exponent, precision="float64")
        r = self._r_grid()
        res = dp_mod.call(r)
        assert_parity(res, pt_mod(torch.from_numpy(r)))
        # boundary contract: E(0)=1, E(r>=rcut)=0 exactly
        np.testing.assert_array_equal(np.asarray(res)[0], 1.0)
        np.testing.assert_array_equal(np.asarray(res)[r[:, 0] >= self.rcut], 0.0)

    def test_envelope_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            C3CutoffEnvelope as DPEnvelope,
        )

        dp_mod = DPEnvelope(rcut=self.rcut, exponent=5, precision="float64")
        dp_mod2 = DPEnvelope.deserialize(dp_mod.serialize())
        r = self._r_grid()
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(r)), np.asarray(dp_mod2.call(r))
        )

    @pytest.mark.parametrize("basis_type", ["bessel", "gaussian"])  # both bases
    @pytest.mark.parametrize("exponent", [5, 7])  # envelope exponent
    def test_radial_basis(self, basis_type, exponent) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialBasis as DPRadialBasis,
        )
        from deepmd.pt.model.descriptor.sezm_nn.radial import (
            RadialBasis as PTRadialBasis,
        )

        n_radial = 16
        pt_mod = PTRadialBasis(
            rcut=self.rcut,
            basis_type=basis_type,
            n_radial=n_radial,
            dtype=torch.float64,
            exponent=exponent,
        )
        # perturb the trained frequencies so parity exercises copied weights,
        # not just identical deterministic init
        rng = np.random.default_rng(2030)
        with torch.no_grad():
            pt_mod.adam_freqs += torch.from_numpy(0.05 * rng.normal(size=(1, n_radial)))
        serialized = pt_mod.serialize()
        # pt state_dict key contract: only the trainable frequencies
        assert list(serialized["@variables"]) == ["adam_freqs"]
        dp_mod = DPRadialBasis.deserialize(serialized)
        r = self._r_grid()
        assert_parity(dp_mod.call(r), pt_mod(torch.from_numpy(r)))

    @pytest.mark.parametrize("basis_type", ["bessel", "gaussian"])  # both bases
    def test_radial_basis_roundtrip(self, basis_type) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialBasis as DPRadialBasis,
        )

        dp_mod = DPRadialBasis(
            rcut=self.rcut,
            basis_type=basis_type,
            n_radial=12,
            precision="float64",
            exponent=7,
        )
        dp_mod2 = DPRadialBasis.deserialize(dp_mod.serialize())
        r = self._r_grid()
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(r)), np.asarray(dp_mod2.call(r))
        )

    @pytest.mark.parametrize(
        "mlp_layers",
        [[16, 32, 24], [16, 24]],
    )  # with hidden layers (Linear+RMSNorm+act) and pure-linear (no hidden) branch
    @pytest.mark.parametrize("activation", ["silu", "tanh"])  # activation mapping
    def test_radial_mlp(self, mlp_layers, activation) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )
        from deepmd.pt.model.descriptor.sezm_nn.radial import (
            RadialMLP as PTRadialMLP,
        )

        pt_mod = PTRadialMLP(
            mlp_layers,
            activation_function=activation,
            dtype=torch.float64,
            seed=11,
        )
        # perturb all parameters (RMSNorm scale inits to ones, which would
        # otherwise make the scale copy untested)
        rng = np.random.default_rng(2031)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        serialized = pt_mod.serialize()
        # pt state_dict key contract: Sequential index 3*i for linear `matrix`,
        # 3*i+1 for RMSNorm `adam_scale` (activation modules are parameter-free)
        n_lin = len(mlp_layers) - 1
        expected_keys = {f"{3 * i}.matrix" for i in range(n_lin)} | {
            f"{3 * i + 1}.adam_scale" for i in range(n_lin - 1)
        }
        assert set(serialized["@variables"]) == expected_keys
        dp_mod = DPRadialMLP.deserialize(serialized)
        x = rng.normal(size=(50, mlp_layers[0]))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    def test_radial_mlp_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )

        dp_mod = DPRadialMLP(
            [16, 32, 24],
            activation_function="silu",
            precision="float64",
            seed=5,
        )
        dp_mod2 = DPRadialMLP.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2032)
        x = rng.normal(size=(50, 16))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_radial_mlp_zero_input_is_zero(self) -> None:
        # bias-free design contract: RadialMLP(0) = 0
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )

        dp_mod = DPRadialMLP([8, 16, 4], precision="float64", seed=3)
        out = dp_mod.call(np.zeros((5, 8), dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(out), 0.0)

    def test_radial_mlp_unsupported_activation(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )

        dp_mod = DPRadialMLP([4, 8, 4], activation_function="nope", seed=0)
        with pytest.raises(NotImplementedError):
            dp_mod.call(np.zeros((2, 4), dtype=np.float64))

    def test_rmsnorm_parity_and_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import RMSNorm as DPRMSNorm
        from deepmd.pt.model.descriptor.sezm_nn.norm import RMSNorm as PTRMSNorm

        channels = 24
        pt_mod = PTRMSNorm(channels=channels, dtype=torch.float64, trainable=True)
        rng = np.random.default_rng(2033)
        with torch.no_grad():
            pt_mod.adam_scale += torch.from_numpy(0.1 * rng.normal(size=(channels,)))
        serialized = pt_mod.serialize()
        assert list(serialized["@variables"]) == ["adam_scale"]
        dp_mod = DPRMSNorm.deserialize(serialized)
        x64 = rng.normal(size=(50, channels))
        assert_parity(dp_mod.call(x64), pt_mod(torch.from_numpy(x64)))
        # input-dtype promotion branch: fp32 input with fp64 params,
        # output cast back to fp32 in both implementations
        x32 = x64.astype(np.float32)
        res32 = dp_mod.call(x32)
        ref32 = pt_mod(torch.from_numpy(x32))
        assert np.asarray(res32).dtype == np.float32
        np.testing.assert_array_equal(np.asarray(res32), ref32.detach().numpy())
        # serialize roundtrip is exact
        dp_mod2 = DPRMSNorm.deserialize(dp_mod.serialize())
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x64)), np.asarray(dp_mod2.call(x64))
        )

    def test_constructor_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            C3CutoffEnvelope as DPEnvelope,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialBasis as DPRadialBasis,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )

        with pytest.raises(ValueError):  # rcut <= 0
            DPEnvelope(rcut=0.0)
        with pytest.raises(ValueError):  # exponent <= 0
            DPEnvelope(rcut=6.0, exponent=0)
        with pytest.raises(ValueError):  # rcut <= 0
            DPRadialBasis(rcut=-1.0)
        with pytest.raises(ValueError):  # n_radial <= 0
            DPRadialBasis(rcut=6.0, n_radial=0)
        with pytest.raises(ValueError):  # unknown basis_type
            DPRadialBasis(rcut=6.0, basis_type="chebyshev")
        with pytest.raises(ValueError):  # mlp_layers too short
            DPRadialMLP([16])

    def test_deserialize_wrong_class(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import RMSNorm as DPRMSNorm
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            C3CutoffEnvelope as DPEnvelope,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialBasis as DPRadialBasis,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
            RadialMLP as DPRadialMLP,
        )

        for klass in (DPEnvelope, DPRadialBasis, DPRadialMLP, DPRMSNorm):
            with pytest.raises(ValueError):
                klass.deserialize({"@class": "Nope", "@version": 1})


def _make_edge_vectors() -> np.ndarray:
    """Random edge vectors plus the polar/eps corner cases of the quaternion charts."""
    rng = np.random.default_rng(1)
    vec = rng.standard_normal((128, 3))
    vec[0] = [0.0, 0.0, 1.0]  # +z axis (rb_small branch in the Wigner path)
    vec[1] = [0.0, 0.0, -1.0]  # -z axis (antiparallel pole, ra_small branch)
    vec[2] = [1e-9, 0.0, 1.0]  # near +z (eps branch of the +z chart)
    vec[3] = [0.0, 1.0, 0.0]  # +y (e3nn polar axis)
    vec[4] = [0.0, -1.0, 0.0]  # -y
    vec[5] = [1e-9, 0.0, -1.0]  # near -z (eps branch of the -z chart)
    vec[6] = [0.0, 0.0, 0.0]  # zero-length edge (eps-floored normalization)
    vec[7] = [0.3, -0.4, 0.0]  # equator (chart blend midpoint region)
    return vec


class TestWignerDParity:
    def test_build_edge_quaternion(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            build_edge_quaternion as dp_build_edge_quaternion,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            build_edge_quaternion as pt_build_edge_quaternion,
        )

        vec = _make_edge_vectors()
        vec_t = torch.tensor(vec, dtype=torch.float64, device=CPU)
        # edge_len omitted branch
        quat_dp = dp_build_edge_quaternion(vec)
        quat_pt = pt_build_edge_quaternion(vec_t)
        assert_parity(quat_dp, quat_pt)
        # edge_len provided branch
        edge_len = np.linalg.norm(vec, axis=-1, keepdims=True)
        quat_dp = dp_build_edge_quaternion(vec, edge_len=edge_len)
        quat_pt = pt_build_edge_quaternion(
            vec_t,
            edge_len=torch.tensor(edge_len, dtype=torch.float64, device=CPU),
        )
        assert_parity(quat_dp, quat_pt)
        # the quaternion rotates the unit edge direction onto local +z
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            quaternion_to_rotation_matrix as dp_quaternion_to_rotation_matrix,
        )

        rot = dp_quaternion_to_rotation_matrix(quat_dp)
        unit = vec / np.sqrt(np.sum(vec * vec, axis=-1, keepdims=True) + 1e-14)
        local = np.einsum("eij,ej->ei", rot, unit)
        scale = np.linalg.norm(unit, axis=-1)  # ~0 for the zero-length edge row
        np.testing.assert_allclose(local[:, 2], scale, atol=1e-10)
        np.testing.assert_allclose(local[:, :2], 0.0, atol=1e-10)

    def test_quaternion_helpers(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn import wignerd as dp_w
        from deepmd.pt.model.descriptor.sezm_nn import wignerd as pt_w

        rng = np.random.default_rng(2)
        q1 = rng.standard_normal((16, 4))
        q2 = rng.standard_normal((16, 4))
        gamma = rng.standard_normal((16,))
        weight = rng.uniform(0.0, 1.0, (16,))
        q1_t = torch.tensor(q1, dtype=torch.float64, device=CPU)
        q2_t = torch.tensor(q2, dtype=torch.float64, device=CPU)
        assert_parity(
            dp_w.quaternion_multiply(q1, q2), pt_w.quaternion_multiply(q1_t, q2_t)
        )
        assert_parity(
            dp_w.quaternion_z_rotation(gamma),
            pt_w.quaternion_z_rotation(
                torch.tensor(gamma, dtype=torch.float64, device=CPU)
            ),
        )
        assert_parity(dp_w.quaternion_normalize(q1), pt_w.quaternion_normalize(q1_t))
        assert_parity(
            dp_w.quaternion_to_rotation_matrix(dp_w.quaternion_normalize(q1)),
            pt_w.quaternion_to_rotation_matrix(pt_w.quaternion_normalize(q1_t)),
        )
        assert_parity(
            dp_w.quaternion_nlerp(q1, q2, weight),
            pt_w.quaternion_nlerp(
                q1_t, q2_t, torch.tensor(weight, dtype=torch.float64, device=CPU)
            ),
        )

    @pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4])  # degree range incl. beyond-core
    def test_quat_and_d(self, lmax) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            WignerDCalculator as DPWignerDCalculator,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            build_edge_quaternion as dp_build_edge_quaternion,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator as PTWignerDCalculator,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            build_edge_quaternion as pt_build_edge_quaternion,
        )

        vec = _make_edge_vectors()
        quat_dp = dp_build_edge_quaternion(vec)
        quat_pt = pt_build_edge_quaternion(
            torch.tensor(vec, dtype=torch.float64, device=CPU)
        )
        assert_parity(quat_dp, quat_pt)

        calc_dp = DPWignerDCalculator(lmax, precision="float64")
        calc_pt = PTWignerDCalculator(lmax, dtype=torch.float64)
        D_dp, Dt_dp = calc_dp(quat_dp)
        D_pt, Dt_pt = calc_pt(quat_pt)
        dim = (lmax + 1) ** 2
        assert D_dp.shape == (vec.shape[0], dim, dim)
        assert_parity(D_dp, D_pt)
        assert_parity(Dt_dp, Dt_pt)
        # rotation property: D @ Dt == I
        eye = np.broadcast_to(np.eye(dim), D_dp.shape)
        np.testing.assert_allclose(D_dp @ Dt_dp, eye, atol=1e-11)

    @pytest.mark.parametrize("lmax", [2, 4])  # calculator degree
    @pytest.mark.parametrize("lmin", [1, 2, 3, 4, 5])  # zonal start degree
    def test_forward_zonal(self, lmax, lmin) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            WignerDCalculator as DPWignerDCalculator,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            build_edge_quaternion as dp_build_edge_quaternion,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator as PTWignerDCalculator,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            build_edge_quaternion as pt_build_edge_quaternion,
        )

        vec = _make_edge_vectors()
        quat_dp = dp_build_edge_quaternion(vec)
        quat_pt = pt_build_edge_quaternion(
            torch.tensor(vec, dtype=torch.float64, device=CPU)
        )
        calc_dp = DPWignerDCalculator(lmax, precision="float64")
        calc_pt = PTWignerDCalculator(lmax, dtype=torch.float64)
        z_dp = calc_dp.forward_zonal(quat_dp, lmin=lmin)
        z_pt = calc_pt.forward_zonal(quat_pt, lmin=lmin)
        n_expected = max((lmax + 1) ** 2 - lmin * lmin, 0)
        assert z_dp.shape == (vec.shape[0], n_expected)
        assert tuple(z_pt.shape) == (vec.shape[0], n_expected)
        assert_parity(z_dp, z_pt)

    def test_call_works_on_torch_tensors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            WignerDCalculator as DPWignerDCalculator,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            build_edge_quaternion as dp_build_edge_quaternion,
        )

        vec = _make_edge_vectors()
        quat_np = dp_build_edge_quaternion(vec)
        quat_t = dp_build_edge_quaternion(
            torch.tensor(vec, dtype=torch.float64, device=CPU)
        )
        assert isinstance(quat_t, torch.Tensor)
        assert_parity(quat_np, quat_t)
        calc_dp = DPWignerDCalculator(3, precision="float64")
        D_np, Dt_np = calc_dp(quat_np)
        D_t, Dt_t = calc_dp(quat_t)
        assert isinstance(D_t, torch.Tensor)
        assert_parity(D_np, D_t)
        assert_parity(Dt_np, Dt_t)
        z_np = calc_dp.forward_zonal(quat_np, lmin=2)
        z_t = calc_dp.forward_zonal(quat_t, lmin=2)
        assert_parity(z_np, z_t)

    def test_serialize(self) -> None:
        # pt WignerDCalculator has buffers, but they are all derived constants:
        # its serialize() emits only {"@class", "@version"} and deserialize()
        # is delegated to the parent (raises NotImplementedError). The dpmodel
        # port mirrors that contract exactly; no @variables roundtrip exists.
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            WignerDCalculator as DPWignerDCalculator,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator as PTWignerDCalculator,
        )

        calc_dp = DPWignerDCalculator(2, precision="float64")
        calc_pt = PTWignerDCalculator(2, dtype=torch.float64)
        assert calc_dp.serialize() == calc_pt.serialize()
        with pytest.raises(NotImplementedError):
            DPWignerDCalculator.deserialize(calc_dp.serialize())
        with pytest.raises(ValueError):
            DPWignerDCalculator.deserialize({"@class": "Nope", "@version": 1})

    def test_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
            WignerDCalculator as DPWignerDCalculator,
        )

        with pytest.raises(ValueError):  # negative lmax
            DPWignerDCalculator(-1, precision="float64")
        calc = DPWignerDCalculator(2, precision="float64")
        with pytest.raises(ValueError):  # lmin < 1
            calc.forward_zonal(np.zeros((4, 4)), lmin=0)


class TestNormParity:
    channels = 8

    def _perturb(self, pt_mod: torch.nn.Module, seed: int) -> None:
        # perturb all parameters (scales init to ones / biases to zeros, which
        # would otherwise make the parameter copy untested)
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    @pytest.mark.parametrize("lmax", [0, 2, 3])  # 0 covers the scalar-only branch
    @pytest.mark.parametrize("n_focus", [1, 2])  # focus streams
    def test_equivariant_rmsnorm(self, lmax, n_focus) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            EquivariantRMSNorm as DPEquivariantRMSNorm,
        )
        from deepmd.pt.model.descriptor.sezm_nn.norm import (
            EquivariantRMSNorm as PTEquivariantRMSNorm,
        )

        pt_mod = PTEquivariantRMSNorm(
            lmax, self.channels, n_focus, dtype=torch.float64, trainable=True
        )
        self._perturb(pt_mod, 2040)
        serialized = pt_mod.serialize()
        # pt state_dict key contract: 2 parameters + 2 persistent buffers
        assert set(serialized["@variables"]) == {
            "adam_scale",
            "bias",
            "expand_index",
            "balance_weight",
        }
        dp_mod = DPEquivariantRMSNorm.deserialize(serialized)
        rng = np.random.default_rng(2041)
        x = rng.normal(size=(17, (lmax + 1) ** 2, n_focus, self.channels))
        x[0] = 0.0  # all-zeros row exercises the eps path
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    def test_equivariant_rmsnorm_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            EquivariantRMSNorm as DPEquivariantRMSNorm,
        )

        dp_mod = DPEquivariantRMSNorm(2, self.channels, 2, precision="float64")
        dp_mod2 = DPEquivariantRMSNorm.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2042)
        x = rng.normal(size=(17, 9, 2, self.channels))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    @pytest.mark.parametrize(
        "lmax,mmax", [(0, 0), (2, 1), (2, 2), (3, 2)]
    )  # (0,0) covers the scalar-only branch; mmax<lmax covers truncation
    @pytest.mark.parametrize("n_focus", [1, 2])  # focus streams
    def test_reduced_equivariant_rmsnorm(self, lmax, mmax, n_focus) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ReducedEquivariantRMSNorm as DPReducedEquivariantRMSNorm,
        )
        from deepmd.pt.model.descriptor.sezm_nn.norm import (
            ReducedEquivariantRMSNorm as PTReducedEquivariantRMSNorm,
        )

        degree_index_m = dp_indexing.build_m_major_l_index(lmax, mmax)
        pt_mod = PTReducedEquivariantRMSNorm(
            lmax=lmax,
            mmax=mmax,
            channels=self.channels,
            degree_index_m=torch.tensor(degree_index_m, dtype=torch.long, device=CPU),
            n_focus=n_focus,
            dtype=torch.float64,
            trainable=True,
        )
        self._perturb(pt_mod, 2043)
        serialized = pt_mod.serialize()
        # pt state_dict key contract: 2 parameters + 2 persistent buffers
        assert set(serialized["@variables"]) == {
            "degree_index_m",
            "balance_weight",
            "adam_scale",
            "bias0",
        }
        dp_mod = DPReducedEquivariantRMSNorm.deserialize(serialized)
        rng = np.random.default_rng(2044)
        x = rng.normal(size=(17, n_focus, degree_index_m.size, self.channels))
        x[0] = 0.0  # all-zeros row exercises the eps path
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    def test_reduced_equivariant_rmsnorm_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ReducedEquivariantRMSNorm as DPReducedEquivariantRMSNorm,
        )

        degree_index_m = dp_indexing.build_m_major_l_index(2, 1)
        dp_mod = DPReducedEquivariantRMSNorm(
            lmax=2,
            mmax=1,
            channels=self.channels,
            degree_index_m=degree_index_m,
            n_focus=2,
            precision="float64",
        )
        dp_mod2 = DPReducedEquivariantRMSNorm.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2045)
        x = rng.normal(size=(17, 2, degree_index_m.size, self.channels))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_reduced_equivariant_rmsnorm_invalid_degree_index(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ReducedEquivariantRMSNorm as DPReducedEquivariantRMSNorm,
        )

        with pytest.raises(ValueError):  # degree 5 > lmax leaves zero weights
            DPReducedEquivariantRMSNorm(
                lmax=2,
                mmax=1,
                channels=4,
                degree_index_m=np.array([0, 1, 5], dtype=np.int64),
                precision="float64",
            )

    @pytest.mark.parametrize("ndim", [2, 3])  # (B, C) and (B, F, C) branches
    def test_scalar_rmsnorm(self, ndim) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ScalarRMSNorm as DPScalarRMSNorm,
        )
        from deepmd.pt.model.descriptor.sezm_nn.norm import (
            ScalarRMSNorm as PTScalarRMSNorm,
        )

        n_focus = 1 if ndim == 2 else 2
        pt_mod = PTScalarRMSNorm(
            channels=self.channels,
            n_focus=n_focus,
            dtype=torch.float64,
            trainable=True,
        )
        self._perturb(pt_mod, 2046)
        serialized = pt_mod.serialize()
        assert list(serialized["@variables"]) == ["adam_scale"]
        dp_mod = DPScalarRMSNorm.deserialize(serialized)
        rng = np.random.default_rng(2047)
        shape = (17, self.channels) if ndim == 2 else (17, n_focus, self.channels)
        x = rng.normal(size=shape)
        x[0] = 0.0  # all-zeros row exercises the eps path
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))
        # serialize roundtrip is exact
        dp_mod2 = DPScalarRMSNorm.deserialize(dp_mod.serialize())
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_norm_fp32_input_branch(self) -> None:
        # input-dtype promotion branch: fp32 input with fp64 params, output is
        # cast back to fp32. The downcast happens at the same point in both
        # implementations, so the results are bit-identical.
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            EquivariantRMSNorm as DPEquivariantRMSNorm,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ScalarRMSNorm as DPScalarRMSNorm,
        )
        from deepmd.pt.model.descriptor.sezm_nn.norm import (
            EquivariantRMSNorm as PTEquivariantRMSNorm,
        )
        from deepmd.pt.model.descriptor.sezm_nn.norm import (
            ScalarRMSNorm as PTScalarRMSNorm,
        )

        rng = np.random.default_rng(2048)
        pt_eq = PTEquivariantRMSNorm(
            2, self.channels, 1, dtype=torch.float64, trainable=True
        )
        dp_eq = DPEquivariantRMSNorm.deserialize(pt_eq.serialize())
        x32 = rng.normal(size=(17, 9, 1, self.channels)).astype(np.float32)
        res = dp_eq.call(x32)
        ref = pt_eq(torch.from_numpy(x32))
        assert np.asarray(res).dtype == np.float32
        np.testing.assert_array_equal(np.asarray(res), ref.detach().numpy())

        pt_sc = PTScalarRMSNorm(
            channels=self.channels, dtype=torch.float64, trainable=True
        )
        dp_sc = DPScalarRMSNorm.deserialize(pt_sc.serialize())
        x32 = rng.normal(size=(17, self.channels)).astype(np.float32)
        res = dp_sc.call(x32)
        ref = pt_sc(torch.from_numpy(x32))
        assert np.asarray(res).dtype == np.float32
        np.testing.assert_array_equal(np.asarray(res), ref.detach().numpy())

    def test_deserialize_wrong_class(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            EquivariantRMSNorm as DPEquivariantRMSNorm,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ReducedEquivariantRMSNorm as DPReducedEquivariantRMSNorm,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.norm import (
            ScalarRMSNorm as DPScalarRMSNorm,
        )

        for klass in (
            DPEquivariantRMSNorm,
            DPReducedEquivariantRMSNorm,
            DPScalarRMSNorm,
        ):
            with pytest.raises(ValueError):
                klass.deserialize({"@class": "Nope", "@version": 1})


class TestSO3LinearParity:
    in_channels = 8
    out_channels = 6

    @pytest.mark.parametrize("lmax", [0, 2, 3])  # 0 covers the scalar-only branch
    @pytest.mark.parametrize("mlp_bias", [False, True])  # l=0 bias branch
    @pytest.mark.parametrize("n_focus", [1, 2])  # focus streams
    def test_so3_linear(self, lmax, mlp_bias, n_focus) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import SO3Linear as DPSO3Linear
        from deepmd.pt.model.descriptor.sezm_nn.so3 import SO3Linear as PTSO3Linear

        pt_mod = PTSO3Linear(
            lmax=lmax,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=n_focus,
            dtype=torch.float64,
            mlp_bias=mlp_bias,
            trainable=True,
            seed=21,
        )
        # bias inits to zeros; perturb so the bias copy is exercised
        rng = np.random.default_rng(2050)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        serialized = pt_mod.serialize()
        expected_keys = {"weight", "expand_index"} | ({"bias"} if mlp_bias else set())
        assert set(serialized["@variables"]) == expected_keys
        dp_mod = DPSO3Linear.deserialize(serialized)
        x = rng.normal(size=(17, (lmax + 1) ** 2, n_focus, self.in_channels))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    @pytest.mark.parametrize("mlp_bias", [False, True])  # l=0 bias branch
    def test_so3_linear_roundtrip(self, mlp_bias) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import SO3Linear as DPSO3Linear

        dp_mod = DPSO3Linear(
            lmax=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=2,
            precision="float64",
            mlp_bias=mlp_bias,
            seed=7,
        )
        dp_mod2 = DPSO3Linear.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2051)
        x = rng.normal(size=(17, 9, 2, self.in_channels))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_so3_linear_init_std_branches(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import SO3Linear as DPSO3Linear

        # init_std=0.0 -> exact zero init
        dp_zero = DPSO3Linear(
            lmax=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            precision="float64",
            init_std=0.0,
        )
        np.testing.assert_array_equal(dp_zero.weight, 0.0)
        # init_std>0 -> normal init (nonzero)
        dp_norm = DPSO3Linear(
            lmax=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            precision="float64",
            seed=3,
            init_std=0.5,
        )
        assert np.any(dp_norm.weight != 0.0)

    @pytest.mark.parametrize("bias", [False, True])  # bias branch
    def test_focus_linear(self, bias) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import FocusLinear as DPFocusLinear
        from deepmd.pt.model.descriptor.sezm_nn.so3 import FocusLinear as PTFocusLinear

        n_focus = 2
        pt_mod = PTFocusLinear(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=n_focus,
            dtype=torch.float64,
            bias=bias,
            trainable=True,
            seed=5,
        )
        rng = np.random.default_rng(2052)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        # pt FocusLinear has no serialize(); copy the state_dict fragment
        # (keys "weight"/"bias") directly, the contract used by nested modules
        state = pt_state_to_numpy(pt_mod)
        assert set(state) == ({"weight", "bias"} if bias else {"weight"})
        dp_mod = DPFocusLinear(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=n_focus,
            precision="float64",
            bias=bias,
            seed=5,
        )
        dp_mod.weight = state["weight"]
        if bias:
            dp_mod.bias = state["bias"]
        x = rng.normal(size=(17, n_focus, self.in_channels))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))
        # serialize roundtrip is exact
        dp_mod2 = DPFocusLinear.deserialize(dp_mod.serialize())
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    @pytest.mark.parametrize("bias", [False, True])  # bias branch
    def test_channel_linear(self, bias) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
            ChannelLinear as DPChannelLinear,
        )
        from deepmd.pt.model.descriptor.sezm_nn.so3 import (
            ChannelLinear as PTChannelLinear,
        )

        pt_mod = PTChannelLinear(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dtype=torch.float64,
            bias=bias,
            trainable=True,
            seed=6,
        )
        rng = np.random.default_rng(2053)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        # pt ChannelLinear has no serialize(); copy the state_dict fragment
        state = pt_state_to_numpy(pt_mod)
        assert set(state) == ({"weight", "bias"} if bias else {"weight"})
        dp_mod = DPChannelLinear(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            precision="float64",
            bias=bias,
            seed=6,
        )
        dp_mod.weight = state["weight"]
        if bias:
            dp_mod.bias = state["bias"]
        # leading axes are batch: exercise a 3D input
        x = rng.normal(size=(17, 4, self.in_channels))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))
        # serialize roundtrip is exact
        dp_mod2 = DPChannelLinear.deserialize(dp_mod.serialize())
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_focus_channel_linear_init_std_branch(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
            ChannelLinear as DPChannelLinear,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import FocusLinear as DPFocusLinear

        # init_std branch: normal(0, init_std) instead of uniform
        dp_f = DPFocusLinear(
            in_channels=64,
            out_channels=64,
            n_focus=1,
            precision="float64",
            seed=8,
            init_std=0.01,
        )
        dp_c = DPChannelLinear(
            in_channels=64,
            out_channels=64,
            precision="float64",
            seed=8,
            init_std=0.01,
        )
        for w in (dp_f.weight, dp_c.weight):
            # uniform init would have std ~ bound/sqrt(3) = 0.072; normal 0.01
            assert np.std(w) < 0.02

    def test_deserialize_wrong_class(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
            ChannelLinear as DPChannelLinear,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import FocusLinear as DPFocusLinear
        from deepmd.dpmodel.descriptor.dpa4_nn.so3 import SO3Linear as DPSO3Linear

        for klass in (DPSO3Linear, DPFocusLinear, DPChannelLinear):
            with pytest.raises(ValueError):
                klass.deserialize({"@class": "Nope", "@version": 1})


class TestGatedActivationParity:
    channels = 8

    def _build_pair(self, *, lmax, mmax, n_focus, mlp_bias, layout, activation):
        from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
            GatedActivation as DPGatedActivation,
        )
        from deepmd.pt.model.descriptor.sezm_nn.activation import (
            GatedActivation as PTGatedActivation,
        )

        pt_mod = PTGatedActivation(
            lmax=lmax,
            mmax=mmax,
            channels=self.channels,
            n_focus=n_focus,
            dtype=torch.float64,
            activation_function=activation,
            mlp_bias=mlp_bias,
            layout=layout,
            trainable=True,
            seed=31,
        )
        # perturb all parameters (gate bias inits to zeros)
        rng = np.random.default_rng(2060)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.05 * rng.normal(size=tuple(p.shape)))
        serialized = pt_mod.serialize()
        expected_keys = {"expand_index"}
        if lmax > 0:
            expected_keys |= {"gate_linear.weight"}
            if mlp_bias:
                expected_keys |= {"gate_linear.bias"}
        assert set(serialized["@variables"]) == expected_keys
        dp_mod = DPGatedActivation.deserialize(serialized)
        return dp_mod, pt_mod

    def _shape(self, lmax, mmax, n_focus, layout):
        if mmax is None:
            ncoeff = (lmax + 1) ** 2
        else:
            ncoeff = dp_indexing.build_m_major_l_index(lmax, mmax).size
        if layout == "nfdc":
            return (17, n_focus, ncoeff, self.channels)
        return (17, ncoeff, n_focus, self.channels)

    @pytest.mark.parametrize("layout", ["nfdc", "ndfc"])  # tensor layout
    @pytest.mark.parametrize("use_gate", [False, True])  # standard vs GLU mode
    def test_gated_activation(self, layout, use_gate) -> None:
        lmax, n_focus = 2, 2
        dp_mod, pt_mod = self._build_pair(
            lmax=lmax,
            mmax=None,
            n_focus=n_focus,
            mlp_bias=False,
            layout=layout,
            activation="silu",
        )
        rng = np.random.default_rng(2061)
        shape = self._shape(lmax, None, n_focus, layout)
        x = rng.normal(size=shape)
        if use_gate:
            gate = rng.normal(size=shape)
            res = dp_mod.call(x, gate=gate)
            ref = pt_mod(torch.from_numpy(x), gate=torch.from_numpy(gate))
        else:
            res = dp_mod.call(x)
            ref = pt_mod(torch.from_numpy(x))
        assert_parity(res, ref)

    @pytest.mark.parametrize("mlp_bias", [False, True])  # gate-linear bias branch
    def test_gated_activation_mmax_reduced(self, mlp_bias) -> None:
        # m-major reduced layout branch (mmax provided) + tanh activation
        lmax, mmax, n_focus = 3, 1, 1
        dp_mod, pt_mod = self._build_pair(
            lmax=lmax,
            mmax=mmax,
            n_focus=n_focus,
            mlp_bias=mlp_bias,
            layout="ndfc",
            activation="tanh",
        )
        rng = np.random.default_rng(2062)
        x = rng.normal(size=self._shape(lmax, mmax, n_focus, "ndfc"))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    @pytest.mark.parametrize("use_gate", [False, True])  # standard vs GLU mode
    def test_gated_activation_lmax0(self, use_gate) -> None:
        # lmax=0 branch: scalar-only output, no gate_linear
        dp_mod, pt_mod = self._build_pair(
            lmax=0,
            mmax=None,
            n_focus=1,
            mlp_bias=False,
            layout="nfdc",
            activation="silu",
        )
        assert dp_mod.gate_linear is None
        rng = np.random.default_rng(2063)
        shape = self._shape(0, None, 1, "nfdc")
        x = rng.normal(size=shape)
        if use_gate:
            gate = rng.normal(size=shape)
            res = dp_mod.call(x, gate=gate)
            ref = pt_mod(torch.from_numpy(x), gate=torch.from_numpy(gate))
        else:
            res = dp_mod.call(x)
            ref = pt_mod(torch.from_numpy(x))
        assert_parity(res, ref)

    def test_gated_activation_fp32_input_branch(self) -> None:
        # input-dtype promotion branch: fp32 input with fp64 gate params;
        # compared at ~1-2 ulp fp32 (downcast happens at different points
        # in the two implementations)
        dp_mod, pt_mod = self._build_pair(
            lmax=2,
            mmax=None,
            n_focus=1,
            mlp_bias=False,
            layout="nfdc",
            activation="silu",
        )
        rng = np.random.default_rng(2064)
        x32 = rng.normal(size=self._shape(2, None, 1, "nfdc")).astype(np.float32)
        res = dp_mod.call(x32)
        ref = pt_mod(torch.from_numpy(x32))
        assert np.asarray(res).dtype == np.float32
        np.testing.assert_allclose(
            np.asarray(res), ref.detach().numpy(), rtol=3e-7, atol=3e-7
        )

    def test_gated_activation_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
            GatedActivation as DPGatedActivation,
        )

        dp_mod = DPGatedActivation(
            lmax=2,
            channels=self.channels,
            n_focus=2,
            precision="float64",
            mlp_bias=True,
            layout="ndfc",
            seed=13,
        )
        dp_mod2 = DPGatedActivation.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2065)
        x = rng.normal(size=(17, 9, 2, self.channels))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_gated_activation_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
            GatedActivation as DPGatedActivation,
        )

        with pytest.raises(ValueError):  # mmax < 0
            DPGatedActivation(lmax=2, mmax=-1, channels=4)
        with pytest.raises(ValueError):  # mmax > lmax
            DPGatedActivation(lmax=2, mmax=3, channels=4)
        with pytest.raises(ValueError):  # invalid layout
            DPGatedActivation(lmax=2, channels=4, layout="cfdn")
        with pytest.raises(ValueError):  # wrong class
            DPGatedActivation.deserialize({"@class": "Nope", "@version": 1})

    def test_swiglu(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as DPSwiGLU
        from deepmd.pt.model.descriptor.sezm_nn.activation import SwiGLU as PTSwiGLU

        rng = np.random.default_rng(2066)
        x = rng.normal(size=(17, 3, 2 * self.channels))
        assert_parity(DPSwiGLU().call(x), PTSwiGLU()(torch.from_numpy(x)))


class TestNoTorchImport:
    def test_dpa4_nn_does_not_import_torch(self) -> None:
        code = (
            "import sys; "
            "import deepmd.dpmodel.descriptor.dpa4_nn.indexing, "
            "deepmd.dpmodel.descriptor.dpa4_nn.utils, "
            "deepmd.dpmodel.descriptor.dpa4_nn.norm, "
            "deepmd.dpmodel.descriptor.dpa4_nn.radial, "
            "deepmd.dpmodel.descriptor.dpa4_nn.so3, "
            "deepmd.dpmodel.descriptor.dpa4_nn.activation, "
            "deepmd.dpmodel.descriptor.dpa4_nn.wignerd; "
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
