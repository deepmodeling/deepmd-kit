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


class TestNoTorchImport:
    def test_dpa4_nn_does_not_import_torch(self) -> None:
        code = (
            "import sys; "
            "import deepmd.dpmodel.descriptor.dpa4_nn.indexing, "
            "deepmd.dpmodel.descriptor.dpa4_nn.utils, "
            "deepmd.dpmodel.descriptor.dpa4_nn.norm, "
            "deepmd.dpmodel.descriptor.dpa4_nn.radial; "
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
