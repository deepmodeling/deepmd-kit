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
        # cast back to fp32. Compared at a few ulp fp32: truncation/downcast
        # points may differ across BLAS/SIMD codegen and environments, so
        # bit-exact equality would be brittle.
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
        np.testing.assert_allclose(
            np.asarray(res), ref.detach().numpy(), rtol=2e-7, atol=2e-7
        )

        pt_sc = PTScalarRMSNorm(
            channels=self.channels, dtype=torch.float64, trainable=True
        )
        dp_sc = DPScalarRMSNorm.deserialize(pt_sc.serialize())
        x32 = rng.normal(size=(17, self.channels)).astype(np.float32)
        res = dp_sc.call(x32)
        ref = pt_sc(torch.from_numpy(x32))
        assert np.asarray(res).dtype == np.float32
        np.testing.assert_allclose(
            np.asarray(res), ref.detach().numpy(), rtol=2e-7, atol=2e-7
        )

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
        # downcast happens at different points in the two implementations,
        # and truncation points vary across BLAS/SIMD codegen, so compare
        # with fp32 round-off headroom rather than a single-machine ulp.
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
            np.asarray(res), ref.detach().numpy(), rtol=2e-6, atol=2e-7
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


class TestS2GridParity:
    channels = 8

    # ---------------------------------------------------------------- helpers
    def _build_projectors(self, lmax, mmax, coefficient_layout):
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            S2GridProjector as DPS2GridProjector,
        )
        from deepmd.pt.model.descriptor.sezm_nn.projection import (
            S2GridProjector as PTS2GridProjector,
        )

        pt_proj = PTS2GridProjector(
            lmax=lmax,
            mmax=mmax,
            dtype=torch.float64,
            coefficient_layout=coefficient_layout,
            grid_method="lebedev",
        )
        dp_proj = DPS2GridProjector(
            lmax=lmax,
            mmax=mmax,
            precision="float64",
            coefficient_layout=coefficient_layout,
            grid_method="lebedev",
        )
        return pt_proj, dp_proj

    def _build_grid_nets(
        self,
        *,
        lmax,
        op_type,
        layout,
        mlp_bias=False,
        n_focus=1,
        mmax=None,
        coefficient_layout="packed",
        grid_branches=1,
        seed=7,
    ):
        """Build a pt S2GridNet, perturb its params, and copy them into dp."""
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            S2GridNet as DPS2GridNet,
        )
        from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
            S2GridNet as PTS2GridNet,
        )

        pt_net = PTS2GridNet(
            lmax=lmax,
            mmax=mmax,
            channels=self.channels,
            n_focus=n_focus,
            mode="self",
            op_type=op_type,
            dtype=torch.float64,
            layout=layout,
            coefficient_layout=coefficient_layout,
            grid_method="lebedev",
            grid_branches=grid_branches,
            mlp_bias=mlp_bias,
            trainable=True,
            seed=seed,
        )
        rng = np.random.default_rng(2100)
        with torch.no_grad():
            for p in pt_net.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        dp_net = DPS2GridNet(
            lmax=lmax,
            mmax=mmax,
            channels=self.channels,
            n_focus=n_focus,
            mode="self",
            op_type=op_type,
            precision="float64",
            layout=layout,
            coefficient_layout=coefficient_layout,
            grid_method="lebedev",
            grid_branches=grid_branches,
            mlp_bias=mlp_bias,
            trainable=True,
            seed=seed,
        )
        # pt S2GridNet has no serialize(); copy the state_dict fragment with
        # the pt key names (the contract used by the dp serialize format)
        state = pt_state_to_numpy(pt_net)
        expected_keys = {"scalar_gate.weight"}
        if mlp_bias:
            expected_keys.add("scalar_gate.bias")
        if op_type == "branch":
            expected_keys |= {
                "grid_op.left_proj.weight",
                "grid_op.right_proj.weight",
                "grid_op.router.weight",
                "grid_op.out_proj.weight",
            }
        assert set(state) == expected_keys
        dp_net.scalar_gate.weight = state["scalar_gate.weight"]
        if mlp_bias:
            dp_net.scalar_gate.bias = state["scalar_gate.bias"]
        if op_type == "branch":
            for name in ("left_proj", "right_proj", "router", "out_proj"):
                getattr(dp_net.grid_op, name).weight = state[f"grid_op.{name}.weight"]
        return pt_net, dp_net

    # ------------------------------------------------- (a) projector constants
    @pytest.mark.parametrize("lmax,mmax", [(2, 2), (3, 3), (3, 2)])  # degree/order
    @pytest.mark.parametrize(
        "coefficient_layout", ["packed", "m_major"]
    )  # coefficient ordering
    def test_projector_constants(self, lmax, mmax, coefficient_layout) -> None:
        pt_proj, dp_proj = self._build_projectors(lmax, mmax, coefficient_layout)
        assert dp_proj.grid_resolution_list == pt_proj.grid_resolution_list
        assert dp_proj.grid_size == pt_proj.grid_size
        assert dp_proj.coeff_dim == pt_proj.coeff_dim
        assert dp_proj.packed_dim == pt_proj.packed_dim
        # validates the numpy-SH replacement of e3nn end-to-end
        assert_parity(dp_proj.to_grid_mat, pt_proj.to_grid_mat)
        assert_parity(dp_proj.from_grid_mat, pt_proj.from_grid_mat)
        # to_grid / from_grid forwards
        rng = np.random.default_rng(2080)
        x = rng.normal(size=(7, dp_proj.coeff_dim, 5))
        assert_parity(dp_proj.to_grid(x), pt_proj.to_grid(torch.from_numpy(x)))
        g = rng.normal(size=(7, dp_proj.grid_size, 5))
        assert_parity(dp_proj.from_grid(g), pt_proj.from_grid(torch.from_numpy(g)))

    @pytest.mark.parametrize("lmax", [2, 3])  # max degree
    def test_projector_quadrature_identity(self, lmax) -> None:
        # Lebedev path: from_grid o to_grid is the identity at machine
        # precision (full mmax == lmax, packed layout)
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            S2GridProjector as DPS2GridProjector,
        )

        dp_proj = DPS2GridProjector(
            lmax=lmax, precision="float64", grid_method="lebedev"
        )
        prod = np.matmul(dp_proj.from_grid_mat, dp_proj.to_grid_mat)
        np.testing.assert_allclose(prod, np.eye(dp_proj.coeff_dim), atol=1e-13)

    def test_projector_serialize(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            S2GridProjector as DPS2GridProjector,
        )

        pt_proj, dp_proj = self._build_projectors(3, 2, "m_major")
        # the serialize contracts are identical
        assert dp_proj.serialize() == pt_proj.serialize()
        # dp deserializes pt's real serialize() output
        dp_from_pt = DPS2GridProjector.deserialize(pt_proj.serialize())
        np.testing.assert_array_equal(dp_from_pt.to_grid_mat, dp_proj.to_grid_mat)
        np.testing.assert_array_equal(dp_from_pt.from_grid_mat, dp_proj.from_grid_mat)
        # dp roundtrip is exact
        dp_proj2 = DPS2GridProjector.deserialize(dp_proj.serialize())
        np.testing.assert_array_equal(dp_proj2.to_grid_mat, dp_proj.to_grid_mat)
        np.testing.assert_array_equal(dp_proj2.from_grid_mat, dp_proj.from_grid_mat)
        with pytest.raises(ValueError):  # wrong class
            DPS2GridProjector.deserialize({"@class": "Nope", "@version": 1})

    @pytest.mark.parametrize("method", ["lebedev", "e3nn"])  # quadrature backend
    @pytest.mark.parametrize("lmax,mmax", [(2, 2), (3, 2), (4, 4)])  # degree/order
    def test_resolve_s2_grid_resolution(self, method, lmax, mmax) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            resolve_s2_grid_resolution as dp_resolve,
        )
        from deepmd.pt.model.descriptor.sezm_nn.projection import (
            resolve_s2_grid_resolution as pt_resolve,
        )

        assert dp_resolve(lmax, mmax, method=method) == pt_resolve(
            lmax, mmax, method=method
        )
        with pytest.raises(ValueError):  # invalid method
            dp_resolve(lmax, mmax, method="cartesian")

    # ------------------------------------------ (b) S2GridNet forward parity
    @pytest.mark.parametrize("lmax", [2, 3])  # max degree
    @pytest.mark.parametrize("op_type", ["glu", "branch"])  # grid operation
    def test_s2_grid_net(self, lmax, op_type) -> None:
        # ffn-style core usage: mode="self", layout="ndfc", packed, n_focus=1
        pt_net, dp_net = self._build_grid_nets(
            lmax=lmax, op_type=op_type, layout="ndfc"
        )
        rng = np.random.default_rng(2081)
        n_coeff = (lmax + 1) ** 2
        x = rng.normal(size=(11, n_coeff, 1, 2 * self.channels))
        assert_parity(dp_net.call(x), pt_net(torch.from_numpy(x)))

    @pytest.mark.parametrize("mlp_bias", [False, True])  # scalar gate bias
    def test_s2_grid_net_nfdc_m_major(self, mlp_bias) -> None:
        # so2-style usage: mode="self", op_type="glu", layout="nfdc",
        # m-major coefficients truncated at mmax < lmax, multiple foci
        lmax, mmax, n_focus = 3, 2, 2
        pt_net, dp_net = self._build_grid_nets(
            lmax=lmax,
            mmax=mmax,
            op_type="glu",
            layout="nfdc",
            mlp_bias=mlp_bias,
            n_focus=n_focus,
            coefficient_layout="m_major",
        )
        n_coeff = dp_net.projector.coeff_dim
        assert n_coeff == pt_net.projector.coeff_dim
        rng = np.random.default_rng(2082)
        x = rng.normal(size=(11, n_focus, n_coeff, 2 * self.channels))
        assert_parity(dp_net.call(x), pt_net(torch.from_numpy(x)))

    def test_s2_grid_net_fp32_input(self) -> None:
        # fp32 input through a float64-precision net exercises the cast
        # branches; the output dtype must match the input dtype as in pt
        pt_net, dp_net = self._build_grid_nets(lmax=2, op_type="branch", layout="ndfc")
        rng = np.random.default_rng(2083)
        x = rng.normal(size=(11, 9, 1, 2 * self.channels)).astype(np.float32)
        dp_out = dp_net.call(x)
        assert dp_out.dtype == np.float32
        pt_out = pt_net(torch.from_numpy(x))
        assert pt_out.dtype == torch.float32
        np.testing.assert_allclose(
            np.asarray(dp_out), pt_out.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
        )

    # ------------------------------------------- (c) GridBranch forward parity
    @pytest.mark.parametrize("n_branches", [1, 2])  # router branches
    def test_grid_branch(self, n_branches) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            GridBranch as DPGridBranch,
        )
        from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
            GridBranch as PTGridBranch,
        )

        pt_mod = PTGridBranch(
            channels=self.channels,
            n_branches=n_branches,
            dtype=torch.float64,
            trainable=True,
            seed=9,
        )
        rng = np.random.default_rng(2084)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
        state = pt_state_to_numpy(pt_mod)
        assert set(state) == {
            "left_proj.weight",
            "right_proj.weight",
            "router.weight",
            "out_proj.weight",
        }
        dp_mod = DPGridBranch(
            channels=self.channels,
            n_branches=n_branches,
            precision="float64",
            seed=9,
        )
        for name in ("left_proj", "right_proj", "router", "out_proj"):
            getattr(dp_mod, name).weight = state[f"{name}.weight"]
        n_batch, n_grid, n_focus = 5, 26, 2
        query = rng.normal(size=(n_batch, n_grid, n_focus, self.channels))
        context = rng.normal(size=(n_batch, n_grid, n_focus, self.channels))
        scalar = rng.normal(size=(n_batch, n_focus, 2 * self.channels))
        assert_parity(
            dp_mod.call(query, context, scalar),
            pt_mod(
                torch.from_numpy(query),
                torch.from_numpy(context),
                torch.from_numpy(scalar),
            ),
        )
        # serialize roundtrip is exact; @variables keys match the pt state dict
        ser = dp_mod.serialize()
        assert set(ser["@variables"]) == set(state)
        dp_mod2 = DPGridBranch.deserialize(ser)
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(query, context, scalar)),
            np.asarray(dp_mod2.call(query, context, scalar)),
        )

    # ------------------------------------------------------ (e) serialization
    @pytest.mark.parametrize("op_type", ["glu", "branch"])  # grid operation
    @pytest.mark.parametrize("mlp_bias", [False, True])  # scalar gate bias
    def test_s2_grid_net_serialize_roundtrip(self, op_type, mlp_bias) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            S2GridNet as DPS2GridNet,
        )

        pt_net, dp_net = self._build_grid_nets(
            lmax=2, op_type=op_type, layout="ndfc", mlp_bias=mlp_bias
        )
        ser = dp_net.serialize()
        # @variables key set equals the pt state_dict key set exactly
        assert set(ser["@variables"]) == set(pt_state_to_numpy(pt_net))
        dp_net2 = DPS2GridNet.deserialize(ser)
        rng = np.random.default_rng(2085)
        x = rng.normal(size=(11, 9, 1, 2 * self.channels))
        np.testing.assert_array_equal(
            np.asarray(dp_net.call(x)), np.asarray(dp_net2.call(x))
        )
        # loading pt's real state_dict values through deserialize also works
        ser_pt = dict(ser)
        ser_pt["@variables"] = pt_state_to_numpy(pt_net)
        dp_net3 = DPS2GridNet.deserialize(ser_pt)
        assert_parity(dp_net3.call(x), pt_net(torch.from_numpy(x)))
        with pytest.raises(ValueError):  # wrong class
            DPS2GridNet.deserialize({"@class": "Nope", "@version": 1})

    def test_grid_branch_deserialize_wrong_class(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            GridBranch as DPGridBranch,
        )

        with pytest.raises(ValueError):
            DPGridBranch.deserialize({"@class": "Nope", "@version": 1})

    # ------------------------------------------------ (d) not-ported guards
    def test_not_ported_guards(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            S2GridNet as DPS2GridNet,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            S2GridProjector as DPS2GridProjector,
        )

        common = {
            "lmax": 2,
            "channels": 4,
            "mode": "self",
            "op_type": "glu",
            "precision": "float64",
            "layout": "ndfc",
            "grid_method": "lebedev",
        }
        with pytest.raises(NotImplementedError, match="lebedev_quadrature"):
            # e3nn product grid (lebedev_quadrature=False) is not ported
            DPS2GridProjector(lmax=2, precision="float64", grid_method="e3nn")
        with pytest.raises(NotImplementedError, match="lebedev_quadrature"):
            DPS2GridNet(**{**common, "grid_method": "e3nn"})
        with pytest.raises(NotImplementedError, match="grid_mlp"):
            # GridMLP (grid_mlp=True) is not ported
            DPS2GridNet(**{**common, "op_type": "mlp"})
        with pytest.raises(NotImplementedError, match="node_wise_s2"):
            # cross mode backs node_wise_s2/message_node_s2 only
            DPS2GridNet(**{**common, "mode": "cross"})
        with pytest.raises(NotImplementedError, match="residual_scale_init"):
            DPS2GridNet(**common, residual_scale_init=1e-3)

    def test_value_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            GridBranch as DPGridBranch,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            S2GridNet as DPS2GridNet,
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
            S2GridProjector as DPS2GridProjector,
        )

        common = {
            "lmax": 2,
            "channels": 4,
            "mode": "self",
            "op_type": "glu",
            "precision": "float64",
            "layout": "ndfc",
            "grid_method": "lebedev",
        }
        with pytest.raises(ValueError):  # unknown grid method
            DPS2GridProjector(lmax=2, grid_method="cartesian")
        with pytest.raises(ValueError):  # negative mmax
            DPS2GridProjector(lmax=2, mmax=-1, grid_method="lebedev")
        with pytest.raises(ValueError):  # mmax > lmax
            DPS2GridProjector(lmax=2, mmax=3, grid_method="lebedev")
        with pytest.raises(ValueError):  # bad coefficient layout
            DPS2GridProjector(
                lmax=2, grid_method="lebedev", coefficient_layout="l_major"
            )
        with pytest.raises(ValueError):  # non-packaged [precision, n_points]
            DPS2GridProjector(
                lmax=2, grid_method="lebedev", grid_resolution_list=[7, 10]
            )
        with pytest.raises(ValueError):  # wrong resolution list length
            DPS2GridProjector(lmax=2, grid_method="lebedev", grid_resolution_list=[7])
        with pytest.raises(ValueError):  # unknown mode
            DPS2GridNet(**{**common, "mode": "pair"})
        with pytest.raises(ValueError):  # unknown op_type
            DPS2GridNet(**{**common, "op_type": "attention"})
        with pytest.raises(ValueError):  # unknown layout
            DPS2GridNet(**{**common, "layout": "cdfn"})
        with pytest.raises(ValueError):  # flat layout is cross-only
            DPS2GridNet(**{**common, "layout": "flat"})
        with pytest.raises(ValueError):  # n_branches must be positive
            DPGridBranch(channels=4, n_branches=0, precision="float64")
        dp_net = DPS2GridNet(**common)
        rng = np.random.default_rng(2086)
        with pytest.raises(ValueError):  # wrong query channel count
            dp_net.call(rng.normal(size=(3, 9, 1, 5)))


def _build_so2_edge_data(
    rng,
    *,
    nloc,
    nnei,
    lmax,
    channels,
    masked="none",
    with_gate=False,
    n_radial=None,
):
    """Build matching pt (sparse) and dp (padded) edge caches.

    The dp cache uses the padded layout (E = nloc * nnei with ``edge_mask``);
    the pt cache keeps only the valid slots (flat sparse edges in the same
    row-major slot order pt's ``torch.nonzero`` would produce). Both sides
    share identical Wigner-D blocks built from the (parity-proven) dpmodel
    ``WignerDCalculator``. Invalid slots intentionally keep garbage (nonzero)
    envelope/feature values so a missing mask shows up as a parity failure.

    ``n_radial``: when not None, ``edge_rbf`` is filled with random values of
    width ``n_radial`` (garbage in masked slots too); otherwise it stays the
    zero (E, 1) placeholder used by consumers that ignore ``edge_rbf``.

    ``masked`` is one of:
    - ``"none"``: all slots valid;
    - ``"slots"``: a few scattered invalid slots;
    - ``"node"``: node 2 fully masked (no incoming edges) plus one extra slot.
    """
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
        build_edge_quaternion,
    )
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )

    n_edge = nloc * nnei
    dim_full = (lmax + 1) ** 2
    src = np.array(
        [(i + 1 + k) % nloc for i in range(nloc) for k in range(nnei)],
        dtype=np.int64,
    )
    dst = np.repeat(np.arange(nloc, dtype=np.int64), nnei)
    mask = np.ones(n_edge, dtype=np.float64)
    if masked == "slots":
        mask[3] = 0.0
        mask[nnei + 1] = 0.0
        mask[-1] = 0.0
    elif masked == "node":
        mask[2 * nnei : 3 * nnei] = 0.0  # node 2: no incoming edges at all
        mask[3] = 0.0
    elif masked != "none":
        raise ValueError(f"unknown masked mode {masked}")
    valid = mask > 0.5
    n_valid = int(valid.sum())

    edge_vec = rng.normal(size=(n_edge, 3))
    edge_vec /= np.linalg.norm(edge_vec, axis=-1, keepdims=True)
    quat = build_edge_quaternion(edge_vec)
    D_full, Dt_full = WignerDCalculator(lmax, precision="float64").call(quat)
    D_full = np.asarray(D_full)
    Dt_full = np.asarray(Dt_full)
    if n_radial is None:
        edge_rbf = np.zeros((n_edge, 1))
    else:
        edge_rbf = rng.normal(size=(n_edge, n_radial))
    edge_env = rng.uniform(0.2, 1.0, size=(n_edge, 1))
    deg = ((edge_env[:, 0] ** 2) * mask).reshape(nloc, nnei).sum(axis=1)
    inv_sqrt_deg = (1.0 / np.sqrt(deg + 1.0)).reshape(nloc, 1, 1)
    edge_src_gate = rng.uniform(0.1, 1.0, size=(n_edge, 1)) if with_gate else None
    radial = rng.normal(size=(n_edge, lmax + 1, channels))
    x = rng.normal(size=(nloc, dim_full, channels))

    t = torch.from_numpy
    pt_cache = EdgeFeatureCache(
        src=t(src[valid]),
        dst=t(dst[valid]),
        edge_type_feat=t(np.zeros((n_valid, channels))),
        edge_vec=t(edge_vec[valid]),
        edge_rbf=t(edge_rbf[valid]),
        edge_env=t(edge_env[valid]),
        deg=t(deg),
        inv_sqrt_deg=t(inv_sqrt_deg),
        D_full=t(D_full[valid]),
        Dt_full=t(Dt_full[valid]),
        edge_src_gate=None if edge_src_gate is None else t(edge_src_gate[valid]),
    )
    dp_cache = EdgeCache(
        src=src,
        dst=dst,
        edge_type_feat=np.zeros((n_edge, channels)),
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_src_gate=edge_src_gate,
        edge_mask=mask,
    )
    return pt_cache, dp_cache, radial, radial[valid], x, valid


class TestSO2Parity:
    nloc = 5
    nnei = 4

    def _perturb(self, pt_mod: torch.nn.Module, seed: int) -> None:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    # ---------- SO2Linear ----------
    @pytest.mark.parametrize(
        "lmax,mmax", [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)]
    )  # degree/order truncations (mmax=0 covers the empty weight_m branch)
    @pytest.mark.parametrize("mlp_bias", [False, True])  # l=0 bias branch
    @pytest.mark.parametrize("n_focus", [1, 2])  # focus streams
    def test_so2_linear(self, lmax, mmax, mlp_bias, n_focus) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPSO2Linear
        from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Linear as PTSO2Linear

        pt_mod = PTSO2Linear(
            lmax=lmax,
            mmax=mmax,
            in_channels=5,
            out_channels=3,
            n_focus=n_focus,
            dtype=torch.float64,
            mlp_bias=mlp_bias,
            seed=11,
            trainable=True,
        )
        self._perturb(pt_mod, 2052)
        serialized = pt_mod.serialize()
        dp_mod = DPSO2Linear.deserialize(serialized)
        rng = np.random.default_rng(2053)
        x = rng.normal(size=(13, n_focus, dp_mod.reduced_dim, 5))
        assert_parity(dp_mod.call(x), pt_mod(torch.from_numpy(x)))

    def test_so2_linear_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPSO2Linear

        dp_mod = DPSO2Linear(
            lmax=3,
            mmax=1,
            in_channels=4,
            out_channels=4,
            n_focus=2,
            precision="float64",
            mlp_bias=True,
            seed=4,
        )
        dp_mod2 = DPSO2Linear.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2054)
        x = rng.normal(size=(9, 2, dp_mod.reduced_dim, 4))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_so2_linear_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPSO2Linear

        with pytest.raises(ValueError):  # mmax > lmax
            DPSO2Linear(lmax=2, mmax=3, in_channels=2, out_channels=2)
        with pytest.raises(ValueError):  # negative mmax
            DPSO2Linear(lmax=2, mmax=-1, in_channels=2, out_channels=2)
        with pytest.raises(ValueError):  # wrong class tag
            DPSO2Linear.deserialize({"@class": "NotSO2Linear", "@version": 1})

    # ---------- DynamicRadialDegreeMixer ----------
    @pytest.mark.parametrize(
        "mode,rank",
        [
            ("degree", 0),  # channel-shared degree kernel
            ("degree_channel", 0),  # full per-channel kernel
            ("degree_channel", 1),  # low-rank factorization (core)
            ("degree_channel", 2),  # low-rank, rank > 1
        ],
    )
    def test_radial_degree_mixer(self, mode, rank) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
            DynamicRadialDegreeMixer as DPMixer,
        )
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            DynamicRadialDegreeMixer as PTMixer,
        )

        pt_mod = PTMixer(
            lmax=3,
            mmax=1,
            channels=4,
            mode=mode,
            rank=rank,
            dtype=torch.float64,
            seed=5,
            trainable=True,
        )
        self._perturb(pt_mod, 2055)
        dp_mod = DPMixer(
            lmax=3,
            mmax=1,
            channels=4,
            mode=mode,
            rank=rank,
            precision="float64",
            seed=5,
        )
        # pt has no standalone serialize(); load the pt state_dict fragment
        dp_mod._load_variables(pt_state_to_numpy(pt_mod))
        rng = np.random.default_rng(2056)
        x_local = rng.normal(size=(17, dp_mod.reduced_dim, 4))
        radial = rng.normal(size=(17, dp_mod.reduced_dim, 4))
        assert_parity(
            dp_mod.call(x_local, radial),
            pt_mod(torch.from_numpy(x_local), torch.from_numpy(radial)),
        )

    def test_radial_degree_mixer_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
            DynamicRadialDegreeMixer as DPMixer,
        )

        dp_mod = DPMixer(
            lmax=3,
            mmax=1,
            channels=4,
            mode="degree_channel",
            rank=1,
            precision="float64",
            seed=6,
        )
        dp_mod2 = DPMixer.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2057)
        x_local = rng.normal(size=(7, dp_mod.reduced_dim, 4))
        radial = rng.normal(size=(7, dp_mod.reduced_dim, 4))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x_local, radial)),
            np.asarray(dp_mod2.call(x_local, radial)),
        )

    def test_radial_degree_mixer_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
            DynamicRadialDegreeMixer as DPMixer,
        )

        common = {"lmax": 2, "mmax": 1, "channels": 4, "precision": "float64"}
        with pytest.raises(ValueError):  # unknown mode
            DPMixer(mode="channel", **common)
        with pytest.raises(ValueError):  # negative rank
            DPMixer(mode="degree_channel", rank=-1, **common)
        with pytest.raises(ValueError):  # non-positive channels
            DPMixer(lmax=2, mmax=1, channels=0, mode="degree")
        with pytest.raises(ValueError):  # mmax > lmax
            DPMixer(lmax=2, mmax=3, channels=4, mode="degree")
        dp_mod = DPMixer(mode="degree", **common)
        rng = np.random.default_rng(2058)
        good = rng.normal(size=(3, dp_mod.reduced_dim, 4))
        with pytest.raises(ValueError):  # shape mismatch between inputs
            dp_mod.call(good, good[:, :, :2])
        with pytest.raises(ValueError):  # incompatible reduced layout
            dp_mod.call(good[:, :3, :], good[:, :3, :])
        with pytest.raises(ValueError):  # wrong class tag
            DPMixer.deserialize({"@class": "NotMixer", "@version": 1})

    # ---------- segment softmax ----------
    @pytest.mark.parametrize(
        "masked", ["none", "slots", "node"]
    )  # padded-slot patterns (node = one all-masked destination)
    @pytest.mark.parametrize("use_src_weight", [False, True])  # SFPG gate branch
    def test_segment_envelope_gated_softmax(self, masked, use_src_weight) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.attention import (
            segment_envelope_gated_softmax as dp_softmax,
        )
        from deepmd.pt.model.descriptor.sezm_nn.attention import (
            segment_envelope_gated_softmax as pt_softmax,
        )

        rng = np.random.default_rng(2059)
        nloc, nnei, n_focus, n_head = self.nloc, self.nnei, 2, 3
        pt_cache, dp_cache, _, _, _, valid = _build_so2_edge_data(
            rng,
            nloc=nloc,
            nnei=nnei,
            lmax=2,
            channels=4,
            masked=masked,
            with_gate=use_src_weight,
        )
        n_edge = nloc * nnei
        logits = rng.normal(size=(n_edge, n_focus, n_head))
        # mixed signs exercise both stable-softplus branches for zeta
        z_bias_raw = rng.normal(size=(n_focus, n_head))
        alpha_dp = dp_softmax(
            logits=logits,
            edge_env=dp_cache.edge_env,
            n_nodes=nloc,
            z_bias_raw=z_bias_raw,
            eps=1e-7,
            src_weight=dp_cache.edge_src_gate,
            edge_mask=dp_cache.edge_mask,
        )
        alpha_pt = pt_softmax(
            logits=torch.from_numpy(logits[valid]),
            edge_env=pt_cache.edge_env,
            dst=pt_cache.dst,
            n_nodes=nloc,
            z_bias_raw=torch.from_numpy(z_bias_raw),
            eps=1e-7,
            src_weight=pt_cache.edge_src_gate,
        )
        alpha_dp = np.asarray(alpha_dp)
        np.testing.assert_allclose(
            alpha_dp[valid],
            alpha_pt.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )
        # invalid slots must produce exactly zero attention weights
        np.testing.assert_array_equal(alpha_dp[~valid], 0.0)
        assert np.all(np.isfinite(alpha_dp))

    def test_segment_softmax_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.attention import (
            segment_envelope_gated_softmax as dp_softmax,
        )

        rng = np.random.default_rng(2062)
        with pytest.raises(ValueError):  # E not a multiple of n_nodes
            dp_softmax(
                logits=rng.normal(size=(7, 1, 1)),
                edge_env=rng.uniform(size=(7, 1)),
                n_nodes=3,
                z_bias_raw=np.zeros((1, 1)),
                eps=1e-7,
            )

    # ---------- SO2Convolution ----------
    def _conv_kwargs(self, **overrides):
        kwargs = {
            "lmax": 3,
            "mmax": 1,
            "kmax": 1,
            "channels": 4,
            "n_focus": 1,
            "focus_dim": 0,
            "focus_compete": True,
            "so2_norm": False,
            "so2_layers": 2,
            "so2_attn_res": "none",
            "layer_scale": False,
            "n_atten_head": 1,
            "radial_so2_mode": "degree_channel",
            "radial_so2_rank": 1,
            "lebedev_quadrature": True,
            "activation_function": "silu",
            "mlp_bias": False,
            "eps": 1e-7,
        }
        kwargs.update(overrides)
        return kwargs

    def _build_conv_pair(self, seed=17, perturb_seed=2060, **overrides):
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as DPSO2Conv
        from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution as PTSO2Conv

        kwargs = self._conv_kwargs(**overrides)
        pt_mod = PTSO2Conv(**kwargs, dtype=torch.float64, seed=seed, trainable=True)
        # post_focus_mix is zero-initialized; perturb so the output is nonzero
        self._perturb(pt_mod, perturb_seed)
        dp_mod = DPSO2Conv.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod, kwargs

    def _assert_conv_parity(
        self, pt_mod, dp_mod, kwargs, *, masked="slots", with_gate=False
    ) -> None:
        rng = np.random.default_rng(2061)
        pt_cache, dp_cache, radial, radial_valid, x, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=kwargs["lmax"],
            channels=kwargs["channels"],
            masked=masked,
            with_gate=with_gate,
        )
        out_dp = dp_mod.call(x, dp_cache, radial)
        out_pt = pt_mod(torch.from_numpy(x), pt_cache, torch.from_numpy(radial_valid))
        assert_parity(out_dp, out_pt)

    @pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
    @pytest.mark.parametrize("so2_layers", [2, 4])  # SO(2) layer loop depth (core=4)
    def test_so2_convolution(self, masked, so2_layers) -> None:
        pt_mod, dp_mod, kwargs = self._build_conv_pair(so2_layers=so2_layers)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)

    def test_so2_convolution_all_masked_node(self) -> None:
        # one destination with zero valid incoming edges
        pt_mod, dp_mod, kwargs = self._build_conv_pair()
        self._assert_conv_parity(pt_mod, dp_mod, kwargs, masked="node")

    @pytest.mark.parametrize(
        "radial_so2_mode,radial_so2_rank",
        [
            ("none", 0),  # elementwise radial modulation
            ("degree", 0),  # channel-shared dynamic degree kernel
            ("degree_channel", 0),  # full per-channel dynamic kernel
        ],
    )
    def test_so2_convolution_radial_modes(
        self, radial_so2_mode, radial_so2_rank
    ) -> None:
        pt_mod, dp_mod, kwargs = self._build_conv_pair(
            radial_so2_mode=radial_so2_mode, radial_so2_rank=radial_so2_rank
        )
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize(
        "n_atten_head", [0, 2]
    )  # 0 = plain envelope sum, 2 = multi-head attention
    def test_so2_convolution_atten_heads(self, n_atten_head) -> None:
        pt_mod, dp_mod, kwargs = self._build_conv_pair(n_atten_head=n_atten_head)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize("focus_compete", [False, True])  # competition branch
    def test_so2_convolution_multi_focus(self, focus_compete) -> None:
        # n_focus=2 also activates the hidden-width ChannelLinear projection
        pt_mod, dp_mod, kwargs = self._build_conv_pair(
            n_focus=2, focus_compete=focus_compete
        )
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    def test_so2_convolution_so2_norm(self) -> None:
        pt_mod, dp_mod, kwargs = self._build_conv_pair(so2_norm=True, so2_layers=3)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    def test_so2_convolution_mlp_bias(self) -> None:
        # exercises bias0 + the layer-0 envelope bias correction
        pt_mod, dp_mod, kwargs = self._build_conv_pair(mlp_bias=True)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize("n_atten_head", [0, 1])  # gate enters both paths
    def test_so2_convolution_src_gate(self, n_atten_head) -> None:
        pt_mod, dp_mod, kwargs = self._build_conv_pair(n_atten_head=n_atten_head)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs, with_gate=True)

    def test_so2_convolution_real_edge_cache(self) -> None:
        # end-to-end: REAL pt build_edge_cache vs REAL dp build_edge_cache
        # feeding the same weight-copied SO2Convolution (no synthetic cache)
        pt_mod, dp_mod, kwargs = self._build_conv_pair()
        rng = np.random.default_rng(2096)
        nf, nloc, nall, nnei = 1, self.nloc, self.nloc + 3, self.nnei
        inputs = _build_real_edge_inputs(
            rng,
            nf=nf,
            nloc=nloc,
            nall=nall,
            nnei=nnei,
            channels=kwargs["channels"],
        )
        pt_cache, dp_cache = _build_real_edge_caches(inputs, lmax=kwargs["lmax"])
        valid = inputs["valid"]
        dim_full = (kwargs["lmax"] + 1) ** 2
        radial = rng.normal(size=(nf * nloc * nnei, kwargs["lmax"] + 1, 4))
        x = rng.normal(size=(nf * nloc, dim_full, kwargs["channels"]))
        out_dp = dp_mod.call(x, dp_cache, radial)
        out_pt = pt_mod(torch.from_numpy(x), pt_cache, torch.from_numpy(radial[valid]))
        assert_parity(out_dp, out_pt)

    def test_so2_convolution_full_mmax(self) -> None:
        # mmax == lmax: rotate_inv_rescale is all ones
        pt_mod, dp_mod, kwargs = self._build_conv_pair(lmax=2, mmax=2)
        self._assert_conv_parity(pt_mod, dp_mod, kwargs)

    def test_so2_convolution_roundtrip(self) -> None:
        _, dp_mod, kwargs = self._build_conv_pair(
            n_focus=2, so2_norm=True, mlp_bias=True
        )
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as DPSO2Conv

        dp_mod2 = DPSO2Conv.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2063)
        _, dp_cache, radial, _, x, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=kwargs["lmax"],
            channels=kwargs["channels"],
            masked="slots",
        )
        out1 = np.asarray(dp_mod.call(x, dp_cache, radial))
        # the D_to_m projections are cached in the EdgeCache dicts; reuse is exact
        out2 = np.asarray(dp_mod2.call(x, dp_cache, radial))
        np.testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("so2_attn_res", "independent"),  # DepthAttnRes
            ("so2_attn_res", "dependent"),  # DepthAttnRes
            ("layer_scale", True),  # per-layer LayerScale
            ("n_atten_head", -1),  # ValueError, not NotImplementedError
            ("atten_f_mix", True),  # focus-merged attention
            ("atten_v_proj", True),  # value projection
            ("atten_o_proj", True),  # output projection
            ("s2_activation", True),  # S2-grid SwiGLU non-linearity
            ("node_wise_s2", True),  # edge-local S2 grid product
            ("node_wise_so3", True),  # edge-local SO(3) grid product
            ("message_node_s2", True),  # post-aggregation S2 grid product
            ("message_node_so3", True),  # post-aggregation SO(3) grid product
        ],
    )
    def test_so2_convolution_guards(self, flag, value) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as DPSO2Conv

        kwargs = self._conv_kwargs(**{flag: value})
        if flag == "n_atten_head":
            with pytest.raises(ValueError):
                DPSO2Conv(**kwargs, precision="float64")
        else:
            with pytest.raises(NotImplementedError, match=flag):
                DPSO2Conv(**kwargs, precision="float64")

    def test_so2_convolution_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as DPSO2Conv

        with pytest.raises(ValueError):  # head count must divide focus width
            DPSO2Conv(**self._conv_kwargs(n_atten_head=3), precision="float64")
        with pytest.raises(ValueError):  # so2_layers must be >= 1
            DPSO2Conv(**self._conv_kwargs(so2_layers=0), precision="float64")
        with pytest.raises(ValueError):  # n_focus must be >= 1
            DPSO2Conv(**self._conv_kwargs(n_focus=0), precision="float64")
        with pytest.raises(ValueError):  # unknown radial mode
            DPSO2Conv(
                **self._conv_kwargs(radial_so2_mode="degree_rank"),
                precision="float64",
            )
        with pytest.raises(ValueError):  # mmax > lmax
            DPSO2Conv(**self._conv_kwargs(mmax=4), precision="float64")
        with pytest.raises(ValueError):  # unknown so2_attn_res token
            DPSO2Conv(**self._conv_kwargs(so2_attn_res="depth"), precision="float64")
        dp_mod = DPSO2Conv(**self._conv_kwargs(), precision="float64", seed=1)
        rng = np.random.default_rng(2064)
        _, dp_cache, radial, _, x, _ = _build_so2_edge_data(
            rng, nloc=self.nloc, nnei=self.nnei, lmax=3, channels=4
        )
        with pytest.raises(ValueError):  # E not a multiple of N
            dp_mod.call(x[:3], dp_cache, radial)
        with pytest.raises(ValueError):  # wrong class tag
            DPSO2Conv.deserialize({"@class": "NotConv", "@version": 1})


class TestEmbeddingParity:
    nloc = 5
    nnei = 4

    def _perturb(self, pt_mod: torch.nn.Module, seed: int) -> None:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    # ---------- SeZMTypeEmbedding ----------
    @pytest.mark.parametrize("padding", [False, True])  # zero padding row branch
    def test_type_embedding(self, padding) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            SeZMTypeEmbedding as DPTypeEmbed,
        )
        from deepmd.pt.model.descriptor.sezm_nn.embedding import (
            SeZMTypeEmbedding as PTTypeEmbed,
        )

        ntypes, embed_dim = 4, 6
        pt_mod = PTTypeEmbed(
            ntypes=ntypes,
            embed_dim=embed_dim,
            dtype=torch.float64,
            seed=21,
            trainable=True,
            padding=padding,
        )
        dp_mod = DPTypeEmbed(
            ntypes=ntypes,
            embed_dim=embed_dim,
            precision="float64",
            seed=21,
            padding=padding,
        )
        state = pt_state_to_numpy(pt_mod)
        # pt has no serialize(); the @variables key set must equal the pt
        # state_dict key set so the weights map one-to-one.
        assert set(dp_mod.serialize()["@variables"]) == set(state)
        assert state["adam_type_embedding"].shape == dp_mod.adam_type_embedding.shape
        dp_mod.adam_type_embedding = state["adam_type_embedding"]
        rng = np.random.default_rng(2070)
        # include the padding row index ntypes when padding=True
        atype = rng.integers(0, ntypes + 1 if padding else ntypes, size=(3, 5))
        assert_parity(dp_mod.call(atype), pt_mod(torch.from_numpy(atype)))
        if padding:
            pad_out = np.asarray(dp_mod.call(np.full((2,), ntypes, dtype=np.int64)))
            np.testing.assert_array_equal(pad_out, 0.0)

    @pytest.mark.parametrize("padding", [False, True])  # zero padding row branch
    def test_type_embedding_roundtrip(self, padding) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            SeZMTypeEmbedding as DPTypeEmbed,
        )

        dp_mod = DPTypeEmbed(
            ntypes=3, embed_dim=4, precision="float64", seed=22, padding=padding
        )
        dp_mod2 = DPTypeEmbed.deserialize(dp_mod.serialize())
        atype = np.array([0, 2, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(atype)), np.asarray(dp_mod2.call(atype))
        )

    def test_type_embedding_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            SeZMTypeEmbedding as DPTypeEmbed,
        )

        with pytest.raises(ValueError):  # non-positive ntypes
            DPTypeEmbed(ntypes=0, embed_dim=4)
        with pytest.raises(ValueError):  # non-positive embed_dim
            DPTypeEmbed(ntypes=3, embed_dim=0)
        with pytest.raises(ValueError):  # wrong class tag
            DPTypeEmbed.deserialize({"@class": "NotTypeEmbed", "@version": 1})
        dp_mod = DPTypeEmbed(ntypes=3, embed_dim=4, precision="float64")
        data = dp_mod.serialize()
        data["@variables"]["adam_type_embedding"] = np.zeros((2, 4))
        with pytest.raises(ValueError):  # table shape mismatch
            DPTypeEmbed.deserialize(data)

    # ---------- GeometricInitialEmbedding ----------
    def _build_gie_pair(self, lmax, channels):
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            GeometricInitialEmbedding as DPGIE,
        )
        from deepmd.pt.model.descriptor.sezm_nn.embedding import (
            GeometricInitialEmbedding as PTGIE,
        )

        pt_mod = PTGIE(lmax=lmax, channels=channels, dtype=torch.float64)
        # pt serialize() is config-only; the dp module is weight-free.
        dp_mod = DPGIE.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod

    @pytest.mark.parametrize(
        "masked", ["none", "slots", "node"]
    )  # padded-slot patterns (node = one all-masked destination)
    @pytest.mark.parametrize(
        "zonal_provided", [False, True]
    )  # zonal_coupling: None (gather from Dt_full) vs provided-zonal input
    #    path with D_node == D_cache only
    @pytest.mark.parametrize("with_gate", [False, True])  # SFPG gate branch
    def test_gie(self, masked, zonal_provided, with_gate) -> None:
        lmax, channels = 2, 4
        pt_mod, dp_mod = self._build_gie_pair(lmax, channels)
        rng = np.random.default_rng(2071)
        pt_cache, dp_cache, radial, radial_valid, _, valid = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=lmax,
            channels=channels,
            masked=masked,
            with_gate=with_gate,
        )
        if zonal_provided:
            # Scope: provided-zonal here uses D_node == D_cache; the genuine
            # lmax_node > lmax_mp (dim_full != ebed_dim) path is exercised by
            # the descriptor-level tests in a later task.
            rows = dp_mod.non_scalar_row_index
            cols = dp_mod.zonal_m0_col_index_for_row
            dp_zonal = np.asarray(dp_cache.Dt_full)[:, rows, cols]
            pt_zonal = pt_cache.Dt_full[
                :, torch.from_numpy(rows), torch.from_numpy(cols)
            ]
        else:
            dp_zonal = pt_zonal = None
        out_dp = dp_mod.call(
            n_nodes=self.nloc,
            edge_cache=dp_cache,
            radial_feat=radial[:, 1:, :],
            zonal_coupling=dp_zonal,
        )
        out_pt = pt_mod(
            n_nodes=self.nloc,
            edge_cache=pt_cache,
            radial_feat=torch.from_numpy(radial_valid[:, 1:, :]),
            zonal_coupling=pt_zonal,
        )
        assert_parity(out_dp, out_pt)
        # l=0 row must be exactly zero (comes from type embedding instead)
        np.testing.assert_array_equal(np.asarray(out_dp)[:, 0, :], 0.0)

    def test_gie_lmax0(self) -> None:
        # lmax=0 short-circuit: all-zero (N, 1, C) on both sides
        pt_mod, dp_mod = self._build_gie_pair(0, 3)
        rng = np.random.default_rng(2072)
        pt_cache, dp_cache, _, _, _, _ = _build_so2_edge_data(
            rng, nloc=self.nloc, nnei=self.nnei, lmax=1, channels=3
        )
        out_dp = np.asarray(
            dp_mod.call(n_nodes=self.nloc, edge_cache=dp_cache, radial_feat=None)
        )
        out_pt = pt_mod(n_nodes=self.nloc, edge_cache=pt_cache, radial_feat=None)
        assert out_dp.shape == (self.nloc, 1, 3)
        np.testing.assert_array_equal(out_dp, out_pt.detach().cpu().numpy())
        np.testing.assert_array_equal(out_dp, 0.0)

    def test_gie_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            GeometricInitialEmbedding as DPGIE,
        )

        dp_mod = DPGIE(lmax=3, channels=4, precision="float64")
        dp_mod2 = DPGIE.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2073)
        _, dp_cache, radial, _, _, _ = _build_so2_edge_data(
            rng, nloc=self.nloc, nnei=self.nnei, lmax=3, channels=4, masked="slots"
        )
        out1 = dp_mod.call(
            n_nodes=self.nloc, edge_cache=dp_cache, radial_feat=radial[:, 1:, :]
        )
        out2 = dp_mod2.call(
            n_nodes=self.nloc, edge_cache=dp_cache, radial_feat=radial[:, 1:, :]
        )
        np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    def test_gie_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            GeometricInitialEmbedding as DPGIE,
        )

        with pytest.raises(ValueError):  # wrong class tag
            DPGIE.deserialize({"@class": "NotGIE", "@version": 1})
        dp_mod = DPGIE(lmax=2, channels=4, precision="float64")
        rng = np.random.default_rng(2074)
        _, dp_cache, radial, _, _, _ = _build_so2_edge_data(
            rng, nloc=self.nloc, nnei=self.nnei, lmax=2, channels=4
        )
        with pytest.raises(ValueError):  # E not a multiple of N
            dp_mod.call(n_nodes=3, edge_cache=dp_cache, radial_feat=radial[:, 1:, :])

    # ---------- EnvironmentInitialEmbedding ----------
    n_radial = 5
    ntypes = 3

    def _env_kwargs(self, **overrides):
        kwargs = {
            "ntypes": self.ntypes,
            "n_radial": self.n_radial,
            "channels": 4,
            "embed_dim": 12,
            "axis_dim": 3,
            "type_dim": 4,
            "hidden_dim": 8,
            "mlp_bias": False,
            "activation_function": "silu",
            "eps": 1e-7,
        }
        kwargs.update(overrides)
        return kwargs

    def _build_env_pair(self, seed=23, perturb_seed=2075, **overrides):
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            EnvironmentInitialEmbedding as DPEnv,
        )
        from deepmd.pt.model.descriptor.sezm_nn.embedding import (
            EnvironmentInitialEmbedding as PTEnv,
        )

        kwargs = self._env_kwargs(**overrides)
        pt_mod = PTEnv(**kwargs, dtype=torch.float64, seed=seed, trainable=True)
        # output_proj is zero-initialized; perturb so the output is nonzero
        self._perturb(pt_mod, perturb_seed)
        dp_mod = DPEnv.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod

    def _assert_env_parity(
        self, pt_mod, dp_mod, *, masked="slots", with_gate=False
    ) -> None:
        rng = np.random.default_rng(2076)
        pt_cache, dp_cache, _, _, _, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=1,
            channels=4,
            masked=masked,
            with_gate=with_gate,
            n_radial=self.n_radial,
        )
        atype = rng.integers(0, self.ntypes, size=(self.nloc,))
        out_dp = dp_mod.call(edge_cache=dp_cache, atype_flat=atype, n_nodes=self.nloc)
        out_pt = pt_mod(
            edge_cache=pt_cache,
            atype_flat=torch.from_numpy(atype),
            n_nodes=self.nloc,
        )
        assert_parity(out_dp, out_pt)

    @pytest.mark.parametrize(
        "masked", ["none", "slots", "node"]
    )  # padded-slot patterns (node = one all-masked destination)
    @pytest.mark.parametrize("mlp_bias", [False, True])  # MLP bias branch
    def test_env_embedding(self, masked, mlp_bias) -> None:
        pt_mod, dp_mod = self._build_env_pair(mlp_bias=mlp_bias)
        self._assert_env_parity(pt_mod, dp_mod, masked=masked)

    def test_env_embedding_src_gate(self) -> None:
        pt_mod, dp_mod = self._build_env_pair()
        self._assert_env_parity(pt_mod, dp_mod, with_gate=True)

    def test_env_embedding_wide_rbf(self) -> None:
        # embed_dim - 2*type_dim > 32 exercises the non-clamped rbf_out_dim
        pt_mod, dp_mod = self._build_env_pair(embed_dim=42, axis_dim=3)
        self._assert_env_parity(pt_mod, dp_mod)

    @pytest.mark.parametrize("mlp_bias", [False, True])  # MLP bias branch
    def test_env_embedding_serialize_keys(self, mlp_bias) -> None:
        pt_mod, dp_mod = self._build_env_pair(mlp_bias=mlp_bias)
        assert set(dp_mod.serialize()["@variables"]) == set(pt_mod.state_dict())

    def test_env_embedding_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            EnvironmentInitialEmbedding as DPEnv,
        )

        _, dp_mod = self._build_env_pair(mlp_bias=True)
        dp_mod2 = DPEnv.deserialize(dp_mod.serialize())
        rng = np.random.default_rng(2077)
        _, dp_cache, _, _, _, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=1,
            channels=4,
            masked="slots",
            n_radial=self.n_radial,
        )
        atype = rng.integers(0, self.ntypes, size=(self.nloc,))
        out1 = dp_mod.call(edge_cache=dp_cache, atype_flat=atype, n_nodes=self.nloc)
        out2 = dp_mod2.call(edge_cache=dp_cache, atype_flat=atype, n_nodes=self.nloc)
        np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    def test_env_embedding_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
            EnvironmentInitialEmbedding as DPEnv,
        )

        with pytest.raises(ValueError):  # axis_dim must be < embed_dim
            DPEnv(**self._env_kwargs(axis_dim=12), precision="float64")
        with pytest.raises(ValueError):  # wrong class tag
            DPEnv.deserialize({"@class": "NotEnv", "@version": 1})
        dp_mod = DPEnv(**self._env_kwargs(), precision="float64", seed=2)
        data = dp_mod.serialize()
        data["@variables"].pop("output_proj.matrix")
        with pytest.raises(ValueError):  # variable key set mismatch
            DPEnv.deserialize(data)
        data = dp_mod.serialize()
        data["@variables"]["output_proj.matrix"] = np.zeros((2, 2))
        with pytest.raises(ValueError):  # variable shape mismatch
            DPEnv.deserialize(data)
        rng = np.random.default_rng(2078)
        _, dp_cache, _, _, _, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=1,
            channels=4,
            n_radial=self.n_radial,
        )
        atype = rng.integers(0, self.ntypes, size=(self.nloc,))
        with pytest.raises(ValueError):  # E not a multiple of N
            dp_mod.call(edge_cache=dp_cache, atype_flat=atype, n_nodes=3)


def _build_real_edge_inputs(
    rng,
    *,
    nf,
    nloc,
    nall,
    nnei,
    channels,
    local_nlist=False,
):
    """Build numpy inputs for a real ``build_edge_cache`` run.

    Includes -1 padding slots, ghosts (``nall > nloc``) mapping back to their
    owners, one broken mapping entry (``mapping == -1``, exercising pt's
    ``src_ok`` drop), and a non-empty type-pair exclusion ``(0, 1)``.
    When ``local_nlist`` is True, neighbor indices are drawn directly from the
    local range and ``mapping`` is ``None`` (pt's mapping-free branch).
    """
    ntypes = 3
    coord = rng.uniform(0.0, 4.0, size=(nf, nall, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    if local_nlist:
        mapping = None
        atype_ext = atype
        hi = nloc
    else:
        n_ghost = nall - nloc
        mapping = np.concatenate(
            [
                np.tile(np.arange(nloc, dtype=np.int64), (nf, 1)),
                rng.integers(0, nloc, size=(nf, n_ghost)),
            ],
            axis=1,
        )
        mapping[:, -1] = -1  # broken ghost: pt drops via src_ok, dp masks
        atype_ext = np.take_along_axis(atype, np.clip(mapping, 0, nloc - 1), axis=1)
        hi = nall
    # neighbors over the extended axis, excluding the center itself
    nlist = rng.integers(0, hi, size=(nf, nloc, nnei))
    center = np.arange(nloc)[None, :, None]
    nlist = np.where(nlist == center, (nlist + 1) % hi, nlist)
    nlist[rng.uniform(size=nlist.shape) < 0.25] = -1  # padding slots
    # pair_keep_mask from exclude pair (0, 1), built on extended types
    nl_safe = np.where(nlist >= 0, nlist, 0)
    nb_type = np.take_along_axis(atype_ext, nl_safe.reshape(nf, -1), axis=1).reshape(
        nf, nloc, nnei
    )
    ct = atype[:, :, None]
    pair_keep_mask = ~(((ct == 0) & (nb_type == 1)) | ((ct == 1) & (nb_type == 0)))
    type_ebed = rng.normal(size=(nf * nloc, channels))
    # expected validity mask, computed independently in numpy
    if local_nlist:
        src_local = nl_safe
    else:
        src_local = np.take_along_axis(
            mapping, nl_safe.reshape(nf, -1), axis=1
        ).reshape(nf, nloc, nnei)
    valid = (
        (nlist >= 0) & pair_keep_mask & (src_local >= 0) & (src_local < nloc)
    ).reshape(-1)
    return {
        "coord": coord,
        "nlist": nlist,
        "mapping": mapping,
        "pair_keep_mask": pair_keep_mask,
        "type_ebed": type_ebed,
        "valid": valid,
    }


def _build_real_edge_caches(
    inputs,
    *,
    lmax,
    rcut=6.0,
    n_radial=8,
    deg_norm_floor=1e-12,
    eps=1e-7,
    random_gamma=False,
    gamma=None,
    seed=2090,
):
    """Run the REAL pt and dp ``build_edge_cache`` on identical inputs.

    The pt ``RadialBasis`` frequencies are perturbed and weight-copied into
    the dp side via ``deserialize`` so parity exercises copied weights.
    Returns ``(pt_cache, dp_cache)``.
    """
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        build_edge_cache as dp_build_edge_cache,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
        C3CutoffEnvelope as DPEnvelope,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
        RadialBasis as DPRadialBasis,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator as DPWigner,
    )
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        build_edge_cache as pt_build_edge_cache,
    )
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        C3CutoffEnvelope as PTEnvelope,
    )
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        RadialBasis as PTRadialBasis,
    )
    from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
        WignerDCalculator as PTWigner,
    )

    pt_rb = PTRadialBasis(rcut=rcut, n_radial=n_radial, dtype=torch.float64)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        pt_rb.adam_freqs += torch.from_numpy(0.05 * rng.normal(size=(1, n_radial)))
    dp_rb = DPRadialBasis.deserialize(pt_rb.serialize())
    pt_env = PTEnvelope(rcut=rcut, dtype=torch.float64)
    dp_env = DPEnvelope(rcut=rcut, precision="float64")
    pt_wig = PTWigner(lmax, dtype=torch.float64)
    dp_wig = DPWigner(lmax, precision="float64")

    t = torch.from_numpy
    mapping = inputs["mapping"]
    pt_cache = pt_build_edge_cache(
        type_ebed=t(inputs["type_ebed"]),
        extended_coord=t(inputs["coord"]),
        nlist=t(inputs["nlist"]),
        mapping=None if mapping is None else t(mapping),
        pair_keep_mask=t(inputs["pair_keep_mask"]),
        eps=eps,
        deg_norm_floor=deg_norm_floor,
        edge_envelope=pt_env,
        radial_basis=pt_rb,
        n_radial=n_radial,
        random_gamma=random_gamma,
        wigner_calc=pt_wig,
    )
    dp_cache = dp_build_edge_cache(
        type_ebed=inputs["type_ebed"],
        extended_coord=inputs["coord"],
        nlist=inputs["nlist"],
        mapping=mapping,
        pair_keep_mask=inputs["pair_keep_mask"],
        eps=eps,
        deg_norm_floor=deg_norm_floor,
        edge_envelope=dp_env,
        radial_basis=dp_rb,
        n_radial=n_radial,
        random_gamma=random_gamma,
        wigner_calc=dp_wig,
        gamma=gamma,
    )
    return pt_cache, dp_cache


class TestEdgeCacheParity:
    nf = 2
    nloc = 6
    nall = 10
    nnei = 12
    channels = 4
    lmax = 2

    def _inputs(self, seed=2086, **overrides):
        rng = np.random.default_rng(seed)
        kwargs = {
            "nf": self.nf,
            "nloc": self.nloc,
            "nall": self.nall,
            "nnei": self.nnei,
            "channels": self.channels,
        }
        kwargs.update(overrides)
        return _build_real_edge_inputs(rng, **kwargs)

    @pytest.mark.parametrize("local_nlist", [False, True])  # mapping None branch
    @pytest.mark.parametrize(
        "deg_norm_floor", [1e-12, 1.0]
    )  # legacy-eps floor vs O(1) floor
    def test_real_build_parity(self, local_nlist, deg_norm_floor) -> None:
        inputs = self._inputs(
            local_nlist=local_nlist,
            nall=self.nloc if local_nlist else self.nall,
        )
        valid = inputs["valid"]
        pt_cache, dp_cache = _build_real_edge_caches(
            inputs, lmax=self.lmax, deg_norm_floor=deg_norm_floor
        )
        # the dp validity mask matches the independently computed mask, and
        # pt's sparse edges occupy exactly those slots (in row-major order)
        np.testing.assert_array_equal(np.asarray(dp_cache.edge_mask), valid)
        assert pt_cache.src.shape[0] == int(valid.sum())
        # padded dst contract: node-contiguous repeat of arange(nf * nloc)
        np.testing.assert_array_equal(
            np.asarray(dp_cache.dst),
            np.repeat(np.arange(self.nf * self.nloc), self.nnei),
        )
        # indices on valid slots are exactly pt's sparse indices
        np.testing.assert_array_equal(
            np.asarray(dp_cache.src)[valid], pt_cache.src.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(dp_cache.dst)[valid], pt_cache.dst.numpy()
        )
        # per-edge fields: compare masked entries against pt's sparse outputs
        for name in (
            "edge_vec",
            "edge_rbf",
            "edge_env",
            "edge_type_feat",
            "edge_quat",
            "D_full",
            "Dt_full",
        ):
            assert_parity(
                np.asarray(getattr(dp_cache, name))[valid], getattr(pt_cache, name)
            )
        # node-level normalization compares directly
        assert_parity(dp_cache.deg, pt_cache.deg)
        assert_parity(dp_cache.inv_sqrt_deg, pt_cache.inv_sqrt_deg)
        # everything is finite, including masked slots
        for name in (
            "edge_vec",
            "edge_rbf",
            "edge_env",
            "edge_type_feat",
            "edge_quat",
            "D_full",
            "Dt_full",
            "deg",
            "inv_sqrt_deg",
        ):
            assert np.isfinite(np.asarray(getattr(dp_cache, name))).all(), name
        # standard path carries no source gate
        assert dp_cache.edge_src_gate is None
        assert dp_cache.D_to_m_cache == {}
        assert dp_cache.Dt_from_m_cache == {}

    def test_masked_edge_inertness(self) -> None:
        # an extra all-(-1) neighbor column must not change the masked-view
        # fields or the degree normalization
        inputs = self._inputs()
        pad = -np.ones((self.nf, self.nloc, 1), dtype=inputs["nlist"].dtype)
        inputs2 = dict(inputs)
        inputs2["nlist"] = np.concatenate([inputs["nlist"], pad], axis=-1)
        inputs2["pair_keep_mask"] = np.concatenate(
            [
                inputs["pair_keep_mask"],
                np.ones((self.nf, self.nloc, 1), dtype=bool),
            ],
            axis=-1,
        )
        _, cache = _build_real_edge_caches(inputs, lmax=self.lmax)
        _, cache2 = _build_real_edge_caches(inputs2, lmax=self.lmax)
        n_nodes = self.nf * self.nloc
        mask = np.asarray(cache.edge_mask).reshape(n_nodes, self.nnei)
        mask2 = np.asarray(cache2.edge_mask).reshape(n_nodes, self.nnei + 1)
        np.testing.assert_array_equal(mask2[:, : self.nnei], mask)
        np.testing.assert_array_equal(mask2[:, self.nnei], False)
        for name in ("edge_vec", "edge_rbf", "edge_env", "edge_quat", "D_full"):
            a = np.asarray(getattr(cache, name))
            b = np.asarray(getattr(cache2, name))
            a = a.reshape(n_nodes, self.nnei, -1)[mask.astype(bool)]
            b = b.reshape(n_nodes, self.nnei + 1, -1)[mask2.astype(bool)]
            np.testing.assert_array_equal(a, b, err_msg=name)
        np.testing.assert_array_equal(np.asarray(cache.deg), np.asarray(cache2.deg))
        np.testing.assert_array_equal(
            np.asarray(cache.inv_sqrt_deg), np.asarray(cache2.inv_sqrt_deg)
        )

    def test_random_gamma(self) -> None:
        # pt draws gamma internally with torch.rand, so the draw cannot be
        # injected identically into both sides; the dp branch is verified by
        # determinism (injected gamma) and gauge properties instead.
        inputs = self._inputs()
        n_edge = self.nf * self.nloc * self.nnei
        gamma = np.random.default_rng(7).uniform(0.0, 2.0 * np.pi, n_edge)
        _, base = _build_real_edge_caches(inputs, lmax=self.lmax)
        _, c1 = _build_real_edge_caches(
            inputs, lmax=self.lmax, random_gamma=True, gamma=gamma
        )
        _, c2 = _build_real_edge_caches(
            inputs, lmax=self.lmax, random_gamma=True, gamma=gamma
        )
        # determinism with an injected gamma
        np.testing.assert_array_equal(np.asarray(c1.D_full), np.asarray(c2.D_full))
        np.testing.assert_array_equal(
            np.asarray(c1.edge_quat), np.asarray(c2.edge_quat)
        )
        # the roll is a gauge choice: D stays orthogonal ...
        d = np.asarray(c1.D_full)
        dt = np.asarray(c1.Dt_full)
        eye = np.broadcast_to(np.eye(d.shape[-1]), d.shape)
        np.testing.assert_allclose(d @ dt, eye, rtol=1e-12, atol=1e-12)
        # ... the l=0 block is unchanged ...
        np.testing.assert_allclose(
            d[:, 0, 0], np.asarray(base.D_full)[:, 0, 0], rtol=1e-12, atol=1e-14
        )
        # ... and the rotation-independent fields are bit-identical
        for name in ("edge_vec", "edge_env", "edge_rbf", "deg", "inv_sqrt_deg"):
            np.testing.assert_array_equal(
                np.asarray(getattr(c1, name)),
                np.asarray(getattr(base, name)),
                err_msg=name,
            )
        # internal-draw branch (gamma=None) runs and stays finite
        _, c3 = _build_real_edge_caches(inputs, lmax=self.lmax, random_gamma=True)
        assert np.isfinite(np.asarray(c3.D_full)).all()
        np.testing.assert_allclose(
            np.asarray(c3.D_full)[:, 0, 0],
            np.asarray(base.D_full)[:, 0, 0],
            rtol=1e-12,
            atol=1e-14,
        )


class TestFFNParity:
    n_node = 11
    channels = 8

    def _perturb(self, pt_mod: torch.nn.Module, seed: int) -> None:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    def _ffn_kwargs(self, **overrides):
        kwargs = {
            "lmax": 3,
            "channels": self.channels,
            "hidden_channels": self.channels,
            "kmax": 1,
            "grid_mlp": False,
            "grid_branch": 0,
            "s2_activation": False,
            "ffn_so3_grid": False,
            "lebedev_quadrature": True,
            "activation_function": "silu",
            "glu_activation": True,
            "mlp_bias": False,
        }
        kwargs.update(overrides)
        return kwargs

    def _build_ffn_pair(self, seed=29, perturb_seed=2110, **overrides):
        from deepmd.dpmodel.descriptor.dpa4_nn.ffn import EquivariantFFN as DPFFN
        from deepmd.pt.model.descriptor.sezm_nn.ffn import EquivariantFFN as PTFFN

        kwargs = self._ffn_kwargs(**overrides)
        pt_mod = PTFFN(**kwargs, dtype=torch.float64, seed=seed, trainable=True)
        # so3_linear_2 is zero-initialized; perturb so the output is nonzero
        self._perturb(pt_mod, perturb_seed)
        dp_mod = DPFFN.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod, kwargs

    def _assert_ffn_parity(self, pt_mod, dp_mod, kwargs, seed=2111) -> None:
        rng = np.random.default_rng(seed)
        dim = (kwargs["lmax"] + 1) ** 2
        x = rng.normal(size=(self.n_node, dim, 1, kwargs["channels"]))
        out_dp = dp_mod.call(x)
        out_pt = pt_mod(torch.from_numpy(x))
        assert out_dp.shape == tuple(out_pt.shape)
        assert_parity(out_dp, out_pt)

    @pytest.mark.parametrize("lmax", [2, 3])  # degree truncation (core=3)
    @pytest.mark.parametrize("s2_activation", [False, True])  # S2 grid path (core=True)
    @pytest.mark.parametrize("glu_activation", [False, True])  # GLU gating (core=True)
    def test_ffn(self, lmax, s2_activation, glu_activation) -> None:
        pt_mod, dp_mod, kwargs = self._build_ffn_pair(
            lmax=lmax, s2_activation=s2_activation, glu_activation=glu_activation
        )
        self._assert_ffn_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize("grid_branch", [0, 1])  # branch mixer off/on (core=1)
    def test_ffn_grid_branch(self, grid_branch) -> None:
        pt_mod, dp_mod, kwargs = self._build_ffn_pair(
            s2_activation=True, grid_branch=grid_branch
        )
        self._assert_ffn_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize("mlp_bias", [False, True])  # l=0 / gate bias branch
    def test_ffn_mlp_bias(self, mlp_bias) -> None:
        pt_mod, dp_mod, kwargs = self._build_ffn_pair(mlp_bias=mlp_bias)
        self._assert_ffn_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize("s2_activation", [False, True])  # both act sub-modules
    def test_ffn_roundtrip(self, s2_activation) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.ffn import EquivariantFFN as DPFFN

        pt_mod, dp_mod, kwargs = self._build_ffn_pair(
            s2_activation=s2_activation, grid_branch=1 if s2_activation else 0
        )
        data = dp_mod.serialize()
        # exact pt state_dict key-set match
        assert set(data["@variables"]) == set(pt_state_to_numpy(pt_mod))
        dp_mod2 = DPFFN.deserialize(data)
        rng = np.random.default_rng(2112)
        dim = (kwargs["lmax"] + 1) ** 2
        x = rng.normal(size=(self.n_node, dim, 1, kwargs["channels"]))
        np.testing.assert_array_equal(
            np.asarray(dp_mod.call(x)), np.asarray(dp_mod2.call(x))
        )

    def test_ffn_guards(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.ffn import EquivariantFFN as DPFFN

        with pytest.raises(NotImplementedError, match="ffn_so3_grid"):
            DPFFN(**self._ffn_kwargs(ffn_so3_grid=True), precision="float64")
        # grid_mlp guard is delegated to S2GridNet's op_type='mlp' NIE
        with pytest.raises(NotImplementedError, match="mlp"):
            DPFFN(
                **self._ffn_kwargs(s2_activation=True, grid_mlp=True),
                precision="float64",
            )

    def test_ffn_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.ffn import EquivariantFFN as DPFFN

        with pytest.raises(ValueError):  # kmax must be non-negative
            DPFFN(**self._ffn_kwargs(kmax=-1), precision="float64")
        with pytest.raises(ValueError):  # grid_branch must be non-negative
            DPFFN(**self._ffn_kwargs(grid_branch=-1), precision="float64")
        with pytest.raises(ValueError):  # wrong class tag
            DPFFN.deserialize({"@class": "NotFFN", "@version": 1})
        dp_mod = DPFFN(**self._ffn_kwargs(), precision="float64", seed=3)
        with pytest.raises(KeyError):  # missing sub-module variables
            dp_mod._load_variables({"so3_linear_1.weight": dp_mod.so3_linear_1.weight})
        with pytest.raises(KeyError):  # unknown variables rejected
            dp_mod._load_variables({**dp_mod._variables(), "extra.weight": 0.0})


class TestBlockParity:
    nloc = 5
    nnei = 4
    channels = 4

    def _perturb(self, pt_mod: torch.nn.Module, seed: int) -> None:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    def _block_kwargs(self, **overrides):
        # core DPA4 config (sandwich_norm=[F,T,T,F], so2 s2 off, ffn s2 on)
        kwargs = {
            "lmax": 3,
            "node_lmax": None,
            "mmax": 1,
            "kmax": 1,
            "channels": self.channels,
            "n_focus": 1,
            "focus_dim": 0,
            "focus_compete": True,
            "so2_norm": False,
            "so2_layers": 4,
            "so2_attn_res": "none",
            "radial_so2_mode": "degree_channel",
            "radial_so2_rank": 1,
            "n_atten_head": 1,
            "so2_pre_norm": False,
            "so2_post_norm": True,
            "ffn_pre_norm": True,
            "ffn_post_norm": False,
            "ffn_neurons": self.channels,
            "ffn_grid_branch": 1,
            "ffn_blocks": 1,
            "ffn_s2_activation": True,
            "so2_lebedev_quadrature": True,
            "ffn_lebedev_quadrature": True,
            "so2_activation_function": "silu",
            "ffn_activation_function": "silu",
            "ffn_glu_activation": True,
            "mlp_bias": False,
            "eps": 1e-7,
        }
        kwargs.update(overrides)
        return kwargs

    def _build_block_pair(self, seed=31, perturb_seed=2120, **overrides):
        from deepmd.dpmodel.descriptor.dpa4_nn.block import (
            SeZMInteractionBlock as DPBlock,
        )
        from deepmd.pt.model.descriptor.sezm_nn.block import (
            SeZMInteractionBlock as PTBlock,
        )

        kwargs = self._block_kwargs(**overrides)
        pt_mod = PTBlock(**kwargs, dtype=torch.float64, seed=seed, trainable=True)
        # zero-initialized residual projections; perturb so the output is nonzero
        self._perturb(pt_mod, perturb_seed)
        dp_mod = DPBlock.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod, kwargs

    def _node_dim(self, kwargs):
        node_lmax = kwargs["node_lmax"]
        if node_lmax is None:
            node_lmax = kwargs["lmax"]
        return (node_lmax + 1) ** 2

    def _assert_block_parity(self, pt_mod, dp_mod, kwargs, *, masked="slots") -> None:
        rng = np.random.default_rng(2121)
        pt_cache, dp_cache, radial, radial_valid, _, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=kwargs["lmax"],
            channels=kwargs["channels"],
            masked=masked,
        )
        node_dim = self._node_dim(kwargs)
        x = rng.normal(size=(self.nloc, node_dim, 1, kwargs["channels"]))
        out_dp = dp_mod.call(x, dp_cache, radial)
        out_pt = pt_mod(torch.from_numpy(x), pt_cache, torch.from_numpy(radial_valid))
        assert out_dp[1:] == (None, None, None)
        assert out_pt[1] is None and out_pt[2] is None and out_pt[3] is None
        assert_parity(out_dp[0], out_pt[0])

    @pytest.mark.parametrize("so2_layers", [2, 4])  # SO(2) layer depth (core=4)
    def test_block(self, so2_layers) -> None:
        pt_mod, dp_mod, kwargs = self._build_block_pair(so2_layers=so2_layers)
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    @pytest.mark.parametrize(
        "sandwich",
        [
            (False, True, True, False),  # core [so2_pre, so2_post, ffn_pre, ffn_post]
            (True, False, False, True),  # flips every norm flag's branch
        ],
    )
    def test_block_sandwich_norm(self, sandwich) -> None:
        so2_pre, so2_post, ffn_pre, ffn_post = sandwich
        pt_mod, dp_mod, kwargs = self._build_block_pair(
            so2_pre_norm=so2_pre,
            so2_post_norm=so2_post,
            ffn_pre_norm=ffn_pre,
            ffn_post_norm=ffn_post,
        )
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    def test_block_ffn_blocks(self) -> None:
        # multiple FFN subblocks exercise the per-subblock loop and seeds
        pt_mod, dp_mod, kwargs = self._build_block_pair(ffn_blocks=2)
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    def test_block_node_lmax(self) -> None:
        # node_lmax > lmax: SO(2) acts on the truncated slice, zero-pads above
        pt_mod, dp_mod, kwargs = self._build_block_pair(lmax=2, node_lmax=3, mmax=1)
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    def test_block_mlp_bias(self) -> None:
        pt_mod, dp_mod, kwargs = self._build_block_pair(mlp_bias=True)
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    def test_block_plain_ffn_act(self) -> None:
        # ffn_s2_activation=False: FFN uses the GatedActivation path
        pt_mod, dp_mod, kwargs = self._build_block_pair(
            ffn_s2_activation=False, ffn_grid_branch=0
        )
        self._assert_block_parity(pt_mod, dp_mod, kwargs)

    def test_block_real_edge_cache(self) -> None:
        # end-to-end: REAL pt build_edge_cache vs REAL dp build_edge_cache
        # feeding the same weight-copied block (no synthetic cache)
        pt_mod, dp_mod, kwargs = self._build_block_pair()
        rng = np.random.default_rng(2122)
        nf, nloc, nall, nnei = 1, self.nloc, self.nloc + 3, self.nnei
        inputs = _build_real_edge_inputs(
            rng,
            nf=nf,
            nloc=nloc,
            nall=nall,
            nnei=nnei,
            channels=kwargs["channels"],
        )
        pt_cache, dp_cache = _build_real_edge_caches(inputs, lmax=kwargs["lmax"])
        valid = inputs["valid"]
        node_dim = self._node_dim(kwargs)
        radial = rng.normal(
            size=(nf * nloc * nnei, kwargs["lmax"] + 1, kwargs["channels"])
        )
        x = rng.normal(size=(nf * nloc, node_dim, 1, kwargs["channels"]))
        out_dp = dp_mod.call(x, dp_cache, radial)
        out_pt = pt_mod(torch.from_numpy(x), pt_cache, torch.from_numpy(radial[valid]))
        assert_parity(out_dp[0], out_pt[0])
        # masked-slot garbage is inert end-to-end: scribble into invalid slots
        # (finite O(10) garbage: the padded layout computes exp() on masked
        # attention logits before zero-weighting them, so the garbage must
        # not overflow exp; see attention.segment_envelope_gated_softmax)
        radial2 = radial.copy()
        radial2[~valid] = 10.0
        out_dp2 = dp_mod.call(x, dp_cache, radial2)
        np.testing.assert_array_equal(np.asarray(out_dp[0]), np.asarray(out_dp2[0]))

    def test_block_roundtrip(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.block import (
            SeZMInteractionBlock as DPBlock,
        )

        pt_mod, dp_mod, kwargs = self._build_block_pair(ffn_blocks=2)
        data = dp_mod.serialize()
        # exact pt state_dict key-set match
        assert set(data["@variables"]) == set(pt_state_to_numpy(pt_mod))
        dp_mod2 = DPBlock.deserialize(data)
        rng = np.random.default_rng(2123)
        _, dp_cache, radial, _, _, _ = _build_so2_edge_data(
            rng,
            nloc=self.nloc,
            nnei=self.nnei,
            lmax=kwargs["lmax"],
            channels=kwargs["channels"],
            masked="slots",
        )
        x = rng.normal(size=(self.nloc, self._node_dim(kwargs), 1, self.channels))
        out1 = np.asarray(dp_mod.call(x, dp_cache, radial)[0])
        out2 = np.asarray(dp_mod2.call(x, dp_cache, radial)[0])
        np.testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("full_attn_res", "independent"),  # block-level DepthAttnRes
            ("full_attn_res", "dependent"),  # block-level DepthAttnRes
            ("block_attn_res", "independent"),  # block-level DepthAttnRes
            ("block_attn_res", "dependent"),  # block-level DepthAttnRes
            ("layer_scale", True),  # block-level FFN LayerScale
            ("so2_s2_activation", True),  # delegated to SO2Convolution
            ("node_wise_s2", True),  # delegated to SO2Convolution
            ("message_node_so3", True),  # delegated to SO2Convolution
            ("ffn_so3_grid", True),  # delegated to EquivariantFFN
        ],
    )
    def test_block_guards(self, flag, value) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.block import (
            SeZMInteractionBlock as DPBlock,
        )

        match = "s2_activation" if flag == "so2_s2_activation" else flag
        with pytest.raises(NotImplementedError, match=match):
            DPBlock(**self._block_kwargs(**{flag: value}), precision="float64")

    def test_block_errors(self) -> None:
        from deepmd.dpmodel.descriptor.dpa4_nn.block import (
            SeZMInteractionBlock as DPBlock,
        )

        with pytest.raises(ValueError):  # node_lmax must be >= lmax
            DPBlock(**self._block_kwargs(node_lmax=2), precision="float64")
        with pytest.raises(ValueError):  # mmax must be <= lmax
            DPBlock(**self._block_kwargs(mmax=4), precision="float64")
        with pytest.raises(ValueError):  # ffn_blocks must be >= 1
            DPBlock(**self._block_kwargs(ffn_blocks=0), precision="float64")
        with pytest.raises(ValueError):  # unknown full_attn_res token
            DPBlock(**self._block_kwargs(full_attn_res="depth"), precision="float64")
        with pytest.raises(ValueError):  # unknown block_attn_res token
            DPBlock(**self._block_kwargs(block_attn_res="depth"), precision="float64")
        with pytest.raises(ValueError):  # negative grid branch count
            DPBlock(**self._block_kwargs(ffn_grid_branch=-1), precision="float64")
        with pytest.raises(ValueError):  # wrong class tag
            DPBlock.deserialize({"@class": "NotBlock", "@version": 1})


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
            "deepmd.dpmodel.descriptor.dpa4_nn.wignerd, "
            "deepmd.dpmodel.descriptor.dpa4_nn.projection, "
            "deepmd.dpmodel.descriptor.dpa4_nn.grid_net, "
            "deepmd.dpmodel.descriptor.dpa4_nn.so2, "
            "deepmd.dpmodel.descriptor.dpa4_nn.attention, "
            "deepmd.dpmodel.descriptor.dpa4_nn.edge_cache, "
            "deepmd.dpmodel.descriptor.dpa4_nn.embedding, "
            "deepmd.dpmodel.descriptor.dpa4_nn.ffn, "
            "deepmd.dpmodel.descriptor.dpa4_nn.block; "
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
