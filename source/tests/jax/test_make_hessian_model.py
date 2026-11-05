# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
)
from deepmd.jax.common import (
    to_jax_array,
)
from deepmd.jax.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.fitting.fitting import (
    EnergyFittingNet,
)
from deepmd.jax.model import (
    EnergyModel,
)

from ..seed import (
    GLOBAL_SEED,
)

dtype = jnp.float64


def finite_hessian(f, x, delta=1e-6):
    """Reuse identical oracle inputs within one finite-difference Hessian."""
    in_shape = x.shape
    assert len(in_shape) == 1
    values = {}

    def evaluate(point):
        """Evaluate one exact input without retaining caller-owned buffers."""
        key = (point.dtype.str, point.shape, point.tobytes())
        if key not in values:
            # A reference evaluator may return a reusable output buffer.
            values[key] = np.array(f(point), copy=True)
        return values[key]

    y0 = evaluate(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape + in_shape)
    for iidx in np.ndindex(*in_shape):
        for jidx in np.ndindex(*in_shape):
            i0 = np.zeros(in_shape)
            i1 = np.zeros(in_shape)
            i2 = np.zeros(in_shape)
            i3 = np.zeros(in_shape)
            i0[iidx] += delta
            i2[iidx] += delta
            i1[iidx] -= delta
            i3[iidx] -= delta
            i0[jidx] += delta
            i1[jidx] += delta
            i2[jidx] -= delta
            i3[jidx] -= delta
            y0 = evaluate(x + i0)
            y1 = evaluate(x + i1)
            y2 = evaluate(x + i2)
            y3 = evaluate(x + i3)
            res[(Ellipsis, *iidx, *jidx)] = (y0 + y3 - y1 - y2) / (4 * delta**2.0)
    return res


class TestFiniteHessian(unittest.TestCase):
    def test_reuses_inputs_without_sharing_between_calls(self):
        """Keep every Hessian entry while evaluating each input only once."""
        x = np.arange(9, dtype=np.float64) / 10
        calls = []
        buffer = np.empty(2)

        def oracle(point):
            """Return two quadratic outputs through a deliberately reused buffer."""
            calls.append(point.copy())
            buffer[:] = (point @ point, np.sum(point) ** 2)
            return buffer

        expected = np.stack((2 * np.eye(9), 2 * np.ones((9, 9))))
        for point in (x, x, x + 1):
            calls.clear()
            actual = finite_hessian(oracle, point, delta=1e-3)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
            self.assertEqual(len(calls), 2 * point.size**2 + 1)
            self.assertEqual(len({value.tobytes() for value in calls}), len(calls))


class HessianTest:
    def test(
        self,
    ) -> None:
        # setup test case
        places = 5
        delta = 1e-3
        natoms = self.nloc
        nf = self.nf
        nv = self.nv
        generator = jax.random.key(GLOBAL_SEED)
        cell0 = jax.random.uniform(generator, [3, 3], dtype=dtype)
        cell0 = 1.0 * (cell0 + cell0.T) + 5.0 * jnp.eye(3)
        cell1 = jax.random.uniform(generator, [3, 3], dtype=dtype)
        cell1 = 1.0 * (cell1 + cell1.T) + 5.0 * jnp.eye(3)
        cell = jnp.stack([cell0, cell1])
        coord = jax.random.uniform(generator, [nf, natoms, 3], dtype=dtype)
        coord = jnp.matmul(coord, cell)
        cell = cell.reshape([nf, 9])
        coord = coord.reshape([nf, natoms * 3])
        atype = jnp.stack(
            [
                jnp.asarray([0, 0, 1], dtype=jnp.int64),
                jnp.asarray([1, 0, 1], dtype=jnp.int64),
            ]
        ).reshape([nf, natoms])
        nfp, nap = 2, 3
        fparam = jax.random.uniform(generator, [nf, nfp], dtype=dtype)
        aparam = jax.random.uniform(generator, [nf, natoms * nap], dtype=dtype)
        # forward hess and value models
        ret_dict0 = self.model_hess(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )
        ret_dict1 = self.model_valu(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )
        # compare hess and value models
        np.testing.assert_allclose(ret_dict0["energy"], ret_dict1["energy"])
        ana_hess = ret_dict0["hessian"]

        # compute finite difference
        fnt_hess = []
        for ii in range(nf):

            def np_infer(
                xx,
            ):
                ret = self.model_valu(
                    to_jax_array(xx)[None, ...],
                    atype[ii][None, ...],
                    box=cell[ii][None, ...],
                    fparam=fparam[ii][None, ...],
                    aparam=aparam[ii][None, ...],
                )
                # detach
                ret = {kk: to_numpy_array(ret[kk]) for kk in ret}
                return ret

            def ff(xx):
                return np_infer(xx)["energy"]

            xx = to_numpy_array(coord[ii])
            fnt_hess.append(finite_hessian(ff, xx, delta=delta).squeeze())

        # compare finite difference with autodiff
        fnt_hess = np.stack(fnt_hess).reshape([nf, natoms * 3, natoms * 3])
        np.testing.assert_almost_equal(
            fnt_hess, to_numpy_array(ana_hess), decimal=places
        )


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestDPModel(unittest.TestCase, HessianTest):
    def setUp(self) -> None:
        jax.random.key(2)
        self.nf = 2
        self.nloc = 3
        self.rcut = 4.0
        self.rcut_smth = 3.0
        self.sel = [10, 10]
        self.nt = 2
        self.nv = 1
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=[2, 4, 8],
            axis_neuron=2,
        )
        ft0 = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            # self.nv,
            mixed_types=ds.mixed_types(),
            neuron=[4, 4, 4],
        )
        type_map = ["foo", "bar"]
        self.model_hess = EnergyModel(ds, ft0, type_map=type_map)
        self.model_hess.enable_hessian()
        self.model_valu = EnergyModel.deserialize(self.model_hess.serialize())

    def test_output_def(self) -> None:
        self.assertTrue(self.model_hess.atomic_output_def()["energy"].r_hessian)
        self.assertFalse(self.model_valu.atomic_output_def()["energy"].r_hessian)
        self.assertTrue(self.model_hess.model_output_def()["energy"].r_hessian)
        self.assertEqual(
            self.model_hess.model_output_def()["energy_derv_r_derv_r"].category,
            OutputVariableCategory.DERV_R_DERV_R,
        )

    def test_enable_hessian_is_idempotent(self) -> None:
        """Restored Hessian models may be enabled again by loss requirements."""
        self.model_valu.enable_hessian()
        enabled_def = self.model_valu.hess_fitting_def

        self.model_valu.enable_hessian()

        self.assertIs(self.model_valu.hess_fitting_def, enabled_def)
        self.assertTrue(self.model_valu.model_output_def()["energy"].r_hessian)
