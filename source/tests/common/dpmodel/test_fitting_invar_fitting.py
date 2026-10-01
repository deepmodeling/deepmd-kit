# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestInvarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            mixed_types,
            od,
            nfp,
            nap,
            et,
        ) in itertools.product(
            [True, False],
            [1, 2],
            [0, 3],
            [0, 4],
            [[], [0], [1]],
        ):
            ifn0 = InvarFitting(
                "energy",
                self.nt,
                ds.dim_out,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                exclude_types=et,
            )
            ifn1 = InvarFitting.deserialize(ifn0.serialize())
            if nfp > 0:
                ifp = rng.normal(size=(self.nf, nfp))
            else:
                ifp = None
            if nap > 0:
                iap = rng.normal(size=(self.nf, self.nloc, nap))
            else:
                iap = None
            ret0 = ifn0(dd[0], atype, fparam=ifp, aparam=iap)
            ret1 = ifn1(dd[0], atype, fparam=ifp, aparam=iap)
            np.testing.assert_allclose(ret0["energy"], ret1["energy"])
            sel_set = set(ifn0.get_sel_type())
            exclude_set = set(et)
            self.assertEqual(sel_set | exclude_set, set(range(self.nt)))
            self.assertEqual(sel_set & exclude_set, set())

    def test_mask(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]
        od = 2
        mixed_types = True
        # exclude type 1
        et = [1]
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            ds.dim_out,
            od,
            mixed_types=mixed_types,
            exclude_types=et,
        )
        ret0 = ifn0(dd[0], atype)
        # atom index 2 is of type 1 that is excluded
        zero_idx = 2
        np.testing.assert_allclose(
            ret0["energy"][0, zero_idx, :],
            np.zeros_like(ret0["energy"][0, zero_idx, :]),
        )
        zero_idx = 0
        np.testing.assert_allclose(
            ret0["energy"][1, zero_idx, :],
            np.zeros_like(ret0["energy"][1, zero_idx, :]),
        )

    def test_self_exception(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            mixed_types,
            od,
            nfp,
            nap,
        ) in itertools.product(
            [True, False],
            [1, 2],
            [0, 3],
            [0, 4],
        ):
            ifn0 = InvarFitting(
                "energy",
                self.nt,
                ds.dim_out,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
            )

            if nfp > 0:
                ifp = rng.normal(size=(self.nf, nfp))
            else:
                ifp = None
            if nap > 0:
                iap = rng.normal(size=(self.nf, self.nloc, nap))
            else:
                iap = None
            with self.assertRaises(ValueError) as context:
                ret0 = ifn0(dd[0][:, :, :-2], atype, fparam=ifp, aparam=iap)
            self.assertIn("input descriptor", str(context.exception))

            if nfp > 0:
                ifp = rng.normal(size=(self.nf, nfp - 1))
                with self.assertRaises(ValueError) as context:
                    ret0 = ifn0(dd[0], atype, fparam=ifp, aparam=iap)
                self.assertIn("input fparam", str(context.exception))

            if nap > 0:
                # restore correct ifp before testing aparam
                if nfp > 0:
                    ifp = rng.normal(size=(self.nf, nfp))
                iap = rng.normal(size=(self.nf, self.nloc, nap - 1))
                with self.assertRaises(ValueError) as context:
                    ifn0(dd[0], atype, fparam=ifp, aparam=iap)
                self.assertIn("input aparam", str(context.exception))

    def test_get_set(self) -> None:
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            3,
            1,
        )
        rng = np.random.default_rng(GLOBAL_SEED)
        foo = rng.normal([3, 4])
        for ii in [
            "bias_atom_e",
            "fparam_avg",
            "fparam_inv_std",
            "aparam_avg",
            "aparam_inv_std",
        ]:
            ifn0[ii] = foo
            np.testing.assert_allclose(foo, ifn0[ii])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_runtime_buffers_follow_torch_descriptor(self) -> None:
        """Portable fitting buffers must materialize on the active backend."""
        fitting = InvarFitting(
            "energy",
            ntypes=2,
            dim_descrpt=3,
            dim_out=1,
            neuron=[5, 5],
            numb_fparam=2,
            numb_aparam=1,
            dim_case_embd=2,
            default_fparam=[0.5, -0.25],
            precision="float64",
            mixed_types=True,
            seed=20260717,
        )
        fitting.bias_atom_e[:] = [[1.5], [-0.75]]
        fitting.fparam_avg[:] = [0.1, -0.2]
        fitting.fparam_inv_std[:] = [2.0, 0.5]
        fitting.aparam_avg[:] = [0.25]
        fitting.aparam_inv_std[:] = [4.0]
        fitting.case_embd[:] = [0.3, -0.6]

        descriptor = np.array(
            [[[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [-0.4, 0.7, 0.1]]],
            dtype=np.float64,
        )
        atype = np.array([[0, 1, 0]], dtype=np.int64)
        aparam = np.array([[[0.5], [0.0], [1.0]]], dtype=np.float64)
        expected = fitting(descriptor, atype, aparam=aparam)["energy"]

        # Backend wrappers eagerly convert these NumPy attributes and would
        # hide the generic dpmodel boundary, so convert only runtime inputs.
        result = fitting(
            torch.as_tensor(descriptor, device="cpu"),
            torch.as_tensor(atype, device="cpu"),
            aparam=torch.as_tensor(aparam, device="cpu"),
        )["energy"]

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device.type, "cpu")
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected)


VACUUM_CONDITIONING = [(0, 0), (2, 0), (0, 1), (2, 1)]


class TestVacuumRef(unittest.TestCase):
    """``vacuum_ref`` references every atom to the isolated atom of its type."""

    ntypes, nd, nf, nloc = 3, 8, 2, 5

    def setUp(self) -> None:
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.descriptor = self.rng.normal(size=(self.nf, self.nloc, self.nd))
        self.vacuum = self.rng.normal(size=(self.ntypes, self.nd))
        self.atype = self.rng.integers(0, self.ntypes, size=(self.nf, self.nloc))
        self.atype[0, : self.ntypes] = np.arange(self.ntypes)
        self.bias = self.rng.normal(size=(self.ntypes, 1))

    def build(self, vacuum_ref: bool, **kwargs) -> InvarFitting:
        ft = InvarFitting(
            "energy",
            self.ntypes,
            self.nd,
            1,
            neuron=[6, 6],
            bias_atom=self.bias,
            vacuum_ref=vacuum_ref,
            seed=GLOBAL_SEED,
            **kwargs,
        )
        if ft.dim_case_embd > 0:
            ft.set_case_embd(1)
        return ft

    def params(self, nfp: int, nap: int) -> dict:
        return {
            "fparam": self.rng.normal(size=(self.nf, nfp)) if nfp else None,
            "aparam": self.rng.normal(size=(self.nf, self.nloc, nap)) if nap else None,
        }

    def test_isolated_atom_gives_bias(self) -> None:
        for mixed_types, (nfp, nap), ncase, mask in itertools.product(
            [True, False], VACUUM_CONDITIONING, [0, 2], [False, True]
        ):
            ft = self.build(
                True,
                mixed_types=mixed_types,
                numb_fparam=nfp,
                numb_aparam=nap,
                dim_case_embd=ncase,
                use_aparam_as_mask=mask,
            )
            out = ft(
                self.vacuum[self.atype],
                self.atype,
                vacuum_descriptor=self.vacuum,
                **self.params(nfp, nap),
            )["energy"]
            np.testing.assert_allclose(
                out, self.bias[self.atype], rtol=1e-10, atol=1e-10
            )

    def test_matches_reference_subtraction(self) -> None:
        for mixed_types, (nfp, nap), ncase in itertools.product(
            [True, False], VACUUM_CONDITIONING, [0, 2]
        ):
            ft_ref = self.build(
                False,
                mixed_types=mixed_types,
                numb_fparam=nfp,
                numb_aparam=nap,
                dim_case_embd=ncase,
            )
            ft_vac = InvarFitting.deserialize(
                {**ft_ref.serialize(), "vacuum_ref": True}
            )
            params = self.params(nfp, nap)
            out = ft_vac(
                self.descriptor, self.atype, vacuum_descriptor=self.vacuum, **params
            )["energy"]
            expected = (
                ft_ref(self.descriptor, self.atype, **params)["energy"]
                - ft_ref(self.vacuum[self.atype], self.atype, **params)["energy"]
                + self.bias[self.atype]
            )
            np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    def test_vacuum_descriptor_required(self) -> None:
        ft = self.build(True)
        with self.assertRaises(ValueError):
            ft(self.descriptor, self.atype)
        with self.assertRaises(ValueError):
            ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum[:, :-1])

    def test_fold_vacuum_reference(self) -> None:
        for mixed_types, ncase in itertools.product([True, False], [0, 2]):
            ft = self.build(True, mixed_types=mixed_types, dim_case_embd=ncase)
            expected = ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum)
            ft.fold_vacuum_reference(self.vacuum)
            self.assertFalse(ft.vacuum_ref)
            np.testing.assert_allclose(
                ft(self.descriptor, self.atype)["energy"],
                expected["energy"],
                rtol=1e-10,
                atol=1e-10,
            )
        # a conditioned fitting stores the table and references from it; the
        # table is a deployment constant that serialization leaves out and a
        # type-map change drops
        ft = self.build(True, numb_aparam=1, type_map=["O", "H", "B"])
        aparam = self.rng.normal(size=(self.nf, self.nloc, 1))
        expected = ft(
            self.descriptor, self.atype, aparam=aparam, vacuum_descriptor=self.vacuum
        )
        ft.fold_vacuum_reference(self.vacuum)
        self.assertTrue(ft.vacuum_ref)
        self.assertFalse(ft.needs_vacuum_descriptor())
        np.testing.assert_allclose(
            ft(self.descriptor, self.atype, aparam=aparam)["energy"],
            expected["energy"],
            rtol=1e-10,
            atol=1e-10,
        )
        restored = InvarFitting.deserialize(ft.serialize())
        self.assertTrue(restored.needs_vacuum_descriptor())
        np.testing.assert_allclose(
            restored(
                self.descriptor,
                self.atype,
                aparam=aparam,
                vacuum_descriptor=self.vacuum,
            )["energy"],
            expected["energy"],
            rtol=1e-10,
            atol=1e-10,
        )
        ft.change_type_map(["B", "O", "H"])
        self.assertTrue(ft.needs_vacuum_descriptor())

    def test_serialization(self) -> None:
        data = self.build(True).serialize()
        self.assertTrue(data["vacuum_ref"])
        self.assertTrue(InvarFitting.deserialize(data).vacuum_ref)
        # a dictionary of the previous version carries no key
        older = {k: v for k, v in data.items() if k != "vacuum_ref"}
        self.assertFalse(InvarFitting.deserialize({**older, "@version": 4}).vacuum_ref)

    def test_atom_ener_is_exclusive(self) -> None:
        self.assertTrue(self.build(True, atom_ener=[None] * self.ntypes).vacuum_ref)
        with self.assertRaises(ValueError):
            self.build(True, atom_ener=[1.0] + [None] * (self.ntypes - 1))
