# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model import (
    DPAtomicModel,
    DPModel,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    FittingNet,
    NativeLayer,
    NativeNet,
    NetworkCollection,
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
    inter2phys,
    load_dp_model,
    save_dp_model,
    to_face_distance,
)


class TestNativeLayer(unittest.TestCase):
    def test_serialize_deserize(self):
        for (
            ni,
            no,
        ), bias, ut, activation_function, resnet, ashp, prec in itertools.product(
            [(5, 5), (5, 10), (5, 9), (9, 5)],
            [True, False],
            [True, False],
            ["tanh", "none"],
            [True, False],
            [None, [4], [3, 2]],
            ["float32", "float64", "single", "double"],
        ):
            nl0 = NativeLayer(
                ni,
                no,
                bias=bias,
                use_timestep=ut,
                activation_function=activation_function,
                resnet=resnet,
                precision=prec,
            )
            nl1 = NativeLayer.deserialize(nl0.serialize())
            inp_shap = [ni]
            if ashp is not None:
                inp_shap = ashp + inp_shap
            inp = np.arange(np.prod(inp_shap)).reshape(inp_shap)
            np.testing.assert_allclose(nl0.call(inp), nl1.call(inp))

    def test_shape_error(self):
        self.w0 = np.full((2, 3), 3.0)
        self.b0 = np.full((2,), 4.0)
        self.b1 = np.full((3,), 4.0)
        self.idt0 = np.full((2,), 4.0)
        with self.assertRaises(ValueError) as context:
            network = NativeLayer.deserialize(
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w0, "b": self.b0},
                }
            )
            assert "not equalt to shape of b" in context.exception
        with self.assertRaises(ValueError) as context:
            network = NativeLayer.deserialize(
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w0, "b": self.b1, "idt": self.idt0},
                }
            )
            assert "not equalt to shape of idt" in context.exception


class TestNativeNet(unittest.TestCase):
    def setUp(self) -> None:
        self.w0 = np.full((2, 3), 3.0)
        self.b0 = np.full((3,), 4.0)
        self.w1 = np.full((3, 4), 3.0)
        self.b1 = np.full((4,), 4.0)

    def test_serialize(self):
        network = NativeNet(
            [
                NativeLayer(2, 3).serialize(),
                NativeLayer(3, 4).serialize(),
            ]
        )
        network[1]["w"] = self.w1
        network[1]["b"] = self.b1
        network[0]["w"] = self.w0
        network[0]["b"] = self.b0
        network[1]["activation_function"] = "tanh"
        network[0]["activation_function"] = "tanh"
        network[1]["resnet"] = True
        network[0]["resnet"] = True
        jdata = network.serialize()
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["w"], self.w0)
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["b"], self.b0)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["w"], self.w1)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["b"], self.b1)
        np.testing.assert_array_equal(jdata["layers"][0]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][1]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][0]["resnet"], True)
        np.testing.assert_array_equal(jdata["layers"][1]["resnet"], True)

    def test_deserialize(self):
        network = NativeNet.deserialize(
            {
                "layers": [
                    {
                        "activation_function": "tanh",
                        "resnet": True,
                        "@variables": {"w": self.w0, "b": self.b0},
                    },
                    {
                        "activation_function": "tanh",
                        "resnet": True,
                        "@variables": {"w": self.w1, "b": self.b1},
                    },
                ],
            }
        )
        np.testing.assert_array_equal(network[0]["w"], self.w0)
        np.testing.assert_array_equal(network[0]["b"], self.b0)
        np.testing.assert_array_equal(network[1]["w"], self.w1)
        np.testing.assert_array_equal(network[1]["b"], self.b1)
        np.testing.assert_array_equal(network[0]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[1]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[0]["resnet"], True)
        np.testing.assert_array_equal(network[1]["resnet"], True)

    def test_shape_error(self):
        with self.assertRaises(ValueError) as context:
            network = NativeNet.deserialize(
                {
                    "layers": [
                        {
                            "activation_function": "tanh",
                            "resnet": True,
                            "@variables": {"w": self.w0, "b": self.b0},
                        },
                        {
                            "activation_function": "tanh",
                            "resnet": True,
                            "@variables": {"w": self.w0, "b": self.b0},
                        },
                    ],
                }
            )
            assert "does not match the dim of layer" in context.exception


class TestEmbeddingNet(unittest.TestCase):
    def test_embedding_net(self):
        for ni, act, idt, prec in itertools.product(
            [1, 10],
            ["tanh", "none"],
            [True, False],
            ["double", "single"],
        ):
            en0 = EmbeddingNet(
                ni,
                activation_function=act,
                precision=prec,
                resnet_dt=idt,
            )
            en1 = EmbeddingNet.deserialize(en0.serialize())
            inp = np.ones([ni])
            np.testing.assert_allclose(en0.call(inp), en1.call(inp))


class TestFittingNet(unittest.TestCase):
    def test_fitting_net(self):
        for ni, no, act, idt, prec, bo in itertools.product(
            [1, 10],
            [1, 7],
            ["tanh", "none"],
            [True, False],
            ["double", "single"],
            [True, False],
        ):
            en0 = FittingNet(
                ni,
                no,
                activation_function=act,
                precision=prec,
                resnet_dt=idt,
                bias_out=bo,
            )
            en1 = FittingNet.deserialize(en0.serialize())
            inp = np.ones([ni])
            en0.call(inp)
            en1.call(inp)
            np.testing.assert_allclose(en0.call(inp), en1.call(inp))


class TestNetworkCollection(unittest.TestCase):
    def setUp(self) -> None:
        w0 = np.full((2, 3), 3.0)
        b0 = np.full((3,), 4.0)
        w1 = np.full((3, 4), 3.0)
        b1 = np.full((4,), 4.0)
        self.network = {
            "layers": [
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": w0, "b": b0},
                },
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": w1, "b": b1},
                },
            ],
        }

    def test_two_dim(self):
        networks = NetworkCollection(ndim=2, ntypes=2)
        networks[(0, 0)] = self.network
        networks[(1, 1)] = self.network
        networks[(0, 1)] = self.network
        with self.assertRaises(RuntimeError):
            networks.check_completeness()
        networks[(1, 0)] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[(0, 0)].serialize(), networks.serialize()["networks"][0]
        )

    def test_one_dim(self):
        networks = NetworkCollection(ndim=1, ntypes=2)
        networks[(0,)] = self.network
        with self.assertRaises(RuntimeError):
            networks.check_completeness()
        networks[(1,)] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[(0,)].serialize(), networks.serialize()["networks"][0]
        )

    def test_zero_dim(self):
        networks = NetworkCollection(ndim=0, ntypes=2)
        networks[()] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[()].serialize(), networks.serialize()["networks"][0]
        )


class TestSaveLoadDPModel(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((3, 2), 3.0)
        self.b = np.full((3,), 4.0)
        self.model_dict = {
            "type": "some_type",
            "layers": [
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w, "b": self.b},
                },
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w, "b": self.b},
                },
            ],
        }
        self.filename = "test_dp_dpmodel.dp"

    def test_save_load_model(self):
        save_dp_model(self.filename, deepcopy(self.model_dict))
        model = load_dp_model(self.filename)
        np.testing.assert_equal(model["model"], self.model_dict)
        assert "software" in model
        assert "version" in model

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.remove(self.filename)


class TestCaseSingleFrameWithNlist:
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 0.4
        self.rcut_smth = 2.2


class TestEnvMat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        em0 = EnvMat(self.rcut, self.rcut_smth)
        em1 = EnvMat.deserialize(em0.serialize())
        mm0, ww0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, davg, dstd)
        mm1, ww1 = em1.call(self.coord_ext, self.atype_ext, self.nlist, davg, dstd)
        np.testing.assert_allclose(mm0, mm1)
        np.testing.assert_allclose(ww0, ww1)


class TestDescrptSeA(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        em0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        em0.davg = davg
        em0.dstd = dstd
        em1 = DescrptSeA.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in [0, 1, 4]:
            np.testing.assert_allclose(mm0[ii], mm1[ii])


class TestInvarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            distinguish_types,
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
                distinguish_types=distinguish_types,
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

    def test_self_exception(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            distinguish_types,
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
                distinguish_types=distinguish_types,
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
                self.assertIn("input descriptor", context.exception)

            if nfp > 0:
                ifp = rng.normal(size=(self.nf, nfp - 1))
                with self.assertRaises(ValueError) as context:
                    ret0 = ifn0(dd[0], atype, fparam=ifp, aparam=iap)
                    self.assertIn("input fparam", context.exception)

            if nap > 0:
                iap = rng.normal(size=(self.nf, self.nloc, nap - 1))
                with self.assertRaises(ValueError) as context:
                    ret0 = ifn0(dd[0], atype, fparam=ifp, aparam=iap)
                    self.assertIn("input aparam", context.exception)

    def test_get_set(self):
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            3,
            1,
        )
        rng = np.random.default_rng()
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


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(ds, ft, type_map=type_map)
        md1 = DPAtomicModel.deserialize(md0.serialize())

        ret0 = md0.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = md1.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)

        np.testing.assert_allclose(ret0["energy"], ret1["energy"])


class TestDPModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPModel(ds, ft, type_map=type_map)
        md1 = DPModel.deserialize(md0.serialize())

        ret0 = md0.call_lower(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = md1.call_lower(self.coord_ext, self.atype_ext, self.nlist)

        np.testing.assert_allclose(ret0["energy"], ret1["energy"])
        np.testing.assert_allclose(ret0["energy_redu"], ret1["energy_redu"])


class TestDPModelFormatNlist(unittest.TestCase):
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 5
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
                [2.3, 0, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.expected_nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.atype_ext = np.array([0, 0, 1, 0, 1], dtype=int).reshape([1, self.nall])
        self.rcut_smth = 0.4
        self.rcut = 2.1

        nf, nloc, nnei = self.expected_nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        self.md = DPModel(ds, ft, type_map=type_map)

    def test_nlist_eq(self):
        # n_nnei == nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_st(self):
        # n_nnei < nnei
        nlist = np.array(
            [
                [1, 3, -1, 2],
                [0, -1, -1, 2],
                [0, 1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_lt(self):
        # n_nnei > nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1, -1, 4],
                [0, -1, 4, -1, -1, 2, -1, 3, -1],
                [0, 1, -1, -1, -1, 4, -1, -1, 3],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)


class TestRegion(unittest.TestCase):
    def setUp(self):
        self.cell = np.array(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]],
        )
        self.cell = np.reshape(self.cell, [1, 1, -1, 3])
        self.cell = np.tile(self.cell, [4, 5, 1, 1])
        self.prec = 1e-8

    def test_inter_to_phys(self):
        rng = np.random.default_rng()
        inter = rng.normal(size=[4, 5, 3, 3])
        phys = inter2phys(inter, self.cell)
        for ii in range(4):
            for jj in range(5):
                expected_phys = np.matmul(inter[ii, jj], self.cell[ii, jj])
                np.testing.assert_allclose(
                    phys[ii, jj], expected_phys, rtol=self.prec, atol=self.prec
                )

    def test_to_face_dist(self):
        cell0 = self.cell[0][0]
        vol = np.linalg.det(cell0)
        # area of surfaces xy, xz, yz
        sxy = np.linalg.norm(np.cross(cell0[0], cell0[1]))
        sxz = np.linalg.norm(np.cross(cell0[0], cell0[2]))
        syz = np.linalg.norm(np.cross(cell0[1], cell0[2]))
        # vol / area gives distance
        dz = vol / sxy
        dy = vol / sxz
        dx = vol / syz
        expected = np.array([dx, dy, dz])
        dists = to_face_distance(self.cell)
        for ii in range(4):
            for jj in range(5):
                np.testing.assert_allclose(
                    dists[ii][jj], expected, rtol=self.prec, atol=self.prec
                )


dtype = np.float64


class TestNeighList(unittest.TestCase):
    def setUp(self):
        self.nf = 3
        self.nloc = 2
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = np.array([[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype)
        self.icoord = np.array([[0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype)
        self.atype = np.array([0, 1], dtype=np.int32)
        [self.cell, self.icoord, self.atype] = [
            np.expand_dims(ii, 0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).reshape([-1, self.nloc * 3])
        self.cell = self.cell.reshape([-1, 9])
        [self.cell, self.coord, self.atype] = [
            np.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        self.ref_nlist = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            ]
        )

    def test_build_notype(self):
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        np.testing.assert_allclose(
            np.sort(nlist_loc, axis=-1),
            np.sort(self.ref_nlist, axis=-1),
        )

    def test_build_type(self):
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            self.nsel,
            distinguish_types=True,
        )
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            np.testing.assert_allclose(
                np.sort(np.split(nlist_loc, self.nsel, axis=-1)[ii], axis=-1),
                np.sort(np.split(self.ref_nlist, self.nsel, axis=-1)[ii], axis=-1),
            )

    def test_build_multiple_nlist(self):
        rcuts = [1.01, 2.01]
        nsels = [20, 80]
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, max(rcuts)
        )
        nlist1 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[1],
            nsels[1] - 1,
            distinguish_types=False,
        )
        pad = -1 * np.ones([self.nf, self.nloc, 1], dtype=nlist1.dtype)
        nlist2 = np.concatenate([nlist1, pad], axis=-1)
        nlist0 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[0],
            nsels[0],
            distinguish_types=False,
        )
        nlists = build_multiple_neighbor_list(ecoord, nlist1, rcuts, nsels)
        for dd in range(2):
            self.assertEqual(
                nlists[get_multiple_nlist_key(rcuts[dd], nsels[dd])].shape[-1],
                nsels[dd],
            )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])],
            nlist0,
        )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])],
            nlist2,
        )

    def test_extend_coord(self):
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # expected ncopy x nloc
        self.assertEqual(list(ecoord.shape), [self.nf, self.nall * 3])
        self.assertEqual(list(eatype.shape), [self.nf, self.nall])
        self.assertEqual(list(mapping.shape), [self.nf, self.nall])
        # check the nloc part is identical with original coord
        np.testing.assert_allclose(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.reshape([-1, self.ns, self.nloc, 3])
            - self.coord.reshape([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.reshape([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = np.matmul(
            shift_vec, np.linalg.inv(self.cell.reshape([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = np.round(shift_vec)
        # check: identical shift vecs
        np.testing.assert_allclose(
            shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = np.unique(shift_vec[0][:, 0], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([30, 30, 30, 30, 30], dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 1], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([30, 30, 30, 30, 30], dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 2], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-1, 0, 1], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([50, 50, 50], dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
