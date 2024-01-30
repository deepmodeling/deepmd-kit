# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.model_format import (
    DescrptSeA,
    EmbeddingNet,
    EnvMat,
    FittingNet,
    InvarFitting,
    NativeLayer,
    NativeNet,
    NetworkCollection,
    load_dp_model,
    save_dp_model,
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


class TestDPModel(unittest.TestCase):
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
        self.filename = "test_dp_model_format.dp"

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
                [0, 1, -1, -1, -1, 0, -1],
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
