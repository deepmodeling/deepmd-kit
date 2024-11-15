# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import pytest

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.nvnmd.data.data import (
    jdata_deepmd_input_v0,
    jdata_deepmd_input_v1,
)
from deepmd.tf.nvnmd.entrypoints.freeze import (
    save_weight,
)
from deepmd.tf.nvnmd.entrypoints.mapt import (
    MapTable,
)
from deepmd.tf.nvnmd.entrypoints.wrap import (
    wrap,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.nvnmd.utils.fio import (
    FioBin,
    FioNpyDic,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.train.trainer import (
    DPTrainer,
)
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)

from .common import (
    tests_path,
)


class TestNvnmdEntrypointsV0(tf.test.TestCase):
    @pytest.mark.run(order=0)
    def test_mapt_cnn_v0(self) -> None:
        config_file = str(tests_path / "nvnmd" / "ref" / "config_v0_cnn.npy")
        weight_file = str(tests_path / "nvnmd" / "ref" / "weight_v0_cnn.npy")
        map_file = str(tests_path / "nvnmd" / "out" / "map_v0_cnn.npy")
        # mapt
        mapObj = MapTable(config_file, weight_file, map_file)
        mapt = mapObj.build_map()
        #
        N = 32
        x = np.reshape(np.arange(N) / N * (8.0**2), [-1, 1])
        pred = mapObj.mapping2(x, {"s": mapt["s"]}, mapt["cfg_u2s"])
        pred = np.reshape(pred["s"], [-1])
        ref_dout = [
            -0.36629248,
            11.73139954,
            7.64562607,
            5.61323166,
            4.28568649,
            3.318367,
            2.57386971,
            1.98331738,
            1.5067997,
            1.11873913,
            0.80147266,
            0.54209423,
            0.33074331,
            0.15961564,
            0.02235527,
            -0.08633655,
            -0.17096317,
            -0.23541415,
            -0.28309417,
            -0.31701875,
            -0.33988452,
            -0.35412383,
            -0.36194587,
            -0.36537147,
            -0.36625886,
            -0.36629248,
            -0.36629248,
            -0.36629248,
            -0.36629248,
            -0.36629248,
            -0.36629248,
            -0.36629248,
            -0.37758207,
            12.93425751,
            8.43843079,
            6.2020607,
            4.7412796,
            3.67687798,
            2.85766029,
            2.20783806,
            1.68349743,
            1.25648975,
            0.90738249,
            0.62197208,
            0.38940978,
            0.20110726,
            0.05007112,
            -0.06952924,
            -0.16264915,
            -0.23356843,
            -0.28603387,
            -0.32336307,
            -0.34852386,
            -0.36419201,
            -0.37279916,
            -0.37656879,
            -0.37754512,
            -0.37758207,
            -0.37758207,
            -0.37758207,
            -0.37758207,
            -0.37758207,
            -0.37758207,
            -0.37758207,
        ]
        np.testing.assert_almost_equal(pred, ref_dout, 8)
        #
        N = 4
        x = np.reshape(np.arange(N) / N * 16, [-1, 1])
        pred = mapObj.mapping2(x, {"g": mapt["g"]}, mapt["cfg_s2g"])
        pred = np.reshape(pred["g"], [-1])
        ref_dout = [
            -2.07704735,
            -0.97188997,
            0.62244987,
            0.13949633,
            -2.37506294,
            0.87449265,
            0.45126176,
            1.25518703,
            0.89931679,
            -1.38976288,
            -0.63327646,
            -1.1182003,
            -0.28332829,
            -0.67501783,
            -0.97833824,
            1.50204563,
            -1.55733299,
            -2.39325142,
            0.50640678,
            -0.21376932,
            -2.35564423,
            0.54273844,
            -1.34427929,
            2.64800453,
            1.21231747,
            -1.76566982,
            -1.49742508,
            0.268327,
            -0.39404368,
            -0.96163464,
            0.15446436,
            0.80911779,
            -1.97592735,
            -0.86313152,
            0.54618216,
            -0.4015007,
            -1.47609615,
            0.73630381,
            1.79491711,
            1.56436729,
            0.61628437,
            -1.10759449,
            -1.38582802,
            -1.38886261,
            0.72347879,
            -0.70347023,
            0.62401485,
            1.97830486,
            -1.64212513,
            -2.15452957,
            0.23154032,
            0.07660228,
            -0.9554491,
            0.29101515,
            0.15927136,
            2.78246689,
            1.1549139,
            -1.555933,
            -1.877285,
            -0.12263685,
            0.34770322,
            -1.12232113,
            1.63235283,
            1.44036388,
            -1.5411787,
            0.20576549,
            0.79139471,
            -1.1107235,
            -0.80872536,
            0.56829691,
            2.01861954,
            1.87240505,
            0.26001358,
            -0.07675731,
            -1.93655968,
            -1.41182518,
            1.55228901,
            -0.44892955,
            1.24660587,
            2.25927925,
            -1.65150547,
            -1.01702881,
            0.26624107,
            -0.06200939,
            0.49566555,
            0.32353425,
            0.52009249,
            2.78927803,
            1.01951599,
            -0.078897,
            -2.12800598,
            -0.25314903,
            1.01961136,
            -0.87592936,
            1.90290642,
            2.03822517,
            -1.40549755,
            0.82007837,
            0.85752344,
            -1.3037796,
            -0.71740103,
            0.46696067,
            1.96389103,
            1.99285221,
            0.08252442,
            0.40720892,
            -2.21003342,
            -1.34711552,
            1.75219822,
            -0.33846807,
            1.36572933,
            2.28229904,
            -1.75278759,
            -0.46196342,
            0.28469324,
            -0.09539914,
            0.79652739,
            0.37996507,
            0.48132706,
            2.78080559,
            0.90426493,
            0.70287371,
            -2.30566406,
            -0.21917295,
            1.23095989,
            -0.74176455,
            1.89975643,
            2.24079704,
            -2.07704735,
            -0.97188854,
            0.6224494,
            0.13949692,
            -2.37506294,
            0.87449265,
            0.45126152,
            1.25518703,
            0.89931774,
            -1.38976383,
            -0.63327646,
            -1.11820126,
            -0.28332829,
            -0.67501879,
            -0.97833729,
            1.50204563,
            -1.55733299,
            -2.39325333,
            0.50640631,
            -0.21377122,
            -2.35564423,
            0.54273939,
            -1.34427929,
            2.64800453,
            1.21231747,
            -1.76566887,
            -1.49742508,
            0.26832724,
            -0.39404345,
            -0.96163607,
            0.15446496,
            0.80911875,
            -1.97592735,
            -0.86312962,
            0.54618168,
            -0.40150023,
            -1.47609615,
            0.73630428,
            1.79491711,
            1.56436729,
            0.61628485,
            -1.10759449,
            -1.38582897,
            -1.38886356,
            0.72347879,
            -0.70347071,
            0.62401533,
            1.97830486,
            -1.64212608,
            -2.15452957,
            0.23153985,
            0.07660151,
            -0.95544958,
            0.29101706,
            0.15927136,
            2.78246689,
            1.1549139,
            -1.55593204,
            -1.877285,
            -0.12263596,
            0.34770393,
            -1.12232208,
            1.63235283,
            1.44036484,
            -1.5411787,
            0.20576644,
            0.79139471,
            -1.1107235,
            -0.80872488,
            0.56829691,
            2.01861954,
            1.87240505,
            0.26001358,
            -0.07675725,
            -1.93656063,
            -1.41182518,
            1.55228901,
            -0.44892955,
            1.24660587,
            2.25927734,
            -1.65150547,
            -1.01702976,
            0.26624107,
            -0.06200951,
            0.49566531,
            0.32353616,
            0.52009296,
            2.78927803,
            1.01951599,
            -0.07889658,
            -2.12800598,
            -0.25314808,
            1.01961136,
            -0.87592983,
            1.90290546,
            2.03822517,
            -1.40549755,
            0.82007933,
            0.85752344,
            -1.30378056,
            -0.71740055,
            0.46696043,
            1.96389198,
            1.99285221,
            0.08252424,
            0.40720892,
            -2.21003342,
            -1.34711552,
            1.75219822,
            -0.33846784,
            1.36572933,
            2.28229904,
            -1.75278759,
            -0.46196342,
            0.28469348,
            -0.0953992,
            0.79652739,
            0.37996674,
            0.4813273,
            2.78080559,
            0.9042654,
            0.70287418,
            -2.30566406,
            -0.21917212,
            1.23095989,
            -0.74176455,
            1.89975548,
            2.24079704,
        ]
        np.testing.assert_almost_equal(pred, ref_dout, 8)

    @pytest.mark.run(order=1)
    def test_model_qnn_v0(self) -> None:
        # without calling test_mapt_cnn_v0, this test will fail when running individually
        self.test_mapt_cnn_v0()

        tf.reset_default_graph()
        # open NVNMD
        jdata_cf = jdata_deepmd_input_v0["nvnmd"]
        jdata_cf["config_file"] = str(
            tests_path / "nvnmd" / "ref" / "config_v0_cnn.npy"
        )
        jdata_cf["weight_file"] = str(
            tests_path / "nvnmd" / "ref" / "weight_v0_cnn.npy"
        )
        jdata_cf["map_file"] = str(tests_path / "nvnmd" / "out" / "map_v0_cnn.npy")
        jdata_cf["enable"] = True
        nvnmd_cfg.init_from_jdata(jdata_cf)
        nvnmd_cfg.init_train_mode("qnn")
        # build trainer
        ntype = nvnmd_cfg.dscp["ntype"]
        jdata = nvnmd_cfg.get_deepmd_jdata()
        run_opt = RunOptions(log_path=None, log_level=20)
        jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
        jdata = normalize(jdata)
        self.trainer = DPTrainer(jdata, run_opt, False)
        self.model = self.trainer.model
        # place holder
        dic_ph = {}
        dic_ph["coord"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], "t_coord")
        dic_ph["box"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], "t_box")
        dic_ph["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        dic_ph["natoms_vec"] = tf.placeholder(tf.int32, [ntype + 2], name="t_natoms")
        dic_ph["default_mesh"] = tf.placeholder(tf.int32, [None], name="t_mesh")
        dic_ph["is_training"] = tf.placeholder(tf.bool)
        # build model
        self.model_pred = self.model.build(
            dic_ph["coord"],
            dic_ph["type"],
            dic_ph["natoms_vec"],
            dic_ph["box"],
            dic_ph["default_mesh"],
            dic_ph,
            frz_model=None,
            suffix="",
            reuse=False,
        )
        # get feed_dict
        crd_file = str(tests_path / "nvnmd" / "ref" / "coord.npy")
        box_file = str(tests_path / "nvnmd" / "ref" / "box.npy")
        type_file = str(tests_path / "nvnmd" / "ref" / "type.raw")
        coord_dat = np.reshape(np.load(crd_file), [-1])
        box_dat = np.reshape(np.load(box_file), [-1])
        type_dat = np.int32(np.reshape(np.loadtxt(type_file), [-1]))
        natoms_vec_dat = np.int32([216, 216, 108, 108])
        mesh_dat = np.int32(np.array([0, 0, 0, 2, 2, 2]))

        feed_dict = {
            dic_ph["coord"]: coord_dat,
            dic_ph["box"]: box_dat,
            dic_ph["type"]: type_dat,
            dic_ph["natoms_vec"]: natoms_vec_dat,
            dic_ph["default_mesh"]: mesh_dat,
        }
        #
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        # get tensordic
        keys = "o_descriptor,o_rmat,o_energy".split(",")
        tensordic = {}
        graph = tf.get_default_graph()
        for key in keys:
            tensordic[key] = graph.get_tensor_by_name(key + ":0")
        # get value
        valuelist = sess.run(list(tensordic.values()), feed_dict=feed_dict)
        valuedic = dict(zip(tensordic.keys(), valuelist))
        # test
        # o_descriptor
        idx = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        pred = np.reshape(valuedic["o_descriptor"], [-1])
        ref_dout = [
            0.00614393,
            -0.00593019,
            0.00424719,
            0.0053246,
            0.03973877,
            0.00422275,
            0.0081141,
            0.01380706,
            0.04038167,
            0.01963985,
            0.01557279,
            0.00587749,
            0.03684294,
            0.02304173,
        ]
        np.testing.assert_almost_equal(pred[idx], ref_dout, 8)
        # o_rmat
        pred = np.reshape(valuedic["o_rmat"], [-1])
        ref_dout = [
            -0.05091095,
            -0.93385935,
            11.98665619,
            14.44672394,
            15.79960632,
            23.92457581,
            36.92721558,
            0.0,
            14.60916138,
            18.10887146,
            40.50817871,
            0.0,
            14.73150635,
            32.27615356,
        ]
        np.testing.assert_almost_equal(pred[idx], ref_dout, 8)
        # o_energy
        pred = valuedic["o_energy"]
        ref_dout = -62.60181403
        np.testing.assert_almost_equal(pred, ref_dout, 8)

    def tearDown(self) -> None:
        # close
        nvnmd_cfg.enable = False


class TestNvnmdEntrypointsV1(tf.test.TestCase):
    @pytest.mark.run(order=0)
    def test_mapt_cnn_v1(self) -> None:
        config_file = str(tests_path / "nvnmd" / "ref" / "config_v1_cnn.npy")
        weight_file = str(tests_path / "nvnmd" / "ref" / "weight_v1_cnn.npy")
        map_file = str(tests_path / "nvnmd" / "out" / "map_v1_cnn.npy")
        # mapt
        mapObj = MapTable(config_file, weight_file, map_file)
        mapObj.Gs_Gt_mode = 0
        mapt = mapObj.build_map()
        #
        N = 32
        x = np.reshape(np.arange(N) / N * (8.0**2), [-1, 1])
        pred = mapObj.mapping2(x, {"s": mapt["s"]}, mapt["cfg_u2s"])
        pred = np.reshape(pred["s"], [-1])
        ref_dout = [
            0.00000000e00,
            6.91349983e-01,
            4.57859278e-01,
            3.41713428e-01,
            2.65847921e-01,
            2.10568190e-01,
            1.68022156e-01,
            1.34273767e-01,
            1.07042134e-01,
            8.48655105e-02,
            6.67345524e-02,
            5.19118309e-02,
            3.98336947e-02,
            3.00542116e-02,
            2.22101510e-02,
            1.59987062e-02,
            1.11625344e-02,
            7.47933984e-03,
            4.75455448e-03,
            2.81585380e-03,
            1.50913559e-03,
            6.95408788e-04,
            2.48396304e-04,
            5.26262156e-05,
            1.91306754e-06,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
        np.testing.assert_almost_equal(pred, ref_dout, 8)
        #
        N = 4
        x = np.reshape(np.arange(N) / N * 16, [-1, 1])
        pred = mapObj.mapping2(x, {"g": mapt["g"]}, mapt["cfg_s2g"])
        pred = np.reshape(pred["g"], [-1])
        ref_dout = [
            -1.0770483,
            0.02810931,
            1.62244892,
            1.1394949,
            -1.37506485,
            1.87449265,
            1.45126152,
            2.25518417,
            1.89931679,
            -0.38976216,
            0.36672592,
            -0.11820012,
            0.71667051,
            0.3249805,
            0.02166232,
            2.50204468,
            -0.55733442,
            -1.39325333,
            1.50640583,
            0.78623056,
            -1.35564613,
            1.54273891,
            -0.34428048,
            3.64800453,
            2.21231842,
            -0.76567078,
            -0.49742508,
            1.26832676,
            0.60595608,
            0.03836584,
            1.15446472,
            1.80911732,
            -0.97592735,
            0.13686812,
            1.54618168,
            0.59849787,
            -0.47609806,
            1.73630333,
            2.79491806,
            2.56436539,
            1.61628342,
            -0.10759199,
            -0.38582754,
            -0.38886118,
            1.72347736,
            0.29652929,
            1.62401485,
            2.978302,
            -0.64212656,
            -1.15452957,
            1.23154068,
            1.07660294,
            0.04454666,
            1.29101658,
            1.15926933,
            3.78246689,
            2.15491486,
            -0.55593348,
            -0.87728548,
            0.87736368,
            1.34770393,
            -0.12231946,
            2.63235283,
            2.44036293,
            -0.54117918,
            1.20576477,
            1.79139519,
            -0.1107251,
            0.19127345,
            1.56829548,
            3.01862144,
            2.87240219,
            1.26001167,
            0.92324543,
            -0.9365592,
            -0.41182208,
            2.55228806,
            0.55107069,
            2.24660492,
            3.25927734,
            -0.65150642,
            -0.01702949,
            1.26624203,
            0.9379921,
            1.49566174,
            1.32353592,
            1.52009106,
            3.78927612,
            2.0195179,
            0.921103,
            -1.12800884,
            0.7468524,
            2.01961136,
            0.12407303,
            2.90290642,
            3.03822327,
            -0.40549946,
            1.8200779,
            1.85752392,
            -0.30378056,
            0.28259802,
            1.466959,
            2.96389198,
            2.99284935,
            1.08252144,
            1.4072113,
            -1.21003246,
            -0.34711146,
            2.75219727,
            0.66153288,
            2.36572838,
            3.28229713,
            -0.75278902,
            0.53803587,
            1.28469372,
            0.904603,
            1.79652405,
            1.37996578,
            1.48132515,
            3.78080368,
            1.90426636,
            1.70287418,
            -1.30566597,
            0.78082895,
            2.23096085,
            0.25823808,
            2.89975548,
            3.24079514,
        ]
        np.testing.assert_almost_equal(pred, ref_dout, 8)

    @pytest.mark.run(order=1)
    def test_model_qnn_v1(self) -> None:
        # without calling test_mapt_cnn_v1, this test will fail when running individually
        self.test_mapt_cnn_v1()

        tf.reset_default_graph()
        # open NVNMD
        jdata_cf = jdata_deepmd_input_v1["nvnmd"]
        jdata_cf["config_file"] = str(
            tests_path / "nvnmd" / "ref" / "config_v1_cnn.npy"
        )
        jdata_cf["weight_file"] = str(
            tests_path / "nvnmd" / "ref" / "weight_v1_cnn.npy"
        )
        jdata_cf["map_file"] = str(tests_path / "nvnmd" / "out" / "map_v1_cnn.npy")
        jdata_cf["enable"] = True
        nvnmd_cfg.init_from_jdata(jdata_cf)
        nvnmd_cfg.init_train_mode("qnn")
        # build trainer
        ntype = nvnmd_cfg.dscp["ntype"]
        jdata = nvnmd_cfg.get_deepmd_jdata()
        run_opt = RunOptions(log_path=None, log_level=20)
        jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
        jdata["model"]["type_embedding"] = {}
        jdata = normalize(jdata)
        jdata["model"]["type_embedding"].update(
            {"activation_function": None, "use_tebd_bias": True}
        )
        self.trainer = DPTrainer(jdata, run_opt, False)
        self.model = self.trainer.model
        # place holder
        dic_ph = {}
        dic_ph["coord"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], "t_coord")
        dic_ph["box"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], "t_box")
        dic_ph["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        dic_ph["natoms_vec"] = tf.placeholder(tf.int32, [ntype + 2], name="t_natoms")
        dic_ph["default_mesh"] = tf.placeholder(tf.int32, [None], name="t_mesh")
        dic_ph["is_training"] = tf.placeholder(tf.bool)
        # build model
        self.model_pred = self.model.build(
            dic_ph["coord"],
            dic_ph["type"],
            dic_ph["natoms_vec"],
            dic_ph["box"],
            dic_ph["default_mesh"],
            dic_ph,
            frz_model=None,
            suffix="",
            reuse=False,
        )
        # get feed_dict
        crd_file = str(tests_path / "nvnmd" / "ref" / "coord.npy")
        box_file = str(tests_path / "nvnmd" / "ref" / "box.npy")
        type_file = str(tests_path / "nvnmd" / "ref" / "type.raw")
        coord_dat = np.reshape(np.load(crd_file), [-1])
        box_dat = np.reshape(np.load(box_file), [-1])
        type_dat = np.int32(np.reshape(np.loadtxt(type_file), [-1]))
        natoms_vec_dat = np.int32([216, 216, 108, 108])
        mesh_dat = np.int32(np.array([0, 0, 0, 2, 2, 2]))

        feed_dict = {
            dic_ph["coord"]: coord_dat,
            dic_ph["box"]: box_dat,
            dic_ph["type"]: type_dat,
            dic_ph["natoms_vec"]: natoms_vec_dat,
            dic_ph["default_mesh"]: mesh_dat,
        }
        #
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        # get tensordic
        keys = "o_descriptor,o_rmat,o_energy".split(",")
        tensordic = {}
        graph = tf.get_default_graph()
        for key in keys:
            tensordic[key] = graph.get_tensor_by_name(key + ":0")
        # get value
        valuelist = sess.run(list(tensordic.values()), feed_dict=feed_dict)
        valuedic = dict(zip(tensordic.keys(), valuelist))
        # test
        # o_descriptor
        idx = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        pred = np.reshape(valuedic["o_descriptor"], [-1])
        ref_dout = [
            -0.03495526,
            0.19181037,
            0.00139165,
            0.03920531,
            0.3982904,
            0.05152893,
            0.28467178,
            1.33868217,
            0.83964777,
            1.16189384,
            1.16278744,
            0.93079185,
            1.37950325,
            0.90435696,
        ]
        np.testing.assert_almost_equal(pred[idx], ref_dout, 8)
        # o_rmat
        pred = np.reshape(valuedic["o_rmat"], [-1])
        ref_dout = [
            2.08768272,
            -1.0802269,
            7.82283401,
            8.53259277,
            11.98665619,
            15.26974487,
            25.43397522,
            36.61801147,
            0.0,
            6.82917404,
            7.45996857,
            6.59498215,
            6.65272522,
            5.89738083,
        ]
        np.testing.assert_almost_equal(pred[idx], ref_dout, 8)
        # o_energy
        pred = valuedic["o_energy"]
        ref_dout = 60.73941362
        np.testing.assert_almost_equal(pred, ref_dout, 8)
        # test freeze
        sess = self.cached_session().__enter__()
        weight_file1 = str(tests_path / "nvnmd" / "ref" / "weight_v1_cnn.npy")
        weight_file2 = str(tests_path / "nvnmd" / "out" / "weight_v1_qnn.npy")
        save_weight(sess, weight_file2)

        d1 = FioNpyDic().load(weight_file1)
        d2 = FioNpyDic().load(weight_file2)
        keys = [
            "type_embed_net.matrix_1",
            "type_embed_net.bias_1",
            "descrpt_attr.t_avg",
            "descrpt_attr.t_std",
            "layer_0.tweight",
            "layer_0.matrix_1",
            "layer_0.bias",
            "layer_1.matrix",
            "layer_1.bias",
            "layer_2.matrix",
            "layer_2.bias",
            "final_layer.matrix",
            "final_layer.bias",
        ]
        for key in keys:
            pred = d2[key]
            ref_dout = d1[key]
            np.testing.assert_almost_equal(pred, ref_dout, 8)

    @pytest.mark.run(order=2)
    def test_wrap_qnn_v1(self) -> None:
        # without calling test_mapt_cnn_v1, this test will fail when running individually
        self.test_mapt_cnn_v1()

        tf.reset_default_graph()
        jdata = {}
        jdata["nvnmd_config"] = str(tests_path / "nvnmd" / "ref" / "config_v1_cnn.npy")
        jdata["nvnmd_weight"] = str(tests_path / "nvnmd" / "ref" / "weight_v1_cnn.npy")
        jdata["nvnmd_map"] = str(tests_path / "nvnmd" / "out" / "map_v1_cnn.npy")
        jdata["nvnmd_model"] = str(tests_path / "nvnmd" / "out" / "model_v1_qnn.pb")
        wrap(**jdata)
        # test
        data = FioBin().load(jdata["nvnmd_model"])
        idx = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = [i + 128 * 4 for i in idx]
        pred = [data[i] for i in idx]
        red_dout = [249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 95, 24, 176]
        np.testing.assert_equal(pred, red_dout)

    def tearDown(self) -> None:
        # close
        nvnmd_cfg.enable = False


if __name__ == "__main__":
    unittest.main()
