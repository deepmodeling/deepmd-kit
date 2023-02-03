import os
import unittest

import numpy as np
from common import (
    tests_path,
)

from deepmd.env import (
    tf,
)
from deepmd.nvnmd.data.data import (
    jdata_deepmd_input,
)
from deepmd.nvnmd.entrypoints.freeze import (
    save_weight,
)
from deepmd.nvnmd.entrypoints.mapt import (
    mapt,
)
from deepmd.nvnmd.entrypoints.train import (
    normalized_input,
    normalized_input_qnn,
)
from deepmd.nvnmd.entrypoints.wrap import (
    wrap,
)
from deepmd.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.nvnmd.utils.fio import (
    FioBin,
    FioJsonDic,
    FioNpyDic,
)


class TestNvnmdFreeze(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.test_session(config=config).__enter__()

    def test_freeze(self):
        #
        namelist1 = [
            "descrpt_attr/t_avg",
            "descrpt_attr/t_std",
            "filter_type_0/matrix_0_0",
            "filter_type_0/bias_0_0",
            "layer_0_type_0/matrix",
            "layer_0_type_0/bias",
            "final_layer_type_0/matrix",
            "final_layer_type_0/bias",
        ]
        namelist2 = ["train_attr/min_nbor_dist"]
        namelist = namelist1 + namelist2
        # crete variable according to namelist
        tvlist = []
        save_path = str(tests_path / "nvnmd" / "out" / "weight.npy")
        vinit = tf.random_normal_initializer(stddev=1.0, seed=0)
        for sname in namelist:
            scope, name = sname.split("/")[0:2]
            with tf.variable_scope(scope, reuse=False):
                if sname in namelist1:
                    # create variable
                    tv = tf.get_variable(name, [1], tf.float32, vinit)
                    tvlist.append(tv)
                elif sname in namelist2:
                    # create constant tensor
                    ts = tf.constant(2.0, name=name, dtype=tf.float64)
        # save variable and test
        self.sess.run(tf.global_variables_initializer())
        save_weight(self.sess, save_path)
        weight = FioNpyDic().load(save_path)
        namelist = [sname.replace("/", ".") for sname in namelist]
        np.testing.assert_equal(namelist, list(weight.keys()))
        tf.reset_default_graph()


class TestNvnmdMapt(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.test_session(config=config).__enter__()

    def test_mapt(self):
        nvnmd_config = str(tests_path / "nvnmd" / "config.npy")
        nvnmd_weight = str(tests_path / "nvnmd" / "weight.npy")
        nvnmd_map = str(tests_path / "nvnmd" / "out" / "map.npy")
        jdata = {
            "nvnmd_config": nvnmd_config,
            "nvnmd_weight": nvnmd_weight,
            "nvnmd_map": nvnmd_map,
        }
        mapt(**jdata)
        #
        data1 = FioNpyDic().load(nvnmd_map)
        #
        nvnmd_map2 = str(tests_path / "nvnmd" / "map.npy")
        data2 = FioNpyDic().load(nvnmd_map2)
        keys = ["cfg_u2s", "cfg_s2g", "s", "s_grad", "h", "h_grad", "g", "g_grad"]
        s1 = np.reshape(np.array(data1["s"]), [-1, 4])
        s2 = np.reshape(np.array(data2["s"]), [-1, 4])
        g1 = np.reshape(np.array(data1["g"]), [-1, 4])
        g2 = np.reshape(np.array(data2["g"]), [-1, 4])
        np.testing.assert_equal(keys, list(data1.keys()))
        np.testing.assert_almost_equal(s1, s2, 5)
        np.testing.assert_almost_equal(g1, g2, 5)
        tf.reset_default_graph()
        # close NVNMD
        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = "none"
        jdata["weight_file"] = "none"
        jdata["map_file"] = "none"
        jdata["enable"] = False
        nvnmd_cfg.init_from_jdata(jdata)


class TestNvnmdTrain(tf.test.TestCase):
    def test_train_input(self):
        # test1: train cnn
        INPUT = str(tests_path / "nvnmd" / "train.json")
        PATH_CNN = "nvnmd_cnn"
        jdata = normalized_input(INPUT, PATH_CNN, "none")
        fn_ref = str(tests_path / "nvnmd" / "out" / "train_cnn.json")
        FioJsonDic().save(fn_ref, jdata)
        # test2: train qnn
        PATH_QNN = "nvnmd_qnn"
        CONFIG_CNN = "none"
        WEIGHT_CNN = "none"
        MAP_CNN = "none"
        jdata = normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN)
        fn_ref = str(tests_path / "nvnmd" / "out" / "train_qnn.json")
        FioJsonDic().save(fn_ref, jdata)
        tf.reset_default_graph()
        # close NVNMD
        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = "none"
        jdata["weight_file"] = "none"
        jdata["map_file"] = "none"
        jdata["enable"] = False
        nvnmd_cfg.init_from_jdata(jdata)


class TestNvnmdWrap(tf.test.TestCase):
    def test_wrap(self):
        nvnmd_config = str(tests_path / "nvnmd" / "config.npy")
        nvnmd_weight = str(tests_path / "nvnmd" / "weight.npy")
        nvnmd_map = str(tests_path / "nvnmd" / "map.npy")
        nvnmd_model = str(tests_path / "nvnmd" / "out" / "model.pb")
        jdata = {
            "nvnmd_config": nvnmd_config,
            "nvnmd_weight": nvnmd_weight,
            "nvnmd_map": nvnmd_map,
            "nvnmd_model": nvnmd_model,
        }
        wrap(**jdata)
        # test
        data = FioBin().load(nvnmd_model)
        idx = [1, 11, 111, 1111, 11111]
        idxx = []
        for ii in range(1, 10):
            idxx.extend([ii * i for i in idx])
        dat = [data[i] for i in idxx]
        dat2 = [
            0,
            0,
            0,
            0,
            48,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            100,
            5,
            150,
            0,
            29,
            41,
            29,
            171,
            196,
            0,
            0,
            94,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            44,
            0,
            0,
            0,
            223,
            17,
            9,
            196,
            0,
            211,
            130,
            24,
        ]
        np.testing.assert_equal(dat, dat2)
        # # close NVNMD
        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = "none"
        jdata["weight_file"] = "none"
        jdata["map_file"] = "none"
        jdata["enable"] = False
        nvnmd_cfg.init_from_jdata(jdata)


if __name__ == "__main__":
    unittest.main()
