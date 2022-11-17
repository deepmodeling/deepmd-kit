import os
import numpy as np
import unittest

from common import tests_path

from deepmd.env import tf
from deepmd.nvnmd.utils.fio import FioNpyDic, FioJsonDic, FioBin
from deepmd.nvnmd.entrypoints.freeze import save_weight
from deepmd.nvnmd.entrypoints.mapt import mapt
from deepmd.nvnmd.entrypoints.train import normalized_input, normalized_input_qnn
from deepmd.nvnmd.entrypoints.wrap import wrap
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.data.data import jdata_deepmd_input


class TestNvnmdFreeze(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()

    def test_freeze(self):
        namelist = (
            "descrpt_attr/t_avg",
            "descrpt_attr/t_std",
            "filter_type_0/matrix_0_0",
            "filter_type_0/bias_0_0",
            "layer_0_type_0/matrix",
            "layer_0_type_0/bias",
            "final_layer_type_0/matrix",
            "final_layer_type_0/bias",
        )
        tvlist = []
        save_path = str(tests_path / os.path.join("nvnmd", "weight.npy"))
        vinit = tf.random_normal_initializer(stddev=1.0, seed=0)
        for sname in namelist:
            scope, name = sname.split('/')[0:2]
            with tf.variable_scope(scope, reuse=False):
                tv = tf.get_variable(name, [1], tf.float32, vinit)
                tvlist.append(tv)
        #
        self.sess.run(tf.global_variables_initializer())
        save_weight(self.sess, save_path)
        weight = FioNpyDic().load(save_path)
        namelist = [sname.replace('/', '.') for sname in namelist]
        print(namelist)
        print(list(weight.keys()))
        np.testing.assert_equal(namelist, list(weight.keys()))
        tf.reset_default_graph()


class TestNvnmdMapt(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()

    def test_mapt(self):
        nvnmd_config = str(tests_path / os.path.join("nvnmd", "config_ref.npy"))
        nvnmd_weight = str(tests_path / os.path.join("nvnmd", "weight_ref.npy"))
        nvnmd_map = str(tests_path / os.path.join("nvnmd", "map.npy"))
        jdata = {
            'nvnmd_config': nvnmd_config,
            'nvnmd_weight': nvnmd_weight,
            'nvnmd_map': nvnmd_map,
        }
        mapt(**jdata)
        #
        data = FioNpyDic().load(nvnmd_map)
        #
        nvnmd_map2 = str(tests_path / os.path.join("nvnmd", "map_ref.npy"))
        data2 = FioNpyDic().load(nvnmd_map2)
        keys = [
            'r2',
            's2',
            's_t0_t0',
            'sr_t0_t0',
            'ds_dr2_t0_t0',
            'dsr_dr2_t0_t0',
            'G_t0_t0',
            'dG_ds_t0_t0',
            's_t0_t1',
            'sr_t0_t1',
            'ds_dr2_t0_t1',
            'dsr_dr2_t0_t1',
            'G_t0_t1',
            'dG_ds_t0_t1'
        ]
        np.testing.assert_equal(keys, list(data.keys()))
        np.testing.assert_almost_equal(data['G_t0_t0'], data2['G_t0_t0'])
        tf.reset_default_graph()
        # close NVNMD
        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = "none"
        jdata['weight_file'] = "none"
        jdata['map_file'] = "none"
        jdata['enable'] = False
        nvnmd_cfg.init_from_jdata(jdata)

class TestNvnmdTrain(tf.test.TestCase):
    def test_train_input(self):
        # test1
        INPUT = str(tests_path / os.path.join("nvnmd", "train_ref.json"))
        PATH_CNN = "nvnmd_cnn"
        jdata = normalized_input(INPUT, PATH_CNN)
        fn_ref = str(tests_path / os.path.join("nvnmd", "train_ref2.json"))
        FioJsonDic().save(fn_ref, jdata)
        # test2
        PATH_QNN = "nvnmd_qnn"
        CONFIG_CNN = "none"
        WEIGHT_CNN = "none"
        MAP_CNN = "none"
        jdata = normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN)
        fn_ref = str(tests_path / os.path.join("nvnmd", "train_ref3.json"))
        FioJsonDic().save(fn_ref, jdata)
        # close NVNMD
        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = "none"
        jdata['weight_file'] = "none"
        jdata['map_file'] = "none"
        jdata['enable'] = False
        nvnmd_cfg.init_from_jdata(jdata)

class TestNvnmdWrap(tf.test.TestCase):
    def test_wrap(self):
        nvnmd_config = str(tests_path / os.path.join("nvnmd", "config_ref.npy"))
        nvnmd_weight = str(tests_path / os.path.join("nvnmd", "weight_ref.npy"))
        nvnmd_map = str(tests_path / os.path.join("nvnmd", "map.npy"))
        nvnmd_model = str(tests_path / os.path.join("nvnmd", "model.pb"))
        jdata = {
            'nvnmd_config': nvnmd_config,
            'nvnmd_weight': nvnmd_weight,
            'nvnmd_map': nvnmd_map,
            'nvnmd_model': nvnmd_model,
        }
        wrap(**jdata)
        # test
        data = FioBin().load(nvnmd_model)
        nvnmd_model2 = str(tests_path / os.path.join("nvnmd", "model_ref.npy"))
        datas = ''.join([hex(d+256).replace('0x1', '') for d in data[::256]])
        data2 = FioNpyDic().load(nvnmd_model2)['ref']
        np.testing.assert_equal(datas, data2)
        # close NVNMD
        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = "none"
        jdata['weight_file'] = "none"
        jdata['map_file'] = "none"
        jdata['enable'] = False
        nvnmd_cfg.init_from_jdata(jdata)


if __name__ == '__main__':
    unittest.main()
