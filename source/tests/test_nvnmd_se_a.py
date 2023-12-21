import os
import unittest

import numpy as np
from common import (
    j_loader,
)

from deepmd.descriptor import (
    DescrptSeA,
)
from deepmd.env import (
    tf,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

#
from common import (
    tests_path,
)

from deepmd.nvnmd.data.data import (
    jdata_deepmd_input,
)
from deepmd.nvnmd.utils.config import (
    nvnmd_cfg,
)


class TestModel(tf.test.TestCase):
    def setUp(self):
        # generate data of GeTe
        a = 3.01585  # crystal constant
        natom = 64
        box = np.zeros([9])
        box[0] = a * 4
        box[4] = a * 4
        box[8] = a * 4
        box = box.reshape([-1, 9])
        types = np.zeros([natom], dtype=np.int32)
        types[32:] = 1
        types = types.reshape([1, natom])
        coord = np.zeros([natom, 3])
        ct = 0
        ct2 = 32
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    if (ix + iy + iz) % 2 == 0:
                        coord[ct] = np.array([ix * a, iy * a, iz * a])
                        ct += 1
                    else:
                        coord[ct2] = np.array([ix * a, iy * a, iz * a])
                        ct2 += 1
        coord = coord.reshape([1, natom * 3])
        natoms = np.array([64, 64, 32, 32])
        mesh = np.array([0, 0, 0, 2, 2, 2])
        #
        self.box = box
        self.types = types
        self.coord = coord
        self.natoms = natoms
        self.mesh = mesh

    def test_descriptor_one_side_qnn(self):
        """: test se_a of NVNMD with quantized value.

        Reference:
            test_descrpt_se_a_type.py

        Note:
        ----
            The test_nvnmd_se_a.py must be run after test_nvnmd_entrypoints.py.
            Because the data file map.npy ia generated in running test_nvnmd_entrypoints.py.
        """
        tf.reset_default_graph()
        # open NVNMD
        jdata_cf = jdata_deepmd_input["nvnmd"]
        jdata_cf["config_file"] = str(tests_path / os.path.join("nvnmd", "config.npy"))
        jdata_cf["weight_file"] = str(tests_path / os.path.join("nvnmd", "weight.npy"))
        jdata_cf["map_file"] = str(tests_path / os.path.join("nvnmd", "map.npy"))
        jdata_cf["enable"] = True
        nvnmd_cfg.init_from_jdata(jdata_cf)
        nvnmd_cfg.quantize_descriptor = True
        nvnmd_cfg.restore_descriptor = True
        # load input
        jfile = str(tests_path / os.path.join("nvnmd", "train_cnn.json"))
        jdata = j_loader(jfile)
        ntypes = nvnmd_cfg.dscp["ntype"]

        # build descriptor
        jdata["model"]["descriptor"].pop("type", None)
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)

        t_coord = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_coord"
        )
        t_type = tf.placeholder(tf.int32, [None, None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            reuse=False,
            suffix="_se_a_nvnmd",
        )
        # data
        feed_dict_test = {
            t_coord: self.coord,
            t_box: self.box,
            t_type: self.types,
            t_natoms: self.natoms,
            t_mesh: self.mesh,
            is_training: False,
        }
        # run
        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        # compare
        ref_dout = [
            0.0136348009,
            0.0083287954,
            -0.0639075041,
            0.0129179955,
            0.0050876141,
            -0.0390378237,
            0.0078909397,
            0.0022796392,
            0.2995386124,
            -0.0605479479,
        ]
        places = 10
        np.testing.assert_almost_equal(model_dout[0:10], ref_dout, places)
        # close NVNMD
        jdata_cf["enable"] = False
        nvnmd_cfg.init_from_jdata(jdata_cf)


if __name__ == "__main__":
    unittest.main()
