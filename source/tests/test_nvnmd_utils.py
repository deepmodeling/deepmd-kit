import os
import unittest

import numpy as np

from deepmd.env import (
    tf,
)
from deepmd.nvnmd.data.data import (
    jdata_deepmd_input,
)
from deepmd.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.nvnmd.utils.network import (
    one_layer,
)


class TestNvnmdNetwork(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.test_session(config=config).__enter__()

    def test_onelayer(self):
        # open NVNMD
        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = "none"
        jdata["weight_file"] = "none"
        jdata["map_file"] = "none"
        jdata["enable"] = True
        nvnmd_cfg.init_from_jdata(jdata)
        w = np.array([-0.313429, 0.783452, -0.423276, 0.832279]).reshape(4, 1)
        b = np.array([0.3482787]).reshape([1, 1])
        nvnmd_cfg.weight = {"nvnmd.matrix": w, "nvnmd.bias": b}
        nvnmd_cfg.quantize_fitting_net = True
        nvnmd_cfg.restore_fitting_net = True
        # build
        x = np.array(
            [
                -0.313429,
                1.436861,
                0.324769,
                -1.4823674,
                0.783452,
                -0.171208,
                -0.033421,
                -1.324673,
            ]
        ).reshape([2, 4])
        y = np.array([0.19909, -0.86702]).reshape([-1])
        ty = one_layer(tf.constant(x), 1, name="nvnmd")
        # run
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        typ = typ.reshape([-1])
        np.testing.assert_almost_equal(typ, y, 5)
        # close NVNMD
        jdata["enable"] = False
        nvnmd_cfg.init_from_jdata(jdata)
        nvnmd_cfg.weight = {}
        nvnmd_cfg.quantize_fitting_net = False
        nvnmd_cfg.restore_fitting_net = False


if __name__ == "__main__":
    unittest.main()
