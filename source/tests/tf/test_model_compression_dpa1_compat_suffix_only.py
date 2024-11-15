# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.common import (
    make_default_mesh,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.descriptor.se_atten import DescrptDPA1Compat as tf_SeAtten
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    tf,
)
from deepmd.tf.utils.sess import (
    run_sess,
)


def build_tf_descriptor(obj, natoms, coords, atype, box, suffix):
    t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
    t_type = tf.placeholder(tf.int32, [None], name="i_type")
    t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
    t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [9], name="i_box")
    t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
    t_des = obj.build(
        t_coord,
        t_type,
        t_natoms,
        t_box,
        t_mesh,
        {},
        suffix=suffix,
    )
    return [t_des], {
        t_coord: coords,
        t_type: atype,
        t_natoms: natoms,
        t_box: box,
        t_mesh: make_default_mesh(True, False),
    }


def build_eval_tf(sess, obj, natoms, coords, atype, box, suffix):
    t_out, feed_dict = build_tf_descriptor(obj, natoms, coords, atype, box, suffix)

    t_out_indentity = [
        tf.identity(tt, name=f"o_{ii}_{suffix}") for ii, tt in enumerate(t_out)
    ]
    run_sess(sess, tf.global_variables_initializer())
    return run_sess(
        sess,
        t_out_indentity,
        feed_dict=feed_dict,
    )


class TestDescriptorSeA(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"
        self.seed = 21
        self.sel = [9, 10]
        self.rcut_smth = 5.80
        self.rcut = 6.00
        self.neuron = [6, 12, 24]
        self.axis_neuron = 3
        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # self.atype = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.suffix = "test"
        self.type_one_side = False
        self.se_a_tf = tf_SeAtten(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.ntypes,
            self.neuron,
            self.axis_neuron,
            type_one_side=self.type_one_side,
            seed=21,
            precision="float32",
            tebd_input_mode="strip",
            temperature=1.0,
            attn_layer=0,
        )

    def test_tf_pt_consistent(
        self,
    ) -> None:
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret = build_eval_tf(
                sess,
                self.se_a_tf,
                self.natoms,
                self.coords,
                self.atype,
                self.box,
                self.suffix,
            )
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.suffix}" for ii, _ in enumerate(ret)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            self.se_a_tf.init_variables(
                new_graph,
                output_graph_def,
                suffix=self.suffix,
            )
            self.se_a_tf.enable_compression(
                1.0,
                new_graph,
                output_graph_def,
                suffix=self.suffix,
            )


if __name__ == "__main__":
    unittest.main()
