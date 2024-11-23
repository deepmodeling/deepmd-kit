# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np

from deepmd.tf.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.infer import (
    DeepPotential,
)
from deepmd.tf.model.linear import (
    LinearEnergyModel,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    DataSystem,
    del_data,
    gen_data,
    infer_path,
)


class TestLinearModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()
        self.data_dir = "system"
        with open(os.path.join(self.data_dir, "type_map.raw"), "w") as f:
            f.write("O\nH")
        self.pbtxts = [
            os.path.join(infer_path, "deeppot.pbtxt"),
            os.path.join(infer_path, "deeppot-1.pbtxt"),
        ]
        self.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in self.pbtxts]
        for pbtxt, pb in zip(self.pbtxts, self.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        self.graphs = [DeepPotential(pb) for pb in self.graph_dirs]

    def test_linear_ener_model(self) -> None:
        numb_test = 1
        data = DataSystem([self.data_dir], "set", 1, 1, 6, run_opt=None)
        test_data = data.get_test()

        model = LinearEnergyModel(
            models=[
                {
                    "type": "frozen",
                    "model_file": model_file,
                }
                for model_file in self.graph_dirs
            ],
            weights="mean",
        )

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [model.get_ntypes() + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            t_fparam,
            suffix="_linear_energy",
            reuse=False,
        )

        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        e = np.reshape(e, [1, -1])
        f = np.reshape(f, [1, -1, 3])
        v = np.reshape(v, [1, 9])

        es = []
        fs = []
        vs = []

        for ii, graph in enumerate(self.graphs):
            ei, fi, vi = graph.eval(
                test_data["coord"][:numb_test, :],
                test_data["box"][:numb_test, :],
                np.reshape(test_data["type"], [-1]),
            )
            es.append(ei)
            fs.append(fi)
            vs.append(vi)

        np.testing.assert_allclose(e, np.mean(es, axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(f, np.mean(fs, axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(v, np.mean(vs, axis=0), rtol=1e-5, atol=1e-5)

    def tearDown(self) -> None:
        for pb in self.graph_dirs:
            os.remove(pb)
        del_data()
