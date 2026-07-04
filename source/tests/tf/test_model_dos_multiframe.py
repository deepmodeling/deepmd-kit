# SPDX-License-Identifier: LGPL-3.0-or-later
"""The TF DOS model global output must equal the per-frame atomic sum.

The DOS model builds its global DOS (``o_dos``) by summing the atomic DOS
(``o_atom_dos``). For a single frame this holds, but the reduction used to
reshape/reduce over the wrong axis, so with more than one frame the global DOS
no longer equaled the per-frame sum of the atomic DOS.
"""

import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.fit import (
    DOSFitting,
)
from deepmd.tf.model import (
    DOSModel,
)

from .common import (
    DataSystem,
    del_data,
    gen_data,
    j_loader,
)


class TestDOSModelMultiFrame(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()

    def tearDown(self) -> None:
        del_data()

    def test_multiframe_global_equals_atomic_sum(self) -> None:
        jfile = "train_dos.json"
        jdata = j_loader(jfile)
        systems = jdata["training"]["systems"]
        rcut = jdata["model"]["descriptor"]["rcut"]
        data = DataSystem(systems, "set", 1, 1, rcut, run_opt=None)
        test_data = data.get_test()
        numb_dos = 20
        natoms = test_data["type"].shape[1]

        jdata["model"]["fitting_net"]["numb_dos"] = numb_dos
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        jdata["model"]["descriptor"].pop("type", None)
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"].pop("type", None)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        fitting = DOSFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        model = DOSModel(descrpt, fitting)

        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)

        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [model.ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")

        model_pred = model.build(
            t_coord, t_type, t_natoms, t_box, t_mesh, None, suffix="dos_mf", reuse=False
        )
        dos = model_pred["dos"]
        atom_dos = model_pred["atom_dos"]

        # two frames (frame 0 repeated) -> global DOS must be the per-frame sum
        nframes = 2
        one_coord = np.reshape(test_data["coord"][:1, :], [-1])
        one_type = np.reshape(test_data["type"][:1, :], [-1])
        feed = {
            t_coord: np.tile(one_coord, nframes),
            t_type: np.tile(one_type, nframes),
            t_natoms: test_data["natoms_vec"],
            t_box: np.tile(test_data["box"][:1, :], (nframes, 1)),
            t_mesh: test_data["default_mesh"],
        }

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            pred_dos, pred_atom_dos = sess.run([dos, atom_dos], feed_dict=feed)

        pred_dos = np.reshape(pred_dos, [nframes, numb_dos])
        pred_atom_dos = np.reshape(pred_atom_dos, [nframes, natoms, numb_dos])
        expected = np.sum(pred_atom_dos, axis=1)
        np.testing.assert_almost_equal(pred_dos, expected, 6)


if __name__ == "__main__":
    tf.test.main()
