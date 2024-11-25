# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit import (
    DOSFitting,
)

from .common import (
    DataSystem,
    gen_data,
    j_loader,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()

    def test_fitting(self) -> None:
        jfile = "train_dos.json"
        jdata = j_loader(jfile)

        systems = jdata["training"]["systems"]
        set_pfx = "set"
        batch_size = jdata["training"]["batch_size"]
        test_size = jdata["training"]["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
        ntypes = len(sel)

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1
        numb_dos = 100

        jdata["model"]["fitting_net"]["numb_dos"] = numb_dos

        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2

        jdata["model"]["descriptor"].pop("type", None)
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)

        jdata["model"]["fitting_net"].pop("type", None)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        fitting = DOSFitting(**jdata["model"]["fitting_net"], uniform_seed=True)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")

        t_atom_dos = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_dos"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        dout = np.array(
            [
                0.0005722682145569174,
                -0.00020202686217742682,
                -0.00020202686217742682,
                7.13250554992363e-05,
                -0.0014770058171250015,
                0.000521468690207748,
                -0.001143865186937176,
                0.0004038453384193948,
                0.0005617335409639567,
                -0.00019831394075147532,
                0.00048086740718842236,
                -0.0001693584775806112,
                -0.0001693584775806112,
                5.966987137476082e-05,
                -0.0012342029581315136,
                0.00043492340851472783,
                -0.0009566016612537016,
                0.00033706767041080107,
                0.00047065988464132244,
                -0.0001657950398095401,
                0.0003647849239740657,
                -0.00013744939018250384,
                -0.00013744939018250384,
                5.1825826955234744e-05,
                -0.00096004206555711,
                0.00036185565262332876,
                -0.0007267433909643961,
                0.0002738914365542745,
                0.00038019365906978136,
                -0.00014322754331896057,
                0.0004675256930823109,
                -0.00017634410399626168,
                -0.00017634410399626168,
                6.652672908755666e-05,
                -0.0012328062885292486,
                0.00046500213384094614,
                -0.0009328887521346069,
                0.0003518668613172834,
                0.0004877847509912577,
                -0.00018396318824508986,
                0.0005154794374703516,
                -0.00019422534512034776,
                -0.00019422534512034776,
                7.318151797939947e-05,
                -0.0013576642997136488,
                0.0005115548790018505,
                -0.0010275333676074971,
                0.00038716440070070385,
                0.0005376426714609369,
                -0.00020257810468163985,
                0.0004482204892297628,
                -0.00016887749501640607,
                -0.00016887749501640607,
                6.364643102775375e-05,
                -0.001181345877677835,
                0.0004452029242063362,
                -0.0008941636427724908,
                0.0003369586197174627,
                0.0004677878512312651,
                -0.00017625260641095753,
            ]
        )

        atype = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)

        dout = dout.reshape([-1, 10])
        atype = atype.reshape([-1])

        natoms = 6
        tmp_dos = np.zeros([numb_test, numb_dos])
        tmp_atom_dos = np.zeros([numb_test, natoms * numb_dos])
        test_data["dos"] = tmp_dos
        test_data["atom_dos"] = tmp_atom_dos

        atom_dos = fitting.build(
            tf.convert_to_tensor(dout),
            t_natoms,
            {
                "atype": tf.convert_to_tensor(atype),
            },
            reuse=False,
            suffix="se_a_dos_fit_",
        )

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_atom_dos: np.reshape(test_data["atom_dos"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [pred_atom_dos] = sess.run([atom_dos], feed_dict=feed_dict_test)

        pred_atom_dos = pred_atom_dos.reshape(-1, numb_dos)

        ref_atom_dos_1 = [
            -0.32495014,
            -0.32495882,
            -0.32496842,
            -0.32495892,
            -0.32495469,
            -0.32496075,
        ]
        ref_atom_dos_2 = [
            0.21549911,
            0.21550413,
            0.21551077,
            0.21550547,
            0.21550303,
            0.21550645,
        ]
        places = 4
        np.testing.assert_almost_equal(pred_atom_dos[:, 0], ref_atom_dos_1, places)
        np.testing.assert_almost_equal(pred_atom_dos[:, 50], ref_atom_dos_2, places)
