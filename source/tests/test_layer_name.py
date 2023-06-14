import numpy as np
from common import (
    DataSystem,
    del_data,
    gen_data,
    j_loader,
)

from deepmd.common import (
    j_must_have,
)
from deepmd.descriptor import (
    DescrptSeA,
)
from deepmd.env import (
    tf,
)
from deepmd.fit import (
    DipoleFittingSeA,
    EnerFitting,
)
from deepmd.model import (
    MultiModel,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    def setUp(self):
        gen_data()

    def tearDown(self):
        del_data()

    def test_model(self):
        """Two fittings which share the same parameters should give the same result."""
        jfile = "water_layer_name.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["multi_task"] = True
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)
        fitting_dict = {}
        fitting_type_dict = {}
        for fitting_key in jdata["model"]["fitting_net_dict"]:
            item_fitting_param = jdata["model"]["fitting_net_dict"][fitting_key]
            item_fitting_type = item_fitting_param.get("type", "ener")
            fitting_type_dict[fitting_key] = item_fitting_type
            item_fitting_param.pop("type", None)
            item_fitting_param.pop("fit_diag", None)
            item_fitting_param["descrpt"] = descrpt
            if item_fitting_type == "ener":
                fitting_dict[fitting_key] = EnerFitting(
                    **item_fitting_param, uniform_seed=True
                )
            elif item_fitting_type == "dipole":
                fitting_dict[fitting_key] = DipoleFittingSeA(
                    **item_fitting_param, uniform_seed=True
                )
            else:
                raise RuntimeError("Test should not be here!")
        model = MultiModel(descrpt, fitting_dict, fitting_type_dict)

        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }

        for fitting_key in jdata["model"]["fitting_net_dict"]:
            model._compute_input_stat(input_data, fitting_key=fitting_key)
        model.descrpt.merge_input_stats(model.descrpt.stat_dict)
        model.descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [model.ntypes + 2], name="i_natoms")
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
            suffix="_layer_name",
            reuse=False,
        )

        e_energy1 = model_pred["water_ener"]["energy"]
        e_force1 = model_pred["water_ener"]["force"]
        e_virial1 = model_pred["water_ener"]["virial"]
        e_energy2 = model_pred["water_ener2"]["energy"]
        e_force2 = model_pred["water_ener2"]["force"]
        e_virial2 = model_pred["water_ener2"]["virial"]
        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_force: np.reshape(test_data["force"][:numb_test, :], [-1]),
            t_virial: np.reshape(test_data["virial"][:numb_test, :], [-1]),
            t_atom_ener: np.reshape(test_data["atom_ener"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [e1, f1, v1, e2, f2, v2] = sess.run(
                [e_energy1, e_force1, e_virial1, e_energy2, e_force2, e_virial2],
                feed_dict=feed_dict_test,
            )
        np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(f1, f2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(v1, v2, rtol=1e-5, atol=1e-5)
