import numpy as np
from common import (
    DataSystem,
    del_data,
    finite_difference,
    gen_data,
    j_loader,
    strerch_box,
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
        jfile = "water_multi.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata, "stop_batch")
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
                RuntimeError("Test should not be here!")
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
            suffix="multi",
            reuse=False,
        )
        e_energy = model_pred["water_ener"]["energy"]
        e_force = model_pred["water_ener"]["force"]
        e_virial = model_pred["water_ener"]["virial"]
        e_atom_ener = model_pred["water_ener"]["atom_ener"]

        d_dipole = model_pred["water_dipole"]["dipole"]
        d_gdipole = model_pred["water_dipole"]["global_dipole"]
        d_force = model_pred["water_dipole"]["force"]
        d_virial = model_pred["water_dipole"]["virial"]
        d_atom_virial = model_pred["water_dipole"]["atom_virial"]

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
        sess = self.test_session().__enter__()

        # test water energy
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([e_energy, e_force, e_virial], feed_dict=feed_dict_test)
        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        refe = [6.135449167779321300e01]
        reff = [
            7.799691562262310585e-02,
            9.423098804815030483e-02,
            3.790560997388224204e-03,
            1.432522403799846578e-01,
            1.148392791403983204e-01,
            -1.321871172563671148e-02,
            -7.318966526325138000e-02,
            6.516069212737778116e-02,
            5.406418483320515412e-04,
            5.870713761026503247e-02,
            -1.605402669549013672e-01,
            -5.089516979826595386e-03,
            -2.554593467731766654e-01,
            3.092063507347833987e-02,
            1.510355029451411479e-02,
            4.869271842355533952e-02,
            -1.446113274345035005e-01,
            -1.126524434771078789e-03,
        ]
        refv = [
            -6.076776685178300053e-01,
            1.103174323630009418e-01,
            1.984250991380156690e-02,
            1.103174323630009557e-01,
            -3.319759402259439551e-01,
            -6.007404107650986258e-03,
            1.984250991380157036e-02,
            -6.007404107650981921e-03,
            -1.200076017439753642e-03,
        ]
        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)

        # test water dipole
        [p, gp] = sess.run([d_dipole, d_gdipole], feed_dict=feed_dict_test)
        p = p.reshape([-1])
        refp = [
            1.616802262298876514e01,
            9.809535439521079425e00,
            3.572312180768947854e-01,
            1.336308874095981203e00,
            1.057908563208963848e01,
            -5.999602350098874881e-01,
        ]
        places = 10
        np.testing.assert_almost_equal(p, refp, places)
        gp = gp.reshape([-1])
        refgp = np.array(refp).reshape(-1, 3).sum(0)
        places = 9
        np.testing.assert_almost_equal(gp, refgp, places)

        # test water dipole : make sure only one frame is used
        feed_dict_single = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:1, :], [-1]),
            t_box: test_data["box"][:1, :],
            t_type: np.reshape(test_data["type"][:1, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        [pf, pv, pav] = sess.run(
            [d_force, d_virial, d_atom_virial], feed_dict=feed_dict_single
        )
        pf, pv = pf.reshape(-1), pv.reshape(-1)
        spv = pav.reshape(1, 3, -1, 9).sum(2).reshape(-1)

        base_dict = feed_dict_single.copy()
        coord0 = base_dict.pop(t_coord)
        box0 = base_dict.pop(t_box)

        fdf = -finite_difference(
            lambda coord: sess.run(
                d_gdipole, feed_dict={**base_dict, t_coord: coord, t_box: box0}
            ).reshape(-1),
            test_data["coord"][:numb_test, :].reshape([-1]),
        ).reshape(-1)
        fdv = -(
            finite_difference(
                lambda box: sess.run(
                    d_gdipole,
                    feed_dict={
                        **base_dict,
                        t_coord: strerch_box(coord0, box0, box),
                        t_box: box,
                    },
                ).reshape(-1),
                test_data["box"][:numb_test, :],
            )
            .reshape([-1, 3, 3])
            .transpose(0, 2, 1)
            @ box0.reshape(3, 3)
        ).reshape(-1)

        delta = 1e-5
        np.testing.assert_allclose(pf, fdf, delta)
        np.testing.assert_allclose(pv, fdv, delta)
        # make sure atomic virial sum to virial
        places = 10
        np.testing.assert_almost_equal(pv, spv, places)
