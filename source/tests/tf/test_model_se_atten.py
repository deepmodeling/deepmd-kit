# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.common import (
    j_must_have,
)
from deepmd.tf.descriptor import (
    DescrptSeAtten,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit import (
    EnerFitting,
)
from deepmd.tf.model import (
    EnerModel,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)

from .common import (
    DataSystem,
    check_smooth_efv,
    finite_difference_fv,
    gen_data,
    j_loader,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestModel(tf.test.TestCase):
    def setUp(self):
        gen_data()
        self.filename = __file__

    def test_model(self):
        jfile = "water_se_atten.json"
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
        jdata["model"]["descriptor"]["ntypes"] = 2
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix="test_model_se_atten",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_ener = model_pred["atom_ener"]

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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        # print(sess.run(model.type_embedding))
        # np.savetxt('tmp.out', sess.run(descrpt.dout, feed_dict = feed_dict_test), fmt='%.10e')
        # # print(sess.run(model.atype_embed, feed_dict = feed_dict_test))
        # print(sess.run(fitting.inputs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.outs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.atype_embed, feed_dict = feed_dict_test))

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        np.savetxt("e.out", e.reshape([1, -1]), delimiter=",")
        np.savetxt("f.out", f.reshape([1, -1]), delimiter=",")
        np.savetxt("v.out", v.reshape([1, -1]), delimiter=",")

        refe = [6.12116933882038480874e01]
        reff = [
            1.15647422509625782266e-02,
            1.75814420518816301453e-02,
            7.13827966845788537686e-04,
            2.37054385620869625950e-02,
            1.68638656843611636771e-02,
            -2.24281688803243482028e-03,
            -7.95826529019691246425e-03,
            9.69601584607941019422e-03,
            1.91505445834121360688e-05,
            8.71431743822387999687e-03,
            -2.71847766570148252629e-02,
            -8.84489238783629392812e-04,
            -4.38853499152154838403e-02,
            5.81882595276563344133e-03,
            2.62678184040523532775e-03,
            7.85911695413897895546e-03,
            -2.27753728780730156644e-02,
            -2.32454225018371036246e-04,
        ]
        refv = [
            -1.05000199239240685212e-01,
            1.67161895068729665637e-02,
            3.44771431604021759421e-03,
            1.67161895068729804414e-02,
            -5.42193765251950954287e-02,
            -1.08055824874348513701e-03,
            3.44771431604021802789e-03,
            -1.08055824874348470332e-03,
            -2.09534775642288966522e-04,
        ]

        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)

    def test_exclude_types(self):
        """In this test, we make type 0 has no interaction with type 0 and type 1,
        so the descriptor should be zero for type 0 atoms.
        """
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        ntypes = 2

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0], [0, 1]]

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        # successful
        descrpt = DescrptSeAtten(ntypes=ntypes, **jdata["model"]["descriptor"])
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        type_embedding = typeebd.build(
            ntypes, suffix=self.filename + "-" + inspect.stack()[0][3]
        )
        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix="_se_atten_exclude_types",
        )

        feed_dict_test1 = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [des] = sess.run([dout], feed_dict=feed_dict_test1)

        np.testing.assert_almost_equal(des[:, 0:2], 0.0, 10)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(des[:, 2:6], 0.0, 10)

    def test_compressible_model(self):
        jfile = "water_se_atten.json"
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
        jdata["model"]["descriptor"]["ntypes"] = 2
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 0
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix="test_model_se_atten_model_compressible",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_ener = model_pred["atom_ener"]

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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        # print(sess.run(model.type_embedding))
        # np.savetxt('tmp.out', sess.run(descrpt.dout, feed_dict = feed_dict_test), fmt='%.10e')
        # # print(sess.run(model.atype_embed, feed_dict = feed_dict_test))
        # print(sess.run(fitting.inputs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.outs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.atype_embed, feed_dict = feed_dict_test))

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        np.savetxt("e.out", e.reshape([1, -1]), delimiter=",")
        np.savetxt("f.out", f.reshape([1, -1]), delimiter=",")
        np.savetxt("v.out", v.reshape([1, -1]), delimiter=",")

        refe = [4.951981086834933166e01]
        reff = [
            3.706988425960650702e00,
            3.375774160760826259e00,
            1.239489759702384758e-01,
            2.575853678437920902e00,
            3.699539279116211166e00,
            -2.069005324163125936e-01,
            -4.258805446260704564e00,
            1.554731495837070154e00,
            2.737673623267052048e-02,
            2.450754822743671735e00,
            -5.057615189705980008e00,
            -1.869152757392671393e-01,
            -5.623845848960147720e00,
            1.555965710447468231e00,
            2.781927025028870237e-01,
            1.149054368078609167e00,
            -5.128395456455598023e00,
            -3.570260655021625928e-02,
        ]
        refv = [
            -1.829433693444095255e01,
            3.911090802878004258e00,
            4.731456035336862320e-01,
            3.911090802878003370e00,
            -1.103569683318792372e01,
            -2.277430677764266387e-01,
            4.731456035336862875e-01,
            -2.277430677764267775e-01,
            -2.613092934079439336e-02,
        ]

        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)

    def test_compressible_exclude_types(self):
        """In this test, we make type 0 has no interaction with type 0 and type 1,
        so the descriptor should be zero for type 0 atoms.
        """
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        ntypes = 2

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0], [0, 1]]

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        # successful
        descrpt = DescrptSeAtten(ntypes=ntypes, **jdata["model"]["descriptor"])
        typeebd_param = jdata["model"]["type_embedding"]
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 0
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        type_embedding = typeebd.build(
            ntypes, suffix=self.filename + "-" + inspect.stack()[0][3]
        )
        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix="_se_atten_compressible_exclude_types",
        )

        feed_dict_test1 = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [des] = sess.run([dout], feed_dict=feed_dict_test1)

        np.testing.assert_almost_equal(des[:, 0:2], 0.0, 10)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(des[:, 2:6], 0.0, 10)

    def test_stripped_type_embedding_model(self):
        jfile = "water_se_atten.json"
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
        jdata["model"]["descriptor"]["ntypes"] = 2
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 2
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "test_model_se_atten_model_compressible",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_ener = model_pred["atom_ener"]

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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        # print(sess.run(model.type_embedding))
        # np.savetxt('tmp.out', sess.run(descrpt.dout, feed_dict = feed_dict_test), fmt='%.10e')
        # # print(sess.run(model.atype_embed, feed_dict = feed_dict_test))
        # print(sess.run(fitting.inputs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.outs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.atype_embed, feed_dict = feed_dict_test))

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        np.savetxt("e.out", e.reshape([1, -1]), delimiter=",")
        np.savetxt("f.out", f.reshape([1, -1]), delimiter=",")
        np.savetxt("v.out", v.reshape([1, -1]), delimiter=",")

        refe = [6.12411774224343261608e01]
        reff = [
            8.63770820796567855016e-03,
            1.62522026393666710331e-02,
            7.22919459978568399415e-04,
            2.46800946249053909654e-02,
            1.50982535714741239463e-02,
            -2.27024703144847314271e-03,
            -6.23780053459390554371e-03,
            9.20020798171328375858e-03,
            -2.07267961671176406842e-05,
            6.19326848220238136006e-03,
            -2.50892401262326376898e-02,
            -7.84679030459893762407e-04,
            -4.10982573216296109830e-02,
            4.73129070889465389027e-03,
            2.56865811814534347399e-03,
            7.82498654115005611021e-03,
            -2.01927147752160932037e-02,
            -2.15924720048428232235e-04,
        ]
        refv = [
            -8.51412082980419621103e-02,
            1.39169542815959605686e-02,
            3.06329019931955021105e-03,
            1.39169542815959553644e-02,
            -4.91657098529515099172e-02,
            -9.54629874035841340107e-04,
            3.06329019931954847633e-03,
            -9.54629874035841340107e-04,
            -2.00155580095981352464e-04,
        ]

        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)

    def test_stripped_type_embedding_exclude_types(self):
        """In this test, we make type 0 has no interaction with type 0 and type 1,
        so the descriptor should be zero for type 0 atoms.
        """
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        ntypes = 2

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0], [0, 1]]

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        # successful
        descrpt = DescrptSeAtten(ntypes=ntypes, **jdata["model"]["descriptor"])
        typeebd_param = jdata["model"]["type_embedding"]
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 2
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        type_embedding = typeebd.build(
            ntypes,
            suffix=self.filename + "-" + inspect.stack()[0][3] + "_type_embedding",
        )
        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_compressible_exclude_types",
        )

        feed_dict_test1 = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [des] = sess.run([dout], feed_dict=feed_dict_test1)

        np.testing.assert_almost_equal(des[:, 0:2], 0.0, 10)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(des[:, 2:6], 0.0, 10)

    def test_smoothness_of_stripped_type_embedding_smooth_model(self):
        """test: auto-diff, continuity of e,f,v."""
        jfile = "water_se_atten.json"
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
        jdata["model"]["descriptor"]["ntypes"] = 2
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["smooth_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 1
        jdata["model"]["descriptor"]["rcut"] = 6.0
        jdata["model"]["descriptor"]["rcut_smth"] = 4.0
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "test_model_se_atten_model_compressible",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_ener = model_pred["atom_ener"]

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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [pe, pf, pv] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        pf, pv = pf.reshape(-1), pv.reshape(-1)

        eps = 1e-4
        delta = 1e-5
        fdf, fdv = finite_difference_fv(
            sess, energy, feed_dict_test, t_coord, t_box, delta=eps
        )
        np.testing.assert_allclose(pf, fdf, delta)
        np.testing.assert_allclose(pv, fdv, delta)

        tested_eps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        for eps in tested_eps:
            deltae = eps
            deltad = eps
            de, df, dv = check_smooth_efv(
                sess,
                energy,
                force,
                virial,
                feed_dict_test,
                t_coord,
                jdata["model"]["descriptor"]["rcut"],
                delta=eps,
            )
            np.testing.assert_allclose(de[0], de[1], rtol=0, atol=deltae)
            np.testing.assert_allclose(df[0], df[1], rtol=0, atol=deltad)
            np.testing.assert_allclose(dv[0], dv[1], rtol=0, atol=deltad)

        for eps in tested_eps:
            deltae = 5.0 * eps
            deltad = 5.0 * eps
            de, df, dv = check_smooth_efv(
                sess,
                energy,
                force,
                virial,
                feed_dict_test,
                t_coord,
                jdata["model"]["descriptor"]["rcut_smth"],
                delta=eps,
            )
            np.testing.assert_allclose(de[0], de[1], rtol=0, atol=deltae)
            np.testing.assert_allclose(df[0], df[1], rtol=0, atol=deltad)
            np.testing.assert_allclose(dv[0], dv[1], rtol=0, atol=deltad)

    def test_smoothness_of_stripped_type_embedding_smooth_model_excluded_types(self):
        """test: auto-diff, continuity of e,f,v."""
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["ntypes"] = 2
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["smooth_type_embedding"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 1
        jdata["model"]["descriptor"]["rcut"] = 6.0
        jdata["model"]["descriptor"]["rcut_smth"] = 4.0
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0], [0, 1]]
        jdata["model"]["descriptor"]["set_davg_zero"] = False
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )
        model = EnerModel(descrpt, fitting, typeebd)

        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
        model.descrpt.bias_atom_e = data.compute_energy_shift()
        # make the original implementation failed
        model.descrpt.davg[:] += 1e-1

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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "test_model_se_atten_model_compressible_excluded_types",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]

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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [pe, pf, pv] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        pf, pv = pf.reshape(-1), pv.reshape(-1)

        eps = 1e-4
        delta = 1e-6
        fdf, fdv = finite_difference_fv(
            sess, energy, feed_dict_test, t_coord, t_box, delta=eps
        )
        np.testing.assert_allclose(pf, fdf, delta)
        np.testing.assert_allclose(pv, fdv, delta)

        tested_eps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        for eps in tested_eps:
            deltae = 1e-15
            deltad = 1e-15
            de, df, dv = check_smooth_efv(
                sess,
                energy,
                force,
                virial,
                feed_dict_test,
                t_coord,
                jdata["model"]["descriptor"]["rcut"],
                delta=eps,
            )
            np.testing.assert_allclose(de[0], de[1], rtol=0, atol=deltae)
            np.testing.assert_allclose(df[0], df[1], rtol=0, atol=deltad)
            np.testing.assert_allclose(dv[0], dv[1], rtol=0, atol=deltad)
