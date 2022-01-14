import os,sys,platform,shutil,dpdata,json
import numpy as np
import unittest
import subprocess as sp

from deepmd.env import tf
from deepmd.infer import DeepPot
from common import j_loader, tests_path
from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.entrypoints.transfer import load_graph, transform_graph

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

def _file_delete(file) :
    if os.path.exists(file):
        os.remove(file)

def _subprocess_run(command):
    popen = sp.Popen(command.split(), shell=False, stdout=sp.PIPE, stderr=sp.STDOUT)
    for line in iter(popen.stdout.readline, b''):
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        line = line.rstrip()
        print(line)
    popen.wait()
    return popen.returncode

class TestTransform(unittest.TestCase) :
    @classmethod
    def setUpClass(self):
        self.old_model = str(tests_path / "dp-old.pb")
        self.raw_model = str(tests_path / "dp-raw.pb")
        self.new_model = str(tests_path / "dp-new.pb")
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot.pbtxt")), self.old_model)
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot-1.pbtxt")), self.raw_model)
        ret = _subprocess_run("dp transfer -O " + self.old_model + " -r " + self.raw_model + " -o " + self.new_model)
        np.testing.assert_equal(ret, 0, 'DP transfer failed!')

        self.dp = DeepPot(self.new_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_e = np.array([-9.275780747115504710e+01,-1.863501786584258468e+02,-1.863392472863538103e+02,-9.279281325486221021e+01,-1.863671545232153903e+02,-1.863619822847602165e+02])
        self.expected_f = np.array([-3.034045420701179663e-01,8.405844663871177014e-01,7.696947487118485642e-02,7.662001266663505117e-01,-1.880601391333554251e-01,-6.183333871091722944e-01,-5.036172391059643427e-01,-6.529525836149027151e-01,5.432962643022043459e-01,6.382357912332115024e-01,-1.748518296794561167e-01,3.457363524891907125e-01,1.286482986991941552e-03,3.757251165286925043e-01,-5.972588700887541124e-01,-5.987006197104716154e-01,-2.004450304880958100e-01,2.495901655353461868e-01])
        self.expected_v = np.array([-2.912234126853306959e-01,-3.800610846612756388e-02,2.776624987489437202e-01,-5.053761003913598976e-02,-3.152373041953385746e-01,1.060894290092162379e-01,2.826389131596073745e-01,1.039129970665329250e-01,-2.584378792325942586e-01,-3.121722367954994914e-01,8.483275876786681990e-02,2.524662342344257682e-01,4.142176771106586414e-02,-3.820285230785245428e-02,-2.727311173065460545e-02,2.668859789777112135e-01,-6.448243569420382404e-02,-2.121731470426218846e-01,-8.624335220278558922e-02,-1.809695356746038597e-01,1.529875294531883312e-01,-1.283658185172031341e-01,-1.992682279795223999e-01,1.409924999632362341e-01,1.398322735274434292e-01,1.804318474574856390e-01,-1.470309318999652726e-01,-2.593983661598450730e-01,-4.236536279233147489e-02,3.386387920184946720e-02,-4.174017537818433543e-02,-1.003500282164128260e-01,1.525690815194478966e-01,3.398976109910181037e-02,1.522253908435125536e-01,-2.349125581341701963e-01,9.515545977581392825e-04,-1.643218849228543846e-02,1.993234765412972564e-02,6.027265332209678569e-04,-9.563256398907417355e-02,1.510815124001868293e-01,-7.738094816888557714e-03,1.502832772532304295e-01,-2.380965783745832010e-01,-2.309456719810296654e-01,-6.666961081213038098e-02,7.955566551234216632e-02,-8.099093777937517447e-02,-3.386641099800401927e-02,4.447884755740908608e-02,1.008593228579038742e-01,4.556718179228393811e-02,-6.078081273849572641e-02])

    @classmethod
    def tearDownClass(self):
        _file_delete(self.old_model)
        _file_delete(self.raw_model)
        _file_delete(self.new_model)

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        # check values
        np.testing.assert_almost_equal(ff.ravel(), self.expected_f.ravel(), default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_1frame_atm(self):
        ee, ff, vv, ae, av = self.dp.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        self.assertEqual(ae.shape, (nframes,natoms,1))
        self.assertEqual(av.shape, (nframes,natoms,9))
        # check values
        np.testing.assert_almost_equal(ff.ravel(), self.expected_f.ravel(), default_places)
        np.testing.assert_almost_equal(ae.ravel(), self.expected_e.ravel(), default_places)
        np.testing.assert_almost_equal(av.ravel(), self.expected_v.ravel(), default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)


    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee, ff, vv, ae, av = self.dp.eval(coords2, box2, self.atype, atomic = True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        self.assertEqual(ae.shape, (nframes,natoms,1))
        self.assertEqual(av.shape, (nframes,natoms,9))
        # check values
        expected_f = np.concatenate((self.expected_f, self.expected_f), axis = 0)
        expected_e = np.concatenate((self.expected_e, self.expected_e), axis = 0)
        expected_v = np.concatenate((self.expected_v, self.expected_v), axis = 0)
        np.testing.assert_almost_equal(ff.ravel(), expected_f.ravel(), default_places)
        np.testing.assert_almost_equal(ae.ravel(), expected_e.ravel(), default_places)
        np.testing.assert_almost_equal(av.ravel(), expected_v.ravel(), default_places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis = 1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis = 1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

