import os,sys,platform,shutil,dpdata
import numpy as np
import unittest

from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.infer import DipoleChargeModifier
from common import tests_path

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

class TestDipoleCharge(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","dipolecharge_d.pbtxt")), "dipolecharge_d.pb")
        self.dp = DipoleChargeModifier(
            "dipolecharge_d.pb", 
            [-1.0, -3.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            4.0,
            0.2
        )
        self.coords = np.array([
            4.6067455554,    8.8719311819,    6.3886531197,
            4.0044515745,    4.2449530507,    7.7902855220,
            2.6453069446,    0.8772647726,    1.2804446790,
            1.1445332290,    0.0067366438,    1.8606485070,
            7.1002867706,    5.0325506787,    3.1805888348,
            4.5352891138,    7.7389683929,    9.4260970128,
            2.1833238914,    9.0916071034,    7.2299906064,
            4.1040157820,    1.0496745045,    5.4748315591,
        ], dtype = np.float64)
            # 1.1445332290,    0.0067366438,    1.8606485070,
            # 2.1833238914,    9.0916071034,    7.2299906064,
            # 4.0044515745,    4.2449530507,    7.7902855220,
            # 7.1002867706,    5.0325506787,    3.1805888348,
        self.atype = np.array([0,3,2,1,3,4,1,4], dtype=int)
        self.box = np.array([10., 0., 0., 0., 10., 0., 0., 0., 10.])
        self.expected_e = np.array([
            3.671081837126222158e+00
        ])
        self.expected_f = np.array([
            8.786854427753210128e-01,-1.590752486903602159e-01,-2.709225006303785932e-01,-4.449513960033193438e-01,-1.564291540964127813e-01,2.139031741772115178e-02,1.219699614140521193e+00,-5.580358618499958734e-02,-3.878662478349682585e-01,-1.286685244990778854e+00,1.886475802950296488e-01,3.904450515493615437e-01,1.605017382138404849e-02,2.138016869742287995e-01,-2.617514921203008965e-02,2.877081057057793712e-01,-3.846449683844421763e-01,3.048855616906603894e-02,-9.075632811311897807e-01,-6.509653472431625731e-03,2.302010972126376787e-01,2.370565856822822726e-01,3.600133435593881881e-01,1.243887532859055609e-02
        ])
        self.expected_v = np.array([
            3.714071471995848417e-01,6.957130186032146613e-01,-1.158289779017217302e+00,6.957130186032139951e-01,-1.400130091653774933e+01,-3.631620234653316626e-01,-1.158289779017217302e+00,-3.631620234653316626e-01,3.805077486043773050e+00
        ])
        self.natoms = self.atype.size
        self.coords = self.coords.reshape([-1, self.natoms, 3])

    def tearDown(self):
        os.remove("dipolecharge_d.pb")

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 5)
        self.assertAlmostEqual(self.dp.get_rcut(), 4.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['A', 'B', 'C', 'D', 'E'])

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, eval_fv = True
)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        self.assertEqual(self.expected_e.shape, (nframes,))
        self.assertEqual(self.expected_f.shape, (nframes*natoms*3,))
        self.assertEqual(self.expected_v.shape, (nframes*9,))
        # np.savetxt('ee.out', ee.reshape([1, -1]), delimiter=',')
        # np.savetxt('ff.out', ff.reshape([1, -1]), delimiter=',')
        # np.savetxt('vv.out', vv.reshape([1, -1]), delimiter=',')
        ee = ee.reshape([-1])
        ff = ff.reshape([-1])
        vv = vv.reshape([-1])        
        for ii in range(ee.size):
            self.assertAlmostEqual(ee[ii], self.expected_e[ii])
        for ii in range(ff.size):
            self.assertAlmostEqual(ff[ii], self.expected_f[ii])
        for ii in range(vv.size):
            self.assertAlmostEqual(vv[ii], self.expected_v[ii])

    def test_2frame(self):
        nframes = 2
        self.coords = np.tile(self.coords, [nframes, 1, 1])
        self.box = np.tile(self.box, [nframes, 1])
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, eval_fv = True
)
        # check shape of the returns
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        self.expected_e = np.tile(self.expected_e, [nframes])
        self.expected_f = np.tile(self.expected_f, [nframes])
        self.expected_v = np.tile(self.expected_v, [nframes])
        self.assertEqual(self.expected_e.shape, (nframes,))
        self.assertEqual(self.expected_f.shape, (nframes*natoms*3,))
        self.assertEqual(self.expected_v.shape, (nframes*9,))
        ee = ee.reshape([-1])
        ff = ff.reshape([-1])
        vv = vv.reshape([-1])
        for ii in range(ee.size):
            self.assertAlmostEqual(ee[ii], self.expected_e[ii])
        for ii in range(ff.size):
            self.assertAlmostEqual(ff[ii], self.expected_f[ii])
        for ii in range(vv.size):
            self.assertAlmostEqual(vv[ii], self.expected_v[ii])

