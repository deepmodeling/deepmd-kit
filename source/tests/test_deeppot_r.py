import os,sys,platform,shutil,dpdata
import numpy as np
import unittest

from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.infer import DeepPot
from common import tests_path

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

class TestDeepPotRPBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot-r.pbtxt")), "deeppot.pb")
        self.dp = DeepPot("deeppot.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_e = np.array([
            -9.320909762801588272e+01,-1.868020345400987878e+02,-1.868011172371355997e+02,-9.320868430396934912e+01,-1.868010398844378415e+02,-1.868016706555875999e+02
        ])
        self.expected_f = np.array([
            6.385312846474267391e-04,-6.460452911141417731e-03,-5.652405655332678417e-04,-7.516468794343579736e-03,1.128804614240160216e-03,5.531937784564192051e-03,1.914138124904981664e-03,5.601819906021693503e-03,-5.131359585752605541e-03,-4.847104424804288617e-03,1.992071550328819614e-03,-4.028159855157302516e-03,1.236340684486603517e-03,-5.373955841338794344e-03,8.312829460571366513e-03,8.574563125108854156e-03,3.111712681889538742e-03,-4.120007238692381148e-03
        ])
        self.expected_v = np.array([
            5.844056241889131371e-03,4.663973497239899614e-04,-2.268382127762904633e-03,4.663973497239897988e-04,2.349338784202595950e-03,-6.908546513234039253e-04,-2.268382127762904633e-03,-6.908546513234039253e-04,2.040499248150800561e-03,4.238130266437327605e-03,-1.539867187443782223e-04,-2.393101333240631613e-03,-1.539867187443782223e-04,4.410341945447907377e-04,9.544239698119633068e-06,-2.393101333240631613e-03,9.544239698119578858e-06,1.877785959095269654e-03,5.798992562057291543e-03,6.943392552230453693e-04,-1.180376879311998773e-03,6.943392552230453693e-04,1.686725132156275536e-03,-1.461632060145726542e-03,-1.180376879311998556e-03,-1.461632060145726325e-03,1.749543733794208444e-03,7.173915604192910439e-03,3.903218041111061569e-04,-5.747400467123527524e-04,3.903218041111061569e-04,1.208289706621179949e-03,-1.826828914132010932e-03,-5.747400467123527524e-04,-1.826828914132011148e-03,2.856960586657185906e-03,4.067553030177322240e-03,-3.267469855253819430e-05,-6.980667859103454904e-05,-3.267469855253830272e-05,1.387653029234650918e-03,-2.096820720698671855e-03,-6.980667859103444062e-05,-2.096820720698671855e-03,3.218305506720191278e-03,4.753992590355240674e-03,1.224911338353675992e-03,-1.683421934571502484e-03,1.224911338353676209e-03,7.332113564901583539e-04,-1.025577052190138451e-03,-1.683421934571502484e-03,-1.025577052190138234e-03,1.456681925652047018e-03
        ])

    def tearDown(self):
        os.remove("deeppot.pb")
    
    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)

    # def test_1frame(self):
    #     ee, ff, vv, ae, av = self.dp.eval(self.coords, self.box, self.atype, atomic = True)
    #     np.savetxt('ee.out', ae.reshape([1, -1]), delimiter=',')
    #     np.savetxt('ff.out', ff.reshape([1, -1]), delimiter=',')
    #     np.savetxt('vv.out', av.reshape([1, -1]), delimiter=',')

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        # check values
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)

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
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(ae.size):
            self.assertAlmostEqual(ae.reshape([-1])[ii], self.expected_e.reshape([-1])[ii], places = default_places)
        for ii in range(av.size):
            self.assertAlmostEqual(av.reshape([-1])[ii], self.expected_v.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)


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
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(ae.size):
            self.assertAlmostEqual(ae.reshape([-1])[ii], expected_e.reshape([-1])[ii], places = default_places)
        for ii in range(av.size):
            self.assertAlmostEqual(av.reshape([-1])[ii], expected_v.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)


class TestDeepPotRNoPBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot-r.pbtxt")), "deeppot.pb")
        self.dp = DeepPot("deeppot.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = None
        self.expected_e = np.array([
            -9.321213823508108476e+01,-1.868044102481340758e+02,-1.868067983858651075e+02,-9.320899631301440991e+01,-1.868014559732615112e+02,-1.868017660713088617e+02
        ])
        self.expected_f = np.array([            
            4.578151103701261042e-03,-1.917874111009987628e-03,-3.464546781179331930e-03,-4.578151103701261042e-03,1.917874111009987628e-03,3.464546781179331930e-03,-2.624402581721222913e-03,3.566275128489623933e-04,-2.859315986763691776e-04,-5.767787273464367384e-03,1.907053583551196647e-03,-3.889064429673861831e-03,1.786820066350549132e-04,-5.327197473636275694e-03,8.236236182834734409e-03,8.213507848550535492e-03,3.063516377236116545e-03,-4.061240154484504865e-03
        ])
        self.expected_v = np.array([        
            1.984979026299632174e-03,-8.315452677741701822e-04,-1.502146290172694243e-03,-8.315452677741700738e-04,3.483500446080982317e-04,6.292774999372096039e-04,-1.502146290172694243e-03,6.292774999372097123e-04,1.136759354725281907e-03,1.402852790439301908e-03,-5.876815743732210226e-04,-1.061618327900012114e-03,-5.876815743732211311e-04,2.461909298049979960e-04,4.447320022283834766e-04,-1.061618327900012331e-03,4.447320022283834766e-04,8.033868427351443728e-04,4.143606961846296385e-03,-5.511382161123719835e-04,4.465413399437045397e-04,-5.511382161123719835e-04,1.082271054025323839e-04,-1.097918001262628728e-04,4.465413399437046481e-04,-1.097918001262628728e-04,1.220966982358671871e-04,5.263952004497593831e-03,2.395243710938091842e-04,-2.830378939414603329e-04,2.395243710938094010e-04,1.189969706598244898e-03,-1.805627331015851201e-03,-2.830378939414602245e-04,-1.805627331015851635e-03,2.801996513751836820e-03,2.208413501170402270e-03,5.331756287635716889e-05,-1.664423506603235218e-04,5.331756287635695205e-05,1.379626072862918072e-03,-2.094132943741625064e-03,-1.664423506603234133e-04,-2.094132943741625064e-03,3.199787996743366607e-03,4.047014004814953811e-03,1.137904999421357000e-03,-1.568106936614101698e-03,1.137904999421357217e-03,7.205982843216952307e-04,-1.011174600268313238e-03,-1.568106936614101698e-03,-1.011174600268313238e-03,1.435226522157425754e-03            
        ])

    def tearDown(self):
        os.remove("deeppot.pb")
    
    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        # check values
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)

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
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(ae.size):
            self.assertAlmostEqual(ae.reshape([-1])[ii], self.expected_e.reshape([-1])[ii], places = default_places)
        for ii in range(av.size):
            self.assertAlmostEqual(av.reshape([-1])[ii], self.expected_v.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)


    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        ee, ff, vv, ae, av = self.dp.eval(coords2, self.box, self.atype, atomic = True)
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
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(ae.size):
            self.assertAlmostEqual(ae.reshape([-1])[ii], expected_e.reshape([-1])[ii], places = default_places)
        for ii in range(av.size):
            self.assertAlmostEqual(av.reshape([-1])[ii], expected_v.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)

    
class TestDeepPotRLargeBoxNoPBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot-r.pbtxt")), "deeppot.pb")
        self.dp = DeepPot("deeppot.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([19., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_e = np.array([
            -9.321213823508108476e+01,-1.868044102481340758e+02,-1.868067983858651075e+02,-9.320899631301440991e+01,-1.868014559732615112e+02,-1.868017660713088617e+02
        ])
        self.expected_f = np.array([            
            4.578151103701261042e-03,-1.917874111009987628e-03,-3.464546781179331930e-03,-4.578151103701261042e-03,1.917874111009987628e-03,3.464546781179331930e-03,-2.624402581721222913e-03,3.566275128489623933e-04,-2.859315986763691776e-04,-5.767787273464367384e-03,1.907053583551196647e-03,-3.889064429673861831e-03,1.786820066350549132e-04,-5.327197473636275694e-03,8.236236182834734409e-03,8.213507848550535492e-03,3.063516377236116545e-03,-4.061240154484504865e-03            
        ])
        self.expected_v = np.array([        
            1.984979026299632174e-03,-8.315452677741701822e-04,-1.502146290172694243e-03,-8.315452677741700738e-04,3.483500446080982317e-04,6.292774999372096039e-04,-1.502146290172694243e-03,6.292774999372097123e-04,1.136759354725281907e-03,1.402852790439301908e-03,-5.876815743732210226e-04,-1.061618327900012114e-03,-5.876815743732211311e-04,2.461909298049979960e-04,4.447320022283834766e-04,-1.061618327900012331e-03,4.447320022283834766e-04,8.033868427351443728e-04,4.143606961846296385e-03,-5.511382161123719835e-04,4.465413399437045397e-04,-5.511382161123719835e-04,1.082271054025323839e-04,-1.097918001262628728e-04,4.465413399437046481e-04,-1.097918001262628728e-04,1.220966982358671871e-04,5.263952004497593831e-03,2.395243710938091842e-04,-2.830378939414603329e-04,2.395243710938094010e-04,1.189969706598244898e-03,-1.805627331015851201e-03,-2.830378939414602245e-04,-1.805627331015851635e-03,2.801996513751836820e-03,2.208413501170402270e-03,5.331756287635716889e-05,-1.664423506603235218e-04,5.331756287635695205e-05,1.379626072862918072e-03,-2.094132943741625064e-03,-1.664423506603234133e-04,-2.094132943741625064e-03,3.199787996743366607e-03,4.047014004814953811e-03,1.137904999421357000e-03,-1.568106936614101698e-03,1.137904999421357217e-03,7.205982843216952307e-04,-1.011174600268313238e-03,-1.568106936614101698e-03,-1.011174600268313238e-03,1.435226522157425754e-03            
        ])

    def tearDown(self):
        os.remove("deeppot.pb")
    
    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        # check values
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)

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
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(ae.size):
            self.assertAlmostEqual(ae.reshape([-1])[ii], self.expected_e.reshape([-1])[ii], places = default_places)
        for ii in range(av.size):
            self.assertAlmostEqual(av.reshape([-1])[ii], self.expected_v.reshape([-1])[ii], places = default_places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis = 1)
        for ii in range(nframes):
            self.assertAlmostEqual(ee.reshape([-1])[ii], expected_se.reshape([-1])[ii], places = default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis = 1)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv.reshape([-1])[ii], expected_sv.reshape([-1])[ii], places = default_places)


