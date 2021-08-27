import os,sys,platform,shutil,dpdata
import numpy as np
import unittest

from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.infer import DeepDipole
from common import tests_path, finite_difference, strerch_box, tf
from packaging.version import parse as parse_version

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

class TestDeepDipolePBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deepdipole.pbtxt")), "deepdipole.pb")
        self.dp = DeepDipole("deepdipole.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_d = np.array([-9.274180565967479195e-01,2.698028341272042496e+00,2.521268387140979117e-01,2.927260638453461628e+00,-8.571926301526779923e-01,1.667785136187720063e+00])

    def tearDown(self):
        os.remove("deepdipole.pb")    

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 4.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_sel_type(), [0])

    def test_1frame_atm(self):
        dd = self.dp.eval(self.coords, self.box, self.atype)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], self.expected_d.reshape([-1])[ii], places = default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        dd = self.dp.eval(coords2, box2, self.atype)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        expected_d = np.concatenate((self.expected_d, self.expected_d))
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], expected_d.reshape([-1])[ii], places = default_places)


class TestDeepDipoleNoPBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deepdipole.pbtxt")), "deepdipole.pb")
        self.dp = DeepDipole("deepdipole.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([20., 0., 0., 0., 20., 0., 0., 0., 20.])
        self.expected_d = np.array([-1.982092647058316e+00, 8.303361089028074e-01, 1.499962003179265e+00, 2.927112547154802e+00, -8.572096473802318e-01, 1.667798310054391e+00])

    def tearDown(self):
        os.remove("deepdipole.pb")    

    def test_1frame_atm(self):
        dd = self.dp.eval(self.coords, None, self.atype)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], self.expected_d.reshape([-1])[ii], places = default_places)

    def test_1frame_atm_large_box(self):
        dd = self.dp.eval(self.coords, self.box, self.atype)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], self.expected_d.reshape([-1])[ii], places = default_places)


@unittest.skipIf(parse_version(tf.__version__) < parse_version("1.15"), 
    f"The current tf version {tf.__version__} is too low to run the new testing model.")
class TestDeepDipoleNewPBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deepdipole_new.pbtxt")), "deepdipole_new.pb")
        self.dp = DeepDipole("deepdipole_new.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.nout = 3
        self.atype = np.array([0, 1, 1, 0, 1, 1])
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_t = np.array([-1.128427726201255282e-01, 2.654103846999197880e-01, 2.625816377288122533e-02, 3.027556488877700680e-01, -7.475444785689989990e-02, 1.526291164572509684e-01])
        self.expected_f = np.array([8.424897862241968738e-02, -3.823566783202275721e-02, 3.570797165027734810e-01, 6.102563129736437997e-02, -1.351209759852018133e-01, -2.438224487466488510e-01, -1.403204771681088869e-01, 1.719596545791735875e-01, -1.136584427103610045e-01, 2.761686212947551955e-02, -7.247860200915196005e-02, 6.208831127377397591e-02, -2.605870723577520809e-01, -4.504074577536486268e-02, 7.340240097998475266e-02, 2.280160774766013809e-01, 1.189163370225677641e-01, -1.350895372995223886e-01, -4.294311497114180337e-02, 1.524802094783661577e-01, 1.070451777645946290e-01, -1.259336332521076574e-01, -2.087610788959351760e-01, 9.447141346538817652e-02, 1.668125597515543457e-01, 5.487037060760904805e-02, -2.014994036104674757e-01, -7.411985441205551361e-02, 3.614456658821710300e-01, 2.901174891391154476e-01, -4.871926969937838414e-02, -1.252747945819455699e-01, -2.555459318266457558e-01, 1.249033125831290059e-01, -2.347603724902655176e-01, -3.458874493198500766e-02, 3.563990394229877290e-01, 1.052342031228763047e-01, 1.907268232932498031e-01, -2.432737821373903708e-01, 1.016781829972335099e-01, -7.707616437996064884e-02, -1.139199805053340564e-01, -2.068592154909300040e-01, -1.156337826476897951e-01, 6.583817133933017596e-02, 2.902207490750204344e-01, 9.945482314729316153e-02, 7.986986504051810098e-02, -2.549975565538568079e-01, 1.275343199697696051e-01, -1.449133131601115787e-01, -3.527636315034351350e-02, -2.250060193826620980e-01])
        self.expected_v = np.array([3.479789535931299138e-02, 4.337414719007849292e-03, -3.647371468256610082e-03, 8.053492919528318708e-03, 1.003834811499279773e-03, -8.441338187607602033e-04, -6.695998268698949256e-03, -8.346286793845711892e-04, 7.018468440279366279e-04, -4.515896716004976635e-02, 1.891794570218296306e-02, 3.417435352652402336e-02, 9.998952222904963771e-02, -4.188750255541257711e-02, -7.566774655171297492e-02, 1.804286120725206444e-01, -7.558495911146115298e-02, -1.365405712981232755e-01, -1.002593446510361419e-01, -1.117945222697993429e-01, 7.449172735713084637e-02, 7.770237313970995707e-02, 1.313723119887387492e-01, -8.655414676270002661e-02, -4.973937467461287537e-02, -8.663006083493235421e-02, 5.703914957966123994e-02, -3.382231967662072125e-02, -4.215813217482468345e-03, 3.545115660155720612e-03, -8.247565860499378454e-03, -1.028025206407854253e-03, 8.644757417520612143e-04, 6.761330949063471332e-03, 8.427721296283078580e-04, -7.086947453692606178e-04, -1.622698090933780493e-02, 1.305372051650728060e-01, -2.082599910094798112e-01, -7.109985131471197733e-03, 2.202585658101286273e-02, -3.554509763049529952e-02, 1.436400379134906459e-02, -3.554915857551419617e-02, 5.763638171798115412e-02, 2.074946305037073946e-01, 5.016353704485233822e-02, -5.700401936915034523e-02, 1.082138666905367308e-01, 2.616159414496492877e-02, -2.972908425564194101e-02, -1.229314789425654392e-01, -2.971969820589494271e-02, 3.377238432488059716e-02, 7.622024445219390681e-03, 9.500540384976005961e-04, -7.989090778275298932e-04, -2.952148931042387209e-02, -3.679732378636401541e-03, 3.094320409307891630e-03, -9.534268115386618486e-04, -1.188407357158671420e-04, 9.993425503379762414e-05, 9.319088860655992679e-02, -3.903942630815338682e-02, -7.052283462118023871e-02, 1.544831983829924038e-01, -6.471593445773991815e-02, -1.169062041817236081e-01, -6.990884596438741438e-02, 2.928613817427033750e-02, 5.290399154061733306e-02, 7.491400658274136037e-02, 1.273824184577304897e-01, -8.391492311946648075e-02, 3.543872837542783732e-02, 4.324623973455964804e-02, -2.873418641045778418e-02, -8.444981234074398768e-02, -1.531171183141288306e-01, 1.007308415346981068e-01, -6.396885751015785743e-03, -7.973455327045167592e-04, 6.704951070469818575e-04, 2.915483242551994078e-02, 3.634030104030812076e-03, -3.055888951116827318e-03, 6.608747470375698129e-04, 8.237532257692081912e-05, -6.927015762150179410e-05, -6.099175331115514430e-03, 2.402310352789886402e-02, -3.861491558256636286e-02, -2.583867422346154685e-02, 6.050621302336450097e-02, -9.822840263095998503e-02, -3.827994718203701213e-02, 1.252239810257823327e-01, -2.018867305507059950e-01, 1.136620144506474833e-01, 2.747872876828840599e-02, -3.122582814578225147e-02, -2.136319389661417989e-01, -5.164728194785846160e-02, 5.869009312256637939e-02, -3.147575788810638014e-02, -7.609523885036708832e-03, 8.647186232996251914e-03, -5.990706138603461330e-03, -7.467169124604876177e-04, 6.279210400235934152e-04, -9.287887182821588476e-04, -1.157696985960763821e-04, 9.735179200124630735e-05, -2.966271471326579340e-02, -3.697335544996301071e-03, 3.109123071928715683e-03, 1.800225987816693740e-01, -7.541487246259104271e-02, -1.362333179969384966e-01, -7.524185541795300192e-02, 3.152023672914239238e-02, 5.693978247845072477e-02, 5.703636164117102669e-02, -2.389361095778780308e-02, -4.316265205277792366e-02, -4.915584336537091176e-02, -8.674240294138457763e-02, 5.709724154860432860e-02, -8.679070528401405804e-02, -1.572017650485294793e-01, 1.034201569997979520e-01, -3.557746655862283752e-02, -8.626268394893003844e-02, 5.645546718878535764e-02, 6.848075985139651621e-03, 8.535845420570665554e-04, -7.177870012752625602e-04, 8.266638576582277997e-04, 1.030402542123569647e-04, -8.664748649675494882e-05, 2.991751925173294011e-02, 3.729095884068693231e-03, -3.135830629785046203e-03, 1.523793442834292522e-02, -3.873020552543556677e-02, 6.275576045602117292e-02, -3.842536616563556329e-02, 1.249268983543572881e-01, -2.014296501045876875e-01, 1.288704808602599873e-02, -6.326999354443738066e-02, 1.014064886873057153e-01, -1.318711149757016143e-01, -3.188092889522457091e-02, 3.622832829002789468e-02, -3.210149046681261276e-02, -7.760799893075580151e-03, 8.819090787585878374e-03, -2.047554776382226327e-01, -4.950132426418570042e-02, 5.625150484566552450e-02])
        self.expected_gt = self.expected_t.reshape(-1, self.nout).sum(0).reshape(-1)
        self.expected_gv = self.expected_v.reshape(1, self.nout, 6, 9).sum(-2).reshape(-1)

    def tearDown(self):
        os.remove("deepdipole_new.pb")    

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 4.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_sel_type(), [0])

    def test_1frame_old(self):
        gt = self.dp.eval(self.coords, self.box, self.atype, atomic=False)
        # check shape of the returns
        nframes = 1
        self.assertEqual(gt.shape, (nframes,self.nout))
        # check values
        for ii in range(gt.size):
            self.assertAlmostEqual(gt.reshape([-1])[ii], self.expected_gt.reshape([-1])[ii], places = default_places)

    def test_1frame_old_atm(self):
        at = self.dp.eval(self.coords, self.box, self.atype)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        # check values
        for ii in range(at.size):
            self.assertAlmostEqual(at.reshape([-1])[ii], self.expected_t.reshape([-1])[ii], places = default_places)

    def test_2frame_old_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        at = self.dp.eval(coords2, box2, self.atype)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        # check values
        expected_d = np.concatenate((self.expected_t, self.expected_t))
        for ii in range(at.size):
            self.assertAlmostEqual(at.reshape([-1])[ii], expected_d.reshape([-1])[ii], places = default_places)

    def test_1frame_full(self):
        gt, ff, vv = self.dp.eval_full(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        # check values
        for ii in range(ff.size):
            self.assertAlmostEqual(ff.reshape([-1])[ii], self.expected_f.reshape([-1])[ii], places = default_places)
        for ii in range(gt.size):
            self.assertAlmostEqual(gt.reshape([-1])[ii], self.expected_gt.reshape([-1])[ii], places = default_places)
        for ii in range(vv.size):
            self.assertAlmostEqual(vv.reshape([-1])[ii], self.expected_gv.reshape([-1])[ii], places = default_places)

    def test_1frame_full_atm(self):
        gt, ff, vv, at, av = self.dp.eval_full(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        self.assertEqual(av.shape, (nframes,self.nout,natoms,9))
        # check values
        np.testing.assert_almost_equal(ff.reshape([-1]), self.expected_f.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(at.reshape([-1]), self.expected_t.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(av.reshape([-1]), self.expected_v.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(gt.reshape([-1]), self.expected_gt.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(vv.reshape([-1]), self.expected_gv.reshape([-1]), decimal = default_places)

    def test_1frame_full_atm_shuffle(self):
        i_sf = [2,1,3,0,5,4]
        isel_sf = [1,0]
        gt, ff, vv, at, av = self.dp.eval_full(self.coords.reshape(-1,3)[i_sf].reshape(-1), self.box, self.atype[i_sf], atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        self.assertEqual(av.shape, (nframes,self.nout,natoms,9))
        # recover the shuffled result
        nff = np.empty_like(ff)
        nav = np.empty_like(av)
        nat = np.empty_like(at)
        nff[:, :, i_sf] = ff
        nav[:, :, i_sf] = av
        nat[:, isel_sf] = at
        # check values
        np.testing.assert_almost_equal(nff.reshape([-1]), self.expected_f.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(nat.reshape([-1]), self.expected_t.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(nav.reshape([-1]), self.expected_v.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(gt.reshape([-1]), self.expected_gt.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(vv.reshape([-1]), self.expected_gv.reshape([-1]), decimal = default_places)

    def test_1frame_num_deriv(self):
        # numerical force
        num_f = - finite_difference(
            lambda coord: self.dp.eval(coord, self.box, self.atype, atomic=False).reshape(-1),
            self.coords
        ).reshape(-1)
        np.testing.assert_allclose(num_f.reshape([-1]), self.expected_f.reshape([-1]), atol=1e-5)
        # numerical virial
        num_v = - (finite_difference(
            lambda box: self.dp.eval(strerch_box(self.coords, self.box, box), box, self.atype, atomic=False).reshape(-1),
            self.box
        ).reshape(-1, 3, 3).transpose(0,2,1) @ self.box.reshape(3,3)).reshape(-1)
        np.testing.assert_allclose(num_v.reshape([-1]), self.expected_gv.reshape([-1]), atol=1e-5)

    def test_2frame_full_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        gt, ff, vv, at, av = self.dp.eval_full(coords2, box2, self.atype, atomic = True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        self.assertEqual(av.shape, (nframes,self.nout,natoms,9))
        # check values
        expected_f = np.tile(self.expected_f.reshape(-1), nframes)
        expected_t = np.tile(self.expected_t.reshape(-1), nframes)
        expected_v = np.tile(self.expected_v.reshape(-1), nframes)
        expected_gt = np.tile(self.expected_gt.reshape(-1), nframes)
        expected_gv = np.tile(self.expected_gv.reshape(-1), nframes)
        np.testing.assert_almost_equal(ff.reshape([-1]), expected_f.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(at.reshape([-1]), expected_t.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(av.reshape([-1]), expected_v.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(gt.reshape([-1]), expected_gt.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(vv.reshape([-1]), expected_gv.reshape([-1]), decimal = default_places)


@unittest.skipIf(parse_version(tf.__version__) < parse_version("1.15"), 
    f"The current tf version {tf.__version__} is too low to run the new testing model.")
class TestDeepDipoleFakePBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deepdipole_fake.pbtxt")), "deepdipole_fake.pb")
        self.dp = DeepDipole("deepdipole_fake.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.nout = 3
        self.atype = np.array([0, 1, 1, 0, 1, 1])
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_t = np.array([-3.186217894664857830e-01, 1.082220317383403296e+00, 5.646623185237639730e-02, 7.426508038929955369e-01, -3.115996324658170114e-01, -5.619108089573777720e-01, -4.181578166874897473e-01, -7.579762930974662805e-01, 4.980618433125854616e-01, 1.059635561913792712e+00, -2.641989315855929332e-01, 5.307984468104405273e-01, -1.484512535335152095e-01, 4.978588497891502374e-01, -8.022467807199461509e-01, -9.165936539882671985e-01, -2.238112120606238209e-01, 2.553133145814526217e-01])
        self.expected_f = np.array([5.041930370187270860e-01, 7.873825190365474347e-03, -4.096376607074713183e-01, -3.904160887819057568e-01, 1.651276463640535541e-01, 2.941164542146633698e-01, -1.137769482368212182e-01, -1.730014715544191672e-01, 1.155212064928080040e-01, 5.863332521864410563e-01, 8.527372103998451247e-02, -6.934420009023331555e-02, -1.225415636916203990e-02, 4.321720022314819165e-02, -7.184309080594213082e-02, -5.740790958172790059e-01, -1.284909212631327180e-01, 1.411872908961754325e-01, 1.394536521676267848e-02, 4.089695733795025712e-01, -8.790828175074971718e-02, 1.594305121314434359e-01, -7.202915091075953735e-02, -1.198685751141350120e-01, -1.733758773482060866e-01, -3.369404224687432281e-01, 2.077768568648848124e-01, 8.892382475507179529e-02, 1.801380487829997712e-01, -3.123469659869602677e-01, 5.864597608298829229e-02, -1.422803757045815187e-01, 2.644907470171818931e-01, -1.475698008380600668e-01, -3.785767307841875901e-02, 4.785621896977837464e-02, -4.108193580732780736e-01, -8.281856742888188405e-02, 3.778676259248315294e-01, 2.952252813797733855e-01, -1.246444286160888204e-01, -2.244502796339041817e-01, 1.155940766935046465e-01, 2.074629960449706489e-01, -1.534173462909272645e-01, -7.510936703550785687e-02, -3.127379668651892319e-01, 4.622598362029770591e-01, -9.621211578064041425e-02, 2.628380090727049923e-01, -4.042471768183623637e-01, 1.713214828161482572e-01, 4.989995779248418417e-02, -5.801265938461462601e-02])
        self.expected_v = np.array([-2.222884841673062051e-01, 9.787686675884660348e-01, -4.154378405125468132e-03, -1.028716279506487613e-01, -5.106807648068932559e-02, 9.617563369584695987e-02, -6.539114125439839109e-02, 8.616465014722822502e-02, 3.804663842399232110e-02, 8.958556637777052023e-01, -3.880568178324209083e-01, -6.754602069672590581e-01, -7.079186190294968484e-02, 2.747611091693637556e-02, 5.399649908930042458e-02, -1.139876669236737639e-01, 5.825425892149401624e-02, 8.421681390884694363e-02, -4.324455921712526130e-01, -7.982113179384198176e-01, 5.178700497729279428e-01, -2.119158650865729521e-02, -5.669958244474895825e-02, 2.880008495593230911e-02, 1.025153878619989092e-02, 3.455330867235743841e-02, -1.531884121903195027e-02, 8.219378927727334361e-01, -3.289162383259068290e-01, 6.075540959886343018e-01, -4.581331025027536585e-02, -2.052131009092891811e-02, 2.750489901219354411e-02, 4.633180549151136307e-02, 2.654757883635484872e-02, -3.696756527480526966e-02, -1.440158444262530923e-01, 4.944364353401542456e-01, -7.963661150769665298e-01, -3.279405043326523786e-03, -2.129463233078606257e-02, 3.328257760760894995e-02, 5.297895300667846037e-03, 3.437606177524311912e-02, -5.372785779467447592e-02, -1.202172148995579004e+00, -2.858130614731594910e-01, 3.226510095110137200e-01, -6.135144302237673097e-02, -7.628488365516866883e-03, 5.476841872267750738e-03, 6.607427030244909794e-02, 5.340677880472323794e-03, -1.357441391258333270e-03, -8.118660176947067875e-02, -5.001362994997625433e-02, 7.779205646059993151e-02, -3.756939173800121767e-01, 9.298080515606454988e-01, 1.339730913665280465e-01, 7.808446283301898050e-02, 6.915261247137938216e-02, -7.891656263643208324e-02, -8.035264423283335067e-02, 3.669461691293440797e-02, 6.021702408564724718e-02, 7.758956893285878786e-01, -3.211906986558734078e-01, -5.879129815844135187e-01, 6.104269012391384808e-02, -2.900814613392431462e-02, -4.552568262646729258e-02, -2.925720146121059406e-02, -6.902319498684716947e-02, 3.795994492146410881e-02, -4.884151777114849047e-01, -8.870211107633522163e-01, 5.820737769422319463e-01, 3.684187251077851444e-02, 8.060668659447538242e-02, -4.657258523345865486e-02, -5.368793987058780026e-02, -2.898503185606490784e-02, 4.002941486858704184e-02, 1.047195951770644173e+00, -2.548621413845133521e-01, 5.147188892651490821e-01, 2.224026955228448205e-02, -3.359454269630585826e-02, 5.544338676867385796e-02, -1.191273887309037081e-03, -2.572624454332552921e-02, 4.050578204667463350e-02, -1.732938335087045867e-01, 5.389208482414027390e-01, -8.697634229876662904e-01, 4.437234466680844980e-02, -8.396020718207411471e-02, 1.373643808601548444e-01, -7.061240859228964939e-02, -6.490608065647092938e-03, 2.687574399814150403e-03, -9.296946571189880215e-01, -2.226700108388965371e-01, 2.521074551855023715e-01, 1.661015709598279849e-02, -1.517347986687963592e-03, 4.175867772300452530e-03, -6.961167479355900856e-02, 8.595942434252096254e-02, 4.162461447266577186e-02, 9.626281426355881576e-02, 7.003654498037747977e-02, -9.432734078079299533e-02, -2.845586320234831934e-01, 9.840080473993093602e-01, 4.702636003956783828e-02, -1.121268620463006793e-01, 5.646007092227271762e-02, 8.300611975708871437e-02, 5.302797712834559501e-02, -2.128036013727904047e-02, -4.031107561971148529e-02, 8.271174343351145319e-01, -3.553740248929939671e-01, -6.241986194331364812e-01, 1.182134083009860406e-02, 3.695184024999947914e-02, -1.710161500383376373e-02, 3.008054412288880750e-02, 7.027591928009153943e-02, -3.889396164699072955e-02, -4.409008808247306677e-01, -8.148107923739302816e-01, 5.281887759440460073e-01, 5.876941218352332852e-02, 3.991562883248954419e-02, -5.674944832716710685e-02, 2.308380369202570059e-02, -3.268790472062921282e-02, 5.410175456271631989e-02, 1.034753757966884624e+00, -2.182612858207719775e-01, 4.555767475016349599e-01, 1.999790463725661591e-03, 4.137558459329451765e-02, -6.513656908661276390e-02, 4.414866304579422029e-02, -8.348549073500094453e-02, 1.365906277014072301e-01, -2.146360657075572775e-01, 6.238014307983194007e-01, -1.008256906299115352e+00, 8.070152934834977365e-02, 3.543449526282398468e-03, 3.048075243036858784e-03, 1.760219621424649605e-02, -1.639238275648761956e-03, 4.474655455192242531e-03, -9.335462888220811273e-01, -2.202218134011651174e-01, 2.478280539571276475e-01])
        self.expected_gt = self.expected_t.reshape(-1, self.nout).sum(0).reshape(-1)
        self.expected_gv = self.expected_v.reshape(1, self.nout, 6, 9).sum(-2).reshape(-1)
        mcoord = self.coords.reshape(2,3,3)
        fake_target = np.stack([
            mcoord[:, 1] + mcoord[:, 2] - 2 * mcoord[:, 0],
            mcoord[:, 0] - mcoord[:, 1],
            mcoord[:, 0] - mcoord[:, 2]
        ], axis=-2)
        fake_target = fake_target - 13 * np.rint(fake_target / 13)
        self.target_t = fake_target.reshape(-1)

    def tearDown(self):
        os.remove("deepdipole_fake.pb")    

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 2.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_sel_type().tolist(), [0, 1])
        np.testing.assert_allclose(self.target_t, self.expected_t, atol=3e-2)

    def test_1frame_full_atm(self):
        gt, ff, vv, at, av = self.dp.eval_full(self.coords, self.box, self.atype, atomic = True)
        for dd in at, ff, av:
            print("\n\n")
            print(", ".join(f"{ii:.18e}" for ii in dd.reshape(-1)))
            print("\n\n")
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = natoms
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        self.assertEqual(av.shape, (nframes,self.nout,natoms,9))
        # check values
        np.testing.assert_almost_equal(ff.reshape([-1]), self.expected_f.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(at.reshape([-1]), self.expected_t.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(av.reshape([-1]), self.expected_v.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(gt.reshape([-1]), self.expected_gt.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(vv.reshape([-1]), self.expected_gv.reshape([-1]), decimal = default_places)

    def test_1frame_full_atm_shuffle(self):
        i_sf = [2,1,3,0,5,4]
        isel_sf = i_sf
        gt, ff, vv, at, av = self.dp.eval_full(self.coords.reshape(-1,3)[i_sf].reshape(-1), self.box, self.atype[i_sf], atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = natoms
        self.assertEqual(gt.shape, (nframes,self.nout))
        self.assertEqual(ff.shape, (nframes,self.nout,natoms,3))
        self.assertEqual(vv.shape, (nframes,self.nout,9))
        self.assertEqual(at.shape, (nframes,nsel,self.nout))
        self.assertEqual(av.shape, (nframes,self.nout,natoms,9))
        # recover the shuffled result
        nff = np.empty_like(ff)
        nav = np.empty_like(av)
        nat = np.empty_like(at)
        nff[:, :, i_sf] = ff
        nav[:, :, i_sf] = av
        nat[:, isel_sf] = at
        # check values
        np.testing.assert_almost_equal(nff.reshape([-1]), self.expected_f.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(nat.reshape([-1]), self.expected_t.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(nav.reshape([-1]), self.expected_v.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(gt.reshape([-1]), self.expected_gt.reshape([-1]), decimal = default_places)
        np.testing.assert_almost_equal(vv.reshape([-1]), self.expected_gv.reshape([-1]), decimal = default_places)