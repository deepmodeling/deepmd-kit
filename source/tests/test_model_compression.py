import os,sys,platform,shutil,dpdata,json
import numpy as np
import unittest

from deepmd.infer import DeepPot
from deepmd.env import MODEL_VERSION
# from deepmd.entrypoints.compress import compress
from common import j_loader, tests_path

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

def _file_delete(file) :
    if os.path.exists(file):
        os.remove(file)

class TestDeepPotAPBC(unittest.TestCase) :
    def setUp(self):
        self.data_file  = str(tests_path / os.path.join("model_compression", "data"))
        self.frozen_model = str(tests_path / "dp-original.pb")
        self.compressed_model = str(tests_path / "dp-compressed.pb")
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
        jdata["training"]["training_data"]["systems"] = self.data_file
        jdata["training"]["validation_data"]["systems"] = self.data_file
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

        ret = os.system("dp train " + self.INPUT)
        assert(ret == 0), "DP train error!"
        ret = os.system("dp freeze -o " + self.frozen_model)
        assert(ret == 0), "DP freeze error!"
        ret = os.system("dp compress " + self.INPUT + " -i " + self.frozen_model + " -o " + self.compressed_model)
        assert(ret == 0), "DP model compression error!"
        
        self.dp_original = DeepPot(self.frozen_model)
        self.dp_compressed = DeepPot(self.compressed_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])

    def tearDown(self):
        _file_delete(self.INPUT)
        _file_delete(self.frozen_model)
        _file_delete(self.compressed_model)
        _file_delete("out.json")
        _file_delete("compress.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")

    def test_attrs(self):
        self.assertEqual(self.dp_original.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_original.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp_original.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp_original.get_dim_fparam(), 0)
        self.assertEqual(self.dp_original.get_dim_aparam(), 0)

        self.assertEqual(self.dp_compressed.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_compressed.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp_compressed.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp_compressed.get_dim_fparam(), 0)
        self.assertEqual(self.dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self):
        ee0, ff0, vv0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = False)
        ee1, ff1, vv1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_1frame_atm(self):
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(coords2, box2, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(coords2, box2, self.atype, atomic = True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))

        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)


class TestDeepPotANoPBC(unittest.TestCase) :
    def setUp(self):
        self.data_file  = str(tests_path / os.path.join("model_compression", "data"))
        self.frozen_model = str(tests_path / "dp-original.pb")
        self.compressed_model = str(tests_path / "dp-compressed.pb")
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
        jdata["training"]["training_data"]["systems"] = self.data_file
        jdata["training"]["validation_data"]["systems"] = self.data_file
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

        ret = os.system("dp train " + self.INPUT)
        assert(ret == 0), "DP train error!"
        ret = os.system("dp freeze -o " + self.frozen_model)
        assert(ret == 0), "DP freeze error!"
        ret = os.system("dp compress " + self.INPUT + " -i " + self.frozen_model + " -o " + self.compressed_model)
        assert(ret == 0), "DP model compression error!"
        
        self.dp_original = DeepPot(self.frozen_model)
        self.dp_compressed = DeepPot(self.compressed_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = None

    def tearDown(self):
        _file_delete(self.INPUT)
        _file_delete(self.frozen_model)
        _file_delete(self.compressed_model)
        _file_delete("out.json")
        _file_delete("compress.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")
    
    def test_1frame(self):
        ee0, ff0, vv0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = False)
        ee1, ff1, vv1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_1frame_atm(self):
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(coords2, self.box, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(coords2, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))

        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    
class TestDeepPotALargeBoxNoPBC(unittest.TestCase) :
    def setUp(self):
        self.data_file  = str(tests_path / os.path.join("model_compression", "data"))
        self.frozen_model = str(tests_path / "dp-original.pb")
        self.compressed_model = str(tests_path / "dp-compressed.pb")
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
        jdata["training"]["training_data"]["systems"] = self.data_file
        jdata["training"]["validation_data"]["systems"] = self.data_file
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

        ret = os.system("dp train " + self.INPUT)
        assert(ret == 0), "DP train error!"
        ret = os.system("dp freeze -o " + self.frozen_model)
        assert(ret == 0), "DP freeze error!"
        ret = os.system("dp compress " + self.INPUT + " -i " + self.frozen_model + " -o " + self.compressed_model)
        assert(ret == 0), "DP model compression error!"
        
        self.dp_original = DeepPot(self.frozen_model)
        self.dp_compressed = DeepPot(self.compressed_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([19., 0., 0., 0., 13., 0., 0., 0., 13.])

    def tearDown(self):
        _file_delete(self.INPUT)
        _file_delete(self.frozen_model)
        _file_delete(self.compressed_model)
        _file_delete("out.json")
        _file_delete("compress.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")
    
    def test_1frame(self):
        ee0, ff0, vv0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = False)
        ee1, ff1, vv1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_1frame_atm(self):
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_ase(self):
        from ase import Atoms
        from deepmd.calculator import DP
        water0 = Atoms('OHHOHH',
                    positions=self.coords.reshape((-1,3)),
                    cell=self.box.reshape((3,3)),
                    calculator=DP(self.frozen_model))
        water1 = Atoms('OHHOHH',
                    positions=self.coords.reshape((-1,3)),
                    cell=self.box.reshape((3,3)),
                    calculator=DP(self.compressed_model))
        ee0 = water0.get_potential_energy()
        ff0 = water0.get_forces()
        ee1 = water1.get_potential_energy()
        ff1 = water1.get_forces()
        nframes = 1
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)

class TestDeepPotAPBCExcludeTypes(unittest.TestCase) :
    def setUp(self):
        self.data_file  = str(tests_path / os.path.join("model_compression", "data"))
        self.frozen_model = str(tests_path / "dp-original.pb")
        self.compressed_model = str(tests_path / "dp-compressed.pb")
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
        jdata["training"]["training_data"]["systems"] = self.data_file
        jdata["training"]["validation_data"]["systems"] = self.data_file
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 1]]
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

        ret = os.system("dp train " + self.INPUT)
        assert(ret == 0), "DP train error!"
        ret = os.system("dp freeze -o " + self.frozen_model)
        assert(ret == 0), "DP freeze error!"
        ret = os.system("dp compress " + self.INPUT + " -i " + self.frozen_model + " -o " + self.compressed_model)
        assert(ret == 0), "DP model compression error!"
        
        self.dp_original = DeepPot(self.frozen_model)
        self.dp_compressed = DeepPot(self.compressed_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])

    def tearDown(self):
        _file_delete(self.INPUT)
        _file_delete(self.frozen_model)
        _file_delete(self.compressed_model)
        _file_delete("out.json")
        _file_delete("compress.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")

    def test_attrs(self):
        self.assertEqual(self.dp_original.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_original.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp_original.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp_original.get_dim_fparam(), 0)
        self.assertEqual(self.dp_original.get_dim_aparam(), 0)

        self.assertEqual(self.dp_compressed.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_compressed.get_rcut(), 6.0, places = default_places)
        self.assertEqual(self.dp_compressed.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp_compressed.get_dim_fparam(), 0)
        self.assertEqual(self.dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self):
        ee0, ff0, vv0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = False)
        ee1, ff1, vv1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_1frame_atm(self):
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(self.coords, self.box, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))
        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(coords2, box2, self.atype, atomic = True)
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(coords2, box2, self.atype, atomic = True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes,1))
        self.assertEqual(ff0.shape, (nframes,natoms,3))
        self.assertEqual(vv0.shape, (nframes,9))
        self.assertEqual(ae0.shape, (nframes,natoms,1))
        self.assertEqual(av0.shape, (nframes,natoms,9))
        self.assertEqual(ee1.shape, (nframes,1))
        self.assertEqual(ff1.shape, (nframes,natoms,3))
        self.assertEqual(vv1.shape, (nframes,9))
        self.assertEqual(ae1.shape, (nframes,natoms,1))
        self.assertEqual(av1.shape, (nframes,natoms,9))

        # check values
        for ii in range(ff0.size):
            self.assertAlmostEqual(ff0.reshape([-1])[ii], ff1.reshape([-1])[ii], places = default_places)
        for ii in range(ae0.size):
            self.assertAlmostEqual(ae0.reshape([-1])[ii], ae1.reshape([-1])[ii], places = default_places)
        for ii in range(av0.size):
            self.assertAlmostEqual(av0.reshape([-1])[ii], av1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes):
            self.assertAlmostEqual(ee0.reshape([-1])[ii], ee1.reshape([-1])[ii], places = default_places)
        for ii in range(nframes, 9):
            self.assertAlmostEqual(vv0.reshape([-1])[ii], vv1.reshape([-1])[ii], places = default_places)