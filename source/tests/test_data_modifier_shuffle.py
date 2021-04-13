import os,sys,platform,json,shutil
import numpy as np
import unittest
import dpdata
from deepmd.env import tf

from deepmd.common import j_must_have, data_requirement
from deepmd.train.run_options import RunOptions
from deepmd.train.trainer import DPTrainer
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.infer.ewald_recp import EwaldRecp
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.infer.deep_dipole import DeepDipole

from common import Data

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5

modifier_datapath = 'data_modifier'


class TestDataModifier (unittest.TestCase) :

    def setUp(self):
        # with tf.variable_scope('load', reuse = False) :
        tf.reset_default_graph()        
        self._setUp()

    def tearDown(self):
        tf.reset_default_graph()        
        if os.path.isdir(os.path.join(modifier_datapath, 'sys_test_0')):
            shutil.rmtree(os.path.join(modifier_datapath, 'sys_test_0'))
        if os.path.isfile(os.path.join(modifier_datapath, 'dipole.pb')):
            os.remove(os.path.join(modifier_datapath, 'dipole.pb'))

    def _setUp(self):
        run_opt = RunOptions(
            restart=None,
            init_model=None,
            log_path=None,
            log_level=30,
            mpi_log="master",
            try_distrib=False
        )
        jdata = self._setUp_jdata()
        self._setUp_data()

        # init model
        model = DPTrainer (jdata, run_opt = run_opt)
        rcut = model.model.get_rcut()

        # init data system
        systems = j_must_have(jdata['training'], 'systems')
        set_pfx = j_must_have(jdata['training'], 'set_prefix')
        batch_size = j_must_have(jdata['training'], 'batch_size')
        test_size = j_must_have(jdata['training'], 'numb_test')    
        data = DeepmdDataSystem(systems, 
                                batch_size, 
                                test_size, 
                                rcut, 
                                set_prefix=set_pfx)
        data.add_dict(data_requirement)

        # clear the default graph
        tf.reset_default_graph()

        # build the model with stats from the first system
        model.build (data)
        
        # freeze the graph
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()
            nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type,model_attr/output_dim,model_attr/model_version"
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                nodes.split(",") 
            )
            output_graph = os.path.join(modifier_datapath, 'dipole.pb')
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def _setUp_data(self):        
        jdata = self._setUp_jdata()
        # sys0
        self.atom_types0 = np.array([0, 3, 2, 1, 3, 4, 1, 4], dtype = int)
        self.natoms = len(self.atom_types0)
        self.nframes = 1
        scale = 10.0
        self.sel_type = jdata['model']['fitting_net']['sel_type']
        self.nsel = 0
        for ii in self.sel_type:
            self.nsel += np.sum(self.atom_types0 == ii)
        self.coords0 = np.random.random([self.nframes, self.natoms * 3]) * scale
        self.dipoles0 = np.random.random([self.nframes, self.nsel * 3]) 
        self.box0 = np.reshape(np.eye(3) * scale, [-1, 9])
        self.box0 = np.tile(self.box0, [self.nframes, 1])
        self._write_sys_data(os.path.join(modifier_datapath, 'sys_test_0'), 
                             self.atom_types0, self.coords0, self.dipoles0, self.box0)
        # sys1
        self.idx_map = np.array([6, 7, 1, 0, 5, 2, 4, 3], dtype = int)
        self.sel_idx_map = np.array([3, 0, 2, 1], dtype = int)
        self.atom_types1 = self.atom_types0[self.idx_map]        
        self.coords1 = np.reshape(self.coords0, [self.nframes, -1, 3])
        self.coords1 = self.coords1[:,self.idx_map,:]
        self.coords1 = np.reshape(self.coords1, [self.nframes, self.natoms*3])
        self.dipoles1 = self.dipoles0[:,self.sel_idx_map]
        self.box1 = self.box0

    def _write_sys_data(self, dirname, atom_types, coords, dipoles, box):
        os.makedirs(dirname, exist_ok = True)
        os.makedirs(dirname+'/set.0', exist_ok = True)
        np.savetxt(os.path.join(dirname, 'type.raw'), atom_types, fmt = '%d')
        np.save(os.path.join(dirname, 'set.0', 'coord.npy'), coords)
        np.save(os.path.join(dirname, 'set.0', 'dipole.npy'), dipoles)
        np.save(os.path.join(dirname, 'set.0', 'box.npy'), box)

    def _setUp_jdata(self):
        aa = {"a":[1,2,3]}
        jdata = {
            "model":{
	        "type_map":		["A", "B", "C", "D", "E"],
	        "descriptor" :{
	            "type":		"se_a",
	            "sel":              [50, 50, 50, 50, 50],
	            "rcut_smth":	3.80,
	            "rcut":		4.00,
	            "neuron":		[2, 4],
	            "resnet_dt":	False,
	            "axis_neuron":	4,
	            "seed":		1,
	        },
	        "fitting_net": {
	            "type":		"dipole",
	            "sel_type":	[1, 3],
	            "neuron":		[10],
	            "resnet_dt":	True,
	            "seed":		1,
	        },
            },
            "learning_rate" :{
	        "type":		"exp",
	        "start_lr":	0.01,
	        "stop_lr":	1e-8,
	        "decay_steps":	5000,
	        "decay_rate":	0.95,
            },
            "training": {
	        "systems":	["data_modifier/sys_test_0"], 
	        "set_prefix":	"set",    
	        "stop_batch":	1000000,
	        "batch_size":	1,
	        "numb_test":	2,
            },
        }
        return jdata


    def test_z_dipole(self):
        dd = DeepDipole(os.path.join(modifier_datapath, "dipole.pb"))
            
        dv0 = dd.eval(self.coords0, self.box0, self.atom_types0)
        dv1 = dd.eval(self.coords1, self.box1, self.atom_types1)

        dv01 = dv0.reshape([self.nframes, -1, 3])
        dv01 = dv01[:,self.sel_idx_map, :]
        dv01 = dv01.reshape([self.nframes, -1])
        dv1 = dv1.reshape([self.nframes, -1])

        for ii in range(self.nframes):
            for jj in range(self.nsel):
                self.assertAlmostEqual(
                    dv01[ii][jj], dv1[ii][jj], 
                    msg = "dipole [%d,%d] dose not match" % (ii, jj))


    def test_modify(self):
        dcm = DipoleChargeModifier(os.path.join(modifier_datapath, "dipole.pb"),
                                   [-1, -3],
                                   [1, 1, 1, 1, 1],
                                   1,
                                   0.25)
        ve0, vf0, vv0 = dcm.eval(self.coords0, self.box0, self.atom_types0)
        ve1, vf1, vv1 = dcm.eval(self.coords1, self.box1, self.atom_types1)
        vf01 = vf0[:,self.idx_map, :]

        for ii in range(self.nframes):
            self.assertAlmostEqual(ve0[ii], ve1[ii], 
                                   msg = 'energy %d should match' % ii)
        for ii in range(self.nframes):
            for jj in range(9):
                self.assertAlmostEqual(vv0[ii][jj], vv1[ii][jj], 
                                       msg = 'virial [%d,%d] should match' % (ii,jj))
        for ii in range(self.nframes):
            for jj in range(self.natoms):
                for dd in range(3):
                    self.assertAlmostEqual(
                        vf01[ii][jj][dd], vf1[ii][jj][dd], 
                        msg = "force [%d,%d,%d] dose not match" % (ii,jj,dd))
                    
        
