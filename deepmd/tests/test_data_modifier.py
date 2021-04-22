import os,sys,platform
import numpy as np
import unittest
from deepmd.env import tf

from deepmd.common import j_must_have, data_requirement, j_loader
from deepmd.RunOptions import RunOptions
from deepmd.Trainer import NNPTrainer
from deepmd.DataSystem import DeepmdDataSystem
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.EwaldRecp import EwaldRecp
from deepmd.DataModifier import DipoleChargeModifier

from common import Data

if global_np_float_precision == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5

modifier_datapath = 'data_modifier'

class Args() :
    INPUT = os.path.join(modifier_datapath, 'dipole.json')
    restart = None
    init_model = None
    inter_threads = 0

class TestDataModifier (unittest.TestCase) :

    def setUp(self):
        # with tf.variable_scope('load', reuse = False) :
        tf.reset_default_graph()        
        self._setUp()

    def tearDown(self):
        tf.reset_default_graph()        

    def _setUp(self):
        args = Args()
        run_opt = RunOptions(args, False)
        jdata = j_loader(args.INPUT)

        # init model
        model = NNPTrainer (jdata, run_opt = run_opt)
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
            nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type"
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                nodes.split(",") 
            )
            output_graph = os.path.join(modifier_datapath, 'dipole.pb')
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def test_fv(self):
        # with tf.variable_scope('load', reuse = False) :
        self._test_fv()
            
    def _test_fv (self):
        dcm = DipoleChargeModifier(os.path.join(modifier_datapath, "dipole.pb"),
                                   [-8],
                                   [6, 1],
                                   1,
                                   0.25)
        data = Data()
        coord, box, atype = data.get_data()
        atype = atype[0]
        ve, vf, vv = dcm.eval(coord, box, atype)

        hh = global_default_fv_hh
        hh=1e-4
        places = global_default_places
        places=1
        nframes = coord.shape[0]
        ndof = coord.shape[1]
        natoms = ndof // 3
        vf = np.reshape(vf, [nframes, -1])
        for ii in range(ndof):
            coordp = np.copy(coord)
            coordm = np.copy(coord)
            coordp[:,ii] += hh
            coordm[:,ii] -= hh
            ep, _, __ = dcm.eval(coordp, box, atype, eval_fv = False)
            em, _, __ = dcm.eval(coordm, box, atype, eval_fv = False)
            num_f = -(ep - em) / (2.*hh)
            for ff in range(nframes):
                self.assertAlmostEqual(vf[ff,ii], num_f[ff], 
                                       places = places,
                                       msg = 'frame %d dof %d does not match' % (ff, ii))

        box3 = np.reshape(box, [nframes, 3,3])
        rbox3 = np.linalg.inv(box3)
        coord3 = np.reshape(coord, [nframes, natoms, 3])
        rcoord3 = np.matmul(coord3, rbox3)
        num_deriv = np.zeros([nframes,3,3])
        for ii in range(3):
            for jj in range(3):
                box3p = np.copy(box3)
                box3m = np.copy(box3)
                box3p[:,ii,jj] = box3[:,ii,jj] + hh
                box3m[:,ii,jj] = box3[:,ii,jj] - hh
                boxp = np.reshape(box3p, [-1,9])
                boxm = np.reshape(box3m, [-1,9])
                coord3p = np.matmul(rcoord3, box3p)
                coord3m = np.matmul(rcoord3, box3m)
                coordp = np.reshape(coord3p, [nframes,-1])
                coordm = np.reshape(coord3m, [nframes,-1])
                ep, _, __ = dcm.eval(coordp, boxp, atype, eval_fv = False)
                em, _, __ = dcm.eval(coordm, boxm, atype, eval_fv = False)
                num_deriv[:,ii,jj] = -(ep - em) / (2.*hh)
        # box3t = np.transpose(box3, [0,2,1])
        # t_esti = np.matmul(num_deriv, box3t)
        num_deriv = np.transpose(num_deriv, [0,2,1])
        t_esti = np.matmul(num_deriv, box3)

        print(t_esti, '\n', vv.reshape([-1, 3, 3]))
        for ff in range(nframes):
            for ii in range(3):
                for jj in range(3):                
                    self.assertAlmostEqual(t_esti[ff][ii][jj], vv[ff,ii*3+jj], 
                                           places = places,
                                           msg = "frame %d virial component [%d,%d] failed" % (ff, ii, jj))
            
