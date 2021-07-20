import os,sys
import numpy as np
import unittest

from deepmd.env import tf
from tensorflow.python.framework import ops

from deepmd.utils.network import embedding_net

from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION

class Inter(unittest.TestCase):
    def setUp (self) :
        self.sess = tf.Session()
        self.inputs = tf.constant([ 0., 1., 2.], dtype = tf.float64)
        self.ndata = 3
        self.inputs = tf.reshape(self.inputs, [-1, 1])
        self.places = 6
        
    def test_enlarger_net(self):
        network_size = [3, 4]
        out = embedding_net(self.inputs, 
                            network_size, 
                            tf.float64,
                            name_suffix = 'enlarger_net',
                            seed = 1)
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        refout = [[-0.1482171,  -0.14177827, -0.76181204,  0.21266767],
                  [-0.27800543, -0.08974353, -0.78784335,  0.3485518 ],
                  [-0.36744368, -0.06285603, -0.80749876,  0.4347974 ]]
        for ii in range(self.ndata):
            for jj in range(network_size[-1]):
                self.assertAlmostEqual(refout[ii][jj], myout[ii][jj], places = self.places)


    def test_enlarger_net_1(self):
        network_size = [4, 4]
        out = embedding_net(self.inputs, 
                            network_size, 
                            tf.float64,
                            name_suffix = 'enlarger_net_1',
                            seed = 1)
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        refout = [[ 0.10842905, -0.61623145, -1.46738788, -0.01921788],
                  [ 0.09376136, -0.75526936, -1.64995884,  0.01076112],
                  [ 0.1033177,  -0.8911794,  -1.75530172,  0.00653156]]
        for ii in range(self.ndata):
            for jj in range(network_size[-1]):
                self.assertAlmostEqual(refout[ii][jj], myout[ii][jj], places = self.places)

    def test_enlarger_net_1_idt(self):
        network_size = [4, 4]
        out = embedding_net(self.inputs, 
                            network_size, 
                            tf.float64,
                            name_suffix = 'enlarger_net_1_idt',
                            resnet_dt = True,
                            seed = 1)
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        refout = [[ 0.10839754, -0.6161336,  -1.46673253, -0.01927138],
                  [ 0.09370214, -0.75516888, -1.64927868,  0.01067603],
                  [ 0.10323835, -0.89107102, -1.75460243,  0.00642493]]
        for ii in range(self.ndata):
            for jj in range(network_size[-1]):
                self.assertAlmostEqual(refout[ii][jj], myout[ii][jj], places = self.places)

    def test_enlarger_net_2(self):
        network_size = [2, 4]
        out = embedding_net(self.inputs, 
                            network_size, 
                            tf.float64,
                            name_suffix = 'enlarger_net_2',
                            seed = 1)
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        refout = [[ 0.24023149, -0.66311811, -0.50951819, -0.36873654],
                  [ 2.00858313, -0.05971232,  0.52272395, -0.12604478],
                  [ 3.39365063,  0.63492697,  1.5780069,   0.46445682]]
        for ii in range(self.ndata):
            for jj in range(network_size[-1]):
                self.assertAlmostEqual(refout[ii][jj], myout[ii][jj], places = self.places)


    def test_enlarger_net_2(self):
        network_size = [2, 4]
        out = embedding_net(self.inputs, 
                            network_size, 
                            tf.float64,
                            name_suffix = 'enlarger_net_2_idt',
                            resnet_dt = True,
                            seed = 1)
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        refout = [[ 0.2403889,  -0.66290763, -0.50883586, -0.36869913],
                  [ 2.00891479, -0.05936574,  0.52351633, -0.12579749],
                  [ 3.3940202,   0.63538459,  1.57887697,  0.46486689]]
        for ii in range(self.ndata):
            for jj in range(network_size[-1]):
                self.assertAlmostEqual(refout[ii][jj], myout[ii][jj], places = self.places)




if __name__ == '__main__':
    unittest.main()
