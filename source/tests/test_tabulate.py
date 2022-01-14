import unittest
import numpy as np
from deepmd.utils.tabulate import DPTabulate
from deepmd.env import op_module
from deepmd.env import tf
from deepmd.common import gelu

# Now just test some OPs utilized by DPTabulate sourced in /opt/deepmd-kit/source/op/unaggregated_grad.cc

class TestDPTabulate(unittest.TestCase):
    def test_op_tanh(self):
        w=tf.constant([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1,1.1,1.2]],dtype='double')
        x=tf.constant([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[1.0,1.1,1.2]],dtype='double')
        b=tf.constant([[0.1],[0.2],[0.3],[0.4]],dtype='double')
        xbar = tf.matmul(x, w) + b
        y=tf.nn.tanh(xbar)
        dy = op_module.unaggregated_dy_dx_s(y, w, xbar, tf.constant(1))
        dy_array = tf.Session().run(dy)
        answer = np.array([[8.008666403121351973e-02, 1.513925729426658651e-01, 2.134733287761668430e-01, 2.661983049806041501e-01], 
                           [4.010658815015744061e-02, 6.306476628799793926e-02, 7.332167904608145881e-02, 7.494218676568849269e-02],
                           [1.561705624394135218e-02, 1.994112926507514427e-02, 1.887519955881525671e-02, 1.576442161040989692e-02],
                           [5.492686739421748753e-03, 5.754985286040992763e-03, 4.493113544969218158e-03, 3.107638130764600777e-03]])
        
        places = 18
        np.testing.assert_almost_equal(dy_array, answer, places)

    def test_op_gelu(self):
        w = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [
                        0.9, 1, 1.1, 1.2]], dtype='double')
        x = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [
                        0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype='double')
        b = tf.constant([[0.1], [0.2], [0.3], [0.4]], dtype='double')
        xbar = tf.matmul(x, w) + b
        y = gelu(xbar)
        dy = op_module.unaggregated_dy_dx_s(y, w, xbar, tf.constant(2))
        dy_array = tf.Session().run(dy)
        answer = np.array([[8.549286163555620821e-02, 1.782905778685600906e-01, 2.776474599997448833e-01, 3.827650237273348965e-01],
                           [1.089906023807040714e-01, 2.230820937721638697e-01, 3.381867859682909927e-01, 4.513008399758057232e-01],
                           [1.124254240556722684e-01, 2.209918074710395253e-01, 3.238894323148118759e-01, 4.220357318198978414e-01],
                           [1.072173273655498138e-01, 2.082159073100979807e-01, 3.059816075270163083e-01, 4.032981557798429595e-01]])

        places = 18
        np.testing.assert_almost_equal(dy_array, answer, places)



if __name__ == '__main__':
    unittest.main()
