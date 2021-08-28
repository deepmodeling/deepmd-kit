import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from deepmd.utils.type_embed import embed_atom_type, TypeEmbedNet

class TestTypeEbd(tf.test.TestCase):
    def test_embed_atom_type(self):
        ntypes = 3
        natoms = tf.constant([5, 5, 3, 0, 2])
        type_embedding = tf.constant(
            [ 
                [1, 2, 3],
                [3, 2, 1],
                [7, 7, 7],
            ])
        expected_out = [[1,2,3],
                        [1,2,3],
                        [1,2,3],
                        [7,7,7],
                        [7,7,7]]            
        atom_embed = embed_atom_type(ntypes, natoms, type_embedding)
        sess = self.test_session().__enter__()
        atom_embed = sess.run(atom_embed)
        for ii in range(5):
            for jj in range(3):                
                self.assertAlmostEqual(
                    atom_embed[ii][jj], expected_out[ii][jj], places=10)

    def test_type_embed_net(self):
        ten = TypeEmbedNet([2, 4, 8], seed = 1, uniform_seed = True)
        type_embedding = ten.build(2)
        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        type_embedding = sess.run(type_embedding)

        expected_out = [
            1.429967002262267917e+00,-9.138175897677495163e-01,-3.799606588218059633e-01,-2.143157692726757046e-01,2.341138114260268743e+00,-1.568346043255314015e+00,8.917082000854256174e-01,-1.500356675378008209e+00,
            8.955885646123034061e-01,-5.835326470989941061e-01,-1.465708662924672057e+00,-4.052047884085572260e-01,1.367825594590430072e+00,-2.736204307656463497e-01,-4.044263041521370394e-01,-9.438057524881729998e-01
        ]
        expected_out = np.reshape(expected_out, [2, 8])

        # 2 types
        self.assertEqual(type_embedding.shape[0], 2)
        # size of embedded vec 8
        self.assertEqual(type_embedding.shape[1], 8)
        # check value
        for ii in range(2):
            for jj in range(8):                
                self.assertAlmostEqual(
                    type_embedding[ii][jj], expected_out[ii][jj], places=10)
