import paddle
from paddle_ops import prod_env_mat_a

import os,sys
import numpy as np
import unittest
import time
class TestProdEnvMat(unittest.TestCase):
    def setUp(self):
        # self.sess = tf.Session()
        self.nframes = 2
        self.dcoord = [
            12.83, 2.56, 2.18,
            12.09, 2.87, 2.74,
            00.25, 3.32, 1.68,
            3.36, 3.00, 1.81,
            3.51, 2.51, 2.60,
            4.27, 3.22, 1.56]
        self.dtype = [0, 1, 1, 0, 1, 1]
        self.dbox = [13., 0., 0., 0., 13., 0., 0., 0., 13.]
        self.dcoord = np.reshape(self.dcoord, [1, -1])
        self.dtype = np.reshape(self.dtype, [1, -1])
        self.dbox = np.reshape(self.dbox, [1, -1])
        self.dcoord = np.tile(self.dcoord, [self.nframes, 1])
        self.dtype = np.tile(self.dtype, [self.nframes, 1])
        self.dbox = np.tile(self.dbox, [self.nframes, 1])        
        self.pbc_expected_output = [
            0.12206, 0.12047, 0.01502, -0.01263, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.02167, -0.77271, 0.32370, 0.58475, 0.99745, 0.41810, 0.75655, -0.49773, 0.10564, 0.10495, -0.00143, 0.01198, 0.03103, 0.03041, 0.00452, -0.00425, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
            1.02167, 0.77271, -0.32370, -0.58475, 0.04135, 0.04039, 0.00123, -0.00880, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, 0.42028, 0.16304, -0.38405, 0.03694, 0.03680, -0.00300, -0.00117, 0.00336, 0.00327, 0.00022, -0.00074, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
            0.99745, -0.41810, -0.75655, 0.49773, 0.19078, 0.18961, -0.01951, 0.00793, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, -0.42028, -0.16304, 0.38405, 0.13499, 0.12636, -0.03140, 0.03566, 0.07054, 0.07049, -0.00175, -0.00210, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
            0.12206, -0.12047, -0.01502, 0.01263, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.06176, 0.16913, -0.55250, 0.89077, 1.03163, 0.96880, 0.23422, -0.26615, 0.19078, -0.18961, 0.01951, -0.00793, 0.04135, -0.04039, -0.00123, 0.00880, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
            1.06176, -0.16913, 0.55250, -0.89077, 0.10564, -0.10495, 0.00143, -0.01198, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, 0.34516, 0.32245, -0.47232, 0.13499, -0.12636, 0.03140, -0.03566, 0.03694, -0.03680, 0.00300, 0.00117, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
            1.03163, -0.96880, -0.23422, 0.26615, 0.03103, -0.03041, -0.00452, 0.00425, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, -0.34516, -0.32245, 0.47232, 0.07054, -0.07049, 0.00175, 0.00210, 0.00336, -0.00327, -0.00022, 0.00074, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        self.nopbc_expected_output = [
            0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1.02167,-0.77271,0.32370,0.58475,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
            1.02167,0.77271,-0.32370,-0.58475,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
            0.19078,0.18961,-0.01951,0.00793,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.13499,0.12636,-0.03140,0.03566,0.07054,0.07049,-0.00175,-0.00210,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
            0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1.06176,0.16913,-0.55250,0.89077,1.03163,0.96880,0.23422,-0.26615,0.19078,-0.18961,0.01951,-0.00793,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
            1.06176,-0.16913,0.55250,-0.89077,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.66798,0.34516,0.32245,-0.47232,0.13499,-0.12636,0.03140,-0.03566,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
1.03163,-0.96880,-0.23422,0.26615,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.66798,-0.34516,-0.32245,0.47232,0.07054,-0.07049,0.00175,0.00210,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000]
        self.sel = [10, 10]
        self.sec = np.array([0, 0, 0], dtype = int)
        self.sec[1:3] = np.cumsum(self.sel)
        self.rcut = 6.
        self.rcut_smth = 0.8
        self.dnatoms = [6, 6, 2, 4]
        self.nloc = self.dnatoms[0]
        self.nall = self.dnatoms[1]
        self.nnei = self.sec[-1]
        self.ndescrpt = 4 * self.nnei
        self.ntypes = np.max(self.dtype) + 1
        self.davg = np.zeros ([self.ntypes, self.ndescrpt])
        self.dstd = np.ones  ([self.ntypes, self.ndescrpt])
        
    def test_pbc_self_built_nlist(self):
        tem, tem_deriv, trij, tnlist \
            = prod_env_mat_a(paddle.to_tensor(self.dcoord, dtype="float64"),
        paddle.to_tensor(self.dtype, dtype='int32'),
        paddle.to_tensor(self.dnatoms, dtype="int32"),
        paddle.to_tensor(self.dbox, dtype="float64"),
        paddle.to_tensor(np.zeros(6, dtype = np.int32), dtype='int32'),
        paddle.to_tensor(self.davg, dtype="float64"),
        paddle.to_tensor(self.dstd, dtype="float64"),
        -1,
        self.rcut,
        self.rcut_smth,
        self.sel,
        [0,0]
        )
        dem = tem.numpy()
        dem_deriv = tem_deriv.numpy()
        drij = trij.numpy()
        dnlist = tnlist.numpy()
        self.assertEqual(dem.shape, (self.nframes, self.nloc*self.ndescrpt))
        self.assertEqual(dem_deriv.shape, (self.nframes, self.nloc*self.ndescrpt*3))
        self.assertEqual(drij.shape, (self.nframes, self.nloc*self.nnei*3))
        self.assertEqual(dnlist.shape, (self.nframes, self.nloc*self.nnei))
        for ff in range(self.nframes):
            for ii in range(self.ndescrpt):
                self.assertAlmostEqual(dem[ff][ii], self.pbc_expected_output[ii], places=5)

    def test_pbc_self_built_nlist_deriv(self):
        hh = 1e-4
        self.check_deriv_numerical_deriv(hh)
    
    def test_nopbc_self_built_nlist(self):
        tem, tem_deriv, trij, tnlist \
            = prod_env_mat_a(paddle.to_tensor(self.dcoord, dtype="float64"),
                paddle.to_tensor(self.dtype, dtype='int32'),
                paddle.to_tensor(self.dnatoms, dtype="int32"),
                paddle.to_tensor(self.dbox, dtype="float64"),
                paddle.to_tensor(np.zeros(0, dtype = np.int32), dtype='int32'),
                paddle.to_tensor(self.davg, dtype="float64"),
                paddle.to_tensor(self.dstd, dtype="float64"),
                -1,
                self.rcut,
                self.rcut_smth,
                self.sel,
                [0,0]
                )
        dem = tem.numpy()
        dem_deriv = tem_deriv.numpy()
        drij = trij.numpy()
        dnlist = tnlist.numpy()

        self.assertEqual(dem.shape, (self.nframes, self.nloc*self.ndescrpt))
        self.assertEqual(dem_deriv.shape, (self.nframes, self.nloc*self.ndescrpt*3))
        self.assertEqual(drij.shape, (self.nframes, self.nloc*self.nnei*3))
        self.assertEqual(dnlist.shape, (self.nframes, self.nloc*self.nnei))
        for ff in range(self.nframes):
            for ii in range(self.ndescrpt):
                self.assertAlmostEqual(dem[ff][ii], self.nopbc_expected_output[ii], places=5)
    
    def test_nopbc_self_built_nlist_deriv(self):
        hh = 1e-4

        self.check_nopbc_deriv_numerical_deriv(hh)
    
    def check_nopbc_deriv_numerical_deriv(self, hh):
        tem, tem_deriv, trij, tnlist \
            = prod_env_mat_a(paddle.to_tensor(self.dcoord, dtype="float64"),
                paddle.to_tensor(self.dtype, dtype='int32'),
                paddle.to_tensor(self.dnatoms, dtype="int32"),
                paddle.to_tensor(self.dbox, dtype="float64"),
                paddle.to_tensor(np.zeros(0, dtype = np.int32), dtype='int32'),
                paddle.to_tensor(self.davg, dtype="float64"),
                paddle.to_tensor(self.dstd, dtype="float64"),
                -1,
                self.rcut,
                self.rcut_smth,
                self.sel,
                [0,0]
                )
        dem_ = tem.numpy()
        dem_deriv_ = tem_deriv.numpy()
        drij_ = trij.numpy()
        dnlist_ = tnlist.numpy()

        ff = 0
        dem = dem_[ff]
        dem_deriv = dem_deriv_[ff]
        dnlist = dnlist_[ff]
        for ii in range(self.dnatoms[0]):            
            for jj in range(self.nnei):
                j_idx = dnlist[ii*self.nnei+jj]
                if j_idx < 0:
                    continue
                for kk in range(4):
                    for dd in range(3):
                        dcoord_0 = np.copy(self.dcoord)
                        dcoord_1 = np.copy(self.dcoord)
                        dcoord_0[ff][j_idx*3+dd] -= hh
                        dcoord_1[ff][j_idx*3+dd] += hh
                        
                        tem_0, tem_deriv_0, trij_0, tnlist_0 \
                            = prod_env_mat_a(paddle.to_tensor(dcoord_0, dtype="float64"),
                                paddle.to_tensor(self.dtype, dtype='int32'),
                                paddle.to_tensor(self.dnatoms, dtype="int32"),
                                paddle.to_tensor(self.dbox, dtype="float64"),
                                paddle.to_tensor(np.zeros(0, dtype = np.int32), dtype='int32'),
                                paddle.to_tensor(self.davg, dtype="float64"),
                                paddle.to_tensor(self.dstd, dtype="float64"),
                                -1,
                                self.rcut,
                                self.rcut_smth,
                                self.sel,
                                [0,0]
                                )
                        dem_0 = tem_0.numpy()
                        dem_deriv_0 = tem_deriv_0.numpy()
                        drij_0 = trij_0.numpy()
                        dnlist_0 = tnlist_0.numpy()

                        tem_1, tem_deriv_1, trij_1, tnlist_1 \
                            = prod_env_mat_a(paddle.to_tensor(dcoord_1, dtype="float64"),
                                paddle.to_tensor(self.dtype, dtype='int32'),
                                paddle.to_tensor(self.dnatoms, dtype="int32"),
                                paddle.to_tensor(self.dbox, dtype="float64"),
                                paddle.to_tensor(np.zeros(0, dtype = np.int32), dtype='int32'),
                                paddle.to_tensor(self.davg, dtype="float64"),
                                paddle.to_tensor(self.dstd, dtype="float64"),
                                -1,
                                self.rcut,
                                self.rcut_smth,
                                self.sel,
                                [0,0]
                                )

                        dem_1 = tem_1.numpy()
                        dem_deriv_1 = tem_deriv_1.numpy()
                        drij_1 = trij_1.numpy()
                        dnlist_1 = tnlist_1.numpy()

                        num_deriv = (dem_1[0][ii*self.nnei*4+jj*4+kk] - dem_0[0][ii*self.ndescrpt+jj*4+kk]) / (2.*hh)
                        ana_deriv = -dem_deriv[ii*self.nnei*4*3+jj*4*3+kk*3+dd]
                        self.assertAlmostEqual(num_deriv, ana_deriv, places = 5)

    def check_deriv_numerical_deriv(self, hh):
        tem, tem_deriv, trij, tnlist \
            = prod_env_mat_a(paddle.to_tensor(self.dcoord, dtype="float64"),
                paddle.to_tensor(self.dtype, dtype='int32'),
                paddle.to_tensor(self.dnatoms, dtype="int32"),
                paddle.to_tensor(self.dbox, dtype="float64"),
                paddle.to_tensor(np.zeros(6, dtype = np.int32), dtype='int32'),
                paddle.to_tensor(self.davg, dtype="float64"),
                paddle.to_tensor(self.dstd, dtype="float64"),
                -1,
                self.rcut,
                self.rcut_smth,
                self.sel,
                [0,0]
                )
        dem_ = tem.numpy()
        dem_deriv_ = tem_deriv.numpy()
        drij_ = trij.numpy()
        dnlist_ = tnlist.numpy()

        ff = 0
        dem = dem_[ff]
        dem_deriv = dem_deriv_[ff]
        dnlist = dnlist_[ff]
        for ii in range(self.dnatoms[0]):            
            for jj in range(self.nnei):
                j_idx = dnlist[ii*self.nnei+jj]
                if j_idx < 0:
                    continue
                for kk in range(4):
                    for dd in range(3):
                        dcoord_0 = np.copy(self.dcoord)
                        dcoord_1 = np.copy(self.dcoord)
                        dcoord_0[ff][j_idx*3+dd] -= hh
                        dcoord_1[ff][j_idx*3+dd] += hh
                        
                        tem_0, tem_deriv_0, trij_0, tnlist_0 \
                            = prod_env_mat_a(paddle.to_tensor(dcoord_0, dtype="float64"),
                                paddle.to_tensor(self.dtype, dtype='int32'),
                                paddle.to_tensor(self.dnatoms, dtype="int32"),
                                paddle.to_tensor(self.dbox, dtype="float64"),
                                paddle.to_tensor(np.zeros(6, dtype = np.int32), dtype='int32'),
                                paddle.to_tensor(self.davg, dtype="float64"),
                                paddle.to_tensor(self.dstd, dtype="float64"),
                                -1,
                                self.rcut,
                                self.rcut_smth,
                                self.sel,
                                [0,0]
                                )
                        dem_0 = tem_0.numpy()
                        dem_deriv_0 = tem_deriv_0.numpy()
                        drij_0 = trij_0.numpy()
                        dnlist_0 = tnlist_0.numpy()

                        tem_1, tem_deriv_1, trij_1, tnlist_1 \
                            = prod_env_mat_a(paddle.to_tensor(dcoord_1, dtype="float64"),
                                paddle.to_tensor(self.dtype, dtype='int32'),
                                paddle.to_tensor(self.dnatoms, dtype="int32"),
                                paddle.to_tensor(self.dbox, dtype="float64"),
                                paddle.to_tensor(np.zeros(6, dtype = np.int32), dtype='int32'),
                                paddle.to_tensor(self.davg, dtype="float64"),
                                paddle.to_tensor(self.dstd, dtype="float64"),
                                -1,
                                self.rcut,
                                self.rcut_smth,
                                self.sel,
                                [0,0]
                                )

                        dem_1 = tem_1.numpy()
                        dem_deriv_1 = tem_deriv_1.numpy()
                        drij_1 = trij_1.numpy()
                        dnlist_1 = tnlist_1.numpy()

                        num_deriv = (dem_1[0][ii*self.nnei*4+jj*4+kk] - dem_0[0][ii*self.ndescrpt+jj*4+kk]) / (2.*hh)
                        ana_deriv = -dem_deriv[ii*self.nnei*4*3+jj*4*3+kk*3+dd]
                        self.assertAlmostEqual(num_deriv, ana_deriv, places = 5)

if __name__ == '__main__':
    unittest.main()