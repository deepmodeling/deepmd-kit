import os,sys
import numpy as np
import unittest

import deepmd.op
from deepmd.env import tf
from deepmd.env import op_module
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION

class TestProdVirial(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
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
        self.dnlist = [33, -1, -1, -1, -1, 1, 32, 34, 35, -1, 
                       0, 33, -1, -1, -1, 32, 34, 35, -1, -1, 
                       6, 3, -1, -1, -1, 7, 4, 5, -1, -1, 
                       6, -1, -1, -1, -1, 4, 5, 2, 7, -1, 
                       3, 6, -1, -1, -1, 5, 2, 7, -1, -1, 
                       3, 6, -1, -1, -1, 4, 2, 7, -1, -1]
        self.dem_deriv = [0.13227682739491875, 0.01648776318803519, -0.013864709953575083, 0.12967498112414713, 0.0204174282700489, -0.017169201045268437, 0.0204174282700489, -0.031583528930688706, -0.0021400703852459233, -0.01716920104526844, -0.0021400703852459233, -0.03232887285478848, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7946522798827726, 0.33289487400494444, 0.6013584820734476, 0.15412158847174678, -0.502001299580599, -0.9068410573068878, -0.502001299580599, -0.833906252681877, 0.3798928753582899, -0.9068410573068878, 0.3798928753582899, -0.3579459969766471, 0.4206262499369199, 0.761133214171572, -0.5007455356391932, -0.6442543005863454, 0.635525177045359, -0.4181086691087898, 0.6355251770453592, 0.15453235677768898,
                          -0.75657759172067, -0.4181086691087898, -0.75657759172067, -0.49771716703202185, 0.12240657396947655, -0.0016631327984983461, 0.013970315507385892, 0.12123416269111335, -0.0020346719145638054, 0.017091244082335703, -0.002034671914563806, -0.028490045221941415, -0.00023221799024912971, 0.017091244082335703, -0.00023221799024912971, -0.026567059102687942, 0.057945707686107975, 0.008613551142529565, -0.008091517739952026, 0.056503423854730866, 0.009417127630974357, -0.008846392623036528, 0.009417127630974357, -0.005448318729873151, -0.0013150043088297543, -0.008846392623036528, -0.0013150043088297541, -0.005612854948377751, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7946522798827726, -0.33289487400494444, -0.6013584820734476, 0.15412158847174678, -0.502001299580599, -0.9068410573068878, -0.502001299580599, -0.833906252681877, 0.3798928753582899, -0.9068410573068878, 0.3798928753582899, -0.3579459969766471, 0.06884320605436924, 0.002095928989945659, -0.01499395354345747, 0.0668001797461137, 0.0023216922720068383, -0.016609029330510533, 0.0023216922720068383, -0.009387797963986713, -0.0005056613145120282, -0.016609029330510533, -0.0005056613145120282, -0.005841058553679004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3025931001933299, 0.11738525438534331, -0.2765074881076981, 0.034913562192579815, 0.15409432322878, -0.3629777391611269, 0.15409432322878003, -0.30252938969021487, -0.14081032984698866, -0.3629777391611269, -0.14081032984698866, -0.030620805157591004, 0.06555082496658332, -0.005338981218997747, -0.002076270474054677, 0.06523884623439505, -0.00599162877720186, -0.0023300778578007205, -0.00599162877720186, -0.007837034455273667, 0.00018978009701544363, -0.0023300778578007205, 0.00018978009701544363, -0.008251237047966105, 0.014091999096200191, 0.0009521621010946066, -0.00321014651226182, 0.013676554858123476, 0.0009667394698497006, -0.0032592930697789946, 0.0009667394698497006, -0.0005658690612028018, -0.00022022250471479668, -0.0032592930697789937, -0.00022022250471479666, 0.00011127514881492382, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          -0.4206262499369199, -0.761133214171572, 0.5007455356391932, -0.6442543005863454, 0.635525177045359, -0.4181086691087898, 0.6355251770453592, 0.15453235677768898, -0.75657759172067, -0.4181086691087898, -0.75657759172067, -0.49771716703202185, 0.17265177804411166, -0.01776481317495682, 0.007216955352326217, 0.1708538944675734, -0.023853120077098278, 0.009690330031321191, -0.02385312007709828, -0.05851427595224925, -0.0009970757588497682, 0.00969033003132119, -0.0009970757588497682, -0.06056355425469288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3025931001933299, -0.11738525438534331, 0.2765074881076981, 0.034913562192579815, 0.15409432322878, -0.3629777391611269, 0.15409432322878003, -0.30252938969021487, -0.14081032984698866, -0.3629777391611269, -0.14081032984698866, -0.030620805157591004, 0.13298898711407747, -0.03304327593938735, 0.03753063440029181, 0.11967949867634801, -0.0393666881596552, 0.044712781613435545, -0.0393666881596552, -0.02897797727002851,
                          -0.01110961751744871, 0.044712781613435545, -0.011109617517448708, -0.026140939946396612, 0.09709214772325653, -0.00241522755530488, -0.0028982730663658636, 0.09699249715361474, -0.0028489422636695603, -0.0034187307164034813, -0.00284894226366956, -0.017464112635362926, 8.504305264685245e-05, -0.003418730716403481, 8.504305264685245e-05, -0.017432930182725747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1322768273949186, -0.016487763188035173, 0.013864709953575069, 0.12967498112414702, 0.020417428270048884, -0.017169201045268423, 0.02041742827004888, -0.03158352893068868, -0.002140070385245921, -0.017169201045268423, -0.002140070385245921, -0.03232887285478844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1802999914938216, -0.5889799722131493, 0.9495799552007915, -1.070225697321266, -0.18728687322613707, 0.30195230581356786, -0.18728687322613707, -0.5157546277429348, -0.9863775323243197, 0.30195230581356786, -0.9863775323243197, 0.4627237303364723, 1.0053013143052718, 0.24303987818369216, -0.2761816797541954, 0.8183357773897718, 0.45521877564245394, -0.517294063230061, 0.45521877564245394, -0.9545617219529918, -0.1250601031984763, -0.517294063230061, -0.1250601031984763, -0.922500859133019, -0.17265177804411166, 0.01776481317495682, -0.007216955352326217, 0.1708538944675734, -0.023853120077098278, 0.009690330031321191, -0.02385312007709828, -0.05851427595224925, -0.0009970757588497682, 0.00969033003132119, -0.0009970757588497682, -0.06056355425469288, -0.06884320605436924, -0.002095928989945659, 0.01499395354345747, 0.0668001797461137, 0.0023216922720068383, -0.016609029330510533, 0.0023216922720068383, -0.009387797963986713, -0.0005056613145120282, -0.016609029330510533, -0.0005056613145120282, -0.005841058553679004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          -0.1802999914938216, 0.5889799722131493, -0.9495799552007915, -1.070225697321266, -0.18728687322613707, 0.30195230581356786, -0.18728687322613707, -0.5157546277429348, -0.9863775323243197, 0.30195230581356786, -0.9863775323243197, 0.4627237303364723, -0.12240657396947667, 0.0016631327984983487, -0.013970315507385913, 0.12123416269111348, -0.002034671914563809, 0.01709124408233573, -0.002034671914563809, -0.028490045221941467, -0.00023221799024913015, 0.01709124408233573, -0.00023221799024913015, -0.026567059102687987, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2602591506940697, 0.24313683814840728, -0.3561441009497795, -0.19841405298242495, 0.23891499072173572, -0.3499599864093028, 0.23891499072173572, -0.23095714382387694, -0.32693630309290145, -0.34995998640930287, -0.32693630309290145, 0.02473856993038946, -0.13298898711407747, 0.03304327593938735, -0.03753063440029181, 0.11967949867634801, -0.0393666881596552, 0.044712781613435545, -0.0393666881596552, -0.02897797727002851,
                          -0.01110961751744871, 0.044712781613435545, -0.011109617517448708, -0.026140939946396612, -0.0655508249665835, 0.005338981218997763, 0.002076270474054683, 0.0652388462343952, -0.005991628777201879, -0.0023300778578007283, -0.005991628777201879, -0.007837034455273709, 0.0001897800970154443, -0.002330077857800728, 0.0001897800970154443, -0.008251237047966148, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0053013143052718, -0.24303987818369216, 0.2761816797541954, 0.8183357773897718, 0.45521877564245394, -0.517294063230061, 0.45521877564245394, -0.9545617219529918, -0.1250601031984763, -0.517294063230061, -0.1250601031984763, -0.922500859133019, -0.057945707686107864, -0.008613551142529548, 0.00809151773995201, 0.05650342385473076, 0.009417127630974336, -0.00884639262303651, 0.009417127630974336, -0.005448318729873148, -0.0013150043088297515, -0.00884639262303651, -0.0013150043088297513, -0.005612854948377747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2602591506940697, -0.24313683814840728, 0.3561441009497795, -0.19841405298242495, 0.23891499072173572, -0.3499599864093028, 0.23891499072173572, -0.23095714382387694, -0.32693630309290145, -0.34995998640930287, -0.32693630309290145, 0.02473856993038946, -0.09709214772325653, 0.00241522755530488, 0.0028982730663658636, 0.09699249715361474, -0.0028489422636695603, -0.0034187307164034813, -0.00284894226366956, -0.017464112635362926, 8.504305264685245e-05, -0.003418730716403481, 8.504305264685245e-05, -0.017432930182725747, -0.014091999096200191, -0.0009521621010946064, 0.0032101465122618194, 0.013676554858123474, 0.0009667394698497003, -0.0032592930697789933, 0.0009667394698497003, -0.0005658690612028016, -0.0002202225047147966, -0.0032592930697789933, -0.0002202225047147966, 0.00011127514881492362, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.drij = [3.5299999999999976, 0.4399999999999995, -0.37000000000000055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.06099789543e-313, 0.0, 0.0, -0.740000000000002, 0.31000000000000005, 0.5599999999999996, 0.41999999999999815, 0.7599999999999993, -0.5000000000000007, 3.6799999999999997, -0.05000000000000071, 0.4199999999999995, 4.439999999999998, 0.6599999999999997, -0.6200000000000006, 1.06099789543e-313, 3.11, -0.31999999999999984, 0.740000000000002, -0.31000000000000005, -0.5599999999999996, 4.27, 0.12999999999999945, -0.9300000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.06099789543e-313, 3.26, -0.81, 1.1600000000000001, 0.4499999999999993, -1.0600000000000003, 4.420000000000002, -0.36000000000000076, -0.14000000000000012, 5.18, 0.34999999999999964, -1.1800000000000002, 0.0, 0.0, 0.0, 1.06099789543e-313, 0.0, 0.0,
                     -0.41999999999999815, -0.7599999999999993, 0.5000000000000007, 3.11, -0.31999999999999984, 0.13000000000000012, 1.0609978957e-313, 2.1219957915e-314, 6.3659873744e-314, 6.3659873744e-314, 0.0, 0.0, 0.0, 0.1499999999999999, -0.4900000000000002, -1.1600000000000001, -0.4499999999999993, 1.0600000000000003, 3.2600000000000002, -0.81, 0.9200000000000002, 4.0200000000000005, -0.09999999999999964, -0.11999999999999988, 0.0, 0.0, 0.0, 0.0, -0.1499999999999999, 0.4900000000000002, -3.529999999999998, -0.4399999999999995, 0.37000000000000055, 0.0, 0.0, 0.0, 5e-324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7599999999999998, 0.7100000000000004, 0.15000000000000036, -0.4900000000000002, 0.79, 0.9100000000000006, 0.2200000000000002, -0.25, -3.11, 0.31999999999999984, -0.13000000000000012, -4.27, -0.12999999999999945, 0.9300000000000002, 0.0, -0.9099999999999997, -0.2200000000000002,
                     -0.15000000000000036, 0.4900000000000002, -0.79, -3.6799999999999984, 0.05000000000000071, -0.4199999999999995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7599999999999998, -0.7100000000000004, 0.7600000000000002, 0.7100000000000004, -1.04, -3.2600000000000002, 0.81, -0.9200000000000002, -4.42, 0.36000000000000076, 0.14000000000000012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9100000000000006, -0.2200000000000002, 0.25, -4.439999999999999, -0.6599999999999997, 0.6200000000000006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7400000000000002, 0.31000000000000005, -0.7600000000000002, -0.7100000000000004, 1.04, -4.0200000000000005, 0.09999999999999964, 0.11999999999999988, -5.180000000000001, -0.34999999999999964, 1.1800000000000002, 0.0, 0.0, 0.0, 0.0, 0.7400000000000002, -0.31000000000000005]
        self.dcoord = np.reshape(self.dcoord, [1, -1])
        self.dtype = np.reshape(self.dtype, [1, -1])
        self.dbox = np.reshape(self.dbox, [1, -1])
        self.dnlist = np.reshape(self.dnlist, [1, -1])
        self.dem_deriv = np.reshape(self.dem_deriv, [1, -1])
        self.drij = np.reshape(self.drij, [1, -1])
        self.dcoord = np.tile(self.dcoord, [self.nframes, 1])
        self.dtype = np.tile(self.dtype, [self.nframes, 1])
        self.dbox = np.tile(self.dbox, [self.nframes, 1])
        self.dnlist = np.tile(self.dnlist, [self.nframes, 1])
        self.dem_deriv = np.tile(self.dem_deriv, [self.nframes, 1])
        self.drij = np.tile(self.drij, [self.nframes, 1])
        self.expected_virial = [100.14628,  7.21146, -24.62874,  6.19651, 23.31547, -19.77773, -26.79150, -20.92554, 38.84203]
        self.expected_atom_virial = [-3.24191,  1.35810,  2.45333, -9.14879,  3.83260,  6.92341, -10.54930,  4.41930,  7.98326, 14.83563, -6.21493, -11.22697,  4.51124, -1.88984, -3.41391,  2.04717, -0.85760, -1.54921, 0.84708, -0.10308,  0.07324,  3.51825, -0.49788,  0.40314,  2.91345, -0.37264,  0.27386, 12.62246, -5.19874,  7.42677,  4.80217, -2.69029,  5.41896,  9.55811, -2.42899,  5.14893, 9.90295,  4.54279, -7.75115, -2.89155, 13.50055, -20.91993,  4.00314, -1.76293,  2.92724, 20.15105,  2.86856, -3.55868, -4.22796, -1.12700,  1.46999, -21.43180, -9.30194, 12.54538, 2.86811,  5.92934, -3.94618,  4.83313,  5.21197, -3.36488,  6.67852,  8.34225, -5.44992, 5.97941,  1.92669, -4.70211,  4.91215,  1.63145, -3.96250,  3.27415,  1.02612, -2.52585,  
                                     0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
                                     0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  
                                     0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  
                                     1.38833,  0.50613, -1.26233,  1.39901,  5.18116, -2.18118, -17.72748, -19.52039, 18.66001, 14.31034,  1.31715, -2.05955, -0.10872,  0.00743,  0.03656, -3.85572, -0.33481,  0.57900, 14.31190, -0.53814,  0.89498, -1.94166,  0.07960, -0.10726, -0.35985,  0.03981,  0.03397,  6.17091,  0.81760, -0.97011,  0.53923,  0.07572, -0.08012, -1.34189, -0.17373,  0.21536,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  
                                     0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000]
        self.sel = [5, 5]
        self.sec = np.array([0, 0, 0], dtype = int)
        self.sec[1:3] = np.cumsum(self.sel)
        self.rcut = 6.
        self.rcut_smth = 0.8
        self.dnatoms = [6, 48, 2, 4]

        self.nloc = self.dnatoms[0]
        self.nall = self.dnatoms[1]
        self.nnei = self.sec[-1]
        self.ndescrpt = 4 * self.nnei
        self.ntypes = np.max(self.dtype) + 1
        self.dnet_deriv=[]
        for ii in range(self.nloc * self.ndescrpt):
            self.dnet_deriv.append(10-ii*0.01)
        self.dnet_deriv = np.reshape(self.dnet_deriv, [1, -1])
        self.dnet_deriv = np.tile(self.dnet_deriv, [self.nframes, 1])

        self.tnet_deriv = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.dnatoms[0] * self.ndescrpt], name='t_net_deriv')
        self.tem_deriv = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.dnatoms[0] * self.ndescrpt * 3], name='t_em_deriv')
        self.trij = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.dnatoms[0] * self.nnei * 3], name='t_rij')
        self.tnlist = tf.placeholder(tf.int32, [None, self.dnatoms[0] * self.nnei], name = "t_nlist")
        self.tnatoms = tf.placeholder(tf.int32, [None], name = "t_natoms")
        
    def test_prod_virial(self):
        tvirial, tatom_virial \
            = op_module.prod_virial_se_a(
                self.tnet_deriv,
                self.tem_deriv,
                self.trij,
                self.tnlist,
                self.tnatoms, 
                n_a_sel=self.nnei,
                n_r_sel=0)
        self.sess.run (tf.global_variables_initializer())
        dvirial, datom_virial = self.sess.run(
            [tvirial, tatom_virial],
            feed_dict = {
                self.tnet_deriv: self.dnet_deriv,
                self.tem_deriv: self.dem_deriv,
                self.trij: self.drij,
                self.tnlist: self.dnlist,
                self.tnatoms: self.dnatoms}
        )
        self.assertEqual(dvirial.shape, (self.nframes, 9))
        self.assertEqual(datom_virial.shape, (self.nframes, self.nall*9))
        for ff in range(self.nframes):
            for ii in range(9):
                self.assertAlmostEqual(dvirial[ff][ii], self.expected_virial[ii], places=5)
            for ii in range(self.nall*9):
                self.assertAlmostEqual(datom_virial[ff][ii], self.expected_atom_virial[ii], places=5)
