# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import numpy as np

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    op_grads_module,
    tf,
)


class TestProdForceGrad(tf.test.TestCase):
    def setUp(self):
        self.sess = self.cached_session().__enter__()
        self.nframes = 2
        self.dcoord = [
            12.83,
            2.56,
            2.18,
            12.09,
            2.87,
            2.74,
            00.25,
            3.32,
            1.68,
            3.36,
            3.00,
            1.81,
            3.51,
            2.51,
            2.60,
            4.27,
            3.22,
            1.56,
        ]
        self.dtype = [0, 1, 1, 0, 1, 1]
        self.dbox = [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0]
        self.dnlist = [
            33,
            -1,
            -1,
            -1,
            -1,
            1,
            32,
            34,
            35,
            -1,
            0,
            33,
            -1,
            -1,
            -1,
            32,
            34,
            35,
            -1,
            -1,
            6,
            3,
            -1,
            -1,
            -1,
            7,
            4,
            5,
            -1,
            -1,
            6,
            -1,
            -1,
            -1,
            -1,
            4,
            5,
            2,
            7,
            -1,
            3,
            6,
            -1,
            -1,
            -1,
            5,
            2,
            7,
            -1,
            -1,
            3,
            6,
            -1,
            -1,
            -1,
            4,
            2,
            7,
            -1,
            -1,
        ]
        self.dem_deriv = [
            0.13227682739491875,
            0.01648776318803519,
            -0.013864709953575083,
            0.12967498112414713,
            0.0204174282700489,
            -0.017169201045268437,
            0.0204174282700489,
            -0.031583528930688706,
            -0.0021400703852459233,
            -0.01716920104526844,
            -0.0021400703852459233,
            -0.03232887285478848,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.7946522798827726,
            0.33289487400494444,
            0.6013584820734476,
            0.15412158847174678,
            -0.502001299580599,
            -0.9068410573068878,
            -0.502001299580599,
            -0.833906252681877,
            0.3798928753582899,
            -0.9068410573068878,
            0.3798928753582899,
            -0.3579459969766471,
            0.4206262499369199,
            0.761133214171572,
            -0.5007455356391932,
            -0.6442543005863454,
            0.635525177045359,
            -0.4181086691087898,
            0.6355251770453592,
            0.15453235677768898,
            -0.75657759172067,
            -0.4181086691087898,
            -0.75657759172067,
            -0.49771716703202185,
            0.12240657396947655,
            -0.0016631327984983461,
            0.013970315507385892,
            0.12123416269111335,
            -0.0020346719145638054,
            0.017091244082335703,
            -0.002034671914563806,
            -0.028490045221941415,
            -0.00023221799024912971,
            0.017091244082335703,
            -0.00023221799024912971,
            -0.026567059102687942,
            0.057945707686107975,
            0.008613551142529565,
            -0.008091517739952026,
            0.056503423854730866,
            0.009417127630974357,
            -0.008846392623036528,
            0.009417127630974357,
            -0.005448318729873151,
            -0.0013150043088297543,
            -0.008846392623036528,
            -0.0013150043088297541,
            -0.005612854948377751,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.7946522798827726,
            -0.33289487400494444,
            -0.6013584820734476,
            0.15412158847174678,
            -0.502001299580599,
            -0.9068410573068878,
            -0.502001299580599,
            -0.833906252681877,
            0.3798928753582899,
            -0.9068410573068878,
            0.3798928753582899,
            -0.3579459969766471,
            0.06884320605436924,
            0.002095928989945659,
            -0.01499395354345747,
            0.0668001797461137,
            0.0023216922720068383,
            -0.016609029330510533,
            0.0023216922720068383,
            -0.009387797963986713,
            -0.0005056613145120282,
            -0.016609029330510533,
            -0.0005056613145120282,
            -0.005841058553679004,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3025931001933299,
            0.11738525438534331,
            -0.2765074881076981,
            0.034913562192579815,
            0.15409432322878,
            -0.3629777391611269,
            0.15409432322878003,
            -0.30252938969021487,
            -0.14081032984698866,
            -0.3629777391611269,
            -0.14081032984698866,
            -0.030620805157591004,
            0.06555082496658332,
            -0.005338981218997747,
            -0.002076270474054677,
            0.06523884623439505,
            -0.00599162877720186,
            -0.0023300778578007205,
            -0.00599162877720186,
            -0.007837034455273667,
            0.00018978009701544363,
            -0.0023300778578007205,
            0.00018978009701544363,
            -0.008251237047966105,
            0.014091999096200191,
            0.0009521621010946066,
            -0.00321014651226182,
            0.013676554858123476,
            0.0009667394698497006,
            -0.0032592930697789946,
            0.0009667394698497006,
            -0.0005658690612028018,
            -0.00022022250471479668,
            -0.0032592930697789937,
            -0.00022022250471479666,
            0.00011127514881492382,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.4206262499369199,
            -0.761133214171572,
            0.5007455356391932,
            -0.6442543005863454,
            0.635525177045359,
            -0.4181086691087898,
            0.6355251770453592,
            0.15453235677768898,
            -0.75657759172067,
            -0.4181086691087898,
            -0.75657759172067,
            -0.49771716703202185,
            0.17265177804411166,
            -0.01776481317495682,
            0.007216955352326217,
            0.1708538944675734,
            -0.023853120077098278,
            0.009690330031321191,
            -0.02385312007709828,
            -0.05851427595224925,
            -0.0009970757588497682,
            0.00969033003132119,
            -0.0009970757588497682,
            -0.06056355425469288,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.3025931001933299,
            -0.11738525438534331,
            0.2765074881076981,
            0.034913562192579815,
            0.15409432322878,
            -0.3629777391611269,
            0.15409432322878003,
            -0.30252938969021487,
            -0.14081032984698866,
            -0.3629777391611269,
            -0.14081032984698866,
            -0.030620805157591004,
            0.13298898711407747,
            -0.03304327593938735,
            0.03753063440029181,
            0.11967949867634801,
            -0.0393666881596552,
            0.044712781613435545,
            -0.0393666881596552,
            -0.02897797727002851,
            -0.01110961751744871,
            0.044712781613435545,
            -0.011109617517448708,
            -0.026140939946396612,
            0.09709214772325653,
            -0.00241522755530488,
            -0.0028982730663658636,
            0.09699249715361474,
            -0.0028489422636695603,
            -0.0034187307164034813,
            -0.00284894226366956,
            -0.017464112635362926,
            8.504305264685245e-05,
            -0.003418730716403481,
            8.504305264685245e-05,
            -0.017432930182725747,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1322768273949186,
            -0.016487763188035173,
            0.013864709953575069,
            0.12967498112414702,
            0.020417428270048884,
            -0.017169201045268423,
            0.02041742827004888,
            -0.03158352893068868,
            -0.002140070385245921,
            -0.017169201045268423,
            -0.002140070385245921,
            -0.03232887285478844,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1802999914938216,
            -0.5889799722131493,
            0.9495799552007915,
            -1.070225697321266,
            -0.18728687322613707,
            0.30195230581356786,
            -0.18728687322613707,
            -0.5157546277429348,
            -0.9863775323243197,
            0.30195230581356786,
            -0.9863775323243197,
            0.4627237303364723,
            1.0053013143052718,
            0.24303987818369216,
            -0.2761816797541954,
            0.8183357773897718,
            0.45521877564245394,
            -0.517294063230061,
            0.45521877564245394,
            -0.9545617219529918,
            -0.1250601031984763,
            -0.517294063230061,
            -0.1250601031984763,
            -0.922500859133019,
            -0.17265177804411166,
            0.01776481317495682,
            -0.007216955352326217,
            0.1708538944675734,
            -0.023853120077098278,
            0.009690330031321191,
            -0.02385312007709828,
            -0.05851427595224925,
            -0.0009970757588497682,
            0.00969033003132119,
            -0.0009970757588497682,
            -0.06056355425469288,
            -0.06884320605436924,
            -0.002095928989945659,
            0.01499395354345747,
            0.0668001797461137,
            0.0023216922720068383,
            -0.016609029330510533,
            0.0023216922720068383,
            -0.009387797963986713,
            -0.0005056613145120282,
            -0.016609029330510533,
            -0.0005056613145120282,
            -0.005841058553679004,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1802999914938216,
            0.5889799722131493,
            -0.9495799552007915,
            -1.070225697321266,
            -0.18728687322613707,
            0.30195230581356786,
            -0.18728687322613707,
            -0.5157546277429348,
            -0.9863775323243197,
            0.30195230581356786,
            -0.9863775323243197,
            0.4627237303364723,
            -0.12240657396947667,
            0.0016631327984983487,
            -0.013970315507385913,
            0.12123416269111348,
            -0.002034671914563809,
            0.01709124408233573,
            -0.002034671914563809,
            -0.028490045221941467,
            -0.00023221799024913015,
            0.01709124408233573,
            -0.00023221799024913015,
            -0.026567059102687987,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2602591506940697,
            0.24313683814840728,
            -0.3561441009497795,
            -0.19841405298242495,
            0.23891499072173572,
            -0.3499599864093028,
            0.23891499072173572,
            -0.23095714382387694,
            -0.32693630309290145,
            -0.34995998640930287,
            -0.32693630309290145,
            0.02473856993038946,
            -0.13298898711407747,
            0.03304327593938735,
            -0.03753063440029181,
            0.11967949867634801,
            -0.0393666881596552,
            0.044712781613435545,
            -0.0393666881596552,
            -0.02897797727002851,
            -0.01110961751744871,
            0.044712781613435545,
            -0.011109617517448708,
            -0.026140939946396612,
            -0.0655508249665835,
            0.005338981218997763,
            0.002076270474054683,
            0.0652388462343952,
            -0.005991628777201879,
            -0.0023300778578007283,
            -0.005991628777201879,
            -0.007837034455273709,
            0.0001897800970154443,
            -0.002330077857800728,
            0.0001897800970154443,
            -0.008251237047966148,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0053013143052718,
            -0.24303987818369216,
            0.2761816797541954,
            0.8183357773897718,
            0.45521877564245394,
            -0.517294063230061,
            0.45521877564245394,
            -0.9545617219529918,
            -0.1250601031984763,
            -0.517294063230061,
            -0.1250601031984763,
            -0.922500859133019,
            -0.057945707686107864,
            -0.008613551142529548,
            0.00809151773995201,
            0.05650342385473076,
            0.009417127630974336,
            -0.00884639262303651,
            0.009417127630974336,
            -0.005448318729873148,
            -0.0013150043088297515,
            -0.00884639262303651,
            -0.0013150043088297513,
            -0.005612854948377747,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2602591506940697,
            -0.24313683814840728,
            0.3561441009497795,
            -0.19841405298242495,
            0.23891499072173572,
            -0.3499599864093028,
            0.23891499072173572,
            -0.23095714382387694,
            -0.32693630309290145,
            -0.34995998640930287,
            -0.32693630309290145,
            0.02473856993038946,
            -0.09709214772325653,
            0.00241522755530488,
            0.0028982730663658636,
            0.09699249715361474,
            -0.0028489422636695603,
            -0.0034187307164034813,
            -0.00284894226366956,
            -0.017464112635362926,
            8.504305264685245e-05,
            -0.003418730716403481,
            8.504305264685245e-05,
            -0.017432930182725747,
            -0.014091999096200191,
            -0.0009521621010946064,
            0.0032101465122618194,
            0.013676554858123474,
            0.0009667394698497003,
            -0.0032592930697789933,
            0.0009667394698497003,
            -0.0005658690612028016,
            -0.0002202225047147966,
            -0.0032592930697789933,
            -0.0002202225047147966,
            0.00011127514881492362,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.dcoord = np.reshape(self.dcoord, [1, -1])
        self.dtype = np.reshape(self.dtype, [1, -1])
        self.dbox = np.reshape(self.dbox, [1, -1])
        self.dnlist = np.reshape(self.dnlist, [1, -1])
        self.dem_deriv = np.reshape(self.dem_deriv, [1, -1])
        self.dcoord = np.tile(self.dcoord, [self.nframes, 1])
        self.dtype = np.tile(self.dtype, [self.nframes, 1])
        self.dbox = np.tile(self.dbox, [self.nframes, 1])
        self.dnlist = np.tile(self.dnlist, [self.nframes, 1])
        self.dem_deriv = np.tile(self.dem_deriv, [self.nframes, 1])
        self.expected_grad_net = [
            -0.12141,
            -0.11963,
            0.01198,
            0.04647,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04188,
            0.37642,
            0.28680,
            0.26547,
            -0.40861,
            0.25610,
            -0.02009,
            1.00344,
            -0.16166,
            -0.16355,
            0.03691,
            0.01165,
            -0.08770,
            -0.08561,
            -0.00398,
            0.02366,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04188,
            -0.37642,
            -0.28680,
            -0.26547,
            -0.03357,
            -0.03151,
            0.00454,
            0.01377,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04304,
            0.05219,
            0.08677,
            0.16032,
            -0.05232,
            -0.05123,
            0.01227,
            0.00935,
            -0.01420,
            -0.01366,
            -0.00022,
            0.00404,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.40861,
            -0.25610,
            0.02009,
            -1.00344,
            -0.04863,
            -0.04701,
            0.02501,
            0.01556,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04304,
            -0.05219,
            -0.08677,
            -0.16032,
            -0.08249,
            -0.07502,
            0.04767,
            -0.00448,
            -0.08260,
            -0.08165,
            0.01821,
            0.01869,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.12141,
            0.11963,
            -0.01198,
            -0.04647,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.16227,
            0.28667,
            0.50683,
            0.06651,
            -0.58330,
            -0.45376,
            0.37464,
            0.93891,
            -0.04863,
            0.04701,
            -0.02501,
            -0.01556,
            -0.03357,
            0.03151,
            -0.00454,
            -0.01377,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.16227,
            -0.28667,
            -0.50683,
            -0.06651,
            -0.16166,
            0.16355,
            -0.03691,
            -0.01165,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04418,
            0.09284,
            0.09569,
            0.19565,
            -0.08249,
            0.07502,
            -0.04767,
            0.00448,
            -0.05232,
            0.05123,
            -0.01227,
            -0.00935,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.58330,
            0.45376,
            -0.37464,
            -0.93891,
            -0.08770,
            0.08561,
            0.00398,
            -0.02366,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.04418,
            -0.09284,
            -0.09569,
            -0.19565,
            -0.08260,
            0.08165,
            -0.01821,
            -0.01869,
            -0.01420,
            0.01366,
            0.00022,
            -0.00404,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
        ]

        self.sel = [5, 5]
        self.sec = np.array([0, 0, 0], dtype=int)
        self.sec[1:3] = np.cumsum(self.sel)
        self.rcut = 6.0
        self.rcut_smth = 0.8
        self.dnatoms = [6, 6, 2, 4]

        self.nloc = self.dnatoms[0]
        self.nall = self.dnatoms[1]
        self.nnei = self.sec[-1]
        self.ndescrpt = 4 * self.nnei
        self.ntypes = np.max(self.dtype) + 1
        self.dnet_deriv = []
        for ii in range(self.nloc * self.ndescrpt):
            self.dnet_deriv.append(10 - ii * 0.01)
        self.dgrad = []
        for ii in range(self.nloc * 3):
            self.dgrad.append(10 - ii * 0.1)
        self.dnet_deriv = np.reshape(self.dnet_deriv, [1, -1])
        self.dgrad = np.reshape(self.dgrad, [1, -1])
        self.dnet_deriv = np.tile(self.dnet_deriv, [self.nframes, 1])
        self.dgrad = np.tile(self.dgrad, [self.nframes, 1])

        self.tgrad = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, self.dnatoms[0] * 3], name="t_grad"
        )
        self.tnet_deriv = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION,
            [None, self.dnatoms[0] * self.ndescrpt],
            name="t_net_deriv",
        )
        self.tem_deriv = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION,
            [None, self.dnatoms[0] * self.ndescrpt * 3],
            name="t_em_deriv",
        )
        self.tnlist = tf.placeholder(
            tf.int32, [None, self.dnatoms[0] * self.nnei], name="t_nlist"
        )
        self.tnatoms = tf.placeholder(tf.int32, [None], name="t_natoms")

    def test_prod_force_grad(self):
        tgrad_net = op_grads_module.prod_force_se_a_grad(
            self.tgrad,
            self.tnet_deriv,
            self.tem_deriv,
            self.tnlist,
            self.tnatoms,
            n_a_sel=self.nnei,
            n_r_sel=0,
        )
        self.sess.run(tf.global_variables_initializer())
        dgrad_net = self.sess.run(
            tgrad_net,
            feed_dict={
                self.tgrad: self.dgrad,
                self.tnet_deriv: self.dnet_deriv,
                self.tem_deriv: self.dem_deriv,
                self.tnlist: self.dnlist,
                self.tnatoms: self.dnatoms,
            },
        )
        self.assertEqual(dgrad_net.shape, (self.nframes, self.nloc * self.ndescrpt))
        for ff in range(self.nframes):
            np.testing.assert_almost_equal(dgrad_net[ff], self.expected_grad_net, 5)
