import os
import unittest

import numpy as np
from common import (
    tests_path,
)

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.infer import (
    DeepDOS,
)
from deepmd.utils.convert import (
    convert_pbtxt_to_pb,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestDeepDOS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(tests_path / os.path.join("infer", "deepdos.pbtxt")), "deepdos.pb"
        )
        cls.dp = DeepDOS("deepdos.pb")

    def setUp(self):
        self.coords = np.array(
            [
                2.288635,
                1.458305,
                3.706535,
                3.475085,
                3.504745,
                0.09779,
                1.573935,
                1.549525,
                1.131545,
                3.006885,
                4.479635,
                2.619155,
                5.152595,
                4.795225,
                2.359665,
                4.564595,
                2.294005,
                1.920635,
                0.271162,
                2.918505,
                3.850855,
                0.407016,
                4.924935,
                5.053735,
            ]
        )
        self.atype = [0, 0, 0, 0, 0, 0, 0, 0]
        self.box = np.array(
            [5.184978, 0.0, 0.0, 0.0, 5.184978, 0.0, 0.0, 0.0, 5.184978]
        )


        self.expected_dos = np.array(
            [-1.39603429e-03, -1.92390955e-03, -2.93336246e-03, -6.89005044e-03,
         -7.84338945e-03, -6.37879461e-03, -1.10690045e-02, -1.57944335e-02,
         -1.41017668e-02, -4.15140057e-03,  7.71792797e-03,  1.99412441e-02,
            5.10548794e-02,  1.01076768e-01,  1.56039938e-01,  2.15395112e-01,
            3.02663312e-01,  3.80252930e-01,  4.75254195e-01,  5.58468628e-01,
            6.54641167e-01,  7.57292255e-01,  8.32860223e-01,  9.14471696e-01,
            9.88996826e-01,  1.04753671e+00,  1.10007427e+00,  1.11869442e+00,
            1.13010925e+00,  1.14578536e+00,  1.12353510e+00,  1.13633460e+00,
            1.14922214e+00,  1.17046880e+00,  1.23263790e+00,  1.30740559e+00,
            1.39474870e+00,  1.47573002e+00,  1.54402758e+00,  1.59417936e+00,
            1.63017159e+00,  1.64617480e+00,  1.64632688e+00,  1.64807479e+00,
            1.65193703e+00,  1.64721726e+00,  1.69176031e+00,  1.72309620e+00,
            1.78413458e+00,  1.80906688e+00,  1.85785015e+00,  1.89456034e+00,
            1.96632172e+00,  2.01793914e+00,  2.05232993e+00,  2.08347003e+00,
            2.09469635e+00,  2.09994438e+00,  2.09880798e+00,  2.08695957e+00,
            2.07824070e+00,  2.08276622e+00,  2.06380779e+00,  2.03929363e+00,
            2.02880899e+00,  2.02322430e+00,  1.99222800e+00,  1.93997333e+00,
            1.88114274e+00,  1.80893034e+00,  1.76219292e+00,  1.82217359e+00,
            1.89333327e+00,  2.02360644e+00,  2.18544345e+00,  2.31464605e+00,
            2.38889812e+00,  2.41743755e+00,  2.39244101e+00,  2.36446368e+00,
            2.35240437e+00,  2.37677639e+00,  2.40832954e+00,  2.42714922e+00,
            2.43265914e+00,  2.39769011e+00,  2.34234329e+00,  2.28782583e+00,
            2.26373179e+00,  2.28309212e+00,  2.30357709e+00,  2.31154708e+00,
            2.29570135e+00,  2.27557353e+00,  2.24059163e+00,  2.24075605e+00,
            2.29794656e+00,  2.37066074e+00,  2.44316172e+00,  2.50178991e+00,
            2.53515486e+00,  2.58569544e+00,  2.67054320e+00,  2.76630915e+00,
            2.87856375e+00,  2.95530073e+00,  3.03032084e+00,  3.10891371e+00,
            3.16266196e+00,  3.23966642e+00,  3.28899912e+00,  3.32381674e+00,
            3.36746587e+00,  3.40019385e+00,  3.42817144e+00,  3.43598214e+00,
            3.47668524e+00,  3.47685799e+00,  3.52705824e+00,  3.58318639e+00,
            3.61960015e+00,  3.66636868e+00,  3.68055774e+00,  3.71591360e+00,
            3.71871289e+00,  3.72753381e+00,  3.72466450e+00,  3.70633333e+00,
            3.67081890e+00,  3.61239068e+00,  3.55272622e+00,  3.55024882e+00,
            3.55061903e+00,  3.56265875e+00,  3.55682624e+00,  3.52874426e+00,
            3.50783896e+00,  3.49618604e+00,  3.49037121e+00,  3.42867476e+00,
            3.35788068e+00,  3.26222434e+00,  3.17601970e+00,  3.07729261e+00,
            3.02038619e+00,  2.98073245e+00,  2.91513464e+00,  2.88749865e+00,
            2.83922788e+00,  2.84838806e+00,  2.84492479e+00,  2.92385605e+00,
            2.92999346e+00,  2.98952428e+00,  3.05588103e+00,  3.10640124e+00,
            3.14875677e+00,  3.21675587e+00,  3.27913677e+00,  3.33546772e+00,
            3.38229410e+00,  3.43984400e+00,  3.47070913e+00,  3.50738767e+00,
            3.55720798e+00,  3.57609687e+00,  3.57008300e+00,  3.57885280e+00,
            3.59893033e+00,  3.61423436e+00,  3.61980550e+00,  3.60556159e+00,
            3.56494389e+00,  3.54140919e+00,  3.54576875e+00,  3.55583969e+00,
            3.55858720e+00,  3.58428521e+00,  3.61107692e+00,  3.60119203e+00,
            3.59449853e+00,  3.57238820e+00,  3.54789758e+00,  3.52535313e+00,
            3.53170033e+00,  3.50967874e+00,  3.48335346e+00,  3.46534439e+00,
            3.42071765e+00,  3.38548036e+00,  3.33026055e+00,  3.28560776e+00,
            3.24771848e+00,  3.23164148e+00,  3.19545771e+00,  3.15457720e+00,
            3.09675198e+00,  3.04579247e+00,  3.01345920e+00,  2.97670851e+00,
            2.95000711e+00,  2.92729969e+00,  2.89379624e+00,  2.85327974e+00,
            2.81009972e+00,  2.77506619e+00,  2.72497897e+00,  2.66778611e+00,
            2.59606369e+00,  2.49898796e+00,  2.40319088e+00,  2.26655584e+00,
            2.09713280e+00,  1.90081697e+00,  1.69550901e+00,  1.47054048e+00,
            1.25949398e+00,  1.05075606e+00,  8.83294485e-01,  7.30385473e-01,
            5.75582455e-01,  4.56838769e-01,  3.50334853e-01,  2.63205822e-01,
            1.90607598e-01,  1.40443324e-01,  9.16355849e-02,  7.32581544e-02,
            4.85474570e-02,  2.66933884e-02,  1.93280518e-02,  1.02097760e-02,
        -2.27192998e-03, -1.34814976e-03,  3.94898606e-03,  6.28424522e-03,
        -5.52494008e-03,  3.76090091e-03, -1.44064397e-03,  2.79929602e-03,
        -2.88968774e-03,  6.90724081e-03, -2.16453825e-03, -2.19639041e-03,
            2.63994592e-05, -4.49649270e-03,  4.30308157e-03, -3.19810785e-04,
            1.06598030e-03, -2.42574160e-04]
        )        

        self.expected_ados_1 = np.array(
            [
                1.14175532e-03,
                -4.19174936e-05,
                -7.21885854e-04,
                -2.80353452e-05,
                2.28109645e-03,
                9.71054959e-04,
                -1.66136145e-03,
                2.41572074e-03,
                7.59108028e-04,
                -1.09641315e-03,
                1.05930884e-03,
                1.22141915e-03,
                7.34257777e-04,
                6.65559142e-03,
                1.37987075e-02,
                2.09233653e-02,
                3.13229430e-02,
                3.90634675e-02,
                4.82889212e-02,
                5.64319923e-02,
                7.06793091e-02,
                7.92214066e-02,
                8.53724891e-02,
                9.83516640e-02,
                1.05937433e-01,
                1.14458508e-01,
                1.23284993e-01,
                1.25192022e-01,
                1.26194526e-01,
                1.31977531e-01,
                1.31374207e-01,
                1.35887189e-01,
                1.40204884e-01,
                1.47443044e-01,
                1.59154415e-01,
                1.67596384e-01,
                1.86427662e-01,
                1.93725971e-01,
                2.01287734e-01,
                2.05798493e-01,
                2.11383466e-01,
                2.10529975e-01,
                2.10948934e-01,
                2.13330547e-01,
                2.10421916e-01,
                2.18339681e-01,
                2.24446963e-01,
                2.33688117e-01,
                2.42512492e-01,
                2.43103645e-01,
                2.51145837e-01,
                2.61808994e-01,
                2.68780114e-01,
                2.77509173e-01,
                2.80643596e-01,
                2.80808795e-01,
                2.82641315e-01,
                2.78653415e-01,
                2.74870187e-01,
                2.70382936e-01,
                2.67680230e-01,
                2.60875725e-01,
                2.54114342e-01,
                2.49571462e-01,
                2.45246974e-01,
                2.37318488e-01,
                2.25764569e-01,
                2.17221817e-01,
                2.00175024e-01,
                1.95963933e-01,
                1.88586498e-01,
                2.05101813e-01,
                2.17773534e-01,
                2.46793989e-01,
                2.70504716e-01,
                2.95048730e-01,
                3.14788577e-01,
                3.26637275e-01,
                3.31674055e-01,
                3.34866098e-01,
                3.31791696e-01,
                3.25904579e-01,
                3.19175448e-01,
                3.14870863e-01,
                3.07347313e-01,
                2.99575478e-01,
                2.88188485e-01,
                2.80888515e-01,
                2.75463219e-01,
                2.75962192e-01,
                2.70914392e-01,
                2.72346524e-01,
                2.66575201e-01,
                2.63365725e-01,
                2.60849191e-01,
                2.61214581e-01,
                2.65058337e-01,
                2.72583484e-01,
                2.75791312e-01,
                2.87116593e-01,
                2.94439358e-01,
                3.07951705e-01,
                3.25241910e-01,
                3.45943998e-01,
                3.69523054e-01,
                3.85090178e-01,
                3.96255952e-01,
                4.07755591e-01,
                4.16952481e-01,
                4.26006603e-01,
                4.33746764e-01,
                4.33862772e-01,
                4.42475912e-01,
                4.47680196e-01,
                4.57403410e-01,
                4.59588233e-01,
                4.71249637e-01,
                4.74479663e-01,
                4.81876616e-01,
                4.88556769e-01,
                4.94491769e-01,
                5.02388004e-01,
                5.04097251e-01,
                5.05881204e-01,
                5.08035046e-01,
                5.01445276e-01,
                4.98505709e-01,
                4.89784972e-01,
                4.81614100e-01,
                4.74807751e-01,
                4.62217755e-01,
                4.60946050e-01,
                4.52659675e-01,
                4.53501336e-01,
                4.45706331e-01,
                4.41109113e-01,
                4.39381973e-01,
                4.36055545e-01,
                4.24106541e-01,
                4.12410809e-01,
                3.96092021e-01,
                3.77029550e-01,
                3.60284659e-01,
                3.37813440e-01,
                3.21487274e-01,
                3.15186497e-01,
                3.01281828e-01,
                2.88366497e-01,
                2.79570150e-01,
                2.78651476e-01,
                2.80986285e-01,
                2.90303718e-01,
                3.00018574e-01,
                3.13242081e-01,
                3.20862051e-01,
                3.37943988e-01,
                3.47843716e-01,
                3.64386614e-01,
                3.73867819e-01,
                3.85681914e-01,
                3.95934276e-01,
                4.10017107e-01,
                4.24607864e-01,
                4.36153253e-01,
                4.45913830e-01,
                4.53163893e-01,
                4.53782804e-01,
                4.55815014e-01,
                4.63449080e-01,
                4.68111779e-01,
                4.72112727e-01,
                4.66982304e-01,
                4.63444524e-01,
                4.58356047e-01,
                4.52449991e-01,
                4.52977039e-01,
                4.55984179e-01,
                4.64264003e-01,
                4.76869597e-01,
                4.76048872e-01,
                4.78388604e-01,
                4.77437466e-01,
                4.75562646e-01,
                4.61257457e-01,
                4.62749451e-01,
                4.53935193e-01,
                4.48675591e-01,
                4.42963669e-01,
                4.35824001e-01,
                4.29616948e-01,
                4.24494137e-01,
                4.17412473e-01,
                4.12379942e-01,
                4.09341508e-01,
                4.03414671e-01,
                3.96561009e-01,
                3.90786014e-01,
                3.83890764e-01,
                3.77173193e-01,
                3.70658765e-01,
                3.66654453e-01,
                3.63236154e-01,
                3.56207521e-01,
                3.50589368e-01,
                3.45146125e-01,
                3.40724451e-01,
                3.31700407e-01,
                3.24164395e-01,
                3.14654255e-01,
                3.02725442e-01,
                2.87242813e-01,
                2.66959043e-01,
                2.44904564e-01,
                2.17484655e-01,
                1.98557229e-01,
                1.69451750e-01,
                1.47235840e-01,
                1.20522212e-01,
                1.02667938e-01,
                8.70284577e-02,
                7.16080345e-02,
                5.96054705e-02,
                4.73006111e-02,
                3.84373330e-02,
                2.82578156e-02,
                2.13776086e-02,
                1.68435434e-02,
                9.78067568e-03,
                7.22150185e-03,
                4.92237110e-03,
                1.10008363e-03,
                2.62159776e-03,
                1.60340975e-03,
                -1.43055047e-04,
                1.06168289e-03,
                2.64765256e-03,
                1.75308536e-03,
                1.03856602e-03,
                -5.73130834e-04,
                1.30352570e-03,
                1.37962860e-04,
                -1.85555443e-03,
                1.08196637e-04,
                1.28182817e-04,
                6.35661243e-04,
                1.73742133e-03,
                -1.75466671e-04,
                -9.39002066e-05,
                -2.47114260e-04,
                2.22984128e-04,
            ]
        )

        self.expected_ados_2 = np.array(
            [
                4.37750618e-04,
                -2.84755141e-04,
                -6.59853131e-04,
                6.73644688e-04,
                2.31276095e-03,
                2.18560151e-04,
                -4.75346698e-04,
                2.79159285e-03,
                1.74560866e-03,
                1.28767705e-03,
                4.45638374e-03,
                8.08237458e-03,
                9.57354233e-03,
                1.64885909e-02,
                2.61870193e-02,
                3.41981514e-02,
                4.56264900e-02,
                5.55092972e-02,
                6.61801887e-02,
                7.55433635e-02,
                8.85846807e-02,
                9.65154491e-02,
                1.02489414e-01,
                1.13540271e-01,
                1.19991412e-01,
                1.24728643e-01,
                1.32181878e-01,
                1.31369680e-01,
                1.29978364e-01,
                1.33587586e-01,
                1.32928488e-01,
                1.36641235e-01,
                1.41997692e-01,
                1.50958132e-01,
                1.63882685e-01,
                1.73442577e-01,
                1.91027716e-01,
                1.99458087e-01,
                2.05133500e-01,
                2.07975486e-01,
                2.09271601e-01,
                2.06500962e-01,
                2.06228352e-01,
                2.06733416e-01,
                2.03140835e-01,
                2.11942604e-01,
                2.17377985e-01,
                2.27001797e-01,
                2.34500837e-01,
                2.35726174e-01,
                2.41983984e-01,
                2.52500402e-01,
                2.57267031e-01,
                2.65919955e-01,
                2.68827238e-01,
                2.69761516e-01,
                2.71938465e-01,
                2.69081037e-01,
                2.65898431e-01,
                2.63100707e-01,
                2.63516146e-01,
                2.58521749e-01,
                2.54745800e-01,
                2.52718934e-01,
                2.49673786e-01,
                2.42329581e-01,
                2.33873310e-01,
                2.28351106e-01,
                2.13443843e-01,
                2.11956908e-01,
                2.04826947e-01,
                2.16954076e-01,
                2.27877326e-01,
                2.53221319e-01,
                2.73915630e-01,
                2.93494715e-01,
                3.06248452e-01,
                3.12142890e-01,
                3.13961714e-01,
                3.16879141e-01,
                3.16319735e-01,
                3.14301737e-01,
                3.11619366e-01,
                3.11177824e-01,
                3.04191740e-01,
                2.98211798e-01,
                2.88849226e-01,
                2.84207353e-01,
                2.81236363e-01,
                2.83824842e-01,
                2.80422275e-01,
                2.80570045e-01,
                2.76432510e-01,
                2.73302267e-01,
                2.71838229e-01,
                2.74293620e-01,
                2.79934281e-01,
                2.89864631e-01,
                2.93365508e-01,
                3.03229377e-01,
                3.09380242e-01,
                3.20525219e-01,
                3.34765143e-01,
                3.51459640e-01,
                3.68188892e-01,
                3.79535850e-01,
                3.87162763e-01,
                3.96112298e-01,
                4.05099458e-01,
                4.14478443e-01,
                4.22900260e-01,
                4.22298613e-01,
                4.29910448e-01,
                4.33504846e-01,
                4.40685090e-01,
                4.41656343e-01,
                4.50383171e-01,
                4.51256548e-01,
                4.57454902e-01,
                4.64650984e-01,
                4.70633353e-01,
                4.76912361e-01,
                4.77658170e-01,
                4.80482014e-01,
                4.84275651e-01,
                4.79409965e-01,
                4.76688684e-01,
                4.71061984e-01,
                4.63358320e-01,
                4.58944927e-01,
                4.47742653e-01,
                4.47493470e-01,
                4.41143643e-01,
                4.44172680e-01,
                4.37940954e-01,
                4.37225087e-01,
                4.36772611e-01,
                4.34943950e-01,
                4.24801770e-01,
                4.16444551e-01,
                4.02852759e-01,
                3.85265358e-01,
                3.70971870e-01,
                3.52902426e-01,
                3.40187438e-01,
                3.38268743e-01,
                3.26935260e-01,
                3.15814985e-01,
                3.10516400e-01,
                3.08000461e-01,
                3.11578892e-01,
                3.17795720e-01,
                3.25833928e-01,
                3.35835206e-01,
                3.42402433e-01,
                3.57610079e-01,
                3.64845861e-01,
                3.78730173e-01,
                3.85307342e-01,
                3.94934826e-01,
                4.02440605e-01,
                4.13596747e-01,
                4.23561135e-01,
                4.32692247e-01,
                4.40203545e-01,
                4.45291010e-01,
                4.44589502e-01,
                4.44198817e-01,
                4.50512056e-01,
                4.56489469e-01,
                4.59647782e-01,
                4.53951937e-01,
                4.50391516e-01,
                4.45091152e-01,
                4.40331850e-01,
                4.41002923e-01,
                4.43363861e-01,
                4.49741329e-01,
                4.58385671e-01,
                4.55776649e-01,
                4.56814334e-01,
                4.56587814e-01,
                4.55485403e-01,
                4.43580785e-01,
                4.46749524e-01,
                4.41314541e-01,
                4.36978785e-01,
                4.31616813e-01,
                4.26233732e-01,
                4.21321668e-01,
                4.17067671e-01,
                4.10033599e-01,
                4.05599102e-01,
                4.01991787e-01,
                3.97867798e-01,
                3.91444255e-01,
                3.85723756e-01,
                3.79363913e-01,
                3.73115658e-01,
                3.66946552e-01,
                3.65185429e-01,
                3.62364934e-01,
                3.56512826e-01,
                3.50874076e-01,
                3.45818623e-01,
                3.41860046e-01,
                3.34184892e-01,
                3.27630577e-01,
                3.19886213e-01,
                3.10514928e-01,
                2.97272557e-01,
                2.80498862e-01,
                2.59844742e-01,
                2.34551682e-01,
                2.15879337e-01,
                1.88100845e-01,
                1.64389883e-01,
                1.36951746e-01,
                1.17093193e-01,
                9.96545109e-02,
                8.40434593e-02,
                6.93282879e-02,
                5.50489160e-02,
                4.56236402e-02,
                3.49793068e-02,
                2.56817776e-02,
                2.06901055e-02,
                1.29793255e-02,
                9.12834860e-03,
                7.23339850e-03,
                2.85341899e-03,
                2.99081369e-03,
                1.94257419e-03,
                2.47909563e-04,
                4.33900286e-04,
                1.65883835e-03,
                1.81092419e-03,
                5.21943727e-04,
                -3.44130907e-04,
                8.94826999e-04,
                4.91047864e-04,
                -1.82618130e-03,
                4.20399394e-04,
                -4.10782601e-04,
                6.59204079e-04,
                1.97465107e-03,
                -4.62844767e-04,
                -1.49835051e-04,
                -5.06931628e-04,
                5.76254430e-04,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        os.remove("deepdos.pb")
        cls.dp = None

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 1)
        self.assertAlmostEqual(self.dp.get_rcut(), 5.0, places=default_places)
        self.assertEqual(self.dp.get_type_map(), ["Si"])
        self.assertEqual(self.dp.get_numb_dos(), 250)

    def test_1frame_atomic(self):
        dd = self.dp.eval(self.coords, self.box, self.atype, atomic=True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        numb_dos = 250
        self.assertEqual(dd[0].shape, (nframes, numb_dos))
        self.assertEqual(dd[1].shape, (nframes, natoms, numb_dos))
        # check values
        ados_list = dd[1].ravel().reshape(natoms, numb_dos)

        np.testing.assert_almost_equal(ados_list[0], self.expected_ados_1, 4)
        np.testing.assert_almost_equal(ados_list[1], self.expected_ados_2, 4)
        np.testing.assert_almost_equal(dd[0].ravel(), self.expected_dos, 4)

    def test_2frame_atomic(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        dd = self.dp.eval(coords2, box2, self.atype, atomic=True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        numb_dos = 250
        self.assertEqual(dd[0].shape, (nframes, numb_dos))
        self.assertEqual(dd[1].shape, (nframes, natoms, numb_dos))
        # check values
        expected_ados1 = np.concatenate((self.expected_ados_1, self.expected_ados_1))
        expected_ados2 = np.concatenate((self.expected_ados_2, self.expected_ados_2))
        expected_total = np.concatenate((self.expected_dos,    self.expected_dos))
        
        self.ados_list = dd[1].ravel().reshape(nframes, natoms, numb_dos)

        np.testing.assert_almost_equal(
            self.ados_list[:, 0, :].reshape(-1), expected_ados1, 4
        )
        np.testing.assert_almost_equal(
            self.ados_list[:, 1, :].reshape(-1), expected_ados2, 4
        )
        np.testing.assert_almost_equal(dd[0].ravel(), expected_total, 4)
