import os
import pathlib

import numpy as np
from common import (
    Data,
    DataSystem,
    gen_data,
    j_loader,
)
from deepmd.utils.convert import (
    convert_pbtxt_to_pb,
)

from deepmd.common import (
    j_must_have,
)
from deepmd.descriptor import (
    DescrptSeA,
    DescrptSeAMask,
)
from deepmd.env import (
    tf,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.utils.type_embed import (
    TypeEmbedNet,
    embed_atom_type,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

tests_path = pathlib.Path(__file__).parent.absolute()


class TestModel(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(tests_path / os.path.join("infer", "dp4mask.pbtxt")), "dp4mask.pb"
        )
        cls.dp = DeepPot("dp4mask.pb")

    def test_dp_mask_model(self):
        dcoord = np.array(
            [
                -6.407,
                51.239,
                21.239,
                -4.533,
                55.791,
                24.103,
                -1.546,
                52.078,
                23.033,
                -3.659,
                55.483,
                18.965,
                -7.267,
                51.116,
                22.571,
                -2.89,
                54.524,
                17.964,
                -3.25,
                56.514,
                24.709,
                -1.393,
                51.214,
                24.346,
                -1.342,
                54.587,
                18.033,
                -8.797,
                51.403,
                22.416,
                -2.945,
                56.22,
                26.207,
                0.683,
                54.251,
                19.4,
                -0.161,
                51.66,
                25.169,
                -10.554,
                52.738,
                21.317,
                -2.728,
                54.512,
                27.98,
                0.826,
                53.412,
                26.593,
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
                -0.663,
                54.867,
                17.04,
                -9.655,
                50.806,
                23.075,
                -2.615,
                57.116,
                26.99,
                0.865,
                50.976,
                25.239,
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
                -3.459,
                53.157,
                18.147,
                -2.12,
                56.187,
                23.802,
                -2.661,
                51.35,
                25.109,
                -0.744,
                54.299,
                19.21,
                -6.981,
                49.739,
                23.059,
                -9.184,
                52.349,
                21.534,
                -3.039,
                54.944,
                26.641,
                -0.251,
                52.838,
                25.827,
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
                -5.425,
                50.76,
                21.427,
                -4.74,
                56.268,
                23.123,
                -1.583,
                53.145,
                23.351,
                -4.432,
                53.23,
                18.391,
                -2.895,
                52.617,
                18.775,
                -1.345,
                54.168,
                19.998,
                -6.894,
                51.833,
                23.296,
                -3.113,
                52.191,
                24.786,
                -4.737,
                55.416,
                18.692,
                -2.408,
                56.404,
                22.862,
                -1.928,
                55.211,
                23.898,
                -0.577,
                51.98,
                22.491,
                -6.888,
                50.55,
                20.505,
                -5.378,
                56.15,
                24.735,
                -8.441,
                52.757,
                20.99,
                -3.365,
                56.516,
                18.67,
                -3.428,
                54.294,
                25.977,
                -3.248,
                50.577,
                24.869,
                -6.093,
                49.461,
                22.691,
                -3.132,
                54.862,
                16.962,
                -3.433,
                57.582,
                24.671,
                -1.226,
                50.179,
                24.075,
                -1.126,
                53.316,
                25.774,
                0.917,
                53.894,
                20.427,
                -7.787,
                49.166,
                22.917,
                1.126,
                55.262,
                19.265,
                1.152,
                53.554,
                18.669,
                -10.607,
                53.534,
                20.543,
                -2.764,
                53.401,
                28.036,
                -11.001,
                53.126,
                22.259,
                -11.154,
                51.866,
                20.973,
                -1.71,
                54.849,
                28.274,
                -3.465,
                54.929,
                28.701,
                0.532,
                54.416,
                26.971,
                1.735,
                53.526,
                25.961,
                1.075,
                52.76,
                27.46,
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
                -3.499,
                55.357,
                20.781,
                -2.882,
                51.791,
                21.816,
                -6.17,
                52.791,
                20.442,
                -4.673,
                53.978,
                24.029,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -4.296,
                53.519,
                21.692,
            ]
        )
        datype = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                5,
            ]
        )
        aparam = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
            ]
        )
        dbox = np.array(
            [20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0]
        )  # Not used in parctice. For interface compatibility.

        expected_f = np.array(
            [
                -3.5993690e01,
                -7.7648770e01,
                1.6714340e01,
                -2.7043980e01,
                5.7522240e01,
                5.0546710e01,
                4.5166060e01,
                2.6040450e01,
                4.8261500e00,
                1.7129000e00,
                1.7982240e01,
                -3.6960410e01,
                1.5873100e01,
                -8.4242400e00,
                -2.5200920e01,
                -2.0512270e01,
                -1.2358140e01,
                3.7627860e01,
                -9.1338400e00,
                -4.0187150e01,
                -2.4780930e01,
                -2.8910740e01,
                2.3607110e01,
                -1.3480000e01,
                7.0969700e00,
                -3.7301300e00,
                -7.6201300e00,
                -8.7067900e00,
                -1.4401450e01,
                6.8929500e00,
                7.7028100e00,
                1.4252170e01,
                7.0598000e00,
                1.2804130e01,
                2.7794000e-01,
                -1.4302700e00,
                9.0479600e00,
                -1.6057170e01,
                2.3842000e-01,
                -1.0324340e01,
                -1.5808700e00,
                1.7572700e00,
                -5.6691000e-01,
                -2.6487700e00,
                1.2565150e01,
                1.0164590e01,
                4.1872300e00,
                7.5286900e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                -8.6003100e00,
                1.0519000e00,
                1.0238500e00,
                1.0853090e01,
                -5.4747600e00,
                1.3128300e00,
                -2.1731100e00,
                -4.3941000e-01,
                -1.0032100e01,
                -8.3830400e00,
                5.1415500e00,
                -5.2965000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                3.1024020e01,
                1.6851350e01,
                -3.4651860e01,
                1.0799920e01,
                2.7493150e01,
                3.7627860e01,
                2.9150200e01,
                -3.6832300e00,
                1.4268510e01,
                -1.1658280e01,
                -5.5203400e00,
                -2.4266600e00,
                -2.1513200e01,
                3.9396740e01,
                2.2037020e01,
                1.8250890e01,
                4.3413000e-01,
                -1.9913100e00,
                5.5734400e00,
                -5.6069800e00,
                -1.6624750e01,
                -8.4110300e00,
                2.3294000e00,
                -7.0068400e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                -1.0464110e01,
                1.0451380e01,
                4.9436700e00,
                9.6216800e00,
                -1.1424330e01,
                8.1045600e00,
                -2.2724300e00,
                -1.4578240e01,
                -2.7763700e00,
                9.2650000e-01,
                -8.4823200e00,
                1.4528340e01,
                -9.0125200e00,
                -5.1504900e00,
                1.3731300e01,
                -9.4581000e00,
                -1.0106400e00,
                1.4434730e01,
                1.6992000e00,
                6.5790400e00,
                1.7111000e00,
                -1.1719670e01,
                3.7697000e00,
                -2.5802300e00,
                1.4025330e01,
                -3.8276400e00,
                3.2638100e00,
                2.5671000e-01,
                -7.5635600e00,
                -8.3684200e00,
                -2.2283000e-01,
                -1.2754010e01,
                -9.4995800e00,
                -1.2458010e01,
                2.1459500e00,
                4.3690100e00,
                1.9534400e00,
                6.4804800e00,
                1.0346300e01,
                4.9571900e00,
                -5.8393700e00,
                -5.3999900e00,
                8.4040800e00,
                1.1162310e01,
                -7.2808400e00,
                -3.1794200e00,
                -8.8637400e00,
                9.2664000e00,
                -7.3975500e00,
                -1.1542260e01,
                -6.2214300e00,
                -1.0755400e01,
                -4.4821800e00,
                -3.2248400e00,
                -2.9953900e00,
                -1.4734280e01,
                -6.6545400e00,
                8.4772000e-01,
                4.0618400e00,
                -4.1332800e00,
                2.7551000e-01,
                4.0179500e00,
                -7.9780000e-02,
                2.2621800e00,
                -3.8397900e00,
                1.4491500e00,
                -6.7421000e00,
                7.2411900e00,
                2.4156400e00,
                -6.9857000e00,
                5.7534200e00,
                -1.5221540e01,
                1.3440600e00,
                -1.5649430e01,
                -6.0455300e00,
                -1.4024600e00,
                -1.3145880e01,
                5.2747000e-01,
                -5.4272300e00,
                1.1973150e01,
                9.5885400e00,
                4.3762200e00,
                -1.1622900e01,
                1.1301080e01,
                -1.0030000e-01,
                1.7600320e01,
                -4.6429600e00,
                2.6303800e00,
                -6.0677900e00,
                -1.2528040e01,
                4.0247000e00,
                1.2156810e01,
                5.9718100e00,
                -1.4261530e01,
                -2.6802700e00,
                -1.8094600e00,
                1.1774170e01,
                -4.7346400e00,
                -4.9402900e00,
                2.0064300e00,
                -1.6625570e01,
                -7.5587800e00,
                -1.0956040e01,
                -2.7341200e00,
                1.0369560e01,
                5.4700000e-02,
                9.6412000e00,
                -1.1201480e01,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                8.4683690e01,
                1.5705003e02,
                -8.6262560e01,
                1.0821187e02,
                -1.3312720e02,
                -1.5838900e00,
                -2.5402410e01,
                5.5693000e01,
                -3.7288700e01,
                -8.3411400e00,
                -2.9747100e00,
                1.4720840e01,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                -1.1806986e02,
                -4.1128690e01,
                6.9734090e01,
            ]
        )

        atom_pref = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
            ]
        )
        atom_pref = np.repeat(atom_pref, 3)

        ee, ff, vv = self.dp.eval(dcoord, dbox, datype, aparam=aparam)
        ff = ff.reshape(expected_f.shape)

        diff_ff = np.multiply(np.square(ff - expected_f), atom_pref)
        normalized_diff_ff = np.sqrt(np.sum(diff_ff) / np.sum(atom_pref))

        assert normalized_diff_ff < 100

    def test_descriptor_se_a_mask(self):
        jfile = "zinc_se_a_mask.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata["training"]["validation_data"], "systems")
        # set_pfx = j_must_have(jdata['validation_data'], "set_prefix")
        set_pfx = "set"
        batch_size = 2
        test_size = 1
        rcut = 20.0  # For DataSystem interface compatibility, not used in this test.
        sel = j_must_have(jdata["model"]["descriptor"], "sel")
        ntypes = len(sel)
        total_atom_num = np.cumsum(sel)[-1]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        assert (
            jdata["model"]["descriptor"]["type"] == "se_a_mask"
        ), "Wrong descriptor type"
        descrpt = DescrptSeAMask(**jdata["model"]["descriptor"], uniform_seed=True)

        t_coord = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_coord"
        )
        t_type = tf.placeholder(tf.int32, [None, None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_aparam = tf.placeholder(tf.int32, [None, None], name="i_aparam")

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            input_dict={"aparam": t_aparam},
            reuse=False,
        )

        # Manually set the aparam to be all zeros. So that all particles are masked as virtual atoms.
        # This is to test the correctness of the mask.
        test_data["aparam"] = np.zeros([numb_test, total_atom_num], dtype=np.int32)
        feed_dict_test = {
            t_coord: test_data["coord"][:numb_test, :],
            t_box: test_data["box"][:numb_test, :],
            t_type: test_data["type"][:numb_test, :],
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            t_aparam: test_data["aparam"][:numb_test, :],
            is_training: False,
        }
        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [op_dout] = sess.run([dout], feed_dict=feed_dict_test)
        op_dout = op_dout.reshape([-1])

        ref_dout = np.zeros(op_dout.shape, dtype=float)

        places = 10
        np.testing.assert_almost_equal(op_dout, ref_dout, places)
