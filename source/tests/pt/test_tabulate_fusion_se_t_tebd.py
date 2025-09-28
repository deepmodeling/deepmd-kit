# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt.utils import (
    env,
)

from ..consistent.common import (
    parameterized,
)


@parameterized((torch.float64, torch.float32))
@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
class TestTabulateFusionSeTTebdOp(unittest.TestCase):
    def setUp(self) -> None:
        (dtype,) = self.param
        if dtype == torch.float64:
            self.prec = 1e-10
        elif dtype == torch.float32:
            # JZ: not sure the reason, but 1e-5 cannot pass the grad test
            self.prec = 1e-3
        self.table_tensor = torch.tensor(
            [
                -1.0600000163027882e02,
                7.7059358807135015e02,
                -5.6954714749735385e03,
                1.2167808756610991e03,
                -7.6199102434332218e01,
                1.0706136029373441e00,
                -1.0600000164528124e02,
                7.7059358630452323e02,
                -5.6954715659539552e03,
                1.2167808757436076e03,
                -7.6199099707724926e01,
                1.0706134206080884e00,
                -1.0600000163027882e02,
                7.7059358807135015e02,
                -5.6954714749735385e03,
                1.2167808756610991e03,
                -7.6199102434332218e01,
                1.0706136029373441e00,
                -1.0600000164528124e02,
                7.7059358630452323e02,
                -5.6954715659539552e03,
                1.2167808757436076e03,
                -7.6199099707724926e01,
                1.0706134206080884e00,
                -9.6000006759336443e01,
                6.2969719646863621e02,
                -4.2053706363664551e03,
                9.0372155784831205e02,
                -5.7600014239472898e01,
                8.6528676197113796e-01,
                -9.6000006828502180e01,
                6.2969718981238339e02,
                -4.2053709121998018e03,
                9.0372156236848912e02,
                -5.7600006817493266e01,
                8.6528625106787871e-01,
                -9.6000006759336443e01,
                6.2969719646863621e02,
                -4.2053706363664551e03,
                9.0372155784831205e02,
                -5.7600014239472898e01,
                8.6528676197113796e-01,
                -9.6000006828502180e01,
                6.2969718981238339e02,
                -4.2053709121998018e03,
                9.0372156236848912e02,
                -5.7600006817493266e01,
                8.6528625106787871e-01,
                -8.6000028021606425e01,
                5.0303296429845562e02,
                -3.0008648248894533e03,
                6.4939597734382562e02,
                -4.2250984019314707e01,
                6.8180015607155764e-01,
                -8.6000028340480625e01,
                5.0303293978396903e02,
                -3.0008656209622986e03,
                6.4939600529391078e02,
                -4.2250965541906716e01,
                6.8179882734268982e-01,
                -8.6000028021606425e01,
                5.0303296429845562e02,
                -3.0008648248894533e03,
                6.4939597734382562e02,
                -4.2250984019314707e01,
                6.8180015607155764e-01,
                -8.6000028340480625e01,
                5.0303293978396903e02,
                -3.0008656209622986e03,
                6.4939600529353049e02,
                -4.2250965541830588e01,
                6.8179882733888086e-01,
                -7.6000116148038558e01,
                3.9060139597613619e02,
                -2.0515743554479322e03,
                4.4772754091167945e02,
                -2.9848087537832814e01,
                5.2014755686537917e-01,
                -7.6000117618125429e01,
                3.9060130821883052e02,
                -2.0515765138621105e03,
                4.4772766653712006e02,
                -2.9848047259266409e01,
                5.2014443989116910e-01,
                -7.6000116148038558e01,
                3.9060139597613619e02,
                -2.0515743554479322e03,
                4.4772754091167945e02,
                -2.9848087537832814e01,
                5.2014755686537917e-01,
                -7.6000117618125742e01,
                3.9060130821877993e02,
                -2.0515765138659344e03,
                4.4772766652483722e02,
                -2.9848047256692499e01,
                5.2014443976043645e-01,
                -6.6000481290731443e01,
                2.9240425245900917e02,
                -1.3271250821434478e03,
                2.9263955624337893e02,
                -2.0087224005740719e01,
                3.8031147992206349e-01,
                -6.6000488067863742e01,
                2.9240394960550276e02,
                -1.3271304743966571e03,
                2.9264002765325057e02,
                -2.0087154325946980e01,
                3.8030522013794582e-01,
                -6.6000481290731443e01,
                2.9240425245900917e02,
                -1.3271250821434478e03,
                2.9263955624337893e02,
                -2.0087224005740719e01,
                3.8031147992206349e-01,
                -6.6000488067883694e01,
                2.9240394960308691e02,
                -1.3271304745319526e03,
                2.9264002727267626e02,
                -2.0087154245656002e01,
                3.8030521605011575e-01,
                -5.6001992867343972e01,
                2.0844745574402617e02,
                -7.9715799906587699e02,
                1.7805563184427194e02,
                -1.2663929104029080e01,
                2.6224978307822894e-01,
                -5.6002024103130161e01,
                2.0844646075692629e02,
                -7.9717003898786652e02,
                1.7805715054974732e02,
                -1.2663864677938077e01,
                2.6224029170957303e-01,
                -5.6001992867343972e01,
                2.0844745574402617e02,
                -7.9715799906587699e02,
                1.7805563184427194e02,
                -1.2663929104029080e01,
                2.6224978307822894e-01,
                -5.6002024104383771e01,
                2.0844646064871867e02,
                -7.9717004324410516e02,
                1.7805714044473001e02,
                -1.2663862524337585e01,
                2.6224018166598279e-01,
                -4.6008230210744550e01,
                1.3874976550319553e02,
                -4.3134867537287749e02,
                9.7902623595157010e01,
                -7.2734403121911884e00,
                1.6589123996688057e-01,
                -4.6008373996710617e01,
                1.3874671965012058e02,
                -4.3137141216256458e02,
                9.7906861443792735e01,
                -7.2735856084076280e00,
                1.6588642735924275e-01,
                -4.6008230210744550e01,
                1.3874976550319553e02,
                -4.3134867537287749e02,
                9.7902623595157010e01,
                -7.2734403121911884e00,
                1.6589123996688057e-01,
                -4.6008374075307870e01,
                1.3874671513440606e02,
                -4.3137152784492957e02,
                9.7906652364871050e01,
                -7.2735401377994249e00,
                1.6588408717348646e-01,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).reshape(8, 22)  # Adjusted for SE_T_TEBD table size
        self.table_info_tensor = torch.tensor(
            [
                -2.1000000000000000e01,
                2.1000000000000000e01,
                1.0500000000000000e02,
                1.0000000000000000e00,
                1.0000000000000000e01,
                -1.0000000000000000e00,
            ],
            dtype=dtype,
            device="cpu",
        )
        self.em_x_tensor = torch.tensor(
            [
                9.3816147034272368e-01,
                -1.6703373029862567e-01,
                -4.4294526064601734e-02,
                -2.8798505489184573e-01,
                -1.6703373029862567e-01,
                9.2489218226366088e-01,
                -2.8928196536572048e-01,
                -4.7833509099876154e-01,
                -4.4294526064601734e-02,
                -2.8928196536572048e-01,
                5.7034320185695120e-01,
                1.8771147911830000e-01,
                -2.8798505489184573e-01,
                -4.7833509099876154e-01,
                1.8771147911830000e-01,
                4.0174654365823070e-01,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).reshape(4, 4)
        self.em_tensor = self.em_x_tensor.reshape(4, 4, 1)  # SE_T_TEBD uses angular information, so 1D
        self.table_info_tensor.requires_grad = False
        self.table_tensor.requires_grad = False
        self.em_x_tensor.requires_grad = True
        self.em_tensor.requires_grad = True
        self.last_layer_size = 4
        self.nloc = 16  # Adjusted for SE_T_TEBD structure
        self.nnei_i = 4
        self.nnei_j = 4

        # Expected results for SE_T_TEBD test
        self.expected_descriptor_tensor = torch.tensor(
            [
                1.2571973325754339e00,
                2.3214997685364109e00,
                2.9394341134078902e00,
                2.0727894815158436e00,
                1.7127738317829568e00,
                2.3288382955492263e00,
                2.9401587802428659e00,
                2.3252400661016079e00,
                8.4806287131835343e-01,
                2.1778589851963829e00,
                2.6273548699126683e00,
                1.7358633427396228e00,
                1.9586806210305824e00,
                2.4256636737020518e00,
                3.1955783231847523e00,
                2.5091329174140033e00,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).reshape(4, 4)

        # Expected gradients for SE_T_TEBD test
        self.expected_dy_dem_x = torch.tensor(
            [
                5.448489055364202,
                -0.7109841888364551,
                -0.14536867097411239,
                -1.0747441933374314,
                -0.7109841888364551,
                5.351778760144183,
                -1.077917429853053,
                -1.579018415609313,
                -0.14536867097411239,
                -1.077917429853053,
                2.866855971667982,
                0.9527786223200397,
                -1.0747441933374314,
                -1.579018415609313,
                0.9527786223200397,
                2.054128070312613,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).reshape(4, 4)

        self.expected_dy_dem = torch.tensor(
            [
                -5.614759600210596,
                -11.731713987629353,
                -11.090122813510595,
                -12.270780785045307,
                -11.731713987629353,
                -5.694533094540567,
                -12.276066372858583,
                -12.975171860091254,
                -11.090122813510595,
                -12.276066372858583,
                -7.703917285588026,
                -9.777251464178656,
                -12.270780785045307,
                -12.975171860091254,
                -9.777251464178656,
                -8.605541422983027,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).reshape(4, 4, 1)

    def test_forward(self) -> None:
        # Call the forward function - SE_T_TEBD uses the same tabulate_fusion_se_t operation
        forward_result = torch.ops.deepmd.tabulate_fusion_se_t(
            self.table_tensor,
            self.table_info_tensor,
            self.em_x_tensor,
            self.em_tensor,
            self.last_layer_size,
        )

        descriptor_tensor = forward_result[0]

        # Check the shape
        self.assertEqual(descriptor_tensor.shape, self.expected_descriptor_tensor.shape)

        # Check the values
        torch.testing.assert_close(
            descriptor_tensor,
            self.expected_descriptor_tensor,
            atol=self.prec,
            rtol=self.prec,
        )

    def test_backward(self) -> None:
        # Call the forward function
        forward_result = torch.ops.deepmd.tabulate_fusion_se_t(
            self.table_tensor,
            self.table_info_tensor,
            self.em_x_tensor,
            self.em_tensor,
            self.last_layer_size,
        )

        descriptor_tensor = forward_result[0]

        # Check the forward
        torch.testing.assert_close(
            descriptor_tensor,
            self.expected_descriptor_tensor,
            atol=self.prec,
            rtol=self.prec,
        )

        # Create a loss and perform backward
        loss = descriptor_tensor.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(self.em_x_tensor.grad)
        self.assertIsNotNone(self.em_tensor.grad)

        # Check the shapes of the gradients
        self.assertEqual(self.em_x_tensor.grad.shape, self.expected_dy_dem_x.shape)
        self.assertEqual(self.em_tensor.grad.shape, self.expected_dy_dem.shape)

        # Check the values of the gradients
        torch.testing.assert_close(
            self.em_x_tensor.grad,
            self.expected_dy_dem_x,
            atol=self.prec,
            rtol=self.prec,
        )

        torch.testing.assert_close(
            self.em_tensor.grad,
            self.expected_dy_dem,
            atol=self.prec,
            rtol=self.prec,
        )


if __name__ == "__main__":
    unittest.main()
