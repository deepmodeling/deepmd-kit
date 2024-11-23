# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import (
    Path,
)

import torch

from deepmd.pt.model.descriptor import (
    DescrptBlockSeAtten,
    DescrptDPA1,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

CUR_DIR = os.path.dirname(__file__)


class TestDPA1(unittest.TestCase):
    def setUp(self) -> None:
        cell = [
            5.122106549439247480e00,
            4.016537340154059388e-01,
            6.951654033828678081e-01,
            4.016537340154059388e-01,
            6.112136112297989143e00,
            8.178091365465004481e-01,
            6.951654033828678081e-01,
            8.178091365465004481e-01,
            6.159552512682983760e00,
        ]
        self.cell = torch.tensor(
            cell, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        ).view(1, 3, 3)
        coord = [
            2.978060152121375648e00,
            3.588469695887098077e00,
            2.792459820604495491e00,
            3.895592322591093115e00,
            2.712091020667753760e00,
            1.366836847133650501e00,
            9.955616170888935690e-01,
            4.121324820711413039e00,
            1.817239061889086571e00,
            3.553661462345699906e00,
            5.313046969500791583e00,
            6.635182659098815883e00,
            6.088601018589653080e00,
            6.575011420004332585e00,
            6.825240650611076099e00,
        ]
        self.coord = torch.tensor(
            coord, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        ).view(1, -1, 3)
        self.atype = torch.tensor(
            [0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE
        ).view(1, -1)
        self.ref_d = torch.tensor(
            [
                8.382518544113587780e-03,
                -3.390120566088597812e-03,
                6.145981571114964362e-03,
                -4.880300873973819273e-03,
                -3.390120566088597812e-03,
                1.372540996564941464e-03,
                -2.484163690574096341e-03,
                1.972313058658722688e-03,
                6.145981571114964362e-03,
                -2.484163690574096341e-03,
                4.507748738021747671e-03,
                -3.579717194906019764e-03,
                -4.880300873973819273e-03,
                1.972313058658722688e-03,
                -3.579717194906019764e-03,
                2.842794615687799838e-03,
                6.733043802494966066e-04,
                -2.721540313345096771e-04,
                4.936158526085561134e-04,
                -3.919743287822345223e-04,
                -1.311123004527576900e-02,
                5.301179352601203924e-03,
                -9.614612349318877454e-03,
                7.634884975521277241e-03,
                8.877088452901006621e-03,
                -3.590945566653638409e-03,
                6.508042782015627942e-03,
                -5.167671664327699171e-03,
                -2.697241463040870365e-03,
                1.091350446825975137e-03,
                -1.976895708961905022e-03,
                1.569671412121975348e-03,
                8.645131636261189911e-03,
                -3.557395265621639355e-03,
                6.298048561552698106e-03,
                -4.999272007935521948e-03,
                -3.557395265621639355e-03,
                1.467866637220284964e-03,
                -2.587004431651147504e-03,
                2.052752235601402672e-03,
                6.298048561552698106e-03,
                -2.587004431651147504e-03,
                4.594085551315935101e-03,
                -3.647656549789176847e-03,
                -4.999272007935521948e-03,
                2.052752235601402672e-03,
                -3.647656549789176847e-03,
                2.896359275520481256e-03,
                6.689620176492027878e-04,
                -2.753606422414641049e-04,
                4.864958810186969444e-04,
                -3.860599754167503119e-04,
                -1.349238259226558101e-02,
                5.547478630961994242e-03,
                -9.835472300819447095e-03,
                7.808197926069362048e-03,
                9.220744348752592245e-03,
                -3.795799103392961601e-03,
                6.716516319358462918e-03,
                -5.331265718473574867e-03,
                -2.783836698392940304e-03,
                1.147461939123531121e-03,
                -2.025013030986024063e-03,
                1.606944814423778541e-03,
                9.280385723343491378e-03,
                -3.515852178447095942e-03,
                7.085282215778941628e-03,
                -5.675852414643783178e-03,
                -3.515852178447095942e-03,
                1.337760635271160884e-03,
                -2.679428786337713451e-03,
                2.145400621815936413e-03,
                7.085282215778941628e-03,
                -2.679428786337713451e-03,
                5.414439648102228192e-03,
                -4.338426468139268931e-03,
                -5.675852414643783178e-03,
                2.145400621815936413e-03,
                -4.338426468139268931e-03,
                3.476467482674507146e-03,
                7.166961981167455130e-04,
                -2.697932188839837972e-04,
                5.474643906631899504e-04,
                -4.386556623669893621e-04,
                -1.480434821331240956e-02,
                5.604647062899507579e-03,
                -1.130745349141585449e-02,
                9.059113563516829268e-03,
                9.758791063112262978e-03,
                -3.701477720487638626e-03,
                7.448215522796466058e-03,
                -5.966057584545172120e-03,
                -2.845102393948158344e-03,
                1.078743584169829543e-03,
                -2.170093031447992756e-03,
                1.738010461687942770e-03,
                9.867599071916231118e-03,
                -3.811041717688905522e-03,
                7.121877634386481262e-03,
                -5.703120290113914553e-03,
                -3.811041717688905522e-03,
                1.474046183772771213e-03,
                -2.747386907428428938e-03,
                2.199711055637492037e-03,
                7.121877634386481262e-03,
                -2.747386907428428938e-03,
                5.145050639440944609e-03,
                -4.120642824501622239e-03,
                -5.703120290113914553e-03,
                2.199711055637492037e-03,
                -4.120642824501622239e-03,
                3.300262321758350853e-03,
                1.370499995344566383e-03,
                -5.313041843655797901e-04,
                9.860110343046961986e-04,
                -7.892505817954784597e-04,
                -1.507686316307561489e-02,
                5.818961290579217904e-03,
                -1.088774506142304276e-02,
                8.719460408506790952e-03,
                9.764630842803939323e-03,
                -3.770134041110058572e-03,
                7.049438389985595785e-03,
                -5.645302934019884485e-03,
                -3.533582373572779437e-03,
                1.367148320603491559e-03,
                -2.546602904764623705e-03,
                2.038882844528267305e-03,
                7.448297038731285964e-03,
                -2.924276815200288742e-03,
                5.355960540523636154e-03,
                -4.280386435083473329e-03,
                -2.924276815200288742e-03,
                1.150311064893848757e-03,
                -2.100635980860638373e-03,
                1.678427895009850001e-03,
                5.355960540523636154e-03,
                -2.100635980860638373e-03,
                3.853607053247790071e-03,
                -3.080076301871465493e-03,
                -4.280386435083473329e-03,
                1.678427895009850001e-03,
                -3.080076301871465493e-03,
                2.461876613756722523e-03,
                9.730712866459405395e-04,
                -3.821759579990726546e-04,
                6.994242056622360787e-04,
                -5.589662297882965055e-04,
                -1.138916742131982317e-02,
                4.469391132927387489e-03,
                -8.192016282448397885e-03,
                6.547234460517113892e-03,
                7.460070829043288082e-03,
                -2.929867802018087421e-03,
                5.363646855497249989e-03,
                -4.286347242903034739e-03,
                -2.643569023340565718e-03,
                1.038826463247002245e-03,
                -1.899910089750410976e-03,
                1.518237240362583541e-03,
            ],
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        with open(Path(CUR_DIR) / "models" / "dpa1.json") as fp:
            self.model_json = json.load(fp)
        self.file_model_param = Path(CUR_DIR) / "models" / "dpa1.pth"
        self.file_type_embed = Path(CUR_DIR) / "models" / "dpa2_tebd.pth"

    def test_descriptor_block(self) -> None:
        # torch.manual_seed(0)
        model_dpa1 = self.model_json
        dparams = model_dpa1["descriptor"]
        ntypes = len(model_dpa1["type_map"])
        assert "se_atten" == dparams.pop("type")
        dparams["ntypes"] = ntypes
        des = DescrptBlockSeAtten(
            **dparams,
        ).to(env.DEVICE)
        state_dict = torch.load(self.file_model_param, weights_only=True)
        # this is an old state dict, modify manually
        state_dict["compress_info.0"] = des.compress_info[0]
        state_dict["compress_data.0"] = des.compress_data[0]
        des.load_state_dict(state_dict)
        coord = self.coord
        atype = self.atype
        box = self.cell
        # handle type_embedding
        type_embedding = TypeEmbedNet(ntypes, 8, use_tebd_bias=True).to(env.DEVICE)
        type_embedding.load_state_dict(
            torch.load(self.file_type_embed, weights_only=True)
        )

        ## to save model parameters
        # torch.save(des.state_dict(), 'model_weights.pth')
        # torch.save(type_embedding.state_dict(), 'model_weights.pth')
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord,
            atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=box,
        )
        descriptor, env_mat, diff, rot_mat, sw = des(
            nlist,
            extended_coord,
            extended_atype,
            type_embedding(extended_atype),
            mapping=None,
        )
        # np.savetxt('tmp.out', descriptor.detach().numpy().reshape(1,-1), delimiter=",")
        self.assertEqual(descriptor.shape[-1], des.get_dim_out())
        self.assertAlmostEqual(6.0, des.get_rcut())
        self.assertEqual(30, des.get_nsel())
        self.assertEqual(2, des.get_ntypes())
        torch.testing.assert_close(
            descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10
        )

    def test_descriptor(self) -> None:
        with open(Path(CUR_DIR) / "models" / "dpa1.json") as fp:
            self.model_json = json.load(fp)
        model_dpa2 = self.model_json
        ntypes = len(model_dpa2["type_map"])
        dparams = model_dpa2["descriptor"]
        dparams["ntypes"] = ntypes
        assert dparams.pop("type") == "se_atten"
        dparams["concat_output_tebd"] = False
        dparams["use_tebd_bias"] = True
        des = DescrptDPA1(
            **dparams,
        ).to(env.DEVICE)
        target_dict = des.state_dict()
        source_dict = torch.load(self.file_model_param, weights_only=True)
        type_embd_dict = torch.load(self.file_type_embed, weights_only=True)
        target_dict = translate_se_atten_and_type_embd_dicts_to_dpa1(
            target_dict,
            source_dict,
            type_embd_dict,
        )
        des.load_state_dict(target_dict)

        coord = self.coord
        atype = self.atype
        box = self.cell
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord,
            atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=box,
        )
        descriptor, env_mat, diff, rot_mat, sw = des(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        self.assertEqual(descriptor.shape[-1], des.get_dim_out())
        self.assertAlmostEqual(6.0, des.get_rcut())
        self.assertEqual(30, des.get_nsel())
        self.assertEqual(2, des.get_ntypes())
        torch.testing.assert_close(
            descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10
        )

        dparams["concat_output_tebd"] = True
        des = DescrptDPA1(
            **dparams,
        ).to(env.DEVICE)
        descriptor, env_mat, diff, rot_mat, sw = des(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        self.assertEqual(descriptor.shape[-1], des.get_dim_out())


def translate_se_atten_and_type_embd_dicts_to_dpa1(
    target_dict,
    source_dict,
    type_embd_dict,
):
    all_keys = list(target_dict.keys())
    record = [False for ii in all_keys]
    for kk, vv in source_dict.items():
        tk = "se_atten." + kk
        record[all_keys.index(tk)] = True
        target_dict[tk] = vv
    assert len(type_embd_dict.keys()) == 2
    it = iter(type_embd_dict.keys())
    for _ in range(2):
        kk = next(it)
        tk = "type_embedding." + kk
        record[all_keys.index(tk)] = True
        target_dict[tk] = type_embd_dict[kk]
    record[all_keys.index("se_atten.compress_data.0")] = True
    record[all_keys.index("se_atten.compress_info.0")] = True
    assert all(record)
    return target_dict
