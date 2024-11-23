# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import (
    Path,
)

import torch

from deepmd.pt.model.descriptor import (
    DescrptDPA2,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

CUR_DIR = os.path.dirname(__file__)


class TestDPA2(unittest.TestCase):
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
                8.435412613327306630e-01,
                -4.717109614540972440e-01,
                -1.812643456954206256e00,
                -2.315248767961955167e-01,
                -7.112973006771171613e-01,
                -4.162041919507591392e-01,
                -1.505159810095323181e00,
                -1.191652416985768403e-01,
                8.439214937875325617e-01,
                -4.712976890460106594e-01,
                -1.812605149396642856e00,
                -2.307222236291133766e-01,
                -7.115427800870099961e-01,
                -4.164729253167227530e-01,
                -1.505483119125936797e00,
                -1.191288524278367872e-01,
                8.286420823261241297e-01,
                -4.535033763979030574e-01,
                -1.787877160970498425e00,
                -1.961763875645104460e-01,
                -7.475459187804838201e-01,
                -5.231446874663764346e-01,
                -1.488399984491664219e00,
                -3.974117581747104583e-02,
                8.283793431613817315e-01,
                -4.551551577556525729e-01,
                -1.789253136645859943e00,
                -1.977673627726055372e-01,
                -7.448826048241211639e-01,
                -5.161350182531234676e-01,
                -1.487589463573479209e00,
                -4.377376017839779143e-02,
                8.295404560710329944e-01,
                -4.492219258475603216e-01,
                -1.784484611185287450e00,
                -1.901182059718481143e-01,
                -7.537407667483000395e-01,
                -5.384371277650709109e-01,
                -1.490368056268364549e00,
                -3.073744832541754762e-02,
            ],
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        self.file_model_param = Path(CUR_DIR) / "models" / "dpa2.pth"
        self.file_type_embed = Path(CUR_DIR) / "models" / "dpa2_tebd.pth"

    def test_descriptor(self) -> None:
        with open(Path(CUR_DIR) / "models" / "dpa2.json") as fp:
            self.model_json = json.load(fp)
        model_dpa2 = self.model_json
        ntypes = len(model_dpa2["type_map"])
        dparams = model_dpa2["descriptor"]
        dparams["ntypes"] = ntypes
        assert dparams.pop("type") == "dpa2"
        dparams["concat_output_tebd"] = False
        dparams["use_tebd_bias"] = True
        des = DescrptDPA2(
            **dparams,
        ).to(env.DEVICE)
        target_dict = des.state_dict()
        source_dict = torch.load(self.file_model_param, weights_only=True)
        # type_embd of repformer is removed
        source_dict.pop("type_embedding.embedding.embedding_net.layers.0.bias")
        type_embd_dict = torch.load(self.file_type_embed, weights_only=True)
        target_dict = translate_type_embd_dicts_to_dpa2(
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
        des = DescrptDPA2(
            **dparams,
        ).to(env.DEVICE)
        descriptor, env_mat, diff, rot_mat, sw = des(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        self.assertEqual(descriptor.shape[-1], des.get_dim_out())


def translate_type_embd_dicts_to_dpa2(
    target_dict,
    source_dict,
    type_embd_dict,
):
    all_keys = list(target_dict.keys())
    record = [False for ii in all_keys]
    for kk, vv in source_dict.items():
        record[all_keys.index(kk)] = True
        target_dict[kk] = vv
    assert len(type_embd_dict.keys()) == 2
    it = iter(type_embd_dict.keys())
    for _ in range(2):
        kk = next(it)
        tk = "type_embedding." + kk
        record[all_keys.index(tk)] = True
        target_dict[tk] = type_embd_dict[kk]
    record[all_keys.index("repinit.compress_data.0")] = True
    record[all_keys.index("repinit.compress_info.0")] = True
    assert all(record)
    return target_dict
