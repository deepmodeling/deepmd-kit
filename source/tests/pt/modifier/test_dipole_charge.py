# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.pt.entrypoints.main import (
    freeze,
    get_trainer,
)
from deepmd.pt.modifier import DipoleChargeModifier as PTDipoleChargeModifier
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.tf.modifier import DipoleChargeModifier as TFDipoleChargeModifier

from ...seed import (
    GLOBAL_SEED,
)


def ref_data():
    all_box = np.load(str(Path(__file__).parent / "water/data/data_0/set.000/box.npy"))
    all_coord = np.load(
        str(Path(__file__).parent / "water/data/data_0/set.000/coord.npy")
    )
    nframe = len(all_box)
    rng = np.random.default_rng(GLOBAL_SEED)
    selected_id = rng.integers(nframe)

    coord = all_coord[selected_id].reshape(1, -1)
    box = all_box[selected_id].reshape(1, -1)
    atype = np.loadtxt(
        str(Path(__file__).parent / "water/data/data_0/type.raw"),
        dtype=int,
    ).reshape(1, -1)
    return coord, box, atype


class TestDipoleChargeModifier(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir.name)
        # setup parameter
        # numerical consistency can only be achieved with high prec
        self.ewald_h = 0.1
        self.ewald_beta = 0.5
        self.model_charge_map = [-8.0]
        self.sys_charge_map = [6.0, 1.0]
        self.descriptor_dict = {
            "type": "se_e2_a",
            "sel": [12, 24],
            "rcut_smth": 0.5,
            "rcut": 4.00,
            "neuron": [6, 12, 24],
        }

        # Train DW model
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json, encoding="utf-8") as f:
            config = json.load(f)
        config["model"]["descriptor"].update(self.descriptor_dict)
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water_tensor/dipole/O78H156"),
        ]
        config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water_tensor/dipole/O96H192")
        ]

        trainer = get_trainer(config)
        trainer.run()
        freeze(
            model="model.ckpt.pt",
            output="dw_model.pth",
            head=None,
        )
        # Convert pb model to pth model
        convert_backend(INPUT="dw_model.pth", OUTPUT="dw_model.pb")

        self.dm_pt = PTDipoleChargeModifier(
            "dw_model.pth",
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )
        self.dm_tf = TFDipoleChargeModifier(
            "dw_model.pb",
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )

    def test_jit(self):
        torch.jit.script(self.dm_pt)

    def test_consistency(self):
        coord, box, atype = ref_data()
        # consistent with the input shape from BaseModifier.modify_data
        t_coord = (
            to_torch_tensor(coord).to(env.GLOBAL_PT_FLOAT_PRECISION).reshape(1, -1, 3)
        )
        t_box = to_torch_tensor(box).to(env.GLOBAL_PT_FLOAT_PRECISION).reshape(1, 3, 3)
        t_atype = to_torch_tensor(atype).to(torch.long).reshape(1, -1)

        pt_data = self.dm_pt(
            coord=t_coord,
            atype=t_atype,
            box=t_box,
        )
        tf_data = {}
        e, f, v = self.dm_tf.eval(
            coord=coord,
            box=box,
            atype=atype.reshape(-1),
        )
        tf_data["energy"] = e
        tf_data["force"] = f
        tf_data["virial"] = v

        for kw in ["energy", "virial"]:
            np.testing.assert_allclose(
                to_numpy_array(pt_data[kw]).reshape(-1),
                tf_data[kw].reshape(-1),
                atol=1e-6,
            )
        kw = "force"
        np.testing.assert_allclose(
            to_numpy_array(pt_data[kw]).reshape(-1),
            tf_data[kw].reshape(-1),
            rtol=1e-6,
        )

    def test_serialize(self):
        """Test the serialize method of DipoleChargeModifier."""
        coord, box, atype = ref_data()
        # consistent with the input shape from BaseModifier.modify_data
        t_coord = (
            to_torch_tensor(coord).to(env.GLOBAL_PT_FLOAT_PRECISION).reshape(1, -1, 3)
        )
        t_box = to_torch_tensor(box).to(env.GLOBAL_PT_FLOAT_PRECISION).reshape(1, 3, 3)
        t_atype = to_torch_tensor(atype).to(torch.long).reshape(1, -1)

        dm0 = self.dm_pt.to(env.DEVICE)
        dm1 = PTDipoleChargeModifier.deserialize(dm0.serialize()).to(env.DEVICE)

        ret0 = dm0(
            coord=t_coord,
            atype=t_atype,
            box=t_box,
        )
        ret1 = dm1(
            coord=t_coord,
            atype=t_atype,
            box=t_box,
        )

        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]), to_numpy_array(ret1["energy"])
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["force"]), to_numpy_array(ret1["force"])
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["virial"]), to_numpy_array(ret1["virial"])
        )

    def test_box_none_error(self):
        """Test that a RuntimeError is raised when box is None."""
        coord, _b, atype = ref_data()
        # consistent with the input shape from BaseModifier.modify_data
        t_coord = (
            to_torch_tensor(coord).to(env.GLOBAL_PT_FLOAT_PRECISION).reshape(1, -1, 3)
        )
        t_atype = to_torch_tensor(atype).to(torch.long).reshape(1, -1)

        with self.assertRaises(RuntimeError) as context:
            self.dm_pt(
                coord=t_coord,
                atype=t_atype,
                box=None,  # Pass None to trigger the error
            )

        self.assertIn(
            "dipole_charge data modifier can only be applied for periodic systems",
            str(context.exception),
        )

    def test_train(self):
        input_json = str(Path(__file__).parent / "water/se_e2_a.json")
        with open(input_json, encoding="utf-8") as f:
            config = json.load(f)
        config["model"]["descriptor"].update(self.descriptor_dict)
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/data_0"),
            str(Path(__file__).parent / "water/data/data_1"),
        ]
        config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single"),
        ]
        config["training"]["numb_steps"] = 1

        trainer = get_trainer(config)
        trainer.run()

    def tearDown(self) -> None:
        os.chdir(self.orig_dir)
        self.test_dir.cleanup()
