# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import torch
import torch.export

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.infer import (
    inference,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

from .test_permutation import (
    model_se_e2_a,
)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.rcut = 6.0
        self.rcut_smth = 5.0
        self.sel = [4, 4]
        self.neuron = [10, 10]
        self.axis_neuron = 4

    def test_export_descriptor_se_a(self):
        """Test DescrptSeA descriptor export."""
        for type_one_side in [True, False]:
            model = DescrptSeA(
                rcut=self.rcut,
                rcut_smth=self.rcut_smth,
                sel=self.sel,
                neuron=self.neuron,
                axis_neuron=self.axis_neuron,
                precision="float32",
                trainable=False,
                type_one_side=type_one_side,
            )
            model.eval()

            nf = 2
            nloc = 5
            nnei = sum(self.sel)
            nall = nloc + 10

            coord_ext = torch.randn(nf, nall * 3, device=env.DEVICE)
            atype_ext = torch.randint(
                0, 2, (nf, nall), dtype=torch.int32, device=env.DEVICE
            )
            nlist = torch.randint(
                0, nall, (nf, nloc, nnei), dtype=torch.int32, device=env.DEVICE
            )

            exported = torch.export.export(model, (coord_ext, atype_ext, nlist))
            self.assertIsNotNone(exported)

    def test_export_energy_model_se_a(self):
        """Test EnergyModel with se_e2_a descriptor export."""
        model_params = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [4, 4],
                "rcut_smth": 0.50,
                "rcut": 4.00,
                "neuron": [10, 20],
                "axis_neuron": 4,
            },
            "fitting_net": {
                "neuron": [10, 10],
            },
        }
        model = get_model(model_params).to(env.DEVICE)
        model.eval()

        nf = 1
        nloc = 3
        nall = 10
        nnei = 8

        coord_ext = torch.randn(nf, nall * 3, device=env.DEVICE)
        atype_ext = torch.randint(
            0, 2, (nf, nall), dtype=torch.int32, device=env.DEVICE
        )
        nlist = torch.randint(
            0, nall, (nf, nloc, nnei), dtype=torch.int32, device=env.DEVICE
        )

        class ForwardLowerWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, extended_coord, extended_atype, nlist):
                return self.model.forward_lower(extended_coord, extended_atype, nlist)

        wrapper = ForwardLowerWrapper(model)
        exported = torch.export.export(wrapper, (coord_ext, atype_ext, nlist))
        self.assertIsNotNone(exported)


class ExportIntegrationTest:
    """Integration test base for torch.export.export following JITTest pattern."""

    def test_export(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        tester = inference.Tester("./model.pt")
        model = tester.model
        model.eval()

        # synthesizing dummy inputs
        nf = 1
        nloc = 10
        nall = 20
        descriptor = model.get_descriptor()
        nnei = sum(descriptor.get_sel())

        coord_ext = torch.randn(nf, nall * 3, device=env.DEVICE, dtype=descriptor.prec)
        atype_ext = torch.randint(
            0, descriptor.get_ntypes(), (nf, nall), device=env.DEVICE, dtype=torch.int32
        )
        nlist = torch.randint(
            0, nall, (nf, nloc, nnei), device=env.DEVICE, dtype=torch.int32
        )

        class ForwardLowerWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, extended_coord, extended_atype, nlist):
                return self.model.forward_lower(extended_coord, extended_atype, nlist)

        wrapper = ForwardLowerWrapper(model)
        exported = torch.export.export(wrapper, (coord_ext, atype_ext, nlist))
        self.assertIsNotNone(exported)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
            if f in ["checkpoint"]:
                os.remove(f)


class TestEnergyModelSeAIntegrationExport(unittest.TestCase, ExportIntegrationTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 2
        self.config["training"]["save_freq"] = 2

    def tearDown(self) -> None:
        ExportIntegrationTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
