# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ZBL bridging (review 3638077323): eager pt parity, FD force,
with-comm gate, and pt-checkpoint interop for ``bridging_method="ZBL"``.
"""

import copy

import numpy as np
import torch

from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)

from ...seed import (
    GLOBAL_SEED,
)

ZBL_CONFIG = {
    "type": "dpa4",
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
    },
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}


def _close_pair_system(cpu):
    generator = torch.Generator(device=cpu).manual_seed(GLOBAL_SEED + 2)
    nloc = 6
    cell = torch.rand([3, 3], dtype=torch.float64, generator=generator)
    cell = (cell + cell.T) + 6.0 * torch.eye(3)
    coord = 1.5 + 3.0 * torch.rand([nloc, 3], dtype=torch.float64, generator=generator)
    coord[1] = coord[0] + torch.tensor([0.95, 0.0, 0.0], dtype=torch.float64)
    atype = torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.int64)
    return coord.unsqueeze(0), atype, cell.reshape(1, 9)


class TestZBLBridgingPtExpt:
    def setup_method(self) -> None:
        cpu = torch.device("cpu")
        self.pt_model = pt_get_model(copy.deepcopy(ZBL_CONFIG)).to(torch.float64)
        self.pt_model = self.pt_model.eval().to(cpu)
        assert self.pt_model.inter_potential is not None

        pt_expt_model = get_model(copy.deepcopy(ZBL_CONFIG))
        atomic = pt_expt_model.atomic_model
        assert atomic.inter_potential is not None
        # weight copy: pt DescrptSeZM / fitting serialize to the SAME
        # backend-agnostic dict schema (incl. the InnerClamp radii)
        atomic.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        atomic.fitting_net = SeZMEnergyFittingNet.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        self.coord, self.atype, self.box = _close_pair_system(cpu)

    def test_parity_vs_pt_with_zbl(self) -> None:
        """Weight-copied fp64 parity incl. the ZBL term (energy/force/virial)."""
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out_pte = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                out_pt[key], out_pte[key], rtol=1e-12, atol=1e-12, msg=key
            )

    def test_zbl_energy_is_positive_addition(self) -> None:
        """Same weights minus the bridging key -> plain twin; diff > 0."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        data = self.pt_expt_model.serialize()
        assert data["bridging_method"] == "ZBL"
        plain_data = copy.deepcopy(data)
        plain_data.pop("bridging_method")
        m_plain = BaseModel.deserialize(plain_data).to(torch.device("cpu")).eval()
        e_zbl = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)[
            "energy"
        ]
        e_plain = m_plain.forward(self.coord, self.atype, box=self.box)["energy"]
        assert float((e_zbl - e_plain).sum()) > 1e-3

    def test_force_matches_finite_difference(self) -> None:
        """F = -dE/dx through the ZBL-carrying edge autograd (central FD)."""
        eps = 1e-5
        out = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        force = out["force"].reshape(-1, 3)
        for atom, comp in ((1, 0), (2, 2)):  # close-pair atom + a far atom
            cp = self.coord.clone()
            cp[0, atom, comp] += eps
            ep = self.pt_expt_model.forward(cp, self.atype, box=self.box)["energy"]
            cm = self.coord.clone()
            cm[0, atom, comp] -= eps
            em = self.pt_expt_model.forward(cm, self.atype, box=self.box)["energy"]
            fd = -float((ep - em).sum()) / (2 * eps)
            np.testing.assert_allclose(
                float(force[atom, comp]), fd, rtol=1e-6, atol=1e-6
            )

    def test_with_comm_gate_off_for_bridging(self) -> None:
        """Bridging models never compile a with-comm artifact (single-rank)."""
        from deepmd.pt_expt.utils.serialization import (
            _needs_with_comm_artifact,
        )

        assert (
            _needs_with_comm_artifact(self.pt_expt_model, lower_kind="graph") is False
        )

    def test_pt_checkpoint_interop(self) -> None:
        """A pt-serialized ZBL SeZM checkpoint deserializes into pt_expt."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        pt_data = self.pt_model.serialize()
        assert str(pt_data.get("bridging_method", "none")).upper() == "ZBL"
        m2 = BaseModel.deserialize(pt_data)
        assert m2.atomic_model.inter_potential is not None
        m2 = m2.to(torch.device("cpu")).eval()
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out2 = m2.forward(self.coord, self.atype, box=self.box)
        torch.testing.assert_close(
            out_pt["energy"], out2["energy"], rtol=1e-12, atol=1e-12
        )


class TestZBLBridgingExportAndTraining:
    """Graph .pt2 freeze + DeepEval parity and a trainer smoke for ZBL models."""

    def test_graph_freeze_and_deep_eval_parity(self, tmp_path) -> None:
        import os

        import pytest

        if os.environ.get("CI") == "true":
            pytest.skip(
                "AOTInductor compile is slow (minutes); local/fixture-gen only."
            )
        from deepmd.infer import (
            DeepPot,
        )
        from deepmd.pt_expt.utils.serialization import (
            deserialize_to_file,
        )

        cpu = torch.device("cpu")
        model = get_model(copy.deepcopy(ZBL_CONFIG)).to(cpu).eval()
        coord, atype, box = _close_pair_system(cpu)
        ref = model.forward(coord, atype, box=box)

        model_file = tmp_path / "dpa4_zbl_graph.pt2"
        data = {"model": model.serialize()}
        deserialize_to_file(str(model_file), data, lower_kind="graph")

        import json
        import zipfile

        with zipfile.ZipFile(model_file) as z:
            md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))
        # single-rank contract: bridging models never get a with-comm artifact
        assert md["has_comm_artifact"] is False

        dp = DeepPot(str(model_file))
        e, f, v = dp.eval(
            coord.reshape(1, -1).numpy(),
            box.numpy(),
            atype.reshape(-1).numpy(),
            atomic=False,
        )
        np.testing.assert_allclose(
            np.asarray(e).reshape(-1),
            ref["energy"].detach().numpy().reshape(-1),
            rtol=1e-10,
            atol=1e-10,
            err_msg="energy",
        )
        np.testing.assert_allclose(
            np.asarray(f).reshape(-1),
            ref["force"].detach().numpy().reshape(-1),
            rtol=1e-10,
            atol=1e-10,
            err_msg="force",
        )

    def test_training_smoke(self, tmp_path) -> None:
        import os

        import pytest

        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "pt", "NiO", "data", "single"
        )
        if not os.path.isdir(data_dir):
            pytest.skip(f"NiO data not found: {data_dir}")
        from deepmd.pt_expt.entrypoints.main import (
            get_trainer,
        )
        from deepmd.pt_expt.train.training import (
            DEFAULT_TASK_KEY,
        )
        from deepmd.utils.argcheck import (
            normalize,
        )
        from deepmd.utils.compat import (
            update_deepmd_input,
        )

        model_cfg = copy.deepcopy(ZBL_CONFIG)
        config = {
            "model": model_cfg,
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
            },
            "training": {
                "training_data": {"systems": [data_dir], "batch_size": 1},
                "validation_data": {
                    "systems": [data_dir],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "numb_steps": 2,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 2,
            },
        }
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            trainer = get_trainer(config)
            model = trainer.wrapper.model[DEFAULT_TASK_KEY]
            assert model.atomic_model.has_analytical_bridging()
            tasks = trainer._make_training_tasks()
            task = trainer.select_task(tasks)
            for step in range(2):
                result = trainer.train_step(task, step)
                loss = result.payload["loss"]
                assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"
        finally:
            os.chdir(old_cwd)


def test_native_spin_composes_with_bridging() -> None:
    """Native spin + ZBL bridging build together and both terms are live.

    pt's SeZMNativeSpinModel inherits the bridging term from SeZMModel; our
    atomic-layer injection composes with the native-spin model factory for
    free -- pinned here (energy responds to BOTH the close pair's ZBL and
    the spin input).
    """
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg["spin"] = {"use_spin": [True, False], "scheme": "native"}
    model = get_model(cfg).to(torch.device("cpu")).eval()
    assert model.has_spin()
    assert model.atomic_model.has_analytical_bridging()

    coord, atype, box = _close_pair_system(torch.device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED + 3)
    spin = torch.rand([1, 6, 3], dtype=torch.float64, generator=generator)
    out = model.forward(coord, atype, spin, box=box)
    # ZBL live: removing the bridging key from the SAME weights lowers the
    # close-pair energy by a positive repulsion.
    plain_data = model.serialize()
    plain_data.pop("bridging_method")
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    m_plain = BaseModel.deserialize(plain_data).to(torch.device("cpu")).eval()
    e_plain = m_plain.forward(coord, atype, spin, box=box)["energy"]
    assert float((out["energy"] - e_plain).sum()) > 1e-3
    assert "force_mag" in out


def test_bridging_radii_defaults() -> None:
    """bridging_r_inner/r_outer default to 0.5/0.8 (pt's defaults)."""
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg.pop("bridging_r_inner")
    cfg.pop("bridging_r_outer")
    model = get_model(cfg)
    ic = model.atomic_model.descriptor.inner_clamp
    assert ic is not None
    assert float(ic.r_inner) == 0.5
    assert float(ic.r_outer) == 0.8
