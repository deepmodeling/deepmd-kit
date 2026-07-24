# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ZBL bridging as COMPOSITION (review 3638077323, redesigned).

``bridging_method: ZBL`` builds a linear composition
(``LinearEnergyModel`` over ``[learned, InterPotentialAtomicModel]`` with
``weights="sum"``); eager values still match pt's flag-architected
``SeZMModel`` bit-for-bit (identical math), pinned here as a value
regression together with FD force, export/DeepEval e2e, training smoke,
and the single-rank with-comm gate.
"""

import copy
import json
import os
import zipfile

import numpy as np
import pytest
import torch

from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt_expt.model.dp_linear_model import (
    LinearEnergyModel,
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


def _analytic_zbl_total(coord, atype, rcut, type_map=("Ni", "O")) -> float:
    """Independent in-test ZBL reference: direct double loop over pairs."""
    import math

    z_of = {"Ni": 28.0, "O": 8.0}
    zs = [z_of[type_map[t]] for t in atype]
    a_coeff = (0.18175, 0.50986, 0.28022, 0.028171)
    b_coeff = (3.1998, 0.94229, 0.4029, 0.20162)
    total = 0.0
    n = len(atype)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coord[i] - coord[j]))
            if r >= rcut:
                continue
            a = 0.88534 * 0.5291772109 / (zs[i] ** 0.23 + zs[j] ** 0.23)
            phi = sum(
                ak * math.exp(-bk * (r / a))
                for ak, bk in zip(a_coeff, b_coeff, strict=True)
            )
            total += 14.3996 * zs[i] * zs[j] / r * phi
    return total


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
        pt_model = pt_get_model(copy.deepcopy(ZBL_CONFIG)).to(torch.float64)
        # JITTER the reference weights: a fresh DPA4 zero-initializes its
        # residual projections and is architecturally input-independent in
        # those paths, which would make the parity below partially vacuous
        # (see dpa4_fixtures.jitter_zero_arrays).
        from deepmd.pt.model.descriptor.sezm import (
            DescrptSeZM,
        )

        from ...dpa4_fixtures import (
            jitter_zero_arrays,
        )

        jittered = jitter_zero_arrays(
            pt_model.atomic_model.descriptor.serialize(), np.random.default_rng(3)
        )
        pt_model.atomic_model.descriptor = DescrptSeZM.deserialize(jittered).to(
            torch.float64
        )
        self.pt_model = pt_model.eval().to(cpu)
        assert self.pt_model.inter_potential is not None

        pt_expt_model = get_model(copy.deepcopy(ZBL_CONFIG))
        assert type(pt_expt_model) is LinearEnergyModel
        dp_child = pt_expt_model.atomic_model.models[0]
        # weight copy into the LEARNED child: pt DescrptSeZM / fitting
        # serialize to the SAME backend-agnostic dict schema (incl. the
        # InnerClamp radii)
        dp_child.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        dp_child.fitting_net = SeZMEnergyFittingNet.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        self.coord, self.atype, self.box = _close_pair_system(cpu)

    def test_parity_vs_pt_with_zbl(self) -> None:
        """Composition == pt's flag architecture on the same weights (values).

        pt adds the raw ZBL to the fitting energy; the composition sums the
        same two per-atom terms -- identical math, pinned at 1e-12 for
        energy/force/virial.
        """
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out_pte = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        # anti-vacuity: the jittered network must produce nontrivial forces,
        # else the parity would compare zeros with zeros.
        assert out_pte["force"].abs().max().item() > 1e-6
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                out_pt[key], out_pte[key], rtol=1e-12, atol=1e-12, msg=key
            )

    def test_zbl_child_adds_positive_energy(self) -> None:
        """Learned child alone vs the composition: positive ZBL repulsion."""
        from deepmd.pt_expt.model.ener_model import (
            EnergyModel,
        )

        m_dp = (
            EnergyModel(atomic_model_=self.pt_expt_model.atomic_model.models[0])
            .to(torch.device("cpu"))
            .eval()
        )
        e_sum = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)[
            "energy"
        ]
        e_dp = m_dp.forward(self.coord, self.atype, box=self.box)["energy"]
        diff = float((e_sum - e_dp).sum())
        # EXACT analytical check, not just positivity: the composition's
        # extra term must equal the independently computed ZBL sum over all
        # pairs within rcut (gas phase: no box, so a direct double loop is
        # the complete reference).
        e_gas_sum = self.pt_expt_model.forward(self.coord, self.atype)["energy"]
        e_gas_dp = m_dp.forward(self.coord, self.atype)["energy"]
        ref = _analytic_zbl_total(
            self.coord[0].numpy(), self.atype[0].numpy(), rcut=4.0
        )
        np.testing.assert_allclose(
            float((e_gas_sum - e_gas_dp).sum()), ref, rtol=1e-10, atol=1e-10
        )
        assert diff > 1e-3

    def test_force_matches_finite_difference(self) -> None:
        """F = -dE/dx through the shared-edge-leaf summed autograd."""
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

    def test_serialize_roundtrip(self) -> None:
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        data = self.pt_expt_model.serialize()
        assert data["type"] == "linear_ener"
        m2 = BaseModel.deserialize(data).to(torch.device("cpu")).eval()
        assert type(m2) is LinearEnergyModel
        out = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        out2 = m2.forward(self.coord, self.atype, box=self.box)
        torch.testing.assert_close(
            out["energy"], out2["energy"], rtol=1e-12, atol=1e-12
        )

    def test_with_comm_gate_off_for_composition(self) -> None:
        """Compositions never compile a with-comm artifact (single-rank)."""
        from deepmd.pt_expt.utils.serialization import (
            _needs_with_comm_artifact,
        )

        assert (
            _needs_with_comm_artifact(self.pt_expt_model, lower_kind="graph") is False
        )

    def test_pt_bridging_checkpoint_rejected(self) -> None:
        """Reject pt's flag-serialized bridging checkpoints.

        pt serializes bridging as a wrapper flag; our architecture is a
        linear composition with a different dict shape -- fail fast instead
        of a silent wrong conversion.
        """
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        with pytest.raises(NotImplementedError, match="bridging_method"):
            BaseModel.deserialize(self.pt_model.serialize())


def test_native_spin_with_bridging_fails_fast() -> None:
    """Native spin + bridging is a follow-up: the builder must not silently
    drop the analytical term.
    """
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg["spin"] = {"use_spin": [True, False], "scheme": "native"}
    with pytest.raises(NotImplementedError, match="native spin"):
        get_model(cfg)


def test_bridging_radii_defaults() -> None:
    """bridging_r_inner/r_outer default to 0.5/0.8 on the learned child."""
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg.pop("bridging_r_inner")
    cfg.pop("bridging_r_outer")
    model = get_model(cfg)
    ic = model.atomic_model.models[0].descriptor.inner_clamp
    assert ic is not None
    assert float(ic.r_inner) == 0.5
    assert float(ic.r_outer) == 0.8


class TestZBLBridgingExportAndTraining:
    """Graph .pt2 freeze + DeepEval parity and a trainer smoke."""

    def test_graph_freeze_and_deep_eval_parity(self, tmp_path) -> None:
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

        with zipfile.ZipFile(model_file) as z:
            md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))
        # single-rank contract: compositions never get a with-comm artifact
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

        config = {
            "model": copy.deepcopy(ZBL_CONFIG),
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
            assert type(model) is LinearEnergyModel
            tasks = trainer._make_training_tasks()
            task = trainer.select_task(tasks)
            for step in range(2):
                result = trainer.train_step(task, step)
                loss = result.payload["loss"]
                assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"
        finally:
            os.chdir(old_cwd)
