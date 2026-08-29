# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ZBL bridging as COMPOSITION (review 3638077323, redesigned).

``bridging_method: ZBL`` builds a linear composition
(``LinearEnergyModel`` over ``[learned, InnerPotentialAtomicModel]`` with
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
        # the flat wire type is "linear" -- the SAME string pt/tf write, so a
        # composition round-trips across backends
        assert data["type"] == "linear"
        m2 = BaseModel.deserialize(data).to(torch.device("cpu")).eval()
        assert type(m2) is LinearEnergyModel
        out = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        out2 = m2.forward(self.coord, self.atype, box=self.box)
        torch.testing.assert_close(
            out["energy"], out2["energy"], rtol=1e-12, atol=1e-12
        )

    def test_with_comm_gate_on_for_composition(self) -> None:
        """The SFPG exchange makes bridged compositions multi-rank
        (issue #5906 Task 2): the graph with-comm artifact is compiled.
        """
        from deepmd.pt_expt.utils.serialization import (
            _needs_with_comm_artifact,
        )

        assert _needs_with_comm_artifact(self.pt_expt_model, lower_kind="graph") is True

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


def _native_spin_zbl_config() -> dict:
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg["spin"] = {"use_spin": [True, False], "scheme": "native"}
    return cfg


def _spin_system():
    """6 atoms (3 Ni spin-active, 3 O) with one close pair driving the ZBL."""
    coord = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [1.9, 1.2, 1.1],  # close to atom 0 -> nontrivial ZBL
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [3.5, 1.0, 2.0],
                [2.0, 2.0, 3.0],
            ]
        ],
        dtype=torch.float64,
    )
    atype = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
    spin = 0.1 * torch.ones_like(coord)
    box = (6.0 * torch.eye(3, dtype=torch.float64)).reshape(1, 9)
    return coord, atype, spin, box


class TestNativeSpinWithBridging:
    """Native spin + analytical bridging compose (review 3649276109).

    ``get_sezm_model`` OWNS the bridging composition (``get_standard_model``
    rejects ``bridging_method``: a composition is not expressible on a
    non-composite model type). ``get_native_spin_model`` routes DPA4/SeZM
    configs there and then RE-CLASSES the returned composition -- so the two
    features combine with no special case: the learned child consumes
    ``spin``, the analytical child accepts and ignores it.
    """

    def test_construction_composes_and_keeps_spin(self) -> None:
        from deepmd.pt_expt.model.native_spin_model import (
            NativeSpinEnergyModel,
        )

        model = get_model(_native_spin_zbl_config())
        assert isinstance(model, NativeSpinEnergyModel)
        assert model.has_spin() is True
        kinds = [type(c).__name__ for c in model.atomic_model.models]
        assert kinds[1] == "InnerPotentialAtomicModel", kinds
        # bridging radii still reach the LEARNED child's descriptor
        assert float(model.atomic_model.models[0].descriptor.inner_clamp.r_inner) == 0.8

    def test_forward_energy_force_force_mag(self) -> None:
        model = get_model(_native_spin_zbl_config()).to(torch.device("cpu")).eval()
        coord, atype, spin, box = _spin_system()
        out = model(coord, atype, spin, box=box)
        for key in ("energy", "force", "force_mag"):
            assert torch.isfinite(out[key]).all(), key
        # mask_mag follows use_spin=[True, False] on atype [0,0,0,1,1,1]
        assert out["mask_mag"].reshape(-1).tolist() == [
            True,
            True,
            True,
            False,
            False,
            False,
        ]
        # anti-vacuity: the analytical term must actually contribute.  The
        # learned child alone is the same model minus the ZBL energy.
        from deepmd.pt_expt.model.native_spin_model import (
            NativeSpinEnergyModel,
        )

        learned_only = (
            NativeSpinEnergyModel(
                atomic_model_=model.atomic_model.models[0], spin=model.spin
            )
            .to(torch.device("cpu"))
            .eval()
        )
        # Gas phase (no box) for the EXACT check: the in-test reference is a
        # direct double loop over pairs, which has no periodic images.
        e_gas = model(coord, atype, spin)["energy"]
        e_gas_learned = learned_only(coord, atype, spin)["energy"]
        zbl_contrib = float((e_gas - e_gas_learned).sum())
        ref = _analytic_zbl_total(coord[0].numpy(), atype[0].numpy(), rcut=4.0)
        np.testing.assert_allclose(zbl_contrib, ref, rtol=1e-10, atol=1e-10)
        assert zbl_contrib > 1e-3

    def test_serialize_roundtrip(self) -> None:
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        model = get_model(_native_spin_zbl_config()).to(torch.device("cpu")).eval()
        data = model.serialize()
        assert data["type"] == "native_spin"
        restored = BaseModel.deserialize(data).to(torch.device("cpu")).eval()
        coord, atype, spin, box = _spin_system()
        out = model(coord, atype, spin, box=box)
        out2 = restored(coord, atype, spin, box=box)
        for key in ("energy", "force", "force_mag"):
            torch.testing.assert_close(
                out[key], out2[key], rtol=1e-12, atol=1e-12, msg=key
            )


def test_native_spin_with_bridging_dpmodel() -> None:
    """Dpmodel twin: same composition, energy-only (no autograd there)."""
    from deepmd.dpmodel.model.model import get_model as dp_get_model
    from deepmd.dpmodel.model.native_spin_model import (
        NativeSpinEnergyModel as NativeSpinEnergyModelDP,
    )

    cfg = _native_spin_zbl_config()
    cfg.pop("type")  # generic builder; the dpa4 alias routes the same way
    model = dp_get_model(cfg)
    assert isinstance(model, NativeSpinEnergyModelDP)
    kinds = [type(c).__name__ for c in model.atomic_model.models]
    assert kinds[1] == "InnerPotentialAtomicModel", kinds

    coord, atype, spin, box = _spin_system()
    out = model.call(coord.numpy(), atype.numpy(), spin.numpy(), box=box.numpy())
    assert np.all(np.isfinite(out["energy"]))
    assert out["mask_mag"].reshape(-1).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


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
        # multi-rank contract (issue #5906): the SFPG partials are completed
        # across ranks, so a bridged composition DOES get a with-comm twin
        assert md["has_comm_artifact"] is True
        with zipfile.ZipFile(model_file) as z:
            assert "model/extra/forward_lower_with_comm.pt2" in z.namelist()

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


class TestInnerPotentialChangeTypeMapPtExpt:
    """pt_expt twin of the dpmodel ``change_type_map`` regression.

    Exercised through the REAL composition: ``LinearEnergyModel`` ->
    ``LinearEnergyAtomicModel`` -> ``InnerPotentialAtomicModel`` ->
    ``InnerPotential``.  Inside a pt_expt module tree the element lookup is a
    wrapped torch buffer, so the rebuild must land on the same
    device/namespace (review 3649295675) -- a numpy rebuild would desync the
    buffer or fail outright on CUDA.
    """

    @staticmethod
    def _zbl_child(model):
        return model.atomic_model.models[1]

    @staticmethod
    def _pair_energy(zbl_child, atype_value: int = 0, r: float = 1.0) -> float:
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
        )
        from deepmd.pt_expt.utils import env as _env

        graph = NeighborGraph(
            n_node=torch.tensor([2], dtype=torch.int64, device=_env.DEVICE),
            edge_index=torch.tensor(
                [[0, 1], [1, 0]], dtype=torch.int64, device=_env.DEVICE
            ),
            edge_vec=torch.tensor(
                [[r, 0.0, 0.0], [-r, 0.0, 0.0]],
                dtype=torch.float64,
                device=_env.DEVICE,
            ),
            edge_mask=torch.ones(2, dtype=torch.bool, device=_env.DEVICE),
        )
        atype = torch.full((2,), atype_value, dtype=torch.int64, device=_env.DEVICE)
        return float(
            zbl_child.forward_common_atomic_graph(graph, atype)["energy"].sum()
        )

    def _build(self, type_map):
        from deepmd.pt_expt.utils import env as _env

        config = copy.deepcopy(ZBL_CONFIG)
        config["type_map"] = list(type_map)
        return get_model(config).to(_env.DEVICE).eval()

    def test_lookup_is_a_wrapped_buffer(self) -> None:
        """Precondition: inside pt_expt the lookup is a torch buffer."""
        from deepmd.pt_expt.utils import env as _env

        z = self._zbl_child(self._build(["Ni", "O"])).potential.atomic_numbers
        assert isinstance(z, torch.Tensor), (
            "the pt_expt wrapper no longer converts the lookup to a tensor; "
            "this test would stop covering the device-safe rebuild"
        )
        assert z.device.type == torch.device(_env.DEVICE).type

    def test_reorder_matches_a_freshly_built_model(self) -> None:
        # NOTE: applied to the ZBL CHILD, not the whole composition -- the
        # DPA4/SeZM learned child does not implement change_type_map at all
        # ("change_type_map is not supported for SeZM"), a separate pre-existing
        # limitation.  The lookup under test belongs to this child.
        child = self._zbl_child(self._build(["Ni", "O"]))
        e_nini = self._pair_energy(child)
        child.change_type_map(["O", "Ni"])
        fresh = self._zbl_child(self._build(["O", "Ni"]))
        e_fresh = self._pair_energy(fresh)
        # anti-vacuity: Ni-Ni and O-O must be far apart, else a stale lookup
        # would be indistinguishable from a rebuilt one
        assert abs(e_fresh - e_nini) > 1.0
        np.testing.assert_allclose(self._pair_energy(child), e_fresh, rtol=1e-12)
        assert [float(v) for v in child.potential.atomic_numbers] == [8.0, 28.0]

    def test_added_element_extends_the_lookup_on_device(self) -> None:
        from deepmd.pt_expt.utils import env as _env

        child = self._zbl_child(self._build(["Ni", "O"]))
        child.change_type_map(["Ni", "O", "H"])
        z = child.potential.atomic_numbers
        assert isinstance(z, torch.Tensor), "the rebuild dropped out of torch"
        assert z.device.type == torch.device(_env.DEVICE).type
        assert [float(v) for v in z] == [28.0, 8.0, 1.0]
        # the new type is addressable -- a stale (length-2) table raises here
        np.testing.assert_allclose(
            self._pair_energy(child, atype_value=2),
            self._pair_energy(self._zbl_child(self._build(["H"]))),
            rtol=1e-12,
        )

    def test_serialize_roundtrip_after_change_type_map(self) -> None:
        """Checkpoint continuity: the restored child must predict the same.

        Serialization records the NEW public map, so a stale in-memory lookup
        and its deserialized twin disagree -- the restart-time symptom.
        """
        from deepmd.dpmodel.atomic_model.base_atomic_model import (
            BaseAtomicModel,
        )

        child = self._zbl_child(self._build(["Ni", "O"]))
        child.change_type_map(["O", "Ni"])
        data = child.serialize()
        # The wire type must still resolve through the shared registry ...
        assert BaseAtomicModel.get_class_by_type(data["type"]) is not None
        # ... but restore through the pt_expt class the child actually is:
        # the dpmodel class is NumPy-backed, and this test feeds device
        # tensors, so a CUDA run would index a NumPy out_bias with a CUDA
        # atype and fail. Same-class restore is also the stricter check.
        restored = type(child).deserialize(data)
        assert restored.get_type_map() == ["O", "Ni"]
        np.testing.assert_allclose(
            self._pair_energy(restored), self._pair_energy(child), rtol=1e-12
        )


def test_native_spin_with_bridging_graph_freeze_and_deep_eval(tmp_path) -> None:
    """Native spin + ZBL freezes to a graph .pt2 and evaluates in parity.

    Both features are graph-route-only, so their combination must survive the
    export seam too, not just eager construction (review 3649276109).
    """
    if os.environ.get("CI") == "true":
        pytest.skip("AOTInductor compile is slow (minutes); local/fixture-gen only.")
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    cpu = torch.device("cpu")
    model = get_model(_native_spin_zbl_config()).to(cpu).eval()
    coord, atype, spin, box = _spin_system()
    ref = model(coord, atype, spin, box=box)

    model_file = tmp_path / "dpa4_native_spin_zbl_graph.pt2"
    # native spin has no dense lower at all, so the graph kind is the only
    # valid one here; since issue #5906 the graph lower additionally carries
    # a with-comm twin (only the dense lower stays single-rank for spin).
    deserialize_to_file(
        str(model_file), {"model": model.serialize()}, lower_kind="graph"
    )
    with zipfile.ZipFile(model_file) as z:
        md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))
        names = z.namelist()
    assert md["is_spin"] is True
    assert md["has_comm_artifact"] is True
    assert "model/extra/forward_lower_with_comm.pt2" in names
    assert md["use_spin"] == [True, False]

    dp = DeepPot(str(model_file))
    assert dp.has_spin
    e, f, v, fm, mm = dp.eval(
        coord.numpy(),
        box.numpy(),
        atype.reshape(-1).numpy(),
        atomic=False,
        spin=spin.numpy(),
    )[:5]
    for got, want, name in (
        (e, ref["energy"], "energy"),
        (f, ref["force"], "force"),
        (fm, ref["force_mag"], "force_mag"),
    ):
        np.testing.assert_allclose(
            np.asarray(got).reshape(-1),
            want.detach().numpy().reshape(-1),
            rtol=1e-10,
            atol=1e-10,
            err_msg=name,
        )


def test_bridged_metadata_carries_charge_spin_dim(tmp_path) -> None:
    """The FROZEN metadata must declare charge_spin for a bridged model.

    This is the consequence the eager forward hides: the learned child
    consumes ``charge_spin`` either way, but the freeze reads the MODEL's
    ``get_dim_chg_spin()``.  While the composition failed to forward it,
    ``dim_chg_spin`` was 0 in metadata -- so the exported ABI had no
    charge_spin slot and the C++ feeder never supplied one, making the
    artifact disagree with its own eager model.  Metadata-only, so no
    inductor compile is needed.
    """
    from deepmd.pt_expt.utils.serialization import (
        _collect_metadata,
    )

    config = copy.deepcopy(ZBL_CONFIG)
    config["descriptor"]["add_chg_spin_ebd"] = True
    model = get_model(config).to(torch.device("cpu")).eval()
    meta = _collect_metadata(model, lower_kind="graph")
    assert meta["dim_chg_spin"] > 0, (
        "the bridged model's metadata dropped charge_spin; the exported "
        "artifact would silently ignore the FiLM conditioning"
    )

    plain = copy.deepcopy(ZBL_CONFIG)
    plain["descriptor"]["add_chg_spin_ebd"] = True
    for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
        plain.pop(key, None)
    plain_model = get_model(plain).to(torch.device("cpu")).eval()
    plain_meta = _collect_metadata(plain_model, lower_kind="graph")
    assert meta["dim_chg_spin"] == plain_meta["dim_chg_spin"]


def test_pair_exclusion_suppresses_the_analytical_term() -> None:
    """Exclusion must remove the analytical term too, not just the learned one.

    Model-level ``pair_exclude_types`` is a neighbor-graph BUILD transform,
    and both children of the bridged composition read the same graph. So an
    excluded pair type must contribute neither a learned nor a ZBL term.

    This is not a free-standing preference: the composition is what drives
    the build (and what the freeze metadata reads). Built without the
    exclusion forwarded, the graph kept the excluded pairs and ZBL went on
    interacting through them -- a large, silent error, since ZBL dominates at
    short range.

    Fixture: an Ni-O dimer at 0.9 A, where (0, 1) is the ONLY pair present,
    so excluding it must remove the analytical term entirely.
    """
    cpu = torch.device("cpu")
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64, device=cpu
    ).reshape(1, -1)
    atype = torch.tensor([[0, 1]], dtype=torch.int64, device=cpu)
    box = (torch.eye(3, dtype=torch.float64, device=cpu) * 20.0).reshape(1, 9)

    def energy(*, bridged: bool, excluded: bool) -> float:
        config = copy.deepcopy(ZBL_CONFIG)
        if not bridged:
            for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
                config.pop(key, None)
        if excluded:
            config["pair_exclude_types"] = [[0, 1]]
        model = get_model(config).to(torch.device("cpu")).eval()
        out = model(coord, atype, box=box)["energy"]
        return float(out.detach().cpu().numpy().reshape(-1)[0])

    # anti-vacuity: without exclusion the analytical term must dominate,
    # otherwise the suppression assertion below would hold trivially.
    zbl_contribution = energy(bridged=True, excluded=False) - energy(
        bridged=False, excluded=False
    )
    assert zbl_contribution > 1.0, (
        f"ZBL contributes only {zbl_contribution} at 0.9 A; the fixture no "
        "longer exercises the analytical term"
    )

    # ... and with the pair type excluded, the bridged model must fall back
    # EXACTLY onto the unbridged one: no ZBL, not merely less ZBL.
    assert energy(bridged=True, excluded=True) == energy(
        bridged=False, excluded=True
    ), "the analytical ZBL term survived a pair-type exclusion"
