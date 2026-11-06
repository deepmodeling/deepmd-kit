# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ZBL bridging as a linear composition.

``bridging_method: ZBL`` builds a ``LinearEnergyModel`` over
``[learned, InnerPotentialAtomicModel]`` with ``weights="sum"``. The analytical
term is the ZBL series truncated at ``bridging_truncation`` times each pair's
covalent bond length, and the learned part reads every pair separation through
the contact clamp of its descriptor: below the data edge ``e`` of a pair the
separation is frozen, so a pair that closes further changes the learned energy
by nothing and the analytical term alone carries the repulsion. Eager values
match pt's flag-architected ``SeZMModel`` on the same weights, and are pinned
here together with the composition of the two terms, the clamp, force against
finite differences, the state and record guards, the recording of the training
contacts, native spin, the export and DeepEval path, and the training smoke.
"""

import contextlib
import copy
import json
import math
import os
import zipfile
from collections.abc import (
    Iterator,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.contact_clamp import (
    ContactClamp,
)
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
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.utils.bridging import (
    DEFAULT_CLAMP,
    DEFAULT_TRUNCATION,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)
from ...seed import (
    GLOBAL_SEED,
)

TYPE_MAP = ["Ni", "O"]
CHARGE = [28.0, 8.0]
RCUT = 4.0
TRUNCATION = 1.05
CLAMP = 0.7
ZBL_CONFIG = {
    "type": "dpa4",
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa4",
        "rcut": RCUT,
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
    "bridging_truncation": TRUNCATION,
    "bridging_clamp": CLAMP,
}
#: Covalent bond length in Å of every pair of types, the end radius of the
#: analytical term before the truncation factor.
BOND = (
    covalent_radii_from_type_map(TYPE_MAP)[:, None]
    + covalent_radii_from_type_map(TYPE_MAP)[None, :]
)
#: Data edges in Å of (Ni-Ni, Ni-O, O-O) with the fixed edge of the padding
#: type. They sit away from the covalent start, so a model that ignored the
#: recorded edges would read different separations.
EDGES = np.array([[1.0, 1.9, 1.0], [1.9, 1.5, 1.0], [1.0, 1.0, 1.0]])


def _zbl_series(zi: float, zj: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes in eV Å and decay rates in Å⁻¹ of the universal ZBL series."""
    a = 0.88534 * 0.5291772109 / (zi**0.23 + zj**0.23)
    amplitude = 14.3996 * zi * zj * np.array([0.18175, 0.50986, 0.28022, 0.028171])
    return amplitude, np.array([3.1998, 0.94229, 0.4029, 0.20162]) / a


def _truncated_pair(r: float, type_i: int, type_j: int) -> float:
    """The ZBL energy of one pair minus its cubic Taylor polynomial at the end.

    The reference is independent of the model: the series and its derivatives
    come from the closed forms of the screened Coulomb terms.
    """
    end = TRUNCATION * float(BOND[type_i, type_j])
    if r >= end:
        return 0.0
    amplitude, rate = _zbl_series(CHARGE[type_i], CHARGE[type_j])

    def derivative(x: float, n: int) -> float:
        factor = sum(
            math.factorial(n) / math.factorial(n - k) * rate ** (n - k) / x ** (k + 1)
            for k in range(n + 1)
        )
        return float((-1) ** n * np.sum(amplitude * np.exp(-rate * x) * factor))

    delta = r - end
    return derivative(r, 0) - sum(
        derivative(end, k) * delta**k / math.factorial(k) for k in range(4)
    )


def _analytic_zbl_total(coord: np.ndarray, atype: np.ndarray) -> float:
    """Truncated ZBL energy of an open-boundary frame: a direct loop over pairs."""
    total = 0.0
    for i in range(len(atype)):
        for j in range(i + 1, len(atype)):
            r = float(np.linalg.norm(coord[i] - coord[j]))
            total += _truncated_pair(r, int(atype[i]), int(atype[j]))
    return total


def _close_pair_system(cpu):
    """Six atoms in a periodic cell, with a close Ni-Ni pair among ordinary ones."""
    generator = torch.Generator(device=cpu).manual_seed(GLOBAL_SEED + 2)
    nloc = 6
    cell = torch.rand([3, 3], dtype=torch.float64, generator=generator)
    cell = (cell + cell.T) + 6.0 * torch.eye(3)
    coord = 1.5 + 3.0 * torch.rand([nloc, 3], dtype=torch.float64, generator=generator)
    coord[1] = coord[0] + torch.tensor([0.95, 0.0, 0.0], dtype=torch.float64)
    atype = torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.int64)
    return coord.unsqueeze(0), atype, cell.reshape(1, 9)


def _dimer(separation: float) -> tuple[torch.Tensor, torch.Tensor]:
    """An Ni-O dimer along x in an open box, with shapes (1, 2, 3) and (1, 2)."""
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]], dtype=torch.float64
    )
    return coord, torch.tensor([[0, 1]], dtype=torch.int64)


def _set_edges(clamp: torch.nn.Module, edges: np.ndarray) -> None:
    """Overwrite the recorded data edges of a contact clamp in place."""
    with torch.no_grad():
        clamp.edges.copy_(torch.as_tensor(edges, dtype=clamp.edges.dtype))


@contextlib.contextmanager
def _without_clamp(descriptor: torch.nn.Module) -> Iterator[None]:
    """Evaluate a bridged descriptor as if it had no contact clamp."""
    clamp = descriptor.contact_clamp
    descriptor.contact_clamp = None
    try:
        yield
    finally:
        descriptor.contact_clamp = clamp


def _make_sensitive_model(config: dict | None = None) -> torch.nn.Module:
    """Build a bridged pt_expt model whose learned part depends on geometry.

    A fresh DPA4 zero-initializes its residual projections and is
    architecturally input-independent in those paths, so the weights are
    jittered (see ``dpa4_fixtures.jitter_zero_arrays``); otherwise every
    comparison of learned energies below would compare constants.
    """
    model = get_model(copy.deepcopy(ZBL_CONFIG if config is None else config))
    learned = model.atomic_model.models[0]
    learned.descriptor = DescrptDPA4.deserialize(
        jitter_zero_arrays(learned.descriptor.serialize(), np.random.default_rng(3))
    )
    return model.to(torch.device("cpu")).eval()


class TestZBLBridgingPtExpt:
    def setup_method(self) -> None:
        cpu = torch.device("cpu")
        pt_model = pt_get_model(copy.deepcopy(ZBL_CONFIG)).to(torch.float64)
        # JITTER the reference weights, see ``_make_sensitive_model``.
        from deepmd.pt.model.descriptor.sezm import (
            DescrptSeZM,
        )

        jittered = jitter_zero_arrays(
            pt_model.atomic_model.descriptor.serialize(), np.random.default_rng(3)
        )
        pt_model.atomic_model.descriptor = DescrptSeZM.deserialize(jittered).to(
            torch.float64
        )
        self.pt_model = pt_model.eval().to(cpu)
        assert self.pt_model.inter_potential is not None
        _set_edges(self.pt_model.atomic_model.descriptor.contact_clamp, EDGES)

        pt_expt_model = get_model(copy.deepcopy(ZBL_CONFIG))
        assert type(pt_expt_model) is LinearEnergyModel
        dp_child = pt_expt_model.atomic_model.models[0]
        # weight copy into the LEARNED child: pt DescrptSeZM / fitting
        # serialize to the SAME backend-agnostic dict schema, including the
        # recorded data edges of the contact clamp
        dp_child.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        dp_child.fitting_net = SeZMEnergyFittingNet.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        self.coord, self.atype, self.box = _close_pair_system(cpu)

    def _descriptor(self, backend: str) -> torch.nn.Module:
        """The descriptor of the learned model of either backend."""
        if backend == "pt":
            return self.pt_model.atomic_model.descriptor
        return self.pt_expt_model.atomic_model.models[0].descriptor

    def _learned_energy(
        self, backend: str, coord: torch.Tensor, atype: torch.Tensor
    ) -> float:
        """Energy in eV of the learned model alone in an open box."""
        if backend == "pt":
            potential = self.pt_model.inter_potential
            self.pt_model.inter_potential = None
            try:
                out = self.pt_model(coord, atype, box=None)
            finally:
                self.pt_model.inter_potential = potential
        else:
            learned = EnergyModel(
                atomic_model_=self.pt_expt_model.atomic_model.models[0]
            ).eval()
            out = learned(coord, atype, box=None)
        return float(out["energy"].detach().reshape(-1)[0])

    def test_parity_vs_pt_with_zbl(self) -> None:
        """Composition == pt's flag architecture on the same weights (values).

        pt adds the raw analytical term to the fitting energy; the composition
        sums the same two per-atom terms -- identical math, pinned at 1e-12 for
        energy/force/virial, with the data edges placing the pairs of the
        fixture in the transition and frozen zones of their clamps.
        """
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out_pte = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        # anti-vacuity: the jittered network must produce nontrivial forces,
        # else the parity would compare zeros with zeros ...
        assert out_pte["force"].abs().max().item() > 1e-6
        # ... and the clamp must act on the fixture, else it would not be
        # part of what the two backends agree on.
        with _without_clamp(self._descriptor("pt_expt")):
            bare = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        assert (bare["energy"] - out_pte["energy"]).abs().max().item() > 1e-6
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                out_pt[key], out_pte[key], rtol=1e-12, atol=1e-12, msg=key
            )

    def test_composition_is_learned_plus_truncated_zbl(self) -> None:
        """Composition energy == learned child + truncated ZBL of every pair.

        In an open box the in-test reference is a direct double loop over the
        pairs, which has no periodic images and so is the complete answer. The
        learned child alone carries the same contact clamp, so the difference
        is the analytical term alone.
        """
        learned = EnergyModel(
            atomic_model_=self.pt_expt_model.atomic_model.models[0]
        ).eval()
        e_sum = self.pt_expt_model.forward(self.coord, self.atype)["energy"]
        e_learned = learned.forward(self.coord, self.atype)["energy"]
        ref = _analytic_zbl_total(self.coord[0].numpy(), self.atype[0].numpy())
        assert ref > 1e-3
        np.testing.assert_allclose(
            float((e_sum - e_learned).sum().detach()), ref, rtol=1e-10, atol=1e-10
        )
        # in the periodic cell the term is positive as well
        e_box = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        e_box_learned = learned.forward(self.coord, self.atype, box=self.box)
        assert float((e_box["energy"] - e_box_learned["energy"]).sum().detach()) > 1e-3

    @pytest.mark.parametrize("backend", ["pt", "pt_expt"])
    def test_a_dimer_below_the_freeze_reads_one_separation(self, backend) -> None:
        """The learned energy of a dimer is constant at and below ``a e``.

        The clamp freezes the separation the descriptor reads at ``(1 + a) e / 2``
        for every separation up to ``a e``, so the learned energy of an Ni-O
        dimer is the same for all of them, while it still follows the pair in
        the transition zone between ``a e`` and ``e``. The composition then
        changes only through the analytical term, whose force along the bond is
        the whole force on the pair there. The unclamped learned model reads
        every one of these separations and varies.
        """
        edge = float(EDGES[0, 1])
        freeze = CLAMP * edge
        frozen = [0.3, 0.6, 1.0, freeze]
        energies = [self._learned_energy(backend, *_dimer(r)) for r in frozen]
        np.testing.assert_allclose(energies, energies[0], rtol=0.0, atol=1e-12)

        # anti-vacuity: the pair is read at one separation only because of the
        # clamp, and the transition zone and the edge are not frozen.
        with _without_clamp(self._descriptor(backend)):
            bare = [self._learned_energy(backend, *_dimer(r)) for r in frozen]
        assert max(abs(e - bare[0]) for e in bare) > 1e-4
        ramp = [0.5 * (freeze + edge), edge]
        followed = [self._learned_energy(backend, *_dimer(r)) for r in ramp]
        assert abs(followed[0] - followed[1]) > 1e-6

        # The analytical term is the only force along the bond in the frozen
        # zone; the learned model reads the same separation at every point.
        if backend == "pt_expt":
            r, step = 0.6, 1.0e-5
            coord, atype = _dimer(r)
            force = self.pt_expt_model(coord, atype, box=None)["force"]
            expected = -(
                _truncated_pair(r + step, 0, 1) - _truncated_pair(r - step, 0, 1)
            ) / (2.0 * step)
            assert force[0, 1, 0].item() == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("backend", ["pt", "pt_expt"])
    def test_contacts_at_or_beyond_the_edge_read_unchanged(self, backend) -> None:
        """With every contact at or beyond its edge the clamp is the identity.

        Lowering the edges below every contact of the fixture makes the learned
        energy equal that of the descriptor without a clamp, whereas the
        recorded edges, which put several contacts inside their transition
        zone, do not.
        """
        coord = self.coord
        atype = self.atype
        reference = self._learned_energy(backend, coord, atype)
        with _without_clamp(self._descriptor(backend)):
            unclamped = self._learned_energy(backend, coord, atype)
        assert abs(reference - unclamped) > 1e-6

        clamp = self._descriptor(backend).contact_clamp
        recorded = clamp.edges.detach().clone()
        try:
            _set_edges(clamp, np.full_like(EDGES, 0.5))
            lowered = self._learned_energy(backend, coord, atype)
        finally:
            _set_edges(clamp, recorded.cpu().numpy())
        assert lowered == pytest.approx(unclamped, abs=1e-12)

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

    def _plain_learned_state(self) -> dict[str, torch.Tensor]:
        """State of an unbridged learned model, which records no contact edges."""
        config = copy.deepcopy(ZBL_CONFIG)
        for key in ("bridging_method", "bridging_truncation", "bridging_clamp"):
            config.pop(key)
        return get_model(config).atomic_model.state_dict()

    def test_legacy_plain_state_loads_into_a_bridged_model(self) -> None:
        """An unbridged state of an earlier version loads into a bridged model.

        It records no edges, so the target keeps the ones it has, and the
        descriptor adopts the latest version.
        """
        state = self._plain_learned_state()
        version_key = "descriptor.version_tensor"
        state[version_key] = torch.full_like(state[version_key], 1.2)
        learned = self.pt_expt_model.atomic_model.models[0]
        edges = learned.descriptor.contact_clamp.edges.detach().clone()
        learned.load_state_dict(state)
        torch.testing.assert_close(
            learned.descriptor.contact_clamp.edges, edges, rtol=0.0, atol=0.0
        )
        assert float(learned.descriptor.version_tensor) == DescrptDPA4.LATEST_VERSION

    def test_current_plain_state_keeps_the_target_edges(self) -> None:
        """An unbridged state of the current version loads into a bridged model.

        It records no edges, so the target keeps the ones it has.
        """
        learned = self.pt_expt_model.atomic_model.models[0]
        edges = learned.descriptor.contact_clamp.edges.detach().clone()
        assert not torch.equal(
            edges, torch.as_tensor(ContactClamp.initial_edges(TYPE_MAP))
        )
        state = self._plain_learned_state()
        assert not any("contact_clamp" in key for key in state)
        learned.load_state_dict(state)
        torch.testing.assert_close(
            learned.descriptor.contact_clamp.edges, edges, rtol=0.0, atol=0.0
        )

    def test_unbridged_record_before_the_clamp_loads(self) -> None:
        """An unbridged record with an empty window loads and adopts the latest."""
        record = self.pt_expt_model.atomic_model.models[0].descriptor.serialize()
        record["@version"] = 1.3
        record["config"].update(
            contact_clamp=None, inner_clamp_f_inner=None, inner_clamp_f_outer=None
        )
        del record["@variables"]["contact_clamp.edges"]
        restored = DescrptDPA4.deserialize(record)
        assert restored.contact_clamp is None
        assert restored.version == DescrptDPA4.LATEST_VERSION

    def test_serialize_roundtrip(self) -> None:
        """The wire form keeps the recorded edges, the clamp and the truncation."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        data = self.pt_expt_model.serialize()
        # the flat wire type is "linear" -- the SAME string pt/tf write, so a
        # composition round-trips across backends
        assert data["type"] == "linear"
        m2 = BaseModel.deserialize(data).to(torch.device("cpu")).eval()
        assert type(m2) is LinearEnergyModel
        restored, original = (
            m.atomic_model.models[0].descriptor.contact_clamp
            for m in (m2, self.pt_expt_model)
        )
        assert restored.inner == CLAMP
        torch.testing.assert_close(restored.edges, original.edges, rtol=0.0, atol=0.0)
        assert not np.array_equal(
            restored.edges.cpu().numpy(), ContactClamp.initial_edges(TYPE_MAP)
        )
        assert m2.atomic_model.models[1].potential.truncation == TRUNCATION
        out = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        out2 = m2.forward(self.coord, self.atype, box=self.box)
        torch.testing.assert_close(
            out["energy"], out2["energy"], rtol=1e-12, atol=1e-12
        )

    def test_with_comm_gate_on_for_composition(self) -> None:
        """Every block of a bridged DPA4 refreshes its ghost rows across ranks,
        so the graph lower of the composition gets a with-comm artifact.
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

    @pytest.mark.parametrize("support", [1, 3])
    def test_training_records_the_contacts(self, support: int) -> None:
        """A training forward records its batch and reads it as evaluation does.

        Every training batch is recorded before it is evaluated, and the clamp
        acts in training as in evaluation. With a support of one frame the
        edges come out as the shortest contacts of each pair of types in the
        batch, so every contact lies at or above its edge and the batch is read
        unchanged; the edges never rise on a later batch, and a pair of types
        the batch does not hold keeps its edge. With a support of three frames
        a single frame leaves every edge at its covalent start, and the batch
        is read through the clamp.
        """
        from deepmd.pt_expt.utils.contact_clamp import ContactClamp as ContactClampPT

        model = copy.deepcopy(self.pt_expt_model)
        descriptor = model.atomic_model.models[0].descriptor
        descriptor.contact_clamp = ContactClampPT(
            TYPE_MAP, inner=CLAMP, support=support
        )
        clamp = descriptor.contact_clamp
        initial = ContactClamp.initial_edges(TYPE_MAP)

        # an Ni-Ni contact at 0.9 Å and two Ni-O contacts, and no oxygen pair
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.4, 1.5, 0.2]]],
            dtype=torch.float64,
        )
        atype = torch.tensor([[0, 0, 1]], dtype=torch.int64)
        points = coord[0].numpy()
        expected = initial.copy()
        if support == 1:
            expected[0, 0] = 0.9
            expected[0, 1] = expected[1, 0] = min(
                np.linalg.norm(points[i] - points[2]) for i in (0, 1)
            )

        def energy(frame: torch.Tensor = coord) -> float:
            out = model(frame, atype, box=None)
            return float(out["energy"].detach().reshape(-1)[0])

        # at its initial edges the geometry lies inside the clamp
        model.eval()
        clamped = energy()
        with _without_clamp(descriptor):
            unclamped = energy()
        assert abs(clamped - unclamped) > 1e-6

        model.train()
        trained = energy()
        np.testing.assert_allclose(
            clamp.edges.cpu().numpy(), expected, rtol=0.0, atol=1e-12
        )
        model.eval()
        assert energy() == pytest.approx(trained, abs=1e-12)
        if support > 1:
            assert trained == pytest.approx(clamped, abs=1e-12)
            return
        # the batch now sits at its edges
        assert trained == pytest.approx(unclamped, abs=1e-12)
        # looser contacts leave the running minimum where it was
        model.train()
        energy(2.0 * coord)
        np.testing.assert_allclose(
            clamp.edges.cpu().numpy(), expected, rtol=0.0, atol=1e-12
        )


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
    """Native spin + analytical bridging compose.

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
        # the clamp reaches the LEARNED child's descriptor and the truncation
        # the analytical child
        learned, analytical = model.atomic_model.models
        assert learned.descriptor.contact_clamp.inner == CLAMP
        assert analytical.potential.truncation == TRUNCATION

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
        # Open boundaries for the EXACT check: the in-test reference is a
        # direct double loop over pairs, which has no periodic images.
        e_gas = model(coord, atype, spin)["energy"]
        e_gas_learned = learned_only(coord, atype, spin)["energy"]
        zbl_contrib = float((e_gas - e_gas_learned).sum())
        ref = _analytic_zbl_total(coord[0].numpy(), atype[0].numpy())
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


def test_bridging_defaults() -> None:
    """Without options the truncation and the clamp take their defaults.

    The truncation reaches the analytical term and the clamp the learned
    child's descriptor, which starts every pair at its covalent bond length;
    ``bridging_clamp: null`` leaves the learned part reading every separation.
    """
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg.pop("bridging_truncation")
    cfg.pop("bridging_clamp")
    learned, analytical = get_model(cfg).atomic_model.models
    assert analytical.potential.truncation == DEFAULT_TRUNCATION
    clamp = learned.descriptor.contact_clamp
    assert clamp.inner == DEFAULT_CLAMP
    np.testing.assert_allclose(
        clamp.edges.cpu().numpy(),
        ContactClamp.initial_edges(TYPE_MAP),
        rtol=0.0,
        atol=1e-12,
    )

    cfg["bridging_clamp"] = None
    learned, analytical = get_model(cfg).atomic_model.models
    assert learned.descriptor.contact_clamp is None
    assert analytical.potential.truncation == DEFAULT_TRUNCATION


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
        model = _make_sensitive_model()
        clamp = model.atomic_model.models[0].descriptor.contact_clamp
        _set_edges(clamp, EDGES)
        coord, atype, box = _close_pair_system(cpu)
        ref = model.forward(coord, atype, box=box)
        # anti-vacuity: the recorded edges put contacts of the fixture inside
        # their clamp, so the frozen artifact has to carry them to agree
        with _without_clamp(model.atomic_model.models[0].descriptor):
            unclamped = model.forward(coord, atype, box=box)
        assert (ref["energy"] - unclamped["energy"]).abs().max().item() > 1e-6

        model_file = tmp_path / "dpa4_zbl_graph.pt2"
        data = {"model": model.serialize()}
        deserialize_to_file(str(model_file), data, lower_kind="graph")

        with zipfile.ZipFile(model_file) as z:
            md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))
        # multi-rank contract: the blocks refresh their ghost rows across
        # ranks, so a bridged composition DOES get a with-comm twin
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

        # With a one-frame support the edges follow the shortest contacts of
        # the training data, which the frame of this set pins down.
        config = {
            "model": {**copy.deepcopy(ZBL_CONFIG), "bridging_support": 1},
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
            # Start the edges, and the covalent start that caps them, above
            # every contact of the frame, so that the training steps have to
            # lower them to the shortest contacts.
            clamp = model.atomic_model.models[0].descriptor.contact_clamp
            high = np.ones((3, 3))
            high[:2, :2] = 10.0
            with torch.no_grad():
                clamp.start.copy_(torch.as_tensor(high, dtype=clamp.start.dtype))
            _set_edges(clamp, high)
            tasks = trainer._make_training_tasks()
            task = trainer.select_task(tasks)
            for step in range(2):
                result = trainer.train_step(task, step)
                loss = result.payload["loss"]
                assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"
            coord = np.load(os.path.join(data_dir, "set.000", "coord.npy"))
            box = np.load(os.path.join(data_dir, "set.000", "box.npy"))
            atype = np.loadtxt(os.path.join(data_dir, "type.raw"), dtype=np.int64)
            np.testing.assert_allclose(
                clamp.edges.cpu().numpy()[:2, :2],
                _shortest_contacts(
                    coord.reshape(-1, 3), atype, box.reshape(3, 3), RCUT
                ),
                rtol=0.0,
                atol=1e-10,
            )
        finally:
            os.chdir(old_cwd)


def _shortest_contacts(
    coord: np.ndarray, atype: np.ndarray, cell: np.ndarray, rcut: float
) -> np.ndarray:
    """Shortest periodic contact of every pair of types within the cutoff.

    Brute force over the lattice images, independent of the neighbor graph:
    the image range suffices for a cell whose heights exceed half the cutoff.

    Parameters
    ----------
    coord : np.ndarray
        Atomic coordinates with shape (N, 3) in Å.
    atype : np.ndarray
        Atom types with shape (N,).
    cell : np.ndarray
        Cell vectors as rows with shape (3, 3) in Å.
    rcut : float
        Cutoff radius in Å.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shortest distances in Å with shape (T, T), ``inf``
        for a pair of types with no contact.
    """
    ntypes = int(atype.max()) + 1
    shortest = np.full((ntypes, ntypes), np.inf)
    shifts = np.array(
        [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]
    )
    offsets = shifts @ cell
    for i in range(len(atype)):
        for j in range(len(atype)):
            distance = np.linalg.norm(coord[j] + offsets - coord[i], axis=-1)
            distance = distance[(distance > 1e-8) & (distance < rcut)]
            if distance.size:
                pair = sorted((int(atype[i]), int(atype[j])))
                shortest[pair[0], pair[1]] = min(
                    shortest[pair[0], pair[1]], distance.min()
                )
                shortest[pair[1], pair[0]] = shortest[pair[0], pair[1]]
    return shortest


class TestInnerPotentialChangeTypeMapPtExpt:
    """pt_expt twin of the dpmodel ``change_type_map`` regression.

    Exercised through the REAL composition: ``LinearEnergyModel`` ->
    ``LinearEnergyAtomicModel`` -> ``InnerPotentialAtomicModel`` ->
    ``InnerPotential``.  Inside a pt_expt module tree the coefficient and
    truncation tables are wrapped torch buffers, so the rebuild must land on
    the same device/namespace -- a numpy rebuild would desync the buffer or
    fail outright on CUDA.
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
        """Precondition: inside pt_expt the tables are torch buffers."""
        from deepmd.pt_expt.utils import env as _env

        potential = self._zbl_child(self._build(["Ni", "O"])).potential
        for table in (potential.series_table, potential.taylor_table):
            assert isinstance(table, torch.Tensor), (
                "the pt_expt wrapper no longer converts the table to a tensor; "
                "this test would stop covering the device-safe rebuild"
            )
            assert table.device.type == torch.device(_env.DEVICE).type

    def test_reorder_matches_a_freshly_built_model(self) -> None:
        # NOTE: applied to the ZBL CHILD, not the whole composition -- the
        # DPA4/SeZM learned child does not implement change_type_map at all
        # ("change_type_map is not supported for SeZM"), a separate pre-existing
        # limitation.  The tables under test belong to this child.
        child = self._zbl_child(self._build(["Ni", "O"]))
        e_nini = self._pair_energy(child)
        child.change_type_map(["O", "Ni"])
        fresh = self._zbl_child(self._build(["O", "Ni"]))
        e_fresh = self._pair_energy(fresh)
        # anti-vacuity: Ni-Ni and O-O must be far apart, else a stale table
        # would be indistinguishable from a rebuilt one
        assert abs(e_fresh - e_nini) > 1.0
        np.testing.assert_allclose(self._pair_energy(child), e_fresh, rtol=1e-12)
        # the series constants and the end radii follow the new order
        for name in ("series_table", "taylor_table"):
            torch.testing.assert_close(
                getattr(child.potential, name), getattr(fresh.potential, name)
            )

    def test_added_element_extends_the_lookup_on_device(self) -> None:
        from deepmd.pt_expt.utils import env as _env

        child = self._zbl_child(self._build(["Ni", "O"]))
        child.change_type_map(["Ni", "O", "H"])
        for table in (child.potential.series_table, child.potential.taylor_table):
            assert isinstance(table, torch.Tensor), "the rebuild dropped out of torch"
            assert table.device.type == torch.device(_env.DEVICE).type
        # one row per ordered pair of the three types plus the padding type
        assert tuple(child.potential.series_table.shape) == (16, 8)
        assert tuple(child.potential.taylor_table.shape) == (16, 5)
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
        assert restored.potential.truncation == TRUNCATION
        np.testing.assert_allclose(
            self._pair_energy(restored), self._pair_energy(child), rtol=1e-12
        )


def test_native_spin_with_bridging_graph_freeze_and_deep_eval(tmp_path) -> None:
    """Native spin + ZBL freezes to a graph .pt2 and evaluates in parity.

    Both features are graph-route-only, so their combination must survive the
    export seam too, not just eager construction.
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
    # valid one here; the graph lower additionally carries a with-comm twin
    # (only the dense lower stays single-rank for spin).
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
    for key in ("bridging_method", "bridging_truncation", "bridging_clamp"):
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
            for key in ("bridging_method", "bridging_truncation", "bridging_clamp"):
                config.pop(key, None)
        if excluded:
            config["pair_exclude_types"] = [[0, 1]]
        model = get_model(config).to(torch.device("cpu")).eval()
        out = model(coord, atype, box=box)["energy"]
        return float(out.detach().cpu().numpy().reshape(-1)[0])

    # anti-vacuity: without exclusion the analytical term must dominate,
    # otherwise the suppression assertion below would hold trivially.
    assert _truncated_pair(0.9, 0, 1) > 1.0
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
