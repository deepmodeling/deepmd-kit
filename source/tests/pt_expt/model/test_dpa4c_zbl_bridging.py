# SPDX-License-Identifier: LGPL-3.0-or-later
"""ZBL bridging of a DPA4C standard model in pt_expt.

``bridging_method: zbl`` on a DPA4C model builds the same linear composition
as DPA4 (``LinearEnergyModel`` over ``[learned, InnerPotentialAtomicModel]``).
The analytical term is the ZBL series truncated at a fraction of each pair's
covalent bond length, and the learned part reads every pair separation through
the contact clamp of its descriptor: below the data edge of a pair the
separation is frozen, so a pair that closes further changes the learned energy
by nothing and the analytical term alone carries the repulsion. The bond
directions and the analytical term read the true geometry.
"""

import copy
import math
from typing import (
    ClassVar,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.atomic_model.inner_potential import (
    ELEMENT_TO_Z,
)
from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
    ef_op_available,
)
from deepmd.pt_expt.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_uses_graph_lower,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt_expt.utils.serialization import (
    _needs_with_comm_artifact,
    _resolve_lower_kind,
    _trace_and_export,
    build_synthetic_graph_inputs,
)
from deepmd.utils.bridging import (
    DEFAULT_TRUNCATION,
    expand_bridging_method,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)

TYPE_MAP = ["O", "H", "He"]
CLAMP = 0.75
RADII = covalent_radii_from_type_map(TYPE_MAP)
PLAIN_CONFIG = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 16,
        "lmax": 2,
        "n_radial": 8,
        "radial_modes": 2,
        "precision": "float64",
        "seed": 7,
    },
    "fitting_net": {"neuron": [16, 16], "precision": "float64", "seed": 7},
}
ZBL_CONFIG = {
    **PLAIN_CONFIG,
    "bridging_method": "zbl",
    "bridging_clamp": CLAMP,
}
# One O-He pair whose separation is set per test, among atoms that keep every
# other pair beyond the data edge of its type pair, where the clamp is the
# identity.
ATYPE = np.array([0, 2, 1, 1])
DIRECTION = np.array([0.8, 0.36, -0.48])
ENVIRONMENT = np.array([[-0.4, 0.9, 0.3], [0.2, -0.5, 1.2]])
#: Magnetic moments of the cluster, in Bohr magnetons.
SPIN = np.array([[0.0, 0.0, 1.4], [0.9, -0.3, 0.2], [-0.5, 1.1, 0.0], [0.3, 0.2, -0.8]])


def _cluster(distance: float) -> np.ndarray:
    return np.concatenate([np.zeros((1, 3)), distance * DIRECTION[None], ENVIRONMENT])


#: An oxygen and a helium atom, whose separation a dimer test sets.
DIMER = np.array([0, 2])


def _dimer(distance: float) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], distance * DIRECTION])


def _evaluate(
    model: torch.nn.Module, coord: np.ndarray, atype: np.ndarray = ATYPE
) -> dict[str, torch.Tensor]:
    """Evaluate one open-boundary frame on the backend device."""
    return model(
        torch.tensor(
            coord[None], dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True),
        torch.tensor(atype[None], dtype=torch.int64, device=env.DEVICE),
        box=None,
    )


def _zbl_series(r: torch.Tensor, charge_i: float, charge_j: float) -> torch.Tensor:
    """Evaluate the ZBL universal screened Coulomb energy of one pair in eV."""
    a_coeff = (0.18175, 0.50986, 0.28022, 0.028171)
    b_coeff = (3.1998, 0.94229, 0.4029, 0.20162)
    length = 0.88534 * 0.5291772109 / (charge_i**0.23 + charge_j**0.23)
    phi = sum(
        ak * torch.exp(-bk * r / length)
        for ak, bk in zip(a_coeff, b_coeff, strict=True)
    )
    return 14.3996 * charge_i * charge_j / r * phi


def _truncated_zbl(distance: float, type_i: int, type_j: int) -> float:
    """Evaluate the truncated ZBL energy of one pair, independently of the model.

    The series ends at ``DEFAULT_TRUNCATION`` times the covalent bond length of
    the pair, by subtracting its cubic Taylor polynomial there; the Taylor
    coefficients come from automatic differentiation of the series.
    """
    charge_i = float(ELEMENT_TO_Z[TYPE_MAP[type_i]])
    charge_j = float(ELEMENT_TO_Z[TYPE_MAP[type_j]])
    end = DEFAULT_TRUNCATION * float(RADII[type_i] + RADII[type_j])
    if distance >= end:
        return 0.0
    anchor = torch.tensor(end, dtype=torch.float64, requires_grad=True)
    derivatives = [_zbl_series(anchor, charge_i, charge_j)]
    for _ in range(3):
        derivatives.append(
            torch.autograd.grad(derivatives[-1], anchor, create_graph=True)[0]
        )
    delta = distance - end
    polynomial = sum(
        derivative.item() / math.factorial(order) * delta**order
        for order, derivative in enumerate(derivatives)
    )
    value = _zbl_series(torch.tensor(distance, dtype=torch.float64), charge_i, charge_j)
    return value.item() - polynomial


def _analytic_zbl_total(coord: np.ndarray, atype: np.ndarray = ATYPE) -> float:
    """Independent reference: direct double loop over open-boundary pairs."""
    total = 0.0
    for i in range(len(atype)):
        for j in range(i + 1, len(atype)):
            distance = float(np.linalg.norm(coord[i] - coord[j]))
            total += _truncated_zbl(distance, int(atype[i]), int(atype[j]))
    return total


def _set_edge(model: torch.nn.Module, type_i: int, type_j: int, edge: float) -> None:
    """Set the data edge of a type pair of the learned child of a bridged model."""
    edges = model.atomic_model.models[0].descriptor.contact_clamp.edges
    with torch.no_grad():
        edges[type_i, type_j] = edges[type_j, type_i] = edge


def _learned_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """State of the learned child of a bridged model, without its contact edges.

    A learned model without a contact clamp has no edges to load.
    """
    return {
        name: value
        for name, value in model.atomic_model.models[0].state_dict().items()
        if "contact_clamp" not in name
    }


def _perturb(model: torch.nn.Module) -> None:
    """Move the weights off their initialization so outputs depend on geometry."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device="cpu",
                )
                parameter.add_(0.3 * noise.to(parameter.device))


@pytest.mark.parametrize("methods", [("none", "zbl"), ("zbl", "none"), ("zbl", "zbl")])
def test_multitask_shared_descriptor(methods: tuple[str, str]) -> None:
    heads = {
        key: {**ZBL_CONFIG, "descriptor": "rep", "bridging_method": method}
        for key, method in zip(("first", "second"), methods, strict=True)
    }
    config, links = preprocess_shared_params(
        {"shared_dict": {"rep": PLAIN_CONFIG["descriptor"]}, "model_dict": heads}
    )
    models = {key: get_model(params) for key, params in config["model_dict"].items()}
    references = {
        key: get_model(params) for key, params in config["model_dict"].items()
    }
    for key, model in models.items():
        _perturb(model)
        references[key].load_state_dict(model.state_dict())

    wrapper = ModelWrapper(models)
    wrapper.share_params(links, {"first": 0.5, "second": 0.5})
    first, second = (model.get_descriptor() for model in models.values())
    assert first is not second
    shared = dict(first.named_parameters())
    assert shared and all(p is shared[name] for name, p in second.named_parameters())

    for key, distance in zip(heads, (0.4, 0.65), strict=True):
        ref = references[key]
        ref_params = dict(ref.get_descriptor().named_parameters())
        with torch.no_grad():
            for name, p in ref_params.items():
                p.copy_(shared[name])
        outputs = _evaluate(models[key], _cluster(distance))
        expected = _evaluate(ref, _cluster(distance))
        for field in ("energy", "force"):
            torch.testing.assert_close(outputs[field], expected[field])
        outputs["energy"].sum().backward()
        expected["energy"].sum().backward()

    ref_grads = [
        dict(m.get_descriptor().named_parameters()) for m in references.values()
    ]
    for name, p in shared.items():
        grads = [params[name].grad for params in ref_grads]
        if all(g is None for g in grads):
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, sum(g for g in grads if g is not None))


def test_linear_descriptor_selection() -> None:
    config = expand_bridging_method(copy.deepcopy(ZBL_CONFIG))
    config["models"].reverse()
    model = get_model(config)
    assert model.get_descriptor() is model.atomic_model.models[1].descriptor

    config["models"][0] = copy.deepcopy(config["models"][1])
    with pytest.raises(AttributeError, match="no unique learned descriptor"):
        get_model(config).get_descriptor()


class TestDPA4CZBLBridging:
    def setup_method(self) -> None:
        self.model = get_model(copy.deepcopy(ZBL_CONFIG)).eval()
        _perturb(self.model)
        self.learned = self.model.atomic_model.models[0]

    def _plain_twin(self) -> torch.nn.Module:
        """The learned model alone, with the same weights and no contact clamp."""
        plain = get_model(copy.deepcopy(PLAIN_CONFIG)).eval()
        plain.atomic_model.load_state_dict(_learned_state(self.model))
        return plain

    def test_builder_composes_a_graph_route_linear_model(self) -> None:
        assert type(self.model) is LinearEnergyModel
        learned, potential = self.model.atomic_model.fused_decomposition()
        assert learned is self.learned
        assert potential is self.model.atomic_model.models[1].potential
        # The composition derives the clamp of the descriptor from its own
        # option, and the analytical term ends at the default fraction of the
        # covalent bond length of every pair.
        assert learned.descriptor.contact_clamp.inner == CLAMP
        assert potential.truncation == DEFAULT_TRUNCATION
        assert model_uses_graph_lower(self.model)
        # DPA4C exchanges no node features, so bridging needs no with-comm
        # artifact.
        assert _needs_with_comm_artifact(self.model, lower_kind="graph") is False

    def test_clamp_starts_from_the_covalent_bond_lengths(self) -> None:
        edges = self.learned.descriptor.contact_clamp.edges
        expected = RADII[:, None] + RADII[None, :]
        np.testing.assert_allclose(
            edges[:3, :3].cpu().numpy(), expected, rtol=0.0, atol=1e-12
        )
        # The padding type keeps its fixed edge.
        assert torch.all(edges[3] == 1.0) and torch.all(edges[:, 3] == 1.0)

    @pytest.mark.parametrize("distance", [0.2, 0.5, 0.8])
    def test_frozen_pair_interacts_through_zbl_alone(self, distance: float) -> None:
        """A pair at or below ``a e`` is read at one separation by the learned part.

        The learned energy of the dimer is the energy of the composition less
        the analytical term. It is the same for every separation in the frozen
        zone, while the learned model without the clamp still sees the pair.
        """
        end = float(RADII[0] + RADII[2]) * CLAMP
        assert distance <= end
        reference = _dimer(0.1)
        learned = _evaluate(self.model, reference, DIMER)["energy"].item() - (
            _analytic_zbl_total(reference, DIMER)
        )
        coord = _dimer(distance)
        energy = _evaluate(self.model, coord, DIMER)["energy"].item()
        assert energy - _analytic_zbl_total(coord, DIMER) == pytest.approx(
            learned, abs=1e-10
        )
        # The learned model without the clamp must read the pair differently, or
        # the comparison is vacuous.
        plain = _evaluate(self._plain_twin(), coord, DIMER)["energy"].item()
        assert abs(plain - learned) > 1e-3

    def test_frozen_pair_feels_the_analytical_force_only(self) -> None:
        """Below ``a e`` the learned energy is flat along the bond."""
        coord = _dimer(0.4)
        force = _evaluate(self.model, coord, DIMER)["force"][0].detach().cpu().numpy()
        step = 1.0e-5
        expected = np.zeros((2, 3))
        for atom in range(2):
            for axis in range(3):
                shift = np.zeros_like(coord)
                shift[atom, axis] = step
                upper = _analytic_zbl_total(coord + shift, DIMER)
                lower = _analytic_zbl_total(coord - shift, DIMER)
                expected[atom, axis] = -(upper - lower) / (2.0 * step)
        # The learned energy depends on the direction of the bond only, which
        # a dimer does not constrain, so the force along the bond is the
        # analytical one.
        bond = DIRECTION / np.linalg.norm(DIRECTION)
        np.testing.assert_allclose(force @ bond, expected @ bond, rtol=1e-6, atol=1e-6)

    def test_pairs_beyond_the_data_edge_add_zbl_to_the_plain_model(self) -> None:
        """Beyond its data edge a pair is read as it is, and the term adds up."""
        edge = float(RADII[0] + RADII[2])
        distance = edge + 0.01
        assert distance < DEFAULT_TRUNCATION * edge
        coord = _cluster(distance)
        assert _truncated_zbl(distance, 0, 2) > 1e-3
        energy = _evaluate(self.model, coord)["energy"].item()
        plain = _evaluate(self._plain_twin(), coord)["energy"].item()
        assert energy == pytest.approx(plain + _analytic_zbl_total(coord), abs=1e-10)

    def test_the_term_ends_at_the_truncation_radius(self) -> None:
        """Beyond the end radius of every pair the composition is the learned model."""
        distance = 1.05 * DEFAULT_TRUNCATION * float(RADII[0] + RADII[2])
        coord = _cluster(distance)
        assert _truncated_zbl(distance, 0, 2) == 0.0
        energy = _evaluate(self.model, coord)["energy"].item()
        plain = _evaluate(self._plain_twin(), coord)["energy"].item()
        assert energy == pytest.approx(plain + _analytic_zbl_total(coord), abs=1e-10)

    @pytest.mark.parametrize("distance", [0.3, 0.5, 0.65, 0.8, 0.95, 1.1, 1.15, 1.25])
    def test_force_matches_finite_difference(self, distance: float) -> None:
        coord = _cluster(distance)
        force = _evaluate(self.model, coord)["force"]
        step = 1.0e-5
        for axis in range(3):
            shift = np.zeros_like(coord)
            shift[1, axis] = step
            upper = _evaluate(self.model, coord + shift)["energy"].item()
            lower = _evaluate(self.model, coord - shift)["energy"].item()
            expected = -(upper - lower) / (2.0 * step)
            assert force[0, 1, axis].item() == pytest.approx(
                expected, rel=1e-6, abs=1e-6
            )

    def test_serialize_roundtrip(self) -> None:
        # The recorded edges belong to the model: moving one puts the pair in
        # the transition zone of its clamp, and the restored model keeps it.
        edges = self.learned.descriptor.contact_clamp.edges
        with torch.no_grad():
            edges[0, 2] = edges[2, 0] = 0.7
        coord = _cluster(0.65)
        data = self.model.serialize()
        assert data["type"] == "linear"
        restored = LinearEnergyModel.deserialize(data).eval()
        torch.testing.assert_close(
            _evaluate(restored, coord)["energy"],
            _evaluate(self.model, coord)["energy"],
            rtol=0.0,
            atol=0.0,
        )
        restored_edges = restored.atomic_model.models[0].descriptor.contact_clamp.edges
        torch.testing.assert_close(restored_edges, edges, rtol=0.0, atol=0.0)

    def test_isolated_atoms_keep_the_vacuum_reference(self) -> None:
        """A preset bias keeps the learned child referencing the isolated atoms.

        The composition owns the output bias, and its preset reaches the
        learned child, whose fitting then still subtracts the network output
        of an atom without neighbors.
        """
        preset = {"energy": [-430.1, -13.6, -79.0]}
        isolated = np.zeros((1, 3))
        energies = []
        for base in (PLAIN_CONFIG, ZBL_CONFIG):
            config = copy.deepcopy(base)
            config["fitting_net"]["vacuum_ref"] = True
            config["preset_out_bias"] = preset
            model = get_model(config).eval()
            _perturb(model)
            fitting = model.atomic_model.fused_decomposition()[0].fitting_net
            assert fitting.vacuum_ref
            energies.append(
                model(
                    torch.tensor(
                        isolated[None], dtype=torch.float64, device=env.DEVICE
                    ),
                    torch.tensor([[0]], dtype=torch.int64, device=env.DEVICE),
                    box=None,
                )["energy"].item()
            )
        # Neither model has computed its statistics, so the reference leaves
        # an isolated atom at the zero output bias, up to float64 roundoff
        # from the separately evaluated network and vacuum reference.
        np.testing.assert_allclose(
            energies, [0.0, 0.0], rtol=0.0, atol=8 * np.finfo(np.float64).eps
        )

    def test_declared_tensors_keep_the_adamw_route(self) -> None:
        """HybridMuon routes the same descriptor tensors as in the plain model.

        The composition carries the learned model one level deeper, so its
        patterns name that path; what must not change is the set of tensors
        they reach.
        """

        def routed(model: torch.nn.Module, path: str) -> list[str]:
            patterns = model.adam_route_patterns()
            assert patterns
            names = [
                name
                for name, _ in model.named_parameters()
                if any(pattern in name.lower() for pattern in patterns)
            ]
            assert names and all(
                name.startswith(f"{path}descriptor.") for name in names
            )
            return [name[len(path) :] for name in names]

        plain = get_model(copy.deepcopy(PLAIN_CONFIG))
        assert routed(self.model, "atomic_model.models.0.") == routed(
            plain, "atomic_model."
        )

    def test_pair_table_is_rebuilt_not_stored(self) -> None:
        potential = self.model.atomic_model.models[1].potential
        stored = list(self.model.state_dict())
        assert not any("pair_table" in key for key in stored)
        assert potential.pair_table.dtype == torch.float64
        table = potential.pair_table.cpu().reshape(4, 4, 16)
        assert table[3].abs().max() == 0.0 and table[:, 3].abs().max() == 0.0
        row = table[0, 2]
        end = DEFAULT_TRUNCATION * float(RADII[0] + RADII[2])
        assert row[8].item() == pytest.approx(end)
        torch.testing.assert_close(row[12:], row[:4] * torch.exp(-row[4:8] * end))


class TestDPA4CZBLBridgingMultiBlock:
    """A model of two blocks evaluates the clamp in its first block.

    The states that the further block passes on are built from the first
    block, whose radial table and envelope read the clamped separation, so the
    clamp reaches every block through them. The block envelope of a further
    block reads the true length.
    """

    MULTI_BLOCK: ClassVar[dict] = {"rcut": [4.0, 3.5], "n_blocks": 2}

    def setup_method(self) -> None:
        config = copy.deepcopy(ZBL_CONFIG)
        config["descriptor"].update(self.MULTI_BLOCK)
        self.model = get_model(config).eval()
        _perturb(self.model)

    @pytest.mark.parametrize("distance", [0.3, 0.5, 0.65, 0.8, 0.95, 1.1, 1.15])
    def test_force_matches_finite_difference(self, distance: float) -> None:
        coord = _cluster(distance)
        force = _evaluate(self.model, coord)["force"]
        step = 1.0e-5
        for atom in (0, 1, 2):
            for axis in range(3):
                shift = np.zeros_like(coord)
                shift[atom, axis] = step
                upper = _evaluate(self.model, coord + shift)["energy"].item()
                lower = _evaluate(self.model, coord - shift)["energy"].item()
                expected = -(upper - lower) / (2.0 * step)
                assert force[0, atom, axis].item() == pytest.approx(
                    expected, rel=1e-6, abs=1e-6
                )


def _native_spin(base: dict) -> dict:
    """The same configuration with a magnetic moment on every type."""
    return {
        **copy.deepcopy(base),
        "spin": {"scheme": "native", "use_spin": [True] * len(TYPE_MAP)},
    }


def _evaluate_spin(
    model: torch.nn.Module, coord: np.ndarray
) -> dict[str, torch.Tensor]:
    """Evaluate one open-boundary frame of a native-spin model."""
    return model(
        torch.tensor(
            coord[None], dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True),
        torch.tensor(ATYPE[None], dtype=torch.int64, device=env.DEVICE),
        torch.tensor(SPIN[None], dtype=torch.float64, device=env.DEVICE),
        box=None,
    )


class TestDPA4CZBLBridgingWithNativeSpin:
    """Native spin and ZBL bridging compose on a DPA4C model.

    The analytical term is a function of the separation alone: it adds its
    energy and its force to the spin-conditioned model and leaves the magnetic
    force untouched.
    """

    def setup_method(self) -> None:
        self.model = get_model(_native_spin(ZBL_CONFIG)).eval()
        _perturb(self.model)
        self.plain = get_model(_native_spin(PLAIN_CONFIG)).eval()
        self.plain.atomic_model.load_state_dict(_learned_state(self.model))
        # The oxygen-helium pair is read as it is from its data edge on, so a
        # dimer of 0.7 Å is a pair that the analytical term acts on strongly
        # and the clamp leaves alone.
        _set_edge(self.model, 0, 2, 0.6)

    def test_the_composition_answers_the_spin_capability(self) -> None:
        learned, potential = self.model.atomic_model.fused_decomposition()
        assert potential is self.model.atomic_model.models[1].potential
        assert self.model.atomic_model.supports_native_spin()
        # The capability comes from the learned child; the analytical term
        # accepts the moments and ignores them.
        assert learned.supports_native_spin()
        assert not self.model.atomic_model.models[1].supports_native_spin()

    def test_the_analytical_term_leaves_the_magnetic_force_alone(self) -> None:
        coord = _cluster(0.7)
        bridged = _evaluate_spin(self.model, coord)
        plain = _evaluate_spin(self.plain, coord)
        assert bridged["energy"].item() == pytest.approx(
            plain["energy"].item() + _analytic_zbl_total(coord), abs=1e-10
        )
        torch.testing.assert_close(
            bridged["force_mag"], plain["force_mag"], rtol=0.0, atol=0.0
        )
        # The analytical force must reach the atoms, or the comparison above
        # holds for a term that does nothing.
        assert (bridged["force"] - plain["force"]).abs().max().item() > 1e-3


def test_children_reference_the_isolated_atoms_by_the_composition_preset() -> None:
    """A child's fitting references the isolated atoms by the composition preset.

    The composition computes the output bias, so a preset in the configuration
    of a child alone fixes nothing and leaves the reference off.
    """
    preset = {"energy": [-430.1, -13.6, -79.0]}
    child = {
        "descriptor": copy.deepcopy(PLAIN_CONFIG["descriptor"]),
        "fitting_net": {
            **copy.deepcopy(PLAIN_CONFIG["fitting_net"]),
            "vacuum_ref": True,
        },
        "preset_out_bias": preset,
    }
    config = {
        "type": "linear_ener",
        "type_map": TYPE_MAP,
        "models": [child, copy.deepcopy(child)],
        "weights": "mean",
    }
    model = get_model(copy.deepcopy(config))
    assert not any(sub.fitting_net.vacuum_ref for sub in model.atomic_model.models)
    config["preset_out_bias"] = preset
    model = get_model(config)
    assert all(sub.fitting_net.vacuum_ref for sub in model.atomic_model.models)


def _compressed_config(bridged: bool) -> dict:
    """A float32 DPA4C model the fused operators can serve."""
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 16,
            "lmax": 3,
            "n_radial": 8,
            "radial_modes": 2,
            "precision": "float32",
            "seed": 17,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [32, 32],
            "activation_function": "silu",
            "resnet_dt": False,
            "precision": "float32",
            "seed": 19,
        },
    }
    if bridged:
        config.update(bridging_method="zbl", bridging_clamp=CLAMP)
    return config


def test_compression_keeps_the_default_charge_state() -> None:
    """A bridged model reports the charge state its compressed tables hold.

    Compression folds the default condition of the learned model into its
    tables, after which the model reads no condition input. The composition
    still reports that default, which an archive of it serves when a caller
    names no condition.
    """
    config = _compressed_config(True)
    config["descriptor"].update(add_chg_spin_ebd=True, default_chg_spin=[0.0, 1.0])
    model = get_model(config).eval()
    model.atomic_model.enable_compression(0.5, 1.0, 0.002, 0.002)
    assert model.get_dim_chg_spin() == 0
    assert model.get_default_chg_spin() == [0.0, 1.0]


class TestCompressedDPA4CZBLBridging:
    """A bridged compressed model keeps the fused and the canonical routes."""

    def setup_method(self) -> None:
        # The data edge of the clamp follows the sample, so that its edges
        # populate the frozen zone, the transition zone and the plain zone of
        # the clamp alike, and the end radius of the analytical term lies
        # inside the sample as well.
        plain = get_model(_compressed_config(False)).to(env.DEVICE).eval()
        self.sample = build_synthetic_graph_inputs(
            plain,
            e_max=None,
            nframes=2,
            nloc=9,
            dtype=torch.float32,
            device=env.DEVICE,
        )
        edge_vec, edge_mask = self.sample[4], self.sample[5]
        length = edge_vec[edge_mask].norm(dim=-1)
        edge = float(torch.quantile(length, 0.5))
        self.model = get_model(_compressed_config(True)).to(env.DEVICE).eval()
        _perturb(self.model)
        descriptor = self.model.atomic_model.models[0].descriptor
        with torch.no_grad():
            descriptor.contact_clamp.edges[:2, :2] = edge
        self.model.atomic_model.enable_compression(0.5, 1.0, 0.002, 0.002)
        lower = CLAMP * edge
        assert ((length > lower) & (length < edge)).sum() > 4
        assert (length <= lower).any() and (length >= edge).any()
        end = self.model.atomic_model.models[1].potential.pair_table[:, 8]
        assert (length < float(end[:2].max())).any()

    def _lower(self, n_local: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        (
            atype,
            n_node,
            sample_n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fparam,
            aparam,
            charge_spin,
        ) = self.sample
        return self.model.forward_common_lower_graph(
            atype,
            n_node,
            sample_n_local if n_local is None else n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            destination_sorted=True,
            do_atomic_virial=True,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )

    def _autograd_lower(
        self,
        monkeypatch: pytest.MonkeyPatch,
        n_local: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The portable composition: learned descriptor plus the ZBL sibling."""
        from deepmd.pt_expt.model import (
            make_model,
        )

        with monkeypatch.context() as patch:
            patch.setattr(make_model, "fused_energy_force_enabled", lambda: False)
            return self._lower(n_local)

    @pytest.mark.skipif(
        not ef_op_available(),
        reason="the fused DPA4C energy-force operators are required",
    )
    @pytest.mark.parametrize("ghosts", [0, 3])
    def test_fused_route_matches_the_autograd_composition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ghosts: int,
    ) -> None:
        monkeypatch.setenv("DP_CUDA_INFER", "2")
        n_local = self.sample[1] - ghosts
        reference = self._autograd_lower(monkeypatch, n_local)
        fused = self._lower(n_local)
        # The analytical term must matter, or the comparison is vacuous.
        learned = self.model.atomic_model.models[0]
        zbl = reference["energy_redu"].detach().sum() - (
            learned.forward_common_atomic_graph(*self._graph_and_types())["energy"]
            .detach()
            .sum()
        )
        assert ghosts or abs(float(zbl)) > 1.0
        # A silent fallback to the autograd lower would compare it with itself.
        assert not torch.equal(fused["energy_derv_r"], reference["energy_derv_r"])
        for key in (
            "energy",
            "energy_redu",
            "energy_derv_r",
            "energy_derv_c",
            "energy_derv_c_redu",
        ):
            torch.testing.assert_close(
                fused[key].double(),
                reference[key].double(),
                atol=2e-4,
                rtol=2e-4,
            )

    def _graph_and_types(self) -> tuple:
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
        )

        atype, n_node, n_local, edge_index, edge_vec, edge_mask = self.sample[:6]
        graph = NeighborGraph(
            n_node=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            n_local=n_local,
        )
        return graph, atype

    @pytest.mark.skipif(
        env.DEVICE.type != "cuda", reason="the compact canonical route is CUDA only"
    )
    @pytest.mark.parametrize("tile", [None, "4"])
    def test_canonical_route_matches_the_autograd_composition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tile: str | None,
    ) -> None:
        if tile is not None:
            monkeypatch.setenv("DP_NODE_TILE", tile)
        reference = self._autograd_lower(monkeypatch)
        atype, n_node, n_local, edge_index, edge_vec = self.sample[:5]
        destination_row_ptr, source_order, source_row_ptr = self.sample[7:10]
        physical = int(destination_row_ptr[-1])
        canonical = self.model.forward_lower_canonical_graph(
            atype,
            n_node,
            n_local,
            edge_index[0][:physical].to(torch.uint32).contiguous(),
            edge_vec[:physical].contiguous(),
            destination_row_ptr,
            source_row_ptr,
            source_order[:physical].to(torch.int64).contiguous(),
            do_atomic_virial=True,
        )
        for key, reference_key in (
            ("atom_energy", "energy"),
            ("energy", "energy_redu"),
            ("force", "energy_derv_r"),
            ("atom_virial", "energy_derv_c"),
            ("virial", "energy_derv_c_redu"),
        ):
            torch.testing.assert_close(
                canonical[key].double().reshape(-1),
                reference[reference_key].double().reshape(-1),
                atol=2e-4,
                rtol=2e-4,
            )

    def test_compressed_model_exports_like_its_learned_part(self) -> None:
        data = {"model": self.model.to("cpu").serialize()}
        expected = "dpa_canonical" if env.DEVICE.type == "cuda" else "graph"
        assert _resolve_lower_kind("model.pt2", data, "auto") == expected
        exported, metadata, _model_json, _output_keys = _trace_and_export(
            data,
            lower_kind="graph",
            do_atomic_virial=True,
        )
        assert isinstance(exported, torch.export.ExportedProgram)
        assert metadata["graph_edge_dtype"] == "float32"
        assert metadata["has_comm_artifact"] is False
