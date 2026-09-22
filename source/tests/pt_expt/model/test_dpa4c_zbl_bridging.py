# SPDX-License-Identifier: LGPL-3.0-or-later
"""ZBL zone bridging of a DPA4C standard model in pt_expt.

``bridging_method: zbl`` on a DPA4C model builds the same linear composition
as DPA4 (``LinearEnergyModel`` over ``[learned, InnerPotentialAtomicModel]``).
DPA4C has no message passing and carries the edge envelope in every edge term,
so its bridging window is one inner switch on that envelope, paired with a
clamp of the length the radial features read: a pair at or below the inner
radius leaves the learned model exactly as if its edge were deleted, and the
analytical term supplies the whole pair interaction.
"""

import copy
import math

import numpy as np
import pytest
import torch

from deepmd.dpmodel.atomic_model.inner_potential import (
    ELEMENT_TO_Z,
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
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    _needs_with_comm_artifact,
    _resolve_lower_kind,
    _trace_and_export,
    build_synthetic_graph_inputs,
)

TYPE_MAP = ["O", "H", "He"]
R_INNER = 0.5
R_OUTER = 0.8
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
    "bridging_r_inner": R_INNER,
    "bridging_r_outer": R_OUTER,
}
# One O-He pair whose separation is set per test, among atoms that keep every
# other pair beyond the window.
ATYPE = np.array([0, 2, 1, 1])
DIRECTION = np.array([0.8, 0.36, -0.48])
ENVIRONMENT = np.array([[-0.4, 0.9, 0.3], [0.2, -0.5, 1.2]])
#: Magnetic moments of the cluster, in Bohr magnetons.
SPIN = np.array([[0.0, 0.0, 1.4], [0.9, -0.3, 0.2], [-0.5, 1.1, 0.0], [0.3, 0.2, -0.8]])


def _cluster(distance: float) -> np.ndarray:
    return np.concatenate([np.zeros((1, 3)), distance * DIRECTION[None], ENVIRONMENT])


def _evaluate(model: torch.nn.Module, coord: np.ndarray) -> dict[str, torch.Tensor]:
    """Evaluate one open-boundary frame on the backend device."""
    return model(
        torch.tensor(
            coord[None], dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True),
        torch.tensor(ATYPE[None], dtype=torch.int64, device=env.DEVICE),
        box=None,
    )


def _analytic_zbl_total(coord: np.ndarray, atype: np.ndarray = ATYPE) -> float:
    """Independent reference: direct double loop over open-boundary pairs."""
    a_coeff = (0.18175, 0.50986, 0.28022, 0.028171)
    b_coeff = (3.1998, 0.94229, 0.4029, 0.20162)
    charge = [float(ELEMENT_TO_Z[TYPE_MAP[t]]) for t in atype]
    total = 0.0
    for i in range(len(charge)):
        for j in range(i + 1, len(charge)):
            r = float(np.linalg.norm(coord[i] - coord[j]))
            a = 0.88534 * 0.5291772109 / (charge[i] ** 0.23 + charge[j] ** 0.23)
            phi = sum(
                ak * math.exp(-bk * r / a)
                for ak, bk in zip(a_coeff, b_coeff, strict=True)
            )
            total += 14.3996 * charge[i] * charge[j] / r * phi
    return total


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


class TestDPA4CZBLBridging:
    def setup_method(self) -> None:
        self.model = get_model(copy.deepcopy(ZBL_CONFIG)).eval()
        _perturb(self.model)
        self.learned = self.model.atomic_model.models[0]

    def _plain_twin(self, exclude: list[tuple[int, int]]) -> torch.nn.Module:
        """The learned model alone, with the same weights and no window."""
        plain = get_model(copy.deepcopy(PLAIN_CONFIG)).eval()
        plain.atomic_model.load_state_dict(self.learned.state_dict())
        plain.atomic_model.descriptor.reinit_exclude(exclude)
        return plain

    def test_builder_composes_a_graph_route_linear_model(self) -> None:
        assert type(self.model) is LinearEnergyModel
        learned, potential = self.model.atomic_model.fused_decomposition()
        assert learned is self.learned
        assert potential is self.model.atomic_model.models[1].potential
        # An explicit window in Å reaches the descriptor as the same pair of
        # fractions measured against a unit length scale.
        assert learned.descriptor.bridging_scale == "absolute"
        assert learned.descriptor.bridging_f_inner == R_INNER
        assert learned.descriptor.bridging_f_outer == R_OUTER
        assert model_uses_graph_lower(self.model)
        # DPA4C exchanges no node features, so bridging needs no with-comm
        # artifact (a bridged DPA4 does, for its source gate).
        assert _needs_with_comm_artifact(self.model, lower_kind="graph") is False

    @pytest.mark.parametrize("distance", [0.2, 0.45])
    def test_frozen_pair_interacts_through_zbl_alone(self, distance: float) -> None:
        coord = _cluster(distance)
        energy = _evaluate(self.model, coord)["energy"].item()
        removed = _evaluate(self._plain_twin([(0, 2), (2, 0)]), coord)["energy"].item()
        visible = _evaluate(self._plain_twin([]), coord)["energy"].item()
        # The plain model must see the pair, or the comparison is vacuous.
        assert abs(visible - removed) > 1e-3
        assert energy == pytest.approx(removed + _analytic_zbl_total(coord), abs=1e-10)

    def test_pairs_beyond_the_window_add_zbl_to_the_plain_model(self) -> None:
        coord = _cluster(R_OUTER + 0.05)
        energy = _evaluate(self.model, coord)["energy"].item()
        plain = _evaluate(self._plain_twin([]), coord)["energy"].item()
        assert energy == pytest.approx(plain + _analytic_zbl_total(coord), abs=1e-10)

    @pytest.mark.parametrize("distance", [0.3, 0.5, 0.65, 0.8, 1.1])
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
        # an isolated atom exactly the zero output bias.
        assert energies == [0.0, 0.0]

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
        assert not any("pair_table" in key for key in self.model.state_dict())
        assert potential.pair_table.dtype == torch.float32
        table = potential.pair_table.double().cpu().reshape(4, 4, 8)
        # Rows and columns of the padding type vanish.
        assert table[3].abs().max() == 0.0 and table[:, 3].abs().max() == 0.0
        radius = 0.7
        series = (table[0, 2, :4] * torch.exp(-table[0, 2, 4:] * radius)).sum() / radius
        pair = np.array([[0.0, 0.0, 0.0], [radius, 0.0, 0.0]])
        assert series.item() == pytest.approx(
            _analytic_zbl_total(pair, np.array([0, 2])), rel=1e-6
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
        self.plain.atomic_model.load_state_dict(
            self.model.atomic_model.models[0].state_dict()
        )

    def test_the_composition_answers_the_spin_capability(self) -> None:
        learned, potential = self.model.atomic_model.fused_decomposition()
        assert potential is self.model.atomic_model.models[1].potential
        assert self.model.atomic_model.supports_native_spin()
        # The capability comes from the learned child; the analytical term
        # accepts the moments and ignores them.
        assert learned.supports_native_spin()
        assert not self.model.atomic_model.models[1].supports_native_spin()

    def test_the_analytical_term_leaves_the_magnetic_force_alone(self) -> None:
        coord = _cluster(R_OUTER + 0.05)
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


def _compressed_config(window: tuple[float, float] | None) -> dict:
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
    if window is not None:
        config.update(
            bridging_method="zbl",
            bridging_r_inner=window[0],
            bridging_r_outer=window[1],
        )
    return config


class TestCompressedDPA4CZBLBridging:
    """A bridged compressed model keeps the fused and the canonical routes."""

    def setup_method(self) -> None:
        # The window follows the sample, so that its edges populate the frozen
        # zone, the transition zone and the plain zone alike.
        plain = get_model(_compressed_config(None)).to(env.DEVICE).eval()
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
        window = tuple(
            float(value)
            for value in torch.quantile(
                length,
                torch.tensor([0.2, 0.5], dtype=length.dtype, device=length.device),
            )
        )
        self.model = get_model(_compressed_config(window)).to(env.DEVICE).eval()
        _perturb(self.model)
        self.model.atomic_model.enable_compression(0.5, 1.0, 0.002, 0.002)
        assert ((length > window[0]) & (length < window[1])).sum() > 4
        assert (length < window[0]).any()

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
            source_order[:physical].to(torch.uint32).contiguous(),
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
        expected = "dpa4c_canonical" if env.DEVICE.type == "cuda" else "graph"
        assert _resolve_lower_kind("model.pt2", data, "auto") == expected
        exported, metadata, _model_json, _output_keys = _trace_and_export(
            data,
            lower_kind="graph",
            do_atomic_virial=True,
        )
        assert isinstance(exported, torch.export.ExportedProgram)
        assert metadata["graph_edge_dtype"] == "float32"
        assert metadata["has_comm_artifact"] is False
