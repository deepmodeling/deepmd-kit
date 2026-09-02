# SPDX-License-Identifier: LGPL-3.0-or-later

import dataclasses
import os
from typing import (
    Any,
)
from unittest import (
    mock,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DPDescrptDPA4C
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    graph_from_dense_quartet,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.pt_expt.utils import (
    env,
)

# Structural variants exercised by the backend-agnostic contracts. They span
# both angular profiles and both radial function classes.
STRUCTURES = [
    {"channels": 16, "lmax": 2, "radial_modes": 0},
    {"channels": 32, "lmax": 4, "radial_modes": 0},
    {"channels": 32, "lmax": 2, "radial_modes": 3},
    {"channels": 16, "lmax": 3, "radial_modes": 3},
]


class TestDPA4C:
    def setup_method(self) -> None:
        self.descriptor = self.build()
        self.coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.2, -0.1],
                    [-0.4, 0.9, 0.3],
                    [0.2, -0.5, 1.2],
                    [-0.7, -0.3, -0.8],
                ]
            ],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        self.atype = torch.tensor(
            [[0, 1, 0, 1, 0]],
            dtype=torch.long,
            device=env.DEVICE,
        )

    @staticmethod
    def build(**structure: Any) -> DescrptDPA4C:
        """Build a descriptor on the backend device.

        The default precision is double, which the numerical contracts need.
        Mixed-precision tests must override it, because CUDA autocast ignores
        double operands and would otherwise leave the region untouched.
        """
        return DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            n_radial=4,
            seed=17,
            **{
                "channels": 16,
                "lmax": 2,
                "precision": "float64",
                **structure,
            },
        ).to(env.DEVICE)

    def _inputs(
        self,
        coord: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return extend_input_and_build_neighbor_list(
            coord,
            self.atype,
            self.descriptor.get_rcut(),
            [8],
            mixed_types=True,
            box=None,
        )

    def _evaluate(
        self,
        descriptor: DescrptDPA4C,
        coord: torch.Tensor,
    ) -> torch.Tensor:
        coord_ext, atype_ext, mapping, nlist = self._inputs(coord)
        return descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]

    def _dimer_probe(
        self,
        descriptor: DescrptDPA4C,
        distance: torch.Tensor,
        *,
        active: bool = True,
    ) -> torch.Tensor:
        """Contract the descriptor of one undirected dimer into a scalar."""
        zero = torch.zeros_like(distance)
        edge_vec = torch.stack(
            [
                torch.stack([distance, zero, zero]),
                torch.stack([-distance, zero, zero]),
            ]
        )
        graph = NeighborGraph(
            n_node=torch.tensor([2], dtype=torch.long, device=env.DEVICE),
            edge_index=torch.tensor(
                [[1, 0], [0, 1]],
                dtype=torch.long,
                device=env.DEVICE,
            ),
            edge_vec=edge_vec,
            edge_mask=torch.full(
                (2,),
                active,
                dtype=torch.bool,
                device=env.DEVICE,
            ),
        )
        atype = torch.zeros(2, dtype=torch.long, device=env.DEVICE)
        output, _ = descriptor.call_graph(graph, atype)
        cotangent = torch.linspace(
            -0.7,
            1.3,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        return (output * cotangent).sum()

    def _dimer_derivatives(
        self,
        descriptor: DescrptDPA4C,
        distance: float,
    ) -> list[torch.Tensor]:
        """Return the value and first three radial derivatives of the probe."""
        radius = torch.tensor(
            distance,
            dtype=torch.float64,
            device=env.DEVICE,
            requires_grad=True,
        )
        derivatives = [self._dimer_probe(descriptor, radius)]
        for _ in range(3):
            derivatives.append(
                torch.autograd.grad(
                    derivatives[-1],
                    radius,
                    create_graph=True,
                )[0]
            )
        return derivatives

    # === Backend equivalence ===

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_numpy_pytorch_parity_and_parameter_gradients(
        self,
        structure: dict[str, int],
    ) -> None:
        descriptor = self.build(**structure)
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        result = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
        reference = DPDescrptDPA4C.deserialize(descriptor.serialize())(
            coord_ext.cpu().numpy(),
            atype_ext.cpu().numpy(),
            nlist.cpu().numpy(),
            mapping=mapping.cpu().numpy(),
        )[0]
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            reference,
            atol=3e-12,
            rtol=3e-12,
        )

        # Every trainable array must remain reachable from the loss.
        result.square().mean().backward()
        for name, parameter in descriptor.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_sharing_keeps_the_branch_local_exclusion_mask(self) -> None:
        """Sharing must not carry the pair-exclusion mask across replicas.

        ``exclude_types`` is absent from the structural signature because each
        branch of a multitask model configures it separately. The mask is a
        submodule, so a backend that rebinds the whole submodule table would
        capture it and silently make the replica evaluate excluded pairs.
        """
        base = self.build(exclude_types=[])
        replica = self.build(exclude_types=[[0, 1]])
        expected = replica.emask.type_mask.detach().clone()
        replica.share_params(base, 0)
        torch.testing.assert_close(replica.emask.type_mask, expected)
        assert replica.exclude_types == [[0, 1]]
        assert replica.readout is base.readout

    def test_compression_rejects_excluded_pairs(self) -> None:
        """The fused kernel has no type-exclusion branch.

        Compression must refuse rather than emit an artifact that can never
        reach the fused path: the re-export would fall back to the plain graph
        lower, which the Kokkos spin pair style in turn refuses to load.
        """
        excluded = self.build(precision="float32", exclude_types=[[0, 1]])
        with pytest.raises(ValueError, match="type-exclusion branch"):
            excluded.enable_compression(0.5)
        # The exclusion is the only thing standing in the way.
        included = self.build(precision="float32")
        included.enable_compression(0.5)
        assert included.compress

    def test_compression_round_trip_preserves_frozen_descriptor(self) -> None:
        """Compression must depend on tensor values, not optimizer registration."""
        frozen = self.build(precision="float32", trainable=False).eval()
        assert not tuple(frozen.radial_embedding.parameters())
        reference = self._evaluate(frozen, self.coord)

        frozen.enable_compression(0.5, table_stride_1=0.2)
        restored = DescrptDPA4C.deserialize(frozen.serialize()).to(env.DEVICE).eval()

        assert restored.trainable is False
        assert restored.compress
        assert not any(parameter.requires_grad for parameter in restored.parameters())
        torch.testing.assert_close(self._evaluate(restored, self.coord), reference)

    def test_serialization_preserves_parameters(self) -> None:
        restored = DescrptDPA4C.deserialize(self.descriptor.serialize()).to(env.DEVICE)
        original_parameters = dict(self.descriptor.named_parameters())
        restored_parameters = dict(restored.named_parameters())
        assert original_parameters.keys() == restored_parameters.keys()
        for name in original_parameters:
            torch.testing.assert_close(
                restored_parameters[name],
                original_parameters[name],
            )

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_torch_export_matches_eager(self, structure: dict[str, int]) -> None:
        descriptor = self.build(**structure).eval()
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        exported = torch.export.export(
            descriptor,
            (coord_ext, atype_ext, nlist),
            kwargs={"mapping": mapping},
            strict=False,
        )
        torch.testing.assert_close(
            exported.module()(coord_ext, atype_ext, nlist, mapping=mapping)[0],
            descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0],
        )

    # === Differentiability ===

    def test_coordinate_gradient_matches_finite_difference(self) -> None:
        coord = self.coord.detach().clone().requires_grad_(True)
        output = self._evaluate(self.descriptor, coord)
        cotangent = torch.linspace(
            -0.8,
            1.1,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        (gradient,) = torch.autograd.grad((output * cotangent).sum(), coord)

        epsilon = 1e-6
        finite_difference = torch.empty_like(coord)
        flat = coord.detach().reshape(-1)
        for index in range(flat.numel()):
            shifted = []
            for sign in (1.0, -1.0):
                probe = flat.clone()
                probe[index] += sign * epsilon
                shifted.append(
                    (
                        self._evaluate(self.descriptor, probe.reshape_as(coord))
                        * cotangent
                    ).sum()
                )
            finite_difference.reshape(-1)[index] = (shifted[0] - shifted[1]) / (
                2.0 * epsilon
            )
        torch.testing.assert_close(
            gradient,
            finite_difference,
            atol=3e-8,
            rtol=3e-8,
        )

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_force_loss_double_backward(self, structure: dict[str, int]) -> None:
        descriptor = self.build(**structure)
        coord = self.coord.detach().clone().requires_grad_(True)
        output = self._evaluate(descriptor, coord)
        (force_gradient,) = torch.autograd.grad(
            output.square().sum(),
            coord,
            create_graph=True,
        )
        named_parameters = list(descriptor.named_parameters())
        parameter_gradients = torch.autograd.grad(
            force_gradient.square().mean(),
            tuple(parameter for _, parameter in named_parameters),
            allow_unused=True,
        )
        for (name, _), gradient in zip(
            named_parameters,
            parameter_gradients,
            strict=True,
        ):
            assert gradient is not None, name
            assert torch.isfinite(gradient).all(), name

    # === Mixed precision ===

    @pytest.mark.skipif(
        env.DEVICE.type != "cuda",
        reason="autocast only engages on CUDA",
    )
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("amp_infer", [False, True])
    @pytest.mark.parametrize("use_amp", [False, True])
    def test_amp_spans_the_edge_stage_and_stops_at_its_boundary(
        self,
        use_amp: bool,
        amp_infer: bool,
        training: bool,
    ) -> None:
        """Autocast must cover the per-edge stage exactly, and only when asked.

        Training follows ``use_amp`` and evaluation follows ``DP_AMP_INFER``;
        the two are independent because mixed precision at inference is a
        throughput choice that must not require the model to have been trained
        with it. Where autocast does engage it has to survive both the radial
        trunk and the mode head, which restore the dtype of their input unless
        the descriptor opts out, and it must not escape into the destination
        reduction or the readout.

        The precision is single because CUDA autocast ignores double operands,
        which would make every assertion below vacuous.
        """
        with mock.patch.dict(
            os.environ,
            {"DP_AMP_INFER": "1" if amp_infer else "0"},
            clear=False,
        ):
            descriptor = self.build(
                channels=32,
                radial_modes=4,
                precision="float32",
                use_amp=use_amp,
            )
        descriptor.train(training)
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )
        graph = dataclasses.replace(
            graph,
            edge_vec=graph.edge_vec.to(torch.float32),
        )

        observed: list[torch.dtype] = []
        handles = [
            layer.register_forward_hook(
                lambda module, args, output: observed.append(output.dtype)
            )
            for layer in (
                *descriptor.radial_embedding.layers,
                descriptor.radial_mode_head,
            )
        ]
        try:
            features = descriptor.build_edge_features(
                graph,
                atype_local,
                descriptor.pair_film.pair_latent(descriptor.type_embedding.call()),
            )[:3]
        finally:
            for handle in handles:
                handle.remove()

        active = use_amp if training else amp_infer
        expected = torch.bfloat16 if active else torch.float32
        assert observed and all(dtype is expected for dtype in observed)
        for feature in features:
            assert feature.dtype == torch.float32
            assert torch.isfinite(feature).all()

    # === Cutoff and regularization ===

    def test_cutoff_is_c3_continuous(self) -> None:
        descriptor = self.build(channels=16, lmax=4).eval()
        inside = self._dimer_derivatives(descriptor, descriptor.rcut - 1.0e-5)
        boundary = self._dimer_derivatives(descriptor, descriptor.rcut)
        outside = self._dimer_derivatives(descriptor, descriptor.rcut + 1.0e-5)
        for order in range(4):
            torch.testing.assert_close(
                boundary[order],
                outside[order],
                atol=1e-14,
                rtol=0.0,
            )
        # The value and its first three derivatives approach the cutoff from
        # inside at the rate set by the p=5 envelope.
        assert inside[1].abs() < 1e-12
        assert inside[2].abs() < 2e-8
        assert inside[3].abs() < 5e-4

    def test_cutoff_edge_matches_removed_topology(self) -> None:
        descriptor = self.build().eval()
        radius = torch.tensor(
            descriptor.rcut,
            dtype=torch.float64,
            device=env.DEVICE,
        )
        torch.testing.assert_close(
            self._dimer_probe(descriptor, radius, active=True),
            self._dimer_probe(descriptor, radius, active=False),
            atol=1e-14,
            rtol=0.0,
        )

    def test_coincident_edge_has_finite_third_derivative(self) -> None:
        descriptor = self.build(lmax=3).eval()
        derivatives = self._dimer_derivatives(descriptor, 0.0)
        for derivative in derivatives:
            assert torch.isfinite(derivative)
        # Direction regularization makes the probe even in the separation, so
        # every odd radial derivative vanishes at coincidence.
        for order in (1, 3):
            torch.testing.assert_close(
                derivatives[order],
                torch.zeros_like(derivatives[order]),
                atol=1e-12,
                rtol=0.0,
            )


class TestDPA4CSpinGate:
    """Torch-side contracts of the spin branch gate."""

    def make_descriptor(self) -> DescrptDPA4C:
        return DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=4,
            precision="float64",
            seed=23,
            use_spin=[True, False],
        ).to(env.DEVICE)

    def test_fresh_descriptor_starts_spin_free(self) -> None:
        assert float(self.make_descriptor().spin.spin_gate.detach()) == 0.0

    def test_closed_gate_still_receives_a_gradient(self) -> None:
        """Zero is a starting point, not a fixed point.

        The invariants are linear in the gate, so its gradient there is the
        branch itself. A factor on the conditioned moment would reach the
        Grams at second and fourth order and could never reopen.
        """
        descriptor = self.make_descriptor()
        generator = torch.Generator(device="cpu").manual_seed(7)
        coord = (
            torch.randn(1, 6, 3, dtype=torch.float64, generator=generator).to(
                env.DEVICE
            )
            * 1.4
        )
        atype = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long, device=env.DEVICE)
        spin = torch.randn(6, 3, dtype=torch.float64, generator=generator).to(
            env.DEVICE
        )
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        graph = build_neighbor_graph(coord, atype, None, 3.0)
        output, _ = descriptor.call_graph(graph, atype.reshape(-1), spin=spin)
        output.sum().backward()
        gradient = descriptor.spin.spin_gate.grad
        assert gradient is not None
        assert float(gradient.abs().max()) > 1e-8

    def test_a_stored_gate_round_trips(self) -> None:
        descriptor = self.make_descriptor()
        with torch.no_grad():
            descriptor.spin.spin_gate.fill_(0.37)
        restored = self.make_descriptor()
        restored.load_state_dict(descriptor.state_dict())
        assert float(restored.spin.spin_gate.detach()) == pytest.approx(0.37)


class TestDPA4CSpin:
    """Torch-side contracts of the native spin branch."""

    def setup_method(self) -> None:
        self.descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=4,
            precision="float64",
            seed=23,
            use_spin=[True, False],
        ).to(env.DEVICE)
        self.descriptor.eval()
        # A fresh descriptor starts spin-free by design; these tests are about
        # the branch behind the gate, which ``TestDPA4CSpinGate`` covers.
        with torch.no_grad():
            self.descriptor.spin.spin_gate.fill_(1.0)
        generator = torch.Generator(device="cpu").manual_seed(5)
        self.coord = (
            torch.randn(1, 6, 3, dtype=torch.float64, generator=generator).to(
                env.DEVICE
            )
            * 1.4
        )
        self.atype = torch.tensor(
            [[0, 1, 0, 1, 0, 1]], dtype=torch.long, device=env.DEVICE
        )
        self.spin = torch.randn(6, 3, dtype=torch.float64, generator=generator).to(
            env.DEVICE
        )
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        self.graph = build_neighbor_graph(self.coord, self.atype, None, 3.0)
        self.flat_atype = self.atype.reshape(-1)

    def probe(self, spin: torch.Tensor) -> torch.Tensor:
        """Reduce the descriptor with fixed weights into a scalar probe."""
        output, _ = self.descriptor.call_graph(self.graph, self.flat_atype, spin=spin)
        weights = torch.arange(
            1, output.shape[1] + 1, dtype=output.dtype, device=output.device
        )
        return (output * weights).sum()

    def test_spin_arrays_are_optimizer_visible(self) -> None:
        names = {name for name, _ in self.descriptor.named_parameters()}
        assert "spin.adam_spin_vector_weight" in names
        assert "spin.adam_spin_quadrupole_weight" in names
        assert "spin.spin_gate" in names
        # The per-type mask and the reference magnitudes are state, not
        # learned quantities.
        buffers = {name for name, _ in self.descriptor.named_buffers()}
        assert {"spin.spin_mask", "spin.spin_reference"} <= buffers

    def test_magnetic_force_matches_finite_differences(self) -> None:
        # The magnetic force is the spin gradient of the energy, so the whole
        # spin branch is validated against numerical differentiation rather
        # than against stored values.
        def descriptor_of_spin(spin: torch.Tensor) -> torch.Tensor:
            return self.descriptor.call_graph(self.graph, self.flat_atype, spin=spin)[0]

        assert torch.autograd.gradcheck(
            descriptor_of_spin,
            (self.spin.clone().requires_grad_(True),),
        )

    def onsite_weights(self) -> list[torch.Tensor]:
        """Return the two per-type on-site spin weights.

        These are the only parameters indexed by atom type that read the
        moment value; the ordered spin tables are shared across types and are
        also weighted by the moment-independent magnetic coordination family.
        """
        return [
            self.descriptor.spin.adam_spin_vector_weight,
            self.descriptor.spin.adam_spin_quadrupole_weight,
        ]

    def test_non_magnetic_types_carry_no_magnetic_degree_of_freedom(self) -> None:
        spin = self.spin.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(self.probe(spin), spin, create_graph=True)
        assert torch.equal(
            gradient[self.flat_atype == 1],
            torch.zeros_like(gradient[self.flat_atype == 1]),
        )
        # The force loss differentiates the magnetic force again, which probes
        # the spin direction even where the value vanishes. A multiplicative
        # gate is what keeps that second derivative exactly zero as well.
        self.descriptor.zero_grad()
        gradient.pow(2).sum().backward()
        for weight in self.onsite_weights():
            assert weight.grad is not None
            assert float(weight.grad[1].abs().max()) == 0.0

    def test_zero_spin_leaves_no_magnetic_force(self) -> None:
        spin = torch.zeros_like(self.spin).requires_grad_(True)
        (gradient,) = torch.autograd.grad(self.probe(spin), spin)
        assert torch.equal(gradient, torch.zeros_like(gradient))
        # Every route that reads the moment value is even in it, so the
        # on-site weights stay dormant in the demagnetized limit.
        self.descriptor.zero_grad()
        self.probe(torch.zeros_like(self.spin)).backward()
        for weight in self.onsite_weights():
            assert weight.grad is None or float(weight.grad.abs().max()) == 0.0

    def test_a_missing_moment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a per-node magnetic"):
            self.descriptor.call_graph(self.graph, self.flat_atype)

    def test_mixed_precision_never_engages(self) -> None:
        # Spin families are quadratic in the moment and feed a fourth-order
        # readout, so the branch stays in compute precision unconditionally.
        for layer in self.descriptor.radial_embedding.layers:
            assert layer.autocast_output is False
        self.descriptor.use_amp = True
        self.descriptor.use_amp_infer = True
        self.descriptor._apply_autocast_policy()
        for layer in self.descriptor.radial_embedding.layers:
            assert layer.autocast_output is False

    def test_compression_covers_the_spin_families(self) -> None:
        """The compiled operator carries spin, so eligibility ignores it.

        Every spin width follows the degree profile rather than a parameter of
        its own, so a spin-conditioned descriptor is covered by the same
        structural set as a spin-free one and its frozen tables are built
        alongside the geometric caches.
        """
        from deepmd.kernels.cuda.dpa4c.graph_compress import (
            mega_eligible,
        )

        single = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=4,
            precision="float32",
            seed=23,
            use_spin=[True, False],
        ).to(env.DEVICE)
        assert mega_eligible(single)
        single.enable_compression(0.5)
        assert single.compress
        assert single.compress_spin_pair.shape == (
            (single.ntypes + 1) ** 2,
            single.spin_channels,
            2,
        )
        assert single.compress_spin_type.shape == (single.ntypes + 1, 4)
