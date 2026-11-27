# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import math
import os
import unittest
from contextlib import (
    nullcontext,
)
from types import (
    SimpleNamespace,
)
from unittest import (
    mock,
)

import torch

from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    C3CutoffEnvelope,
    DynamicRadialDegreeMixer,
    EdgeCartesianTensorProduct,
    ForceEmbedding,
    NodeCartesianTensorProduct,
    RadialBasis,
    RadialMLP,
    SeZMDirectForceHead,
    SO2Linear,
    SpinEmbedding,
    WignerDCalculator,
    build_cartesian_basis,
    build_edge_cartesian_tensors,
    build_edge_quaternion,
    build_gie_zonal_index,
    build_m_major_l_index,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    safe_norm,
    segment_envelope_gated_softmax,
)
from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)


def _random_quaternion(
    n_batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample normalized quaternions in ``(w, x, y, z)`` order."""
    sample_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    q = torch.randn(n_batch, 4, device=device, dtype=sample_dtype)
    q = q / torch.sqrt(
        torch.sum(q * q, dim=-1, keepdim=True).clamp_min(torch.finfo(sample_dtype).eps)
    )
    return q.to(dtype=dtype)


def _tiny_two_atom_system(
    device: torch.device,
    dtype: torch.dtype,
    *,
    padded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a minimal two-atom system for descriptor tests."""
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, -1, 3)
    atype = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    nlist_values = [[[1, -1], [0, -1]]] if padded else [[[1, 1], [0, 0]]]
    nlist = torch.tensor(nlist_values, dtype=torch.int64, device=device)
    return coord, atype, nlist


def _spin_edge_system(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A two-atom edge system with one magnetic and one non-magnetic atom."""
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype, device=device
    ).view(1, -1, 3)
    atype = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
    edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long, device=device)
    edge_vec = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=dtype, device=device
    )
    edge_mask = torch.ones(2, dtype=torch.bool, device=device)
    return coord, atype, edge_index, edge_vec, edge_mask


def _descriptor_kwargs(**overrides) -> dict:
    """Build a compact SeZM descriptor config for tests."""
    kwargs = {
        "rcut": 3.0,
        "sel": [1, 1],
        "ntypes": 2,
        "l_schedule": [1, 0],
        "channels": 4,
        "n_radial": 3,
        "radial_mlp": [6],
        "ffn_neurons": 8,
        "ffn_blocks": 1,
        "random_gamma": False,
        "n_atten_head": 0,
        "mlp_bias": True,
        "precision": "float32",
        "trainable": True,
    }
    kwargs.update(overrides)
    return kwargs


def _perturb_parameters(model: torch.nn.Module, seed: int, scale: float = 0.1) -> None:
    """Move every parameter off its initial value; fresh output projections are zero."""
    generator = torch.Generator(device=env.DEVICE).manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                scale
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
            )


def _attention_descriptor_kwargs(
    *,
    precision: str = "float32",
    seed: int | None = None,
    **overrides,
) -> dict:
    """Build a richer attention-enabled SeZM descriptor config for tests."""
    kwargs = _descriptor_kwargs(
        l_schedule=[1, 1, 0],
        channels=8,
        n_focus=2,
        focus_dim=0,
        n_radial=4,
        radial_mlp=[8],
        so2_layers=2,
        full_attn_res="dependent",
        so2_attn_res="dependent",
        ffn_neurons=16,
        ffn_blocks=2,
        layer_scale=False,
        precision=precision,
        seed=seed,
    )
    kwargs.update(overrides)
    return kwargs


def _forward_tols(dtype: torch.dtype) -> tuple[float, float]:
    """Return output comparison tolerances for one dtype."""
    if dtype == torch.float64:
        return 1e-10, 1e-10
    if dtype == torch.float32:
        return 5e-5, 5e-5
    return 5e-3, 5e-3


def _parameter_tols(dtype: torch.dtype) -> tuple[float, float]:
    """Return parameter comparison tolerances for one dtype."""
    if dtype == torch.float64:
        return 1e-10, 1e-10
    if dtype == torch.float32:
        return 1e-6, 1e-6
    return 1e-3, 1e-3


class _SeZMTestCase(unittest.TestCase):
    """Base test case with the shared device setup."""

    def setUp(self) -> None:
        self.device = env.DEVICE


class TestDescrptSeZM(_SeZMTestCase):
    """Test the SeZM descriptor."""

    def _assert_forward_backward_smoke(self, **model_kwargs) -> DescrptSeZM:
        """Run a compact forward/backward smoke test and return the model."""
        coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=torch.float32)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
        model = DescrptSeZM(**model_kwargs)
        desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
        self.assertEqual(desc.shape, (1, 2, model_kwargs["channels"]))
        self.assertEqual(desc.dtype, env.GLOBAL_PT_FLOAT_PRECISION)
        desc.sum().backward()
        self.assertIsNotNone(extended_coord.grad)
        self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))
        return model

    def test_train_and_eval_amp_switches_are_independent(self) -> None:
        """Training follows ``use_amp``, evaluation follows ``DP_AMP_INFER``.

        Neither switch may leak into the other's mode: mixed precision at
        inference is a throughput choice that must not require the model to
        have been trained with it, and a model trained under AMP must still
        deploy at full precision.
        """
        models = {}
        for amp_infer in (False, True):
            with mock.patch.dict(
                os.environ, {"DP_AMP_INFER": "1" if amp_infer else "0"}, clear=False
            ):
                for use_amp in (False, True):
                    models[amp_infer, use_amp] = DescrptSeZM(
                        **_descriptor_kwargs(use_amp=use_amp)
                    )

        for (amp_infer, use_amp), model in models.items():
            for training in (False, True):
                expected = use_amp if training else amp_infer
                with self.subTest(
                    amp_infer=amp_infer, use_amp=use_amp, training=training
                ):
                    model.train(training)
                    with mock.patch(
                        "torch.autocast", return_value=nullcontext()
                    ) as autocast_mock:
                        with model._compute_mode_ctx(torch.device("cuda")):
                            pass
                    if expected:
                        autocast_mock.assert_called_once_with(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=True,
                        )
                    else:
                        autocast_mock.assert_not_called()

    def test_cartesian_config_wiring(self) -> None:
        """Each Cartesian/mixing config builds the intended submodules.

        Guards against silent fallback: a mis-wired flag would still run and stay
        equivariant, so the structural assertions are the only safeguard that the
        requested path is actually taken.
        """
        # edge_cartesian replaces the SO(2) core and elides the full Wigner-D.
        edge_model = self._assert_forward_backward_smoke(
            **_descriptor_kwargs(
                l_schedule=[2, 1], edge_cartesian=True, channels=4, n_focus=2
            )
        )
        self.assertFalse(edge_model._need_full_wigner)
        for block in edge_model.blocks:
            conv = block.so2_conv
            self.assertTrue(conv.edge_cartesian)
            self.assertTrue(hasattr(conv, "edge_cartesian_tp"))
            self.assertFalse(hasattr(conv, "so2_linears"))

        # node_cartesian adds a per-node product on top of the SO(2) core, leaving
        # the per-edge message path (and the full Wigner-D) intact.
        node_model = self._assert_forward_backward_smoke(
            **_descriptor_kwargs(
                l_schedule=[2, 1], node_cartesian="parity:2", channels=4, n_focus=2
            )
        )
        self.assertTrue(node_model._need_full_wigner)
        for block in node_model.blocks:
            tp = block.so2_conv.node_cartesian_tp
            self.assertIsNotNone(tp)
            self.assertTrue(tp.symmetric)
            self.assertEqual(tp.n_layers, 2)

        # mixing_layers=0 without a degree mixer skips the edge-aligned frame.
        radial_model = self._assert_forward_backward_smoke(
            **_descriptor_kwargs(
                l_schedule=[2, 1],
                mixing_layers=0,
                radial_so2_mode="none",
                channels=4,
                n_focus=2,
            )
        )
        for block in radial_model.blocks:
            conv = block.so2_conv
            self.assertFalse(conv.needs_local_frame)
            self.assertEqual(len(conv.so2_linears), 0)

    def test_cartesian_rotation_invariance(self) -> None:
        """Descriptor scalar output is rotation-invariant across Cartesian modes.

        This end-to-end check covers every per-edge and per-node path introduced
        by the Cartesian options, so it also guards the edge/node engine
        equivariance within the full pipeline.
        """
        dtype = torch.float64
        coord = torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.2, 0.1], [0.1, 1.0, 0.3]],
            dtype=dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor(
            [[[1, 2], [0, 2], [0, 1]]], dtype=torch.int64, device=self.device
        )
        rot = quaternion_to_rotation_matrix(
            _random_quaternion(1, device=self.device, dtype=dtype)
        )[0]
        configs = {
            "edge": {"edge_cartesian": True},
            "node_default": {"node_cartesian": "default:1"},
            "node_parity": {"node_cartesian": "parity:2"},
            "edge_and_node": {"edge_cartesian": True, "node_cartesian": "parity:1"},
            "mixing0_radial": {"mixing_layers": 0, "radial_so2_mode": "none"},
            "mixing0_degree": {"mixing_layers": 0, "radial_so2_mode": "degree_channel"},
        }
        for name, override in configs.items():
            with self.subTest(config=name):
                model = DescrptSeZM(
                    **_descriptor_kwargs(
                        rcut=3.0,
                        sel=[2, 2],
                        l_schedule=[2, 2],
                        channels=4,
                        n_focus=2,
                        n_atten_head=1,
                        precision="float64",
                        use_amp=False,
                        random_gamma=False,
                        **override,
                    )
                )
                model.eval()
                with torch.no_grad():
                    desc, *_ = model(coord.reshape(1, -1), atype, nlist)
                    coord_rot = (rot @ coord.reshape(-1, 3).T).T.reshape(1, -1)
                    desc_rot, *_ = model(coord_rot, atype, nlist)
                torch.testing.assert_close(desc, desc_rot, atol=1e-10, rtol=1e-10)

    def test_so3_readout_empty_edge_shrinking_schedule(self) -> None:
        """so3_readout glu/mlp must handle a frame without edges.

        With a shrinking ``l_schedule`` and no edges (every atom isolated), the
        blocks run on an empty edge set and the readout receives the final node
        degree ``node_ebed_dims[-1]`` (what ``output_ffn`` is built for).
        """
        coord, atype, _ = _tiny_two_atom_system(self.device, dtype=torch.float32)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
        # all neighbors masked out -> an empty edge set
        nlist = torch.full((1, 2, 2), -1, dtype=torch.int64, device=self.device)
        for readout in ("glu", "mlp"):
            with self.subTest(so3_readout=readout):
                model = DescrptSeZM(
                    **_descriptor_kwargs(l_schedule=[2, 1], so3_readout=readout)
                )
                desc, *_ = model(
                    extended_coord, atype, nlist, mapping=None, comm_dict=None
                )
                self.assertEqual(desc.shape, (1, 2, 4))
                self.assertTrue(torch.all(torch.isfinite(desc)))

    def test_edge_free_frame_continues_the_cutoff_limit(self) -> None:
        """A frame without edges is the limit of a frame whose last edge leaves the cutoff."""
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        empty_nlist = torch.full((1, 2, 2), -1, dtype=torch.int64, device=self.device)
        pair_nlist = torch.tensor(
            [[[1, -1], [0, -1]]], dtype=torch.int64, device=self.device
        )

        def descriptor(distance: float, nlist: torch.Tensor) -> torch.Tensor:
            coord = torch.tensor(
                [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
                dtype=torch.float64,
                device=self.device,
            ).reshape(1, -1)
            return model(coord, atype, nlist, mapping=None, comm_dict=None)[0]

        for options in (
            {},
            {"node_wise_s2": True},
            {"node_wise_so3": True},
            {"s2_activation": [True, False]},
        ):
            with self.subTest(options=options):
                model = DescrptSeZM(
                    **_descriptor_kwargs(precision="float64", seed=5, **options)
                )
                model = model.to(self.device).eval()
                _perturb_parameters(model, seed=5)
                isolated = descriptor(10.0, empty_nlist)
                torch.testing.assert_close(
                    descriptor(model.rcut - 1e-6, pair_nlist),
                    isolated,
                    rtol=0.0,
                    atol=1e-12,
                )
                self.assertFalse(
                    torch.allclose(descriptor(model.rcut - 0.5, pair_nlist), isolated)
                )

    def test_so3_readout_scalar_path_matches_full_output(self) -> None:
        """The scalar-specialized final FFN matches slicing its full output."""
        dtype = torch.float64
        for readout in ("glu", "mlp"):
            with self.subTest(so3_readout=readout):
                descriptor = DescrptSeZM(
                    **_descriptor_kwargs(
                        l_schedule=[2, 2],
                        kmax=1,
                        so3_readout=readout,
                        precision="float64",
                        use_amp=False,
                        seed=19,
                    )
                )
                ffn = descriptor.output_ffn
                generator = torch.Generator(device=self.device).manual_seed(23)
                with torch.no_grad():
                    for parameter in ffn.parameters():
                        parameter.add_(
                            torch.randn(
                                parameter.shape,
                                dtype=parameter.dtype,
                                device=parameter.device,
                                generator=generator,
                            )
                            * 0.1
                        )

                shape = (3, descriptor.node_readout_dim, 1, descriptor.channels)
                full_input = torch.randn(
                    shape,
                    dtype=dtype,
                    device=self.device,
                    generator=generator,
                    requires_grad=True,
                )
                scalar_input = full_input.detach().clone().requires_grad_(True)
                probe = torch.randn(
                    (3, 1, 1, descriptor.channels),
                    dtype=dtype,
                    device=self.device,
                    generator=generator,
                )

                full = ffn(full_input)[:, 0:1, :, :]
                scalar = ffn.forward_scalar(scalar_input)
                torch.testing.assert_close(full, scalar, atol=1e-12, rtol=1e-12)

                parameters = tuple(ffn.parameters())
                full_grads = torch.autograd.grad(
                    torch.sum(full * probe),
                    (full_input, *parameters),
                    allow_unused=True,
                    create_graph=True,
                )
                scalar_grads = torch.autograd.grad(
                    torch.sum(scalar * probe),
                    (scalar_input, *parameters),
                    allow_unused=True,
                    create_graph=True,
                )
                for full_grad, scalar_grad in zip(
                    full_grads, scalar_grads, strict=True
                ):
                    self.assertEqual(full_grad is None, scalar_grad is None)
                    if full_grad is not None:
                        torch.testing.assert_close(
                            full_grad,
                            scalar_grad,
                            atol=1e-12,
                            rtol=1e-12,
                        )

                tangents = tuple(
                    None
                    if grad is None
                    else torch.randn(
                        grad.shape,
                        dtype=grad.dtype,
                        device=grad.device,
                        generator=generator,
                    )
                    for grad in full_grads
                )
                full_grad_probe = sum(
                    torch.sum(grad * tangent)
                    for grad, tangent in zip(full_grads, tangents, strict=True)
                    if grad is not None and grad.requires_grad
                )
                scalar_grad_probe = sum(
                    torch.sum(grad * tangent)
                    for grad, tangent in zip(scalar_grads, tangents, strict=True)
                    if grad is not None and grad.requires_grad
                )
                full_second_grads = torch.autograd.grad(
                    full_grad_probe,
                    (full_input, *parameters),
                    allow_unused=True,
                )
                scalar_second_grads = torch.autograd.grad(
                    scalar_grad_probe,
                    (scalar_input, *parameters),
                    allow_unused=True,
                )
                for full_grad, scalar_grad in zip(
                    full_second_grads,
                    scalar_second_grads,
                    strict=True,
                ):
                    self.assertEqual(full_grad is None, scalar_grad is None)
                    if full_grad is not None:
                        torch.testing.assert_close(
                            full_grad,
                            scalar_grad,
                            atol=1e-12,
                            rtol=1e-12,
                        )

    def test_zero_block_descriptor(self) -> None:
        """``n_blocks=0`` builds the interaction-free descriptor end to end.

        Covers the schedule collapse (empty schedules, backbone degrees falling
        back to ``lmax`` plus ``extra_node_l``, no blocks), both forward entry
        points, the conservative-force path, and serialization. The
        coordinate-carrying paths (FiLM, GIE, read-out) are zero-initialized, so
        a parameter perturbation activates them before the force check;
        ``so3_readout`` glu/mlp fold the GIE ``l>0`` geometry into the scalar,
        while ``none`` keeps only the env-seed scalar.
        """
        dtype = torch.float32
        coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
        atol, rtol = _forward_tols(dtype)
        for readout, extra in (("mlp", 0), ("glu", 0), ("none", 0), ("mlp", 1)):
            with self.subTest(so3_readout=readout, extra_node_l=extra):
                model = DescrptSeZM(
                    **_descriptor_kwargs(
                        l_schedule=None,
                        n_blocks=0,
                        lmax=2,
                        mmax=1,
                        kmax=1,
                        extra_node_l=extra,
                        use_env_seed=True,
                        so3_readout=readout,
                    )
                )
                # Empty schedules; backbone degrees collapse onto lmax(+extra).
                self.assertEqual(model.n_blocks, 0)
                self.assertEqual(model.l_schedule, [])
                self.assertEqual(model.m_schedule, [])
                self.assertEqual(len(model.blocks), 0)
                self.assertEqual(model.node_init_lmax, 2 + extra)
                self.assertEqual(model.node_readout_lmax, 2 + extra)
                self.assertTrue(model.use_gie)

                # Activate the zero-initialized geometry paths so the force check
                # exercises genuine coordinate dependence rather than the
                # all-zero gradient seen at initialization.
                torch.manual_seed(0)
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.2)

                extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
                desc, *_ = model(extended_coord, atype, nlist)
                self.assertEqual(desc.shape, (1, 2, 4))
                self.assertTrue(torch.all(torch.isfinite(desc)))
                desc.sum().backward()
                self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))
                self.assertGreater(extended_coord.grad.abs().max().item(), 0.0)

                # Serialization restores the empty schedule and reproduces output.
                data = model.serialize()
                self.assertEqual(data["config"]["n_blocks"], 0)
                self.assertEqual(data["config"]["l_schedule"], [])
                restored = DescrptSeZM.deserialize(data)
                desc2, *_ = restored(coord.reshape(1, -1), atype, nlist)
                torch.testing.assert_close(desc.detach(), desc2, atol=atol, rtol=rtol)

                # Sparse-edge entry point: the latent keeps the initial degree
                # ``(node_init_lmax + 1) ** 2`` since no pyramid shrinks it.
                flat = coord.reshape(-1, 3)
                edge_index = torch.tensor(
                    [[1, 0], [0, 1]], dtype=torch.long, device=self.device
                )
                edge_vec = flat[edge_index[0]] - flat[edge_index[1]]
                edge_mask = torch.ones(2, dtype=torch.bool, device=self.device)
                desc_e, latent, _ = model.forward_with_edges(
                    extended_coord=coord.reshape(1, -1),
                    extended_atype=atype,
                    edge_index=edge_index,
                    edge_vec=edge_vec,
                    edge_mask=edge_mask,
                )
                self.assertEqual(desc_e.shape, (1, 2, 4))
                self.assertEqual(latent.shape, (2, (2 + extra + 1) ** 2, 1, 4))

    def test_zero_block_without_env_seed_is_geometry_free(self) -> None:
        """Without env-seed the zero-block descriptor loses all geometry.

        With no blocks, no env FiLM, and no GIE, the scalar output depends only
        on the type embedding and is independent of the coordinates. This
        documents that ``use_env_seed=True`` is required for a meaningful
        zero-block descriptor, since the single ``use_env_seed`` switch gates the
        only geometry source (GIE).
        """
        coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=torch.float32)
        model = DescrptSeZM(
            **_descriptor_kwargs(
                l_schedule=None,
                n_blocks=0,
                lmax=2,
                use_env_seed=False,
                so3_readout="mlp",
            )
        )
        self.assertFalse(model.use_gie)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
        desc, *_ = model(extended_coord, atype, nlist)
        self.assertTrue(torch.all(torch.isfinite(desc)))
        # The descriptor does not depend on coordinates, so there is no force path.
        (grad,) = torch.autograd.grad(desc.sum(), extended_coord, allow_unused=True)
        self.assertIsNone(grad)

    def test_forward_with_descriptor_variants(self) -> None:
        """Test forward/backward smoke paths for compact descriptor variants."""
        cases = {
            "focus_dim_zero": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
            ),
            "focus_dim_explicit": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=3,
                so2_layers=2,
            ),
            "focus_dim_zero_s2": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                s2_activation=[False, True],
            ),
            "focus_dim_zero_s2_lebedev": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                s2_activation=[False, True],
                lebedev_quadrature=[False, True],
            ),
            "message_node_s2": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                message_node_s2=True,
            ),
            "gaussian_basis": _descriptor_kwargs(
                channels=4,
                basis_type="gaussian",
            ),
            "extra_node_l": _descriptor_kwargs(
                channels=4,
                extra_node_l=1,
                s2_activation=[False, True],
            ),
            "radial_so2_degree": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                radial_so2_mode="degree",
            ),
            "radial_so2_degree_channel_rank2": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                radial_so2_mode="degree_channel",
                radial_so2_rank=2,
            ),
        }
        for name, model_kwargs in cases.items():
            with self.subTest(mode=name):
                self._assert_forward_backward_smoke(**model_kwargs)

    def test_fixed_basis_types_freeze_the_basis(self) -> None:
        """``/fix`` basis types keep the basis parameters out of training only."""
        for family in ("bessel", "gaussian"):
            with self.subTest(family=family):
                torch.manual_seed(7)
                free = self._assert_forward_backward_smoke(
                    **_descriptor_kwargs(channels=4, basis_type=family)
                )
                torch.manual_seed(7)
                fixed = self._assert_forward_backward_smoke(
                    **_descriptor_kwargs(channels=4, basis_type=f"{family}/fix")
                )
                self.assertEqual(fixed.radial_basis.basis_type, f"{family}/fix")
                self.assertEqual(fixed.radial_basis.basis_family, family)
                self.assertFalse(fixed.radial_basis.adam_freqs.requires_grad)
                self.assertTrue(free.radial_basis.adam_freqs.requires_grad)
                torch.testing.assert_close(
                    fixed.radial_basis.adam_freqs, free.radial_basis.adam_freqs
                )
                # Every other parameter is unaffected by the suffix.
                frozen = [
                    name for name, p in fixed.named_parameters() if not p.requires_grad
                ]
                self.assertEqual(frozen, ["radial_basis.adam_freqs"])
                # The same weights give the same descriptor, and the suffix
                # survives a serialization round trip.
                fixed.load_state_dict(free.state_dict())
                coord, atype, nlist = _tiny_two_atom_system(
                    self.device, dtype=torch.float32
                )
                extended_coord = coord.reshape(1, -1)
                torch.testing.assert_close(
                    fixed(extended_coord, atype, nlist, mapping=None, comm_dict=None)[
                        0
                    ],
                    free(extended_coord, atype, nlist, mapping=None, comm_dict=None)[0],
                )
                self.assertEqual(
                    fixed.serialize()["config"]["basis_type"], f"{family}/fix"
                )
        with self.assertRaises(ValueError):
            DescrptSeZM(**_descriptor_kwargs(channels=4, basis_type="gaussian-frozen"))

    def test_forward_with_attention_variants(self) -> None:
        """Test forward/backward smoke paths for attention-based variants."""
        cases = {
            "full_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
            ),
            "block_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                full_attn_res="none",
                block_attn_res="dependent",
            ),
            "full_attention_s2": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                s2_activation=[False, True],
                extra_node_l=1,
            ),
            "mixed_so2_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                n_atten_head=4,
                atten_f_mix=True,
            ),
            "standard_single_head_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                n_atten_head=1,
                atten_v_proj=True,
                atten_o_proj=True,
            ),
        }
        for name, model_kwargs in cases.items():
            with self.subTest(mode=name):
                self._assert_forward_backward_smoke(**model_kwargs)

    def test_forward_backward_second_order_fixed_edges(self) -> None:
        """Test fixed-shape edge path matches nlist for fwd/bwd/2nd order."""
        dtype = torch.float32
        coord = torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 0.7, 0.2]],
            dtype=dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int64, device=self.device)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)

        model = DescrptSeZM(
            **_attention_descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=1,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
            )
        )

        desc_nlist, _, _, _, sw_nlist = model(
            extended_coord, atype, nlist, mapping=None, comm_dict=None
        )

        # Fixed-shape edge list for n_node=2, nsel=2
        edge_index = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 1]],
            dtype=torch.long,
            device=self.device,
        )
        coord_view = extended_coord.view(1, 2, 3)
        valid_nlist = nlist >= 0
        gather_index = torch.where(valid_nlist, nlist, torch.zeros_like(nlist))
        index = gather_index.view(1, 4, 1).expand(-1, -1, 3)
        nei_pos = torch.gather(coord_view, 1, index).view(1, 2, 2, 3)
        atom_pos = coord_view[:, :2].unsqueeze(2)
        diff = nei_pos - atom_pos
        edge_vec = diff.reshape(4, 3)
        edge_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool, device=self.device)

        desc_edge, _, _, _, sw_edge = model(
            extended_coord,
            atype,
            nlist,
            mapping=None,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            comm_dict=None,
        )

        torch.testing.assert_close(desc_nlist, desc_edge, atol=1e-6, rtol=1e-6)

        loss_nlist = desc_nlist.sum()
        loss_edge = desc_edge.sum()

        (grad_nlist,) = torch.autograd.grad(
            loss_nlist, extended_coord, create_graph=True
        )
        (grad_edge,) = torch.autograd.grad(loss_edge, extended_coord, create_graph=True)
        torch.testing.assert_close(grad_nlist, grad_edge, atol=1e-6, rtol=1e-6)

        (grad2_nlist,) = torch.autograd.grad(
            grad_nlist.sum(), extended_coord, create_graph=False
        )
        (grad2_edge,) = torch.autograd.grad(
            grad_edge.sum(), extended_coord, create_graph=False
        )
        torch.testing.assert_close(grad2_nlist, grad2_edge, atol=1e-6, rtol=1e-6)

    def test_force_embedding_and_vector_head_roundtrip_l1_basis(self) -> None:
        """Force encoding basis should match the vector-head Cartesian decode."""
        force_input = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, -3.0],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        encoder = ForceEmbedding(
            lmax=1,
            channels=1,
            precision="float64",
            mlp_bias=False,
            trainable=True,
            seed=123,
        ).to(self.device)
        decoder = SeZMDirectForceHead(
            lmax=1,
            channels=1,
            precision="float64",
            mlp_bias=False,
            trainable=True,
            seed=456,
        ).to(self.device)

        with torch.no_grad():
            encoder.proj.weight.zero_()
            encoder.proj.weight[1, 0, 0] = 1.0
            if encoder.proj.bias is not None:
                encoder.proj.bias.zero_()
            decoder.proj.weight.zero_()
            decoder.proj.weight[1, 0, 0] = math.sqrt(3.0)
            if decoder.proj.bias is not None:
                decoder.proj.bias.zero_()

        latent = encoder(force_input)
        decoded = decoder(latent)
        torch.testing.assert_close(decoded, force_input, atol=1e-10, rtol=1e-10)

    def test_backward_gradient(self) -> None:
        """Test backward gradient through coordinates."""
        for prec in ["float64", "float32", "bfloat16"]:
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
            model = DescrptSeZM(
                **_descriptor_kwargs(
                    ffn_blocks=2,
                    layer_scale=False,
                    precision=prec,
                )
            )
            desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
            loss = desc.sum()
            loss.backward()
            self.assertIsNotNone(extended_coord.grad)
            self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))

    def test_serialization_deserialization(self) -> None:
        """Test serialization and deserialization preserves model state."""
        cases = {
            "focus_dim_zero": _attention_descriptor_kwargs(
                precision="float32",
                focus_dim=0,
            ),
            "focus_dim_explicit": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=3,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
            ),
            "focus_dim_zero_s2": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                s2_activation=[False, True],
            ),
            "focus_dim_zero_s2_lebedev": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                s2_activation=[False, True],
                lebedev_quadrature=[False, True],
            ),
            "message_node_s2": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                message_node_s2=True,
            ),
            "extra_node_l": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                extra_node_l=1,
                s2_activation=[False, True],
            ),
            "radial_so2_degree": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                radial_so2_mode="degree",
            ),
            "radial_so2_degree_channel_rank2": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                radial_so2_mode="degree_channel",
                radial_so2_rank=2,
            ),
            "cartesian": _descriptor_kwargs(
                precision="float32",
                l_schedule=[2, 1],
                edge_cartesian=True,
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
            ),
            "native_spin": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                use_spin=[True, False],
                use_env_seed=True,
            ),
        }
        dtype = PRECISION_DICT["float32"]
        for case_name, model_kwargs in cases.items():
            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1)
            with self.subTest(mode=case_name):
                model = DescrptSeZM(**model_kwargs)

                desc1, _, _, _, sw1 = model(extended_coord, atype, nlist)
                data = model.serialize()
                model_restored = DescrptSeZM.deserialize(data)
                desc2, _, _, _, sw2 = model_restored(extended_coord, atype, nlist)
                atol, rtol = _forward_tols(dtype)

                torch.testing.assert_close(
                    desc1,
                    desc2,
                    atol=atol,
                    rtol=rtol,
                    msg="Descriptor mismatch after deserialization",
                )
                torch.testing.assert_close(
                    sw1,
                    sw2,
                    atol=atol,
                    rtol=rtol,
                    msg="Smooth weight mismatch after deserialization",
                )

    def test_charge_spin_sparse_edge_conditioning(self) -> None:
        """Charge/spin conditions should affect the sparse-edge descriptor path."""
        coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=torch.float32)
        extended_coord = coord.reshape(1, -1)
        edge_index = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.long, device=self.device
        )
        edge_vec = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        edge_mask = torch.ones(2, dtype=torch.bool, device=self.device)
        model = DescrptSeZM(
            **_descriptor_kwargs(
                add_chg_spin_ebd=True,
                default_chg_spin=[0.0, 1.0],
                seed=123,
            )
        )

        desc_default, *_ = model(extended_coord, atype, nlist)
        desc_explicit, *_ = model(
            extended_coord,
            atype,
            nlist,
            charge_spin=torch.tensor([[0.0, 1.0]], device=self.device),
        )
        desc_ref, _, _ = model.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            charge_spin=torch.tensor([[0.0, 1.0]], device=self.device),
        )
        desc_shifted, _, _ = model.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            charge_spin=torch.tensor([[1.0, 1.0]], device=self.device),
        )
        restored = DescrptSeZM.deserialize(model.serialize())
        desc_restored, _, _ = restored.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            charge_spin=torch.tensor([[0.0, 1.0]], device=self.device),
        )

        self.assertTrue(restored.add_chg_spin_ebd)
        self.assertEqual(restored.get_default_chg_spin(), [0.0, 1.0])
        torch.testing.assert_close(desc_default, desc_explicit, atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(desc_ref, desc_shifted))
        torch.testing.assert_close(desc_ref, desc_restored, atol=1e-6, rtol=1e-6)

    def test_seed_reproducibility(self) -> None:
        """Test that fixed seed produces identical model initialization."""
        for prec in ["float64", "float32", "bfloat16"]:
            dtype = PRECISION_DICT[prec]
            seed = 12345

            model_kwargs = _attention_descriptor_kwargs(precision=prec, seed=seed)
            model1 = DescrptSeZM(**model_kwargs)
            model2 = DescrptSeZM(**model_kwargs)
            param_atol, param_rtol = _parameter_tols(dtype)

            for (n1, p1), (n2, p2) in zip(
                model1.named_parameters(), model2.named_parameters(), strict=False
            ):
                self.assertEqual(n1, n2, msg="Parameter name mismatch")
                torch.testing.assert_close(
                    p1,
                    p2,
                    atol=param_atol,
                    rtol=param_rtol,
                    msg=f"Parameter {n1} differs between models with same seed",
                )

            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            desc1, _, _, _, sw1 = model1(extended_coord, atype, nlist)
            desc2, _, _, _, sw2 = model2(extended_coord, atype, nlist)
            forward_atol, forward_rtol = _forward_tols(dtype)

            torch.testing.assert_close(
                desc1,
                desc2,
                atol=forward_atol,
                rtol=forward_rtol,
                msg="Forward output differs for models with same seed",
            )
            torch.testing.assert_close(
                sw1,
                sw2,
                atol=forward_atol,
                rtol=forward_rtol,
                msg="Smooth weight differs for models with same seed",
            )

    def test_shared_spin_descriptors_preserve_the_branch_scheme(self) -> None:
        config = _descriptor_kwargs(use_env_seed=True, use_spin=[True, False])
        orbit = DescrptSeZM(**config, spin_mode="orbit")
        exchange = DescrptSeZM(**config, spin_mode="exchange")
        original = exchange.type_embedding
        with self.assertRaisesRegex(ValueError, "spin modes"):
            exchange.share_params(orbit, 0)
        self.assertIs(exchange.type_embedding, original)
        exchange.share_params(orbit, 1)
        self.assertIs(exchange.type_embedding, orbit.type_embedding)
        self.assertIsNot(exchange.spin_embedding, orbit.spin_embedding)

    def test_shared_descriptor_binds_fused_radial_to_shared_basis(self) -> None:
        """Fused radial inference of a linked branch reads the shared basis."""
        from deepmd.pt_expt.kernels.cuda.dpa4 import (
            edge_radial,
        )

        if self.device.type != "cuda" or not edge_radial.op_available():
            self.skipTest("DPA4 CUDA radial operators are unavailable")
        levels = {
            "DP_CUDA_INFER": "1",
            "DP_TRITON_INFER": "0",
            "DP_CUTILE_INFER": "0",
            "DP_CUTE_INFER": "0",
        }
        with mock.patch.dict(os.environ, levels):
            base = DescrptSeZM(**_descriptor_kwargs()).to(self.device)
            link = DescrptSeZM(**_descriptor_kwargs()).to(self.device)
        self.assertIsNotNone(link._cuda_radial_fn)
        with torch.no_grad():
            base.radial_basis.adam_freqs += 0.5
        link.share_params(base, 0)
        self.assertIs(link.radial_basis, base.radial_basis)
        edge_len = torch.linspace(0.5, 2.9, 16, device=self.device).reshape(-1, 1)
        keep = torch.ones_like(edge_len)
        shared = link._cuda_radial_fn(edge_len, keep)
        reference = base._cuda_radial_fn(edge_len, keep)
        for value, expected in zip(shared, reference, strict=True):
            torch.testing.assert_close(value, expected)

    def test_exportable_backend_options_are_refused(self) -> None:
        """Options of the exportable backend load at their defaults and fail otherwise.

        The refusal comes before the neighbor statistics as well, which read
        ``rcut`` before the descriptor is built.
        """
        defaults = {"density_floor": None, "readout_norm": False, "parity": False}
        DescrptSeZM(**_descriptor_kwargs(**defaults))
        for options in (
            {"density_floor": 8.0},
            {"readout_norm": True},
            {"readout_kmax": 1},
            {"parity": True},
            {"rcut": [2.0, 3.0]},
            {"mixing_layers": [1, 2]},
            {"moe": {}},
        ):
            config = _descriptor_kwargs(**options)
            for build in (
                lambda: DescrptSeZM(**config),
                lambda: DescrptSeZM.update_sel(None, None, config),
            ):
                with (
                    self.subTest(options=options),
                    self.assertRaisesRegex(NotImplementedError, "pt_expt backend only"),
                ):
                    build()


class TestSeZMSpinEmbedding(_SeZMTestCase):
    """Test the native per-atom spin embedding and its descriptor injection."""

    def test_time_even_pair_features_and_signed_exchange(self) -> None:
        """Global reversal preserves the features; a local reversal retains exchange sign."""
        dtype = torch.float64
        atype = torch.tensor([0, 0, 1], device=self.device)
        spin = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=dtype,
            device=self.device,
        )
        edge = SimpleNamespace(
            src=torch.tensor([1, 2], device=self.device),
            dst=torch.tensor([0, 2], device=self.device),
            edge_vec=torch.tensor(
                [[1.0, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=dtype, device=self.device
            ),
            edge_env=torch.ones(2, 1, dtype=dtype, device=self.device),
        )
        module = SpinEmbedding(
            ntypes=2,
            channels=4,
            n_radial=3,
            use_spin=[True, False],
            spin_mode="exchange",
            dtype=dtype,
            seed=2,
        ).to(self.device)
        with torch.no_grad():
            module.pair_layer.matrix.copy_(
                torch.eye(3, dtype=dtype, device=self.device)
            )
        pair = module.edge_features(spin, atype, edge)
        torch.testing.assert_close(pair[0], spin.new_tensor([1.0, 4.0, 2.0]))
        torch.testing.assert_close(pair[1], torch.zeros_like(pair[1]))
        torch.testing.assert_close(module.edge_features(-spin, atype, edge), pair)
        flipped = spin.clone()
        flipped[1] *= -1
        torch.testing.assert_close(
            module.edge_features(flipped, atype, edge)[0],
            spin.new_tensor([1.0, 4.0, -2.0]),
        )
        zero = torch.zeros_like(spin, requires_grad=True)
        (grad,) = torch.autograd.grad(
            module.edge_features(zero, atype, edge).sum(), zero
        )
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0.0, rtol=0.0)

    def test_descriptor_spin_joint_rotation_invariance(self) -> None:
        """The scalar descriptor is invariant under a joint rotation of geometry and spin.

        Covers on-site, pair and neighbor spin conditioning.
        """
        dtype = torch.float64
        coord, atype, edge_index, edge_vec, edge_mask = _spin_edge_system(
            self.device, dtype
        )
        spin = torch.zeros(1, 2, 3, dtype=dtype, device=self.device)
        spin[0, 0] = torch.tensor([0.3, -0.7, 0.5], dtype=dtype, device=self.device)
        quat = _random_quaternion(1, device=self.device, dtype=dtype)
        rot = quaternion_to_rotation_matrix(quat)[0]  # (3, 3)

        model = DescrptSeZM(
            **_descriptor_kwargs(
                precision="float64",
                use_spin=[True, False],
                use_env_seed=True,
                seed=7,
            )
        )
        # Perturb away from the near-identity initialization so the
        # rotation check exercises a non-trivial spin-dependent
        # descriptor (env-seed output_proj is otherwise zero-init).
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        model.eval()

        desc, _, _ = model.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            spin=spin,
        )
        desc_rot, _, _ = model.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=torch.einsum("ij,ej->ei", rot, edge_vec),
            edge_mask=edge_mask,
            spin=torch.einsum("ij,nkj->nki", rot, spin),
        )
        torch.testing.assert_close(desc, desc_rot, atol=1e-9, rtol=1e-9)

        # Spin actually changes the descriptor (injection is not a no-op).
        desc_zero, _, _ = model.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            spin=torch.zeros_like(spin),
        )
        self.assertFalse(torch.allclose(desc, desc_zero, atol=1e-6))


class TestSeZMEnvSeedSpinProjection(_SeZMTestCase):
    """The independent spin projection has a trainable zero initialization."""

    def _kwargs(self) -> dict:
        """Config of a spin descriptor with the env-seed route enabled."""
        return _descriptor_kwargs(
            precision="float64",
            use_spin=[True, False],
            use_env_seed=True,
            seed=7,
        )

    def _descriptor(self) -> DescrptSeZM:
        """A spin descriptor with non-route weights moved off initialization.

        The env-seed output projection is zero-initialized, so an unperturbed
        descriptor is insensitive to the environment matrix. The
        spin output projections stay at zero so the tests exercise the
        initialization used by a fresh native-spin model.
        """
        model = DescrptSeZM(**self._kwargs())
        spin_routes = {
            "spin_embedding.mag_layer2.matrix",
            "spin_embedding.pair_layer.matrix",
            "env_seed_embedding.spin_output_proj.matrix",
        }
        torch.manual_seed(0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in spin_routes:
                    parameter.copy_(torch.randn_like(parameter) * 0.1)
        model.eval()
        return model

    def _inputs(self) -> tuple[dict, torch.Tensor]:
        """Edge-route keyword arguments and a spin on the magnetic atom."""
        dtype = torch.float64
        coord, atype, edge_index, edge_vec, edge_mask = _spin_edge_system(
            self.device, dtype
        )
        spin = torch.zeros(1, 2, 3, dtype=dtype, device=self.device)
        spin[0, 0] = torch.tensor([0.3, -0.7, 0.5], dtype=dtype, device=self.device)
        return {
            "extended_coord": coord,
            "extended_atype": atype,
            "edge_index": edge_index,
            "edge_vec": edge_vec,
            "edge_mask": edge_mask,
        }, spin

    def test_zero_projection_receives_a_gradient(self) -> None:
        """The zero projection receives gradients from nonzero spin statistics."""
        model = self._descriptor()
        kwargs, spin = self._inputs()
        desc, _, _ = model.forward_with_edges(**kwargs, spin=spin)
        desc.sum().backward()
        projection_grad = model.env_seed_embedding.spin_output_proj.matrix.grad
        self.assertIsNotNone(projection_grad)
        self.assertGreater(float(projection_grad.abs().max()), 1e-8)


class TestBuildEdgeQuaternion(_SeZMTestCase):
    """Test the stable edge-quaternion chart used by SeZM."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 1e-4, 1e-4
        return 5e-3, 5e-3

    def _safe_norm(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True).clamp(min=eps))

    def _assert_quaternion_invariants(
        self, edge_quat: torch.Tensor, edge_vec: torch.Tensor
    ) -> None:
        atol, rtol = self._get_tols(edge_vec.dtype)
        rot_mat = quaternion_to_rotation_matrix(edge_quat)
        n_edge = int(edge_vec.shape[0])
        eye = torch.eye(3, device=self.device, dtype=edge_vec.dtype).expand(
            n_edge, 3, 3
        )
        torch.testing.assert_close(
            rot_mat @ rot_mat.transpose(-1, -2),
            eye,
            atol=atol,
            rtol=rtol,
        )

        edge_unit = edge_vec / self._safe_norm(edge_vec)
        ez = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=edge_vec.dtype
        ).expand(n_edge, 3)
        rotated = (rot_mat @ edge_unit.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(rotated, ez, atol=atol, rtol=rtol)

        det_mat = rot_mat.float() if rot_mat.dtype == torch.bfloat16 else rot_mat
        det = torch.linalg.det(det_mat)
        torch.testing.assert_close(
            det,
            torch.ones_like(det),
            atol=atol,
            rtol=rtol,
        )

    def test_invariants_random_edges(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            edge_vec = torch.randn(512, 3, device=self.device, dtype=dtype)
            edge_quat = build_edge_quaternion(edge_vec)
            self._assert_quaternion_invariants(edge_quat, edge_vec)

    def test_invariants_near_poles(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            delta = torch.tensor(
                [-1.0e-3, -1.0e-4, 0.0, 1.0e-4, 1.0e-3],
                device=self.device,
                dtype=dtype,
            )
            for sign in [1.0, -1.0]:
                edge_vec = torch.stack(
                    [delta, torch.zeros_like(delta), torch.full_like(delta, sign)],
                    dim=-1,
                )
                edge_quat = build_edge_quaternion(edge_vec)
                self._assert_quaternion_invariants(edge_quat, edge_vec)


class TestWignerDCalculator(_SeZMTestCase):
    """Test the quaternion-driven Wigner-D calculator."""

    def setUp(self) -> None:
        super().setUp()
        self.batch = 8
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 5e-5, 5e-5
        return 5e-3, 5e-3

    def _extract_l_block(
        self,
        D_full: torch.Tensor,
        l: int,
    ) -> torch.Tensor:
        """Extract the l-block from D_full."""
        s, e = l * l, (l + 1) * (l + 1)
        return D_full[:, s:e, s:e]

    def test_orthogonality(self) -> None:
        """Test D @ D^T = I for random quaternions."""
        for dtype, lmax in itertools.product(
            [torch.float64, torch.float32], [1, 3, 5, 6, 8, 10]
        ):
            atol, rtol = self._get_tols(dtype)
            wigner = WignerDCalculator(lmax=lmax, dtype=dtype).to(self.device)
            edge_quat = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            D_full, Dt_full = wigner(edge_quat)

            for l in range(lmax + 1):
                dim = 2 * l + 1
                eye = torch.eye(dim, device=self.device, dtype=dtype).expand(
                    self.batch, dim, dim
                )
                D_l = self._extract_l_block(D_full, l)
                Dt_l = self._extract_l_block(Dt_full, l)
                torch.testing.assert_close(
                    D_l @ Dt_l,
                    eye,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Orthogonality failed for WignerDCalculator, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_group_property(self) -> None:
        """Test group property in quaternion composition order."""
        for dtype, lmax in itertools.product(
            [torch.float64, torch.float32], [1, 3, 5, 6, 8, 10]
        ):
            atol = 1e-10 if dtype == torch.float64 else 5e-4
            rtol = 1e-10 if dtype == torch.float64 else 5e-4
            wigner = WignerDCalculator(lmax=lmax, dtype=dtype).to(self.device)

            q1 = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            q2 = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            q12 = quaternion_multiply(q1, q2)

            D1_full, _ = wigner(q1)
            D2_full, _ = wigner(q2)
            D12_full, _ = wigner(q12)

            for l in range(lmax + 1):
                D1_l = self._extract_l_block(D1_full, l)
                D2_l = self._extract_l_block(D2_full, l)
                D12_l = self._extract_l_block(D12_full, l)
                torch.testing.assert_close(
                    D12_l,
                    D1_l @ D2_l,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Group property failed for WignerDCalculator, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_l1_matches_vector_representation(self) -> None:
        """Test that the l=1 block matches the Cartesian vector representation."""
        for dtype in [torch.float64, torch.float32]:
            atol, rtol = self._get_tols(dtype)
            S = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            S_batch = S.unsqueeze(0).expand(self.batch, 3, 3)

            edge_quat = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            rot = quaternion_to_rotation_matrix(edge_quat)
            wigner = WignerDCalculator(lmax=1, dtype=dtype).to(self.device)
            D_full, Dt_full = wigner(edge_quat)
            D1 = self._extract_l_block(D_full, 1)
            Dt1 = self._extract_l_block(Dt_full, 1)

            expected = S_batch @ rot @ S_batch.transpose(-1, -2)
            torch.testing.assert_close(
                D1,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"l=1 block mismatch for WignerDCalculator, dtype={dtype}",
            )
            torch.testing.assert_close(
                Dt1,
                expected.transpose(-1, -2),
                atol=atol,
                rtol=rtol,
                msg=f"l=1 transpose block mismatch for WignerDCalculator, dtype={dtype}",
            )

    def test_zonal_matches_full_wigner_gather(self) -> None:
        """Test GIE zonal coupling matches the full Wigner-D gather."""
        for dtype, lmax in itertools.product(
            [torch.float64, torch.float32], [0, 1, 3, 5, 6, 8, 10]
        ):
            atol, rtol = self._get_tols(dtype)
            wigner = WignerDCalculator(lmax=lmax, dtype=dtype).to(self.device)
            if lmax < 11:
                self.assertFalse(hasattr(wigner, "poly_coeffs"))
            edge_quat = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            _, dt_full = wigner(edge_quat)
            node_row_index, node_zonal_m0_col_index, _ = build_gie_zonal_index(
                lmax,
                device=self.device,
            )
            expected = dt_full[:, node_row_index, node_zonal_m0_col_index]
            actual = wigner.forward_zonal(edge_quat)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            if lmax >= 2:
                lmin = 2
                suffix_start = lmin * lmin - 1
                actual_suffix = wigner.forward_zonal(edge_quat, lmin=lmin)
                torch.testing.assert_close(
                    actual_suffix,
                    expected[:, suffix_start:],
                    atol=atol,
                    rtol=rtol,
                )

    def test_special_blocks_match_generic_reference_on_cpu(self) -> None:
        """Test specialized l=2..10 Wigner blocks against the generic reference."""
        device = torch.device("cpu")
        dtype = torch.float64
        edge_quat = _random_quaternion(4, device=device, dtype=dtype)
        wigner = WignerDCalculator(lmax=10, dtype=dtype).to(device)
        d_full, _ = wigner(edge_quat)
        reference = WignerDCalculator._compute_generic_reference_blocks(
            edge_quat,
            lmax=10,
            dtype=dtype,
            device=device,
        )
        for degree in range(2, 11):
            torch.testing.assert_close(
                self._extract_l_block(d_full, degree),
                reference[degree],
                atol=1.0e-12,
                rtol=1.0e-10,
                msg=f"Special Wigner block mismatch for l={degree}",
            )

    def test_pole_path_gradient_matches_finite_difference(self) -> None:
        """Check one pole-crossing Wigner probe against finite differences."""
        for dtype in [torch.float64, torch.float32]:
            wigner = WignerDCalculator(lmax=6, dtype=dtype).to(self.device)
            atol = 5.0e-8 if dtype == torch.float64 else 2.0e-6
            rtol = 1.0e-6 if dtype == torch.float64 else 2.0e-4
            for sign in [1.0, -1.0]:
                delta = torch.linspace(
                    -0.02,
                    0.02,
                    257,
                    device=self.device,
                    dtype=dtype,
                    requires_grad=True,
                )
                edge_vec = torch.stack(
                    [delta, torch.zeros_like(delta), torch.full_like(delta, sign)],
                    dim=-1,
                )
                edge_quat = build_edge_quaternion(edge_vec)
                D_full, _ = wigner(edge_quat)
                probe = D_full[:, 5, 7] + D_full[:, 17, 19]
                grad = torch.autograd.grad(probe.sum(), delta)[0]
                delta_detached = delta.detach()
                probe_detached = probe.detach()
                numerical_grad = (probe_detached[2:] - probe_detached[:-2]) / (
                    2.0 * (delta_detached[1] - delta_detached[0])
                )
                torch.testing.assert_close(
                    grad[1:-1].detach(),
                    numerical_grad,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Pole-path Wigner gradient mismatch for dtype={dtype}, sign={sign}"
                    ),
                )

    def test_y_crossing_overlap_has_no_large_wigner_jump(self) -> None:
        """Check chart-overlap continuity for a path that crosses y=0."""
        for dtype in [torch.float64, torch.float32]:
            wigner = WignerDCalculator(lmax=4, dtype=dtype).to(self.device)
            max_allowed = 1.0e-2 if dtype == torch.float64 else 1.5e-2
            y_vals = torch.tensor(
                [-1.0e-3, -5.0e-4, -1.0e-4, 0.0, 1.0e-4, 5.0e-4, 1.0e-3],
                device=self.device,
                dtype=dtype,
            )
            edge_vec = torch.stack(
                [
                    torch.full_like(y_vals, 0.35),
                    y_vals,
                    torch.full_like(y_vals, 0.25),
                ],
                dim=-1,
            )
            edge_quat = build_edge_quaternion(edge_vec)
            D_full, _ = wigner(edge_quat)
            step = (D_full[1:] - D_full[:-1]).abs().amax(dim=(1, 2))
            self.assertLess(
                step.max().item(),
                max_allowed,
                msg=f"Large Wigner jump across y=0 for dtype={dtype}",
            )


class TestSO2LinearEquivariance(_SeZMTestCase):
    """Test SO2Linear z-rotation equivariance: SO2Linear(Z @ x) = Z @ SO2Linear(x)."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 1e-5, 1e-5
        # bf16 has only 7-bit mantissa; use looser tolerance for equivariance tests.
        return 2e-2, 2e-2

    def _build_m_major_z_rotation(
        self, angles: torch.Tensor, lmax: int, mmax: int
    ) -> torch.Tensor:
        """
        Build block z-rotation matrix for the m-major truncated layout.

        Parameters
        ----------
        angles
            Rotation angles with shape (batch,).
        lmax
            Maximum degree.
        mmax
            Maximum order (|m|). Must satisfy mmax <= lmax.

        Returns
        -------
        torch.Tensor
            Z matrix with shape (batch, dim_red, dim_red).
        """
        batch = angles.shape[0]
        m0_size = lmax + 1
        dim_red = m0_size
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            dim_red += 2 * num_l

        Z = angles.new_zeros(batch, dim_red, dim_red)
        eye0 = torch.eye(m0_size, device=self.device, dtype=angles.dtype).expand(
            batch, m0_size, m0_size
        )
        Z[:, :m0_size, :m0_size] = eye0

        offset = m0_size
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            eye = torch.eye(num_l, device=self.device, dtype=angles.dtype).expand(
                batch, num_l, num_l
            )
            cos_m = torch.cos(m * angles).view(batch, 1, 1)
            sin_m = torch.sin(m * angles).view(batch, 1, 1)

            # In m-major layout, each m group is stored as [neg(l), pos(l)] with two halves.
            # Rotation is [[cos I, -sin I], [sin I, cos I]] for the (neg, pos) pair.
            Z[:, offset : offset + num_l, offset : offset + num_l] = cos_m * eye
            Z[
                :,
                offset : offset + num_l,
                offset + num_l : offset + 2 * num_l,
            ] = -sin_m * eye
            Z[
                :,
                offset + num_l : offset + 2 * num_l,
                offset : offset + num_l,
            ] = sin_m * eye
            Z[
                :,
                offset + num_l : offset + 2 * num_l,
                offset + num_l : offset + 2 * num_l,
            ] = cos_m * eye
            offset += 2 * num_l

        return Z

    def test_equivariance_random_angles(self) -> None:
        """Test SO2Linear(Z @ x) = Z @ SO2Linear(x) for random z-rotations."""
        for dtype, lmax, mmax in itertools.product(
            [torch.float64, torch.float32, torch.bfloat16],
            [1, 2, 3],
            [1, 2, 3],
        ):
            if mmax > lmax:
                continue
            atol, rtol = self._get_tols(dtype)
            batch = 16
            channels_in = 8
            channels_out = 12

            so2_linear = SO2Linear(
                lmax=lmax,
                mmax=mmax,
                in_channels=channels_in,
                out_channels=channels_out,
                dtype=dtype,
                seed=123,
                trainable=True,
            )

            dim_red = so2_linear.reduced_dim
            # SO2Linear consumes the focus-major ``(F, E, D_m, C)`` contract, so
            # the edge axis (the per-edge z-rotation batch) is axis 1.
            x = torch.randn(
                1, batch, dim_red, channels_in, device=self.device, dtype=dtype
            )

            angles = torch.rand(batch, device=self.device, dtype=dtype) * 2 * 3.14159
            Z = self._build_m_major_z_rotation(angles, lmax, mmax)

            x_rotated = torch.einsum("eij,fejc->feic", Z, x)
            lhs = so2_linear(x_rotated)
            rhs = torch.einsum("eij,fejc->feic", Z, so2_linear(x))

            torch.testing.assert_close(
                lhs,
                rhs,
                atol=atol,
                rtol=rtol,
                msg=f"SO2Linear equivariance failed for dtype={dtype}, lmax={lmax}, mmax={mmax}",
            )

    def test_dynamic_radial_degree_mixer_equivariance(self) -> None:
        """Dynamic radial degree mixer should commute with local z-rotations."""
        for mode, rank in (
            ("degree", 0),
            ("degree_channel", 0),
            ("degree_channel", 2),
        ):
            with self.subTest(mode=mode, rank=rank):
                dtype = torch.float32
                lmax = 2
                mmax = 2
                batch = 6
                channels = 5
                mixer = DynamicRadialDegreeMixer(
                    lmax=lmax,
                    mmax=mmax,
                    channels=channels,
                    mode=mode,
                    rank=rank,
                    dtype=dtype,
                    seed=123,
                    trainable=True,
                )
                degree_index_m = build_m_major_l_index(lmax, mmax, device=self.device)
                dim_red = int(degree_index_m.numel())
                x = torch.randn(
                    batch, dim_red, channels, device=self.device, dtype=dtype
                )
                radial_base = torch.randn(
                    batch, lmax + 1, channels, device=self.device, dtype=dtype
                )
                radial_feat = radial_base[:, degree_index_m, :]
                angles = (
                    torch.rand(batch, device=self.device, dtype=dtype) * 2 * math.pi
                )
                Z = self._build_m_major_z_rotation(angles, lmax, mmax)

                x_rotated = torch.einsum("bij,bjc->bic", Z, x)
                lhs = mixer(x_rotated, radial_feat)
                rhs = torch.einsum("bij,bjc->bic", Z, mixer(x, radial_feat))

                torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


class TestCartesianTensorProduct(_SeZMTestCase):
    """Test the Cartesian rank-2 tensor-product building blocks."""

    def test_basis_intertwines_wigner(self) -> None:
        """cart(D @ sh) == R cart(sh) R^T links Wigner-D to Cartesian rotation.

        Guards the hand-written sign/ordering convention of
        ``build_cartesian_basis``: the change of basis must intertwine the packed
        Wigner-D rotation with the Cartesian conjugation ``X -> R X R^T``.
        """
        for lmax in (1, 2):
            dim = (lmax + 1) ** 2
            basis = build_cartesian_basis(lmax, dtype=torch.float64, device=self.device)
            wigner = WignerDCalculator(lmax=lmax, dtype=torch.float64).to(self.device)
            quat = _random_quaternion(6, device=self.device, dtype=torch.float64)
            d_full, _ = wigner(quat)
            rot = quaternion_to_rotation_matrix(quat)
            sh = torch.randn(6, dim, dtype=torch.float64, device=self.device)
            cart = torch.einsum("bd,dij->bij", sh, basis)
            lhs = torch.einsum(
                "bd,dij->bij", torch.einsum("bde,be->bd", d_full, sh), basis
            )
            rhs = rot @ cart @ rot.transpose(-1, -2)
            torch.testing.assert_close(lhs, rhs, atol=1e-9, rtol=1e-9)

    def test_edge_cartesian_matches_dense_reference(self) -> None:
        """Channel-shared edge evaluation equals the naive per-channel ``Y @ T_e``.

        Covers the ``mixing_layers > 0`` stack and the ``mixing_layers == 0``
        single-modulation path; equivariance of the edge path is covered
        end-to-end by ``test_cartesian_rotation_invariance``.
        """
        for lmax in (1, 2):
            for n_layers in (0, 3):
                dim = (lmax + 1) ** 2
                n_focus, focus_dim, n_edge = 2, 4, 16
                width = n_focus * focus_dim
                engine = EdgeCartesianTensorProduct(
                    lmax=lmax,
                    focus_dim=focus_dim,
                    n_focus=n_focus,
                    n_layers=n_layers,
                    activation_function="silu",
                    mlp_bias=True,
                    eps=1e-7,
                    dtype=torch.float64,
                    seed=7,
                    trainable=True,
                ).to(self.device)
                x = torch.randn(
                    n_edge, dim, width, dtype=torch.float64, device=self.device
                )
                edge_vec = torch.randn(
                    n_edge, 3, dtype=torch.float64, device=self.device
                )
                rad = torch.randn(
                    n_edge, lmax + 1, width, dtype=torch.float64, device=self.device
                )
                out = engine(x, edge_vec, rad)
                ref = self._dense_cartesian_reference(engine, x, edge_vec, rad)
                torch.testing.assert_close(out, ref, atol=1e-11, rtol=1e-11)

    @staticmethod
    def _dense_cartesian_reference(
        engine: EdgeCartesianTensorProduct,
        x: torch.Tensor,
        edge_vec: torch.Tensor,
        rad: torch.Tensor,
    ) -> torch.Tensor:
        """Reference Cartesian product via explicit per-channel ``Y @ T_e``."""
        n_edge = x.shape[0]
        f, cf = engine.n_focus, engine.focus_dim
        basis = build_cartesian_basis(engine.lmax, dtype=x.dtype, device=x.device)
        r_hat = edge_vec / safe_norm(edge_vec, engine.eps)
        a0, s0 = build_edge_cartesian_tensors(r_hat)
        eye = torch.eye(3, dtype=x.dtype, device=x.device) / math.sqrt(3.0)
        a0 = a0 / math.sqrt(2.0)
        f_iso = rad[:, 0, :].reshape(n_edge, f, cf, 1, 1)
        f_aniso = rad[:, 1, :].reshape(n_edge, f, cf, 1, 1)
        t_e = f_iso * eye + f_aniso * a0[:, None, None, :, :]
        if engine.lmax == 2:
            s0 = s0 / math.sqrt(2.0 / 3.0)
            f_sym = rad[:, 2, :].reshape(n_edge, f, cf, 1, 1)
            t_e = t_e + f_sym * s0[:, None, None, :, :]

        def modulate(coeff: torch.Tensor) -> torch.Tensor:
            cart = torch.einsum("edfc,dij->efcij", coeff, basis)
            return torch.einsum("efcij,dij->edfc", torch.matmul(cart, t_e), basis)

        h = x.reshape(n_edge, engine.ebed_dim, f, cf)
        if engine.n_layers == 0:
            # Single modulation ``x @ T_e`` with no learnable channel-mixing layer.
            return modulate(h).reshape(n_edge, engine.ebed_dim, f * cf)
        for linear, activation in zip(engine.linears, engine.activations, strict=True):
            h = h + activation(modulate(linear(h)))
        return h.reshape(n_edge, engine.ebed_dim, f * cf)

    def test_node_engine_rotation_equivariance(self) -> None:
        """NodeCartesianTensorProduct commutes with a global Wigner-D rotation."""
        for lmax in (1, 2):
            for symmetric in (False, True):
                dim = (lmax + 1) ** 2
                n_focus, focus_dim, n_node = 2, 4, 16
                width = n_focus * focus_dim
                engine = NodeCartesianTensorProduct(
                    lmax=lmax,
                    focus_dim=focus_dim,
                    n_focus=n_focus,
                    n_layers=3,
                    symmetric=symmetric,
                    activation_function="silu",
                    mlp_bias=True,
                    dtype=torch.float64,
                    seed=7,
                    trainable=True,
                ).to(self.device)
                message = torch.randn(
                    n_node, dim, width, dtype=torch.float64, device=self.device
                )
                node = torch.randn(
                    n_node, dim, width, dtype=torch.float64, device=self.device
                )
                wigner = WignerDCalculator(lmax=lmax, dtype=torch.float64).to(
                    self.device
                )
                quat = _random_quaternion(1, device=self.device, dtype=torch.float64)
                d_mat = wigner(quat)[0][0]  # (D, D), one shared global rotation
                out = engine(message, node)
                out_rot = engine(
                    torch.einsum("ij,njc->nic", d_mat, message),
                    torch.einsum("ij,njc->nic", d_mat, node),
                )
                torch.testing.assert_close(
                    out_rot,
                    torch.einsum("ij,njc->nic", d_mat, out),
                    atol=1e-9,
                    rtol=1e-9,
                )


class TestCutoffNumerics(_SeZMTestCase):
    """Numerical stability at cutoff-vanishing attention boundaries."""

    def test_high_logit_edge_vanishes_continuously(self) -> None:
        """A zero-weight high-logit edge must leave the segment continuously."""
        logits = torch.tensor(
            [[[0.0]], [[20.0]]], dtype=torch.float64, device=self.device
        )
        dst = torch.zeros(2, dtype=torch.int64, device=self.device)
        z_bias_raw = torch.tensor(
            [[math.log(math.expm1(1.0))]], dtype=torch.float64, device=self.device
        )
        eps = 1.0e-7

        def evaluate(crossing_envelope: float) -> torch.Tensor:
            edge_env = torch.tensor(
                [[1.0], [crossing_envelope]],
                dtype=torch.float64,
                device=self.device,
            )
            return segment_envelope_gated_softmax(
                logits, edge_env, dst, 1, z_bias_raw, eps
            )

        near = evaluate(1.0e-12)
        zero = evaluate(0.0)
        near_mass = math.exp(20.0) * 1.0e-24
        near_denominator = 2.0 + eps + near_mass
        torch.testing.assert_close(
            near[1, 0, 0],
            torch.tensor(
                near_mass / near_denominator,
                dtype=torch.float64,
                device=self.device,
            ),
            rtol=1.0e-12,
            atol=0.0,
        )
        expected_stable = 1.0 / (2.0 + eps)
        torch.testing.assert_close(
            near[0, 0, 0],
            torch.tensor(expected_stable, dtype=torch.float64, device=self.device),
        )
        torch.testing.assert_close(
            zero[0, 0, 0],
            torch.tensor(expected_stable, dtype=torch.float64, device=self.device),
        )
        self.assertEqual(float(zero[1, 0, 0]), 0.0)

    def test_envelope_nextafter_cutoff_attention(self) -> None:
        """Adjacent float32 distances must not create a spurious attention edge."""
        envelope_fn = C3CutoffEnvelope(6.0, exponent=5, dtype=torch.float32)
        rcut = torch.tensor(6.0, dtype=torch.float32, device=self.device)
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        r_near = torch.nextafter(rcut, zero)
        r_inner = torch.nextafter(r_near, zero)
        distances = torch.stack([r_near, r_inner])[:, None]
        envelope = envelope_fn(distances)

        distance64 = distances[:, 0].to(torch.float64)
        u = (6.0 - distance64) / 6.0
        x = 1.0 - u
        reference = u**4 * (1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x))))
        torch.testing.assert_close(
            envelope[:, 0].to(torch.float64),
            reference,
            rtol=1.0e-6,
            atol=0.0,
        )
        self.assertTrue(bool((envelope >= 0.0).all()))

        logits = torch.tensor(
            [[[0.0]], [[20.0]]], dtype=torch.float32, device=self.device
        )
        dst = torch.zeros(2, dtype=torch.int64, device=self.device)
        z_bias_raw = torch.tensor(
            [[math.log(math.expm1(1.0))]], dtype=torch.float32, device=self.device
        )
        for edge_envelope in envelope[:, 0]:
            edge_env = torch.stack([torch.ones_like(edge_envelope), edge_envelope])[
                :, None
            ]
            alpha = segment_envelope_gated_softmax(
                logits, edge_env, dst, 1, z_bias_raw, 1.0e-7
            )
            self.assertLess(float(alpha[1, 0, 0]), 1.0e-30)


class TestEdgeNorm(_SeZMTestCase):
    """The ``edge_norm`` switch and its effect on cutoff smoothness.

    The descriptor exposes a single ``edge_norm`` flag; internally it drives the
    ``RadialMLP.radial_norm`` hidden RMSNorm, the FiLM scale/shift norms, and the
    cross-focus competition norm, and selects the post-SO(2) residual scaling
    floor. The RadialMLP-level tests exercise the radial mechanism directly; the
    descriptor test checks the umbrella propagation.
    """

    def setUp(self) -> None:
        super().setUp()
        self.dtype = torch.float64
        self.rcut = 6.0

    def _radial_feature_curve(self, *, radial_norm: bool, seed: int) -> torch.Tensor:
        """Radial features over a near-cutoff distance sweep.

        ``RadialBasis`` bakes in the C^3 envelope, so ``edge_rbf`` vanishes at
        ``rcut``. With ``radial_norm=True`` the hidden RMSNorm divides that
        envelope out and its ``eps`` floor is crossed near ``rcut``, injecting a
        localized curvature spike; ``radial_norm=False`` drops the RMSNorm so the
        feature stays smooth. Both variants share the same linear weights (same
        ``seed``), isolating the effect of the norm.

        Parameters
        ----------
        radial_norm : bool
            Whether the RadialMLP keeps its hidden RMSNorm.
        seed : int
            Seed shared by both variants so the linear weights match.

        Returns
        -------
        torch.Tensor
            Radial features with shape (N, out_dim) over the distance sweep.
        """
        basis = RadialBasis(rcut=self.rcut, n_radial=8, exponent=7, dtype=self.dtype)
        mlp = RadialMLP(
            [8, 12, 8], radial_norm=radial_norm, dtype=self.dtype, seed=seed
        )
        r = torch.linspace(
            0.5 * self.rcut,
            0.9995 * self.rcut,
            4000,
            dtype=self.dtype,
            device=self.device,
        ).view(-1, 1)
        with torch.no_grad():
            return mlp(basis(r))

    @staticmethod
    def _peak_curvature(feat: torch.Tensor) -> float:
        """Peak absolute second finite difference of the feature L2 norm."""
        y = feat.norm(dim=1)
        return float((y[2:] - 2.0 * y[1:-1] + y[:-2]).abs().max())

    def test_radial_norm_false_removes_cutoff_curvature_spike(self) -> None:
        """``radial_norm=False`` suppresses the eps-crossing kink near ``rcut``."""
        feat_norm = self._radial_feature_curve(radial_norm=True, seed=3)
        feat_smooth = self._radial_feature_curve(radial_norm=False, seed=3)
        # Both vanish at rcut: edge_rbf -> 0 and RadialMLP(0) = 0 (bias=False).
        self.assertLess(feat_smooth[-1].abs().max().item(), 1.0e-8)
        # The normalized variant floor-crosses just inside rcut; dropping the
        # RMSNorm removes that localized curvature spike by a wide margin.
        self.assertLess(
            self._peak_curvature(feat_smooth) * 5.0,
            self._peak_curvature(feat_norm),
        )

    def test_radial_norm_structure_and_serialization(self) -> None:
        """The flag toggles the hidden RMSNorm and round-trips through serialize."""
        for radial_norm in (True, False):
            with self.subTest(radial_norm=radial_norm):
                mlp = RadialMLP(
                    [8, 12, 8], radial_norm=radial_norm, dtype=self.dtype, seed=5
                )
                has_norm = any(type(m).__name__ == "RMSNorm" for m in mlp.net)
                self.assertEqual(has_norm, radial_norm)

                restored = RadialMLP.deserialize(mlp.serialize())
                self.assertEqual(restored.radial_norm, radial_norm)
                x = torch.rand(16, 8, dtype=self.dtype, device=self.device)
                with torch.no_grad():
                    torch.testing.assert_close(mlp(x), restored(x))

    def test_edge_norm_gates_all_cutoff_vanishing_norms(self) -> None:
        """``edge_norm`` controls every cutoff-vanishing normalization path.

        A bool switches the radial, FiLM and focus norms together; a list of
        three bools ``[radial, film, focus]`` switches them individually. The
        post-SO(2) residual scaling follows the radial switch.
        """
        cases = [
            (True, (True, True, True)),
            (False, (False, False, False)),
            ([False, True, False], (False, True, False)),
            ([True, False, True], (True, False, True)),
        ]
        for edge_norm, (radial_on, film_on, focus_on) in cases:
            with self.subTest(edge_norm=edge_norm):
                desc = DescrptSeZM(
                    **_descriptor_kwargs(
                        edge_norm=edge_norm,
                        use_env_seed=True,
                        n_focus=2,
                        sandwich_norm=[True, True, True, True],
                        precision="float64",
                    )
                )
                # radial MLP hidden RMSNorm
                radial_has_norm = any(
                    type(m).__name__ == "RMSNorm" for m in desc.radial_embedding.net
                )
                self.assertEqual(radial_has_norm, radial_on)
                # env-seed FiLM scale/shift norms
                self.assertEqual(
                    type(desc.film_scale_norm).__name__ == "ScalarRMSNorm", film_on
                )
                self.assertEqual(
                    type(desc.film_shift_norm).__name__ == "ScalarRMSNorm", film_on
                )
                # cross-focus competition norm (n_focus>1 -> competition active)
                focus_norm_mod = desc.blocks[0].so2_conv.focus_compete_norm
                self.assertEqual(
                    type(focus_norm_mod).__name__ == "ScalarRMSNorm", focus_on
                )
                # Only the post-SO(2) residual branch uses unit-floor scaling,
                # bound to the radial switch.
                expected_eps = 1.0e-5 if radial_on else 1.0
                self.assertEqual(desc.blocks[0].post_so2_norm.eps, expected_eps)
                self.assertEqual(desc.blocks[0].pre_so2_norm.eps, 1.0e-5)
                self.assertEqual(desc.blocks[0].pre_ffn_norms[0].eps, 1.0e-5)
                self.assertEqual(desc.blocks[0].post_ffn_norms[0].eps, 1.0e-5)
                canonical = [radial_on, film_on, focus_on]
                self.assertEqual(desc.serialize()["config"]["edge_norm"], canonical)
                restored = DescrptSeZM.deserialize(desc.serialize())
                self.assertEqual(
                    [
                        restored.radial_norm,
                        restored.film_norm,
                        restored.focus_norm,
                    ],
                    canonical,
                )

    def test_edge_norm_legacy_checkpoint_formats(self) -> None:
        """Serialized data from older checkpoints loads unchanged.

        The oldest checkpoints predate the ``edge_norm`` option and carry no
        such config key (the norms were always built, matching the default
        ``True``); intermediate checkpoints store a plain bool. Both must
        deserialize into the same module structure and reproduce the source
        model's output exactly.
        """
        dtype = PRECISION_DICT["float64"]
        cases = [
            ("missing_key", True, None, (True, True, True)),
            ("bool_true", True, True, (True, True, True)),
            ("bool_false", False, False, (False, False, False)),
        ]
        for case_name, build_edge_norm, stored_edge_norm, expected in cases:
            with self.subTest(case=case_name):
                model = DescrptSeZM(
                    **_descriptor_kwargs(
                        edge_norm=build_edge_norm,
                        use_env_seed=True,
                        n_focus=2,
                        precision="float64",
                    )
                )
                data = model.serialize()
                if stored_edge_norm is None:
                    data["config"].pop("edge_norm")
                else:
                    data["config"]["edge_norm"] = stored_edge_norm
                restored = DescrptSeZM.deserialize(data)
                self.assertEqual(
                    (
                        restored.radial_norm,
                        restored.film_norm,
                        restored.focus_norm,
                    ),
                    expected,
                )
                coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
                extended_coord = coord.reshape(1, -1)
                desc1, _, _, _, sw1 = model(extended_coord, atype, nlist)
                desc2, _, _, _, sw2 = restored(extended_coord, atype, nlist)
                atol, rtol = _forward_tols(dtype)
                torch.testing.assert_close(desc1, desc2, atol=atol, rtol=rtol)
                torch.testing.assert_close(sw1, sw2, atol=atol, rtol=rtol)


class _EnergyCurveFixture(_SeZMTestCase):
    """Test PES smoothness from scaled symmetric eight-atom probes."""

    RANDOM_WEIGHT_BASE_SEED = 184
    RANDOM_WEIGHT_STD = 0.1
    N_DISPLACEMENT_POINTS = 201
    MAX_DISPLACEMENT = 0.1
    # Probe a bit inside the cutoff (rcut=5.0) rather than right at the edge.
    # The smooth descriptor correctly makes near-rcut contributions vanish as
    # ~(rcut - r)^4, so the energy bowl is only resolvable above noise a little
    # inside the cutoff; 4.5 keeps a robust smooth-bowl signal for this guard.
    RCUT_NEAR_DISTANCE = 4.5
    ENERGY_SPAN_MARGIN = 1.0e-7
    FIRST_DERIVATIVE_MARGIN = 5.0e-7
    SECOND_DERIVATIVE_MARGIN = 5.0e-4
    EXTREMUM_MARGIN = 1.0e-9

    def setUp(self) -> None:
        super().setUp()
        self.dtype = torch.float64
        self.symmetry_frac_coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=self.dtype,
            device=self.device,
        ).view(1, 8, 3)
        self.symmetry_atype = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1]],
            dtype=torch.int32,
            device=self.device,
        )

    def _build_model_params(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int = 1,
        edge_cartesian: bool = False,
        bridging_method: str = "none",
    ) -> dict:
        """Build the SeZM probe model used for one PES scan."""
        params = {
            "type": "SeZM",
            "type_map": ["Na", "Cl"],
            "descriptor": {
                "type": "SeZM",
                "sel": [16, 16],
                "rcut": 5.0,
                "channels": 16,
                "n_focus": n_focus,
                "focus_dim": 0,
                "n_radial": 6,
                "radial_mlp": [16],
                "use_env_seed": True,
                "edge_cartesian": edge_cartesian,
                "l_schedule": [2, 1] if edge_cartesian else [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": n_atten_head,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 16,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": False,
                "use_amp": use_amp,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [],
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
        }
        if bridging_method.lower() != "none":
            params["bridging_method"] = bridging_method
        return params

    def _build_scaled_symmetric_structure(
        self,
        nearest_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale one symmetric eight-atom template to a target nearest-neighbor distance."""
        lattice = 2.0 * nearest_distance
        coord = self.symmetry_frac_coord * lattice
        box = torch.tensor(
            [[lattice, 0.0, 0.0, 0.0, lattice, 0.0, 0.0, 0.0, lattice]],
            dtype=self.dtype,
            device=self.device,
        )
        return coord, box

    def _stable_parameter_seed(self, name: str) -> int:
        """Return an order-independent seed derived from one parameter name."""
        seed_value = self.RANDOM_WEIGHT_BASE_SEED
        for idx, char in enumerate(name):
            seed_value += (idx + 1) * ord(char)
        return seed_value

    def _build_random_weight_model(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int = 1,
        edge_cartesian: bool = False,
        bridging_method: str = "none",
    ) -> torch.nn.Module:
        """Build a probe model and randomize all floating-point parameters."""
        model = get_sezm_model(
            self._build_model_params(
                n_atten_head,
                use_amp=use_amp,
                n_focus=n_focus,
                edge_cartesian=edge_cartesian,
                bridging_method=bridging_method,
            )
        ).to(
            device=self.device,
            dtype=self.dtype,
        )
        randomized = 0

        # Assign deterministic random weights by parameter name.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.is_floating_point():
                    continue
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self._stable_parameter_seed(name))
                param.normal_(
                    mean=0.0,
                    std=self.RANDOM_WEIGHT_STD,
                    generator=generator,
                )
                randomized += 1

        self.assertGreater(
            randomized, 0, "No floating-point parameters were randomized"
        )
        model.eval()
        return model


class TestDescriptorEnergyCurveSmoothness(_EnergyCurveFixture):
    def _scan_total_energy_curve(
        self,
        model: torch.nn.Module,
        *,
        nearest_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan total energies while displacing atom 0 along x."""
        displacements = torch.linspace(
            -self.MAX_DISPLACEMENT,
            self.MAX_DISPLACEMENT,
            self.N_DISPLACEMENT_POINTS,
            dtype=self.dtype,
            device=self.device,
        )
        coord0, box = self._build_scaled_symmetric_structure(nearest_distance)
        coord = coord0.repeat(displacements.shape[0], 1, 1)
        coord[:, 0, 0] += displacements
        result = model(
            coord,
            self.symmetry_atype.expand(displacements.shape[0], -1),
            box=box.expand(displacements.shape[0], -1),
        )
        return displacements, result["energy"][:, 0].detach()

    def _collect_curve_statistics(
        self,
        energies: torch.Tensor,
        displacements: torch.Tensor,
    ) -> dict[str, float | str]:
        """Compute derivative and extremum statistics for one energy curve."""
        step = displacements[1] - displacements[0]
        first = (energies[2:] - energies[:-2]) / (2.0 * step)
        second = (energies[2:] - 2.0 * energies[1:-1] + energies[:-2]) / (step * step)
        center_idx = energies.shape[0] // 2
        deriv_center_idx = first.shape[0] // 2
        left_first = first[:deriv_center_idx]
        right_first = first[deriv_center_idx + 1 :]
        center_energy = energies[center_idx]
        other_energies = torch.cat((energies[:center_idx], energies[center_idx + 1 :]))

        bowl_up = bool(
            torch.all(left_first < -self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(right_first > self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(second > self.SECOND_DERIVATIVE_MARGIN)
            and torch.all(center_energy <= other_energies + self.EXTREMUM_MARGIN)
        )
        bowl_down = bool(
            torch.all(left_first > self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(right_first < -self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(second < -self.SECOND_DERIVATIVE_MARGIN)
            and torch.all(center_energy >= other_energies - self.EXTREMUM_MARGIN)
        )

        if bowl_up:
            curve_kind = "minimum"
        elif bowl_down:
            curve_kind = "maximum"
        else:
            curve_kind = "invalid"

        return {
            "curve_kind": curve_kind,
            "energy_span": float((energies.max() - energies.min()).abs()),
            "left_abs_min": float(left_first.abs().min()),
            "right_abs_min": float(right_first.abs().min()),
            "curvature_abs_min": float(second.abs().min()),
        }

    def _assert_curve_has_usable_signal(
        self,
        stats: dict[str, float | str],
        *,
        label: str,
        n_atten_head: int,
    ) -> None:
        """Check that the scanned curve has enough signal above numerical noise."""
        self.assertGreater(
            stats["energy_span"],
            self.ENERGY_SPAN_MARGIN,
            f"{label} energy curve became nearly flat for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["left_abs_min"],
            self.FIRST_DERIVATIVE_MARGIN,
            f"{label} left-branch slope is too small for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["right_abs_min"],
            self.FIRST_DERIVATIVE_MARGIN,
            f"{label} right-branch slope is too small for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["curvature_abs_min"],
            self.SECOND_DERIVATIVE_MARGIN,
            f"{label} curvature is too small for n_atten_head={n_atten_head}: {stats}",
        )

    def _assert_cutoff_near_energy_curve_is_smooth(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int,
        edge_cartesian: bool = False,
    ) -> None:
        """Check that the non-bridged near-cutoff probe keeps one smooth extremum."""
        model = self._build_random_weight_model(
            n_atten_head,
            use_amp=use_amp,
            n_focus=n_focus,
            edge_cartesian=edge_cartesian,
        )
        displacements, energies = self._scan_total_energy_curve(
            model,
            nearest_distance=self.RCUT_NEAR_DISTANCE,
        )
        self.assertTrue(torch.isfinite(energies).all().item())

        stats = self._collect_curve_statistics(energies, displacements)
        self._assert_curve_has_usable_signal(
            stats,
            label=f"Near-cutoff (use_amp={use_amp}, n_focus={n_focus})",
            n_atten_head=n_atten_head,
        )
        self.assertIn(
            stats["curve_kind"],
            {"minimum", "maximum"},
            (
                "Near-cutoff energy curve is not a single smooth bowl "
                f"for n_atten_head={n_atten_head}, use_amp={use_amp}, n_focus={n_focus}: {stats}"
            ),
        )

    def test_scaled_cutoff_near_energy_curve_is_smooth_across_attention_modes(
        self,
    ) -> None:
        """Check the non-bridged near-cutoff PES shape across attention and AMP modes."""
        for use_amp in (False, True):
            for n_atten_head in (0, 1, 2):
                for n_focus in (1, 2):
                    with self.subTest(
                        n_atten_head=n_atten_head, use_amp=use_amp, n_focus=n_focus
                    ):
                        self._assert_cutoff_near_energy_curve_is_smooth(
                            n_atten_head,
                            use_amp=use_amp,
                            n_focus=n_focus,
                        )

    def test_cartesian_near_cutoff_energy_curve_is_smooth(self) -> None:
        """The Cartesian path keeps a single smooth near-cutoff PES extremum."""
        for n_atten_head in (0, 1):
            for n_focus in (1, 2):
                with self.subTest(n_atten_head=n_atten_head, n_focus=n_focus):
                    self._assert_cutoff_near_energy_curve_is_smooth(
                        n_atten_head,
                        use_amp=False,
                        n_focus=n_focus,
                        edge_cartesian=True,
                    )

    def _assert_bridging_force_consistent_across_the_junction(
        self,
        model: torch.nn.Module,
        *,
        eps: float = 1.0e-5,
    ) -> None:
        """Assert the bridged force matches a finite difference of the energy.

        A finite, isolated cluster keeps the neighbor count bounded and equal to
        the physical neighbors within ``rcut``, so the check is independent of
        the periodic-image count and of ``sel``. The Na-Cl probe pair closes
        from beyond the end of the truncated term (1.1 x 2.54 Å) through the
        contact clamp (its edge starts at the 2.54 Å covalent bond length) to
        well inside its freeze, while the spectators stay beyond both. A kink
        at the truncation end or at either end of the clamp ramp would make
        the analytical force (``-dE/dx``) diverge from the central difference.
        """
        distances = torch.linspace(1.0, 3.2, 45, dtype=self.dtype, device=self.device)
        clamp = model.atomic_model.descriptor.contact_clamp
        edge = clamp.edges[0, 1]
        end = model.inter_potential.pair_table[1, 8]
        junctions = torch.stack((edge * clamp.inner, edge, end)).to(distances)
        distances = torch.cat((distances, junctions - eps, junctions + eps))
        template = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.6, 0.0],
                [0.0, 0.0, 3.8],
                [-4.0, 0.2, 0.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        atype = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.int32, device=self.device)
        # Three coordinate frames per scan point: the probe distance and its
        # +/- eps neighbors for a central difference of the total energy.
        offsets = torch.tensor([0.0, eps, -eps], dtype=self.dtype, device=self.device)
        probe_x = (distances.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
        n_frame = probe_x.shape[0]
        coord = template.unsqueeze(0).repeat(n_frame, 1, 1)
        coord[:, 1, 0] = probe_x
        result = model(coord, atype.expand(n_frame, -1), box=None)
        energy = result["energy"][:, 0].reshape(-1, 3)
        self.assertTrue(torch.isfinite(energy).all().item())
        fd_force = -(energy[:, 1] - energy[:, 2]) / (2.0 * eps)
        forces = result["force"][:, 1, 0].reshape(-1, 3)
        analytical_force = forces[:, 0]
        # Extrapolate the one-sided forces from h and 2h to each junction.
        left, right = forces[-6:-3], forces[-3:]
        torch.testing.assert_close(
            2 * left[:, 0] - left[:, 2],
            2 * right[:, 0] - right[:, 1],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(analytical_force, fd_force, atol=1.0e-6, rtol=1.0e-4)
        # Below the freeze the learned energy no longer depends on the pair
        # separation and the truncated ZBL term dominates, so the closest
        # probe must be pushed outward (+x).
        self.assertGreater(
            analytical_force[0].item(),
            0.0,
            "bridged short-range force should be repulsive",
        )

    def test_bridging_force_consistent_across_the_junction(self) -> None:
        """The bridged total energy stays conservative and C1-smooth through
        the truncation end and the clamp ramp on a bounded, sel-independent
        isolated cluster, and is repulsive at short range.
        """
        for use_amp in (False, True):
            for n_atten_head in (0, 1, 2):
                for n_focus in (1, 2):
                    with self.subTest(
                        n_atten_head=n_atten_head, use_amp=use_amp, n_focus=n_focus
                    ):
                        model = self._build_random_weight_model(
                            n_atten_head,
                            use_amp=use_amp,
                            n_focus=n_focus,
                            bridging_method="ZBL",
                        )
                        self._assert_bridging_force_consistent_across_the_junction(
                            model
                        )


class TestContactClampFreeze(_EnergyCurveFixture):
    """
    The learned energy of a bridged model stops depending on a closing pair.

    Two atoms ``A`` and ``B`` close along the x axis with no other atom
    inside the cutoff, so their separation is the only geometric quantity
    that changes. The contact clamp freezes the separation the descriptor
    reads below ``inner`` times the pair's data edge, and every message about
    the pair is built from what the descriptor reads. Below the freeze the
    learned part of both atomic energies is therefore strictly constant, and
    the energy of the pair changes only through the analytical term, which is
    evaluated on the true separation.
    """

    N_SCAN_POINTS = 9
    INVARIANCE_TOLERANCE = 1.0e-10

    def _atom_energies(
        self, model: torch.nn.Module, separations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom bridged and learned energies along the A-B scan."""
        descriptor = model.atomic_model.descriptor
        previous_random_gamma = descriptor.random_gamma
        descriptor.random_gamma = False
        try:
            coord = torch.zeros(
                (separations.shape[0], 2, 3), dtype=self.dtype, device=self.device
            )
            coord[:, 1, 0] = separations
            atype = torch.tensor(
                [[0, 1]], dtype=torch.int32, device=self.device
            ).expand(separations.shape[0], -1)
            bridged = model(coord, atype, box=None)["atom_energy"].detach()
            potential = model.inter_potential
            model.inter_potential = None
            try:
                learned = model(coord, atype, box=None)["atom_energy"].detach()
            finally:
                model.inter_potential = potential
        finally:
            descriptor.random_gamma = previous_random_gamma
        return bridged, learned

    def test_learned_energy_is_frozen_below_the_clamp(self) -> None:
        model = self._build_random_weight_model(
            n_atten_head=2,
            use_amp=False,
            n_focus=2,
            bridging_method="ZBL",
        )
        clamp = model.atomic_model.descriptor.contact_clamp
        edge = float(clamp.edges[0, 1])
        freeze = clamp.inner * edge
        below = torch.linspace(
            0.3 * freeze,
            freeze,
            self.N_SCAN_POINTS,
            dtype=self.dtype,
            device=self.device,
        )
        bridged, learned = self._atom_energies(model, below)
        span = float((learned - learned[:1]).abs().max())
        self.assertLess(span, self.INVARIANCE_TOLERANCE)
        # The analytical term adds a repulsion that grows as the pair closes.
        added = (bridged - learned).sum(dim=(1, 2))
        self.assertTrue(torch.all(added > 0.0))
        self.assertTrue(torch.all(added[:-1] > added[1:]))

        # Inside the ramp the learned energy still follows the pair.
        ramp = torch.tensor(
            [0.5 * (freeze + edge), edge], dtype=self.dtype, device=self.device
        )
        _, learned_ramp = self._atom_energies(model, ramp)
        self.assertGreater(
            float((learned_ramp[0] - learned_ramp[1]).abs().max()),
            self.INVARIANCE_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
