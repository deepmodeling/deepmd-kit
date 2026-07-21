# SPDX-License-Identifier: LGPL-3.0-or-later
import math
import os
import tempfile
import unittest
import warnings
from pathlib import (
    Path,
)
from unittest import (
    mock,
)

import h5py
import numpy as np
import torch
from packaging.version import parse as parse_version

from deepmd.pt.loss import (
    DeNSLoss,
    EnergySpinLoss,
    EnergyStdLoss,
    PropertyLoss,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    GatedActivation,
    LoRASO2,
    LoRASO3,
    SO2Linear,
    SO3Linear,
    apply_lora_to_sezm,
    build_edge_cache,
    build_edge_cache_from_edges,
    build_merged_state_dict,
)
from deepmd.pt.model.model import (
    get_model,
    get_sezm_model,
)
from deepmd.pt.model.model.sezm_model import (
    InterPotential,
    SeZMModel,
)
from deepmd.pt.model.model.sezm_native_spin_model import (
    SeZMNativeSpinModel,
)
from deepmd.pt.model.model.sezm_property_model import (
    SeZMPropertyModel,
)
from deepmd.pt.train.training import (
    prepare_model_for_loss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.utils.path import (
    DPPath,
)

warnings.filterwarnings(
    # Keep the compile-test warning summary focused on strict-tolerance drift.
    # PyTorch's AOTAutograd cache emits an internal Python 3.14 deprecation
    # warning that is unrelated to SeZM numerical correctness.
    "ignore",
    category=DeprecationWarning,
    module=r"torch\._functorch\._aot_autograd\.autograd_cache",
)

# SeZM's ``torch.compile`` / AOT-export code paths are validated on torch
# 2.11.x and 2.12.x, the releases the compile pipeline supports (see
# ``deepmd.pt.utils.compile_compat``). Other torch versions can segfault or
# drift, so the compile-parity tests are skipped there.
_TORCH_VERSION = parse_version(torch.__version__)
_SKIP_OFF_COMPILE_TORCH = (_TORCH_VERSION.major, _TORCH_VERSION.minor) not in {
    (2, 11),
    (2, 12),
}
_SKIP_OFF_COMPILE_TORCH_REASON = (
    "SeZM's torch.compile path is only supported on torch 2.11.x and 2.12.x; "
    f"current torch is {torch.__version__}."
)


def _assert_close_with_strict_warning(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    strict_atol: float = 1.0e-6,
    strict_rtol: float = 1.0e-6,
    atol: float,
    rtol: float,
    msg: str,
) -> None:
    """Warn on strict compile drift, fail only outside relaxed tolerance."""
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=strict_atol,
            rtol=strict_rtol,
            msg=msg,
        )
    except AssertionError as err:
        warnings.warn(
            f"{msg} exceeds strict tolerance "
            f"(atol={strict_atol:g}, rtol={strict_rtol:g}) but is checked "
            f"against relaxed tolerance (atol={atol:g}, rtol={rtol:g}): {err}",
            RuntimeWarning,
            stacklevel=2,
        )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)


def _build_m_major_z_rotation(
    angles: torch.Tensor, lmax: int, mmax: int, device: torch.device
) -> torch.Tensor:
    """Build the m-major block z-rotation matrix used by SO(2) equivariance tests.

    Given per-sample rotation ``angles`` and the ``(lmax, mmax)`` truncation,
    return a tensor with shape ``(batch, dim_red, dim_red)`` where ``dim_red``
    is the truncated coefficient dimension of the m-major layout.
    """
    batch = angles.shape[0]
    m0_size = lmax + 1
    dim_red = m0_size
    for m in range(1, mmax + 1):
        num_l = lmax - m + 1
        dim_red += 2 * num_l

    Z = angles.new_zeros(batch, dim_red, dim_red)
    eye0 = torch.eye(m0_size, device=device, dtype=angles.dtype).expand(
        batch, m0_size, m0_size
    )
    Z[:, :m0_size, :m0_size] = eye0

    offset = m0_size
    for m in range(1, mmax + 1):
        num_l = lmax - m + 1
        eye = torch.eye(num_l, device=device, dtype=angles.dtype).expand(
            batch, num_l, num_l
        )
        cos_m = torch.cos(m * angles).view(batch, 1, 1)
        sin_m = torch.sin(m * angles).view(batch, 1, 1)

        # Each m group stores the coefficients as [neg(l), pos(l)]; the rotation
        # is [[cos I, -sin I], [sin I, cos I]] for the (neg, pos) pair.
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


def _build_lora_sezm_model_params(**overrides) -> dict:
    """Minimal SeZMModel config suitable for LoRA injection tests.

    Uses ``s2_activation=[False, False]`` so the model keeps a
    ``GatedActivation`` (the override-freeze policy is easier to exercise) and
    sets ``use_compile=False`` by default; set ``use_compile=True`` via
    ``overrides`` to exercise the compile path.
    """
    params = {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, False],
            "mlp_bias": True,
            "layer_scale": True,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float32",
            "seed": 7,
        },
        "use_compile": False,
    }
    params.update(overrides)
    return params


class TestSeZMModelCompile(unittest.TestCase):
    """Test SeZM model compile path consistency."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    @staticmethod
    def _randomize_params(model: torch.nn.Module, seed: int = 1234) -> None:
        """Fill all parameters with small random values.

        Zero-initialized parameters mask second-order gradient bugs because
        many multiplicative paths collapse to zero.
        """
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)

    def _build_model_params(
        self, *, use_compile: bool, edge_cartesian: bool = False
    ) -> dict:
        return {
            "type": "SeZM",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "edge_cartesian": edge_cartesian,
                "l_schedule": [2, 1] if edge_cartesian else [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 1,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "s2_activation": [False, True],
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": use_compile,
        }

    def _make_tiny_frame(
        self,
        nframe: int = 1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build deterministic tiny frames with force and virial labels.

        Parameters
        ----------
        nframe
            Number of frames to build.

        Returns
        -------
        coord : torch.Tensor
            Coordinates with shape (nframe, nloc, 3).
        atype : torch.Tensor
            Atom types with shape (nframe, nloc).
        box : torch.Tensor
            Box tensor with shape (nframe, 9).
        energy : torch.Tensor
            Energy with shape (nframe, 1).
        force : torch.Tensor
            Forces with shape (nframe, nloc, 3).
        virial : torch.Tensor
            Virial tensor with shape (nframe, 9).
        """
        if nframe <= 0:
            raise ValueError("nframe must be positive")

        frame_shift = torch.arange(
            nframe, device=self.device, dtype=torch.float32
        ).view(nframe, 1, 1)
        coord = (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.1, 0.3, 0.0],
                        [0.2, 1.5, 0.4],
                        [1.7, 1.2, 0.2],
                        [2.3, 0.1, 1.0],
                        [0.8, 2.2, 1.1],
                        [2.6, 1.8, 1.5],
                    ],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            + 0.05 * frame_shift
        )
        atype = torch.tensor(
            [[0, 1, 0, 1, 0, 1, 0]], device=self.device, dtype=torch.int32
        ).repeat(nframe, 1)
        box = torch.tensor(
            [[8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]],
            device=self.device,
            dtype=torch.float32,
        ).repeat(nframe, 1)
        energy = torch.tensor(
            [[0.25]], device=self.device, dtype=torch.float32
        ) + 0.01 * frame_shift.view(nframe, 1)
        force = (
            torch.tensor(
                [
                    [
                        [0.2, -0.1, 0.0],
                        [-0.3, 0.4, 0.1],
                        [0.1, -0.3, -0.1],
                        [0.0, 0.2, -0.2],
                        [-0.2, -0.1, 0.3],
                        [0.3, 0.0, -0.1],
                        [-0.1, -0.1, 0.0],
                    ],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            + 0.02 * frame_shift
        )
        virial = torch.tensor(
            [[0.3, 0.01, -0.02, 0.01, -0.2, 0.04, -0.02, 0.04, 0.1]],
            device=self.device,
            dtype=torch.float32,
        ) + 0.03 * frame_shift.view(nframe, 1)
        return coord, atype, box, energy, force, virial

    def _train_steps(
        self,
        model: torch.nn.Module,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        energy: torch.Tensor,
        force: torch.Tensor,
        virial: torch.Tensor | None = None,
        steps: int = 3,
    ) -> dict[str, torch.Tensor]:
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-7)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(coord, atype, box=box)
            loss_energy = torch.mean(
                (out["energy"] - energy.to(out["energy"].dtype)) ** 2
            )
            loss_force = torch.mean((out["force"] - force.to(out["force"].dtype)) ** 2)
            loss = loss_energy + loss_force
            if virial is not None and "virial" in out:
                loss_virial = torch.mean(
                    (out["virial"] - virial.to(out["virial"].dtype)) ** 2
                )
                loss = loss + loss_virial
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return {
            name: param.detach().clone() for name, param in model.named_parameters()
        }

    def _make_frame_with_natoms(
        self, nloc: int, *, seed: int = 20240613
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a compact ``nloc``-atom frame with neighbours inside ``rcut``.

        Atoms are placed in a tight cluster so the ``sel=[2, 2]`` neighbour list
        is saturated and the edge count is comfortably larger than ``nloc``.
        """
        torch.manual_seed(seed + nloc)
        coord = torch.rand(1, nloc, 3, device=self.device, dtype=torch.float32) * 2.5
        atype = (
            (torch.arange(nloc, device=self.device) % 2).view(1, nloc).to(torch.int32)
        )
        box = torch.tensor(
            [[8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]],
            dtype=torch.float32,
            device=self.device,
        )
        return coord, atype, box

    def test_trace_pad_dim_trim_returns_contiguous(self) -> None:
        """Trimmed trace inputs stay contiguous so strides mirror runtime layout.

        A sliced (non-contiguous) trim leaks the pre-trim length into the tensor
        stride; ``make_fx`` duck-shaping can then fuse that stale stride with the
        edge-count symbol and corrupt the compiled shape guards.
        """
        from deepmd.pt.utils.compile_compat import (
            trace_pad_dim,
        )

        base = torch.arange(5 * 13, device=self.device).view(5, 13)
        trimmed = trace_pad_dim(base, 1, 7)
        self.assertEqual(tuple(trimmed.shape), (5, 7))
        self.assertTrue(trimmed.is_contiguous())
        padded = trace_pad_dim(base, 1, 20)
        self.assertEqual(tuple(padded.shape), (5, 20))
        self.assertTrue(padded.is_contiguous())

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_eval_compile_first_frame_nloc_matches_trace_edge_count(self) -> None:
        """First eval frame with ``nloc`` equal to the trace edge count compiles.

        The symbolic trace pads the edge axis to ``next_safe_prime`` (13 for the
        two-type forbidden set ``{1, 2, 3, 9}`` -> primes 5/7/11/13) and trims
        ``atype`` to ``trace_nloc`` (7). A first frame with ``nloc == 13`` leaves
        the trimmed ``atype`` carrying ``stride(0) == 13``; previously that stale
        stride was duck-shaped onto the edge-count symbol, so every edge tensor
        was guarded against ``nloc`` and ``assert_size_stride`` failed once the
        real edge count differed. Pins the contiguous-trace + eval-only
        duck-shape-off fix.
        """
        nloc = 13  # == next_safe_prime edge count for the two-type forbidden set
        coord, atype, box = self._make_frame_with_natoms(nloc)

        model_dyn = get_sezm_model(self._build_model_params(use_compile=False))
        self._randomize_params(model_dyn)
        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_cmp = get_sezm_model(self._build_model_params(use_compile=True))
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.eval()
        model_cmp.eval()

        out_dyn = model_dyn(coord, atype, box=box)
        # The compiled eval path must trace, lower and run without tripping
        # ``assert_size_stride`` on the edge tensors.
        out_cmp = model_cmp(coord, atype, box=box)
        self.assertIn((False, False), model_cmp.compiled_core_compute_cache)
        _assert_close_with_strict_warning(
            out_dyn["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval energy mismatch when first-frame nloc == trace edge count",
        )
        _assert_close_with_strict_warning(
            out_dyn["force"],
            out_cmp["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval force mismatch when first-frame nloc == trace edge count",
        )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_cache_slots_and_eval_shape_change(self) -> None:
        """Compile cache slots should coexist while eval handles batch-size growth."""
        coord_1, atype_1, box_1, _, _, _ = self._make_tiny_frame()
        coord_2, atype_2, box_2, _, _, _ = self._make_tiny_frame(nframe=2)

        # === Step 1. Build paired models with shared random weights ===
        model_dyn = get_sezm_model(self._build_model_params(use_compile=False))
        self._randomize_params(model_dyn)
        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_cmp = get_sezm_model(self._build_model_params(use_compile=True))
        model_cmp.load_state_dict(model_dyn.state_dict())

        # Compile cache key is (training, has_coord_corr).
        train_key = (True, False)
        eval_key = (False, False)

        # === Step 2. Train-mode forward fills the training slot. ===
        model_cmp.train()
        model_cmp(coord_1, atype_1, box=box_1)
        self.assertIn(train_key, model_cmp.compiled_core_compute_cache)
        self.assertNotIn(eval_key, model_cmp.compiled_core_compute_cache)
        callable_train_first = model_cmp.compiled_core_compute_cache[train_key]

        # === Step 3. First eval call adds the eval slot without evicting train. ===
        model_dyn.eval()
        model_cmp.eval()
        out_dyn_1 = model_dyn(coord_1, atype_1, box=box_1)
        out_cmp_1 = model_cmp(coord_1, atype_1, box=box_1)
        self.assertIn(train_key, model_cmp.compiled_core_compute_cache)
        self.assertIn(eval_key, model_cmp.compiled_core_compute_cache)
        self.assertIs(
            model_cmp.compiled_core_compute_cache[train_key], callable_train_first
        )
        callable_eval_first = model_cmp.compiled_core_compute_cache[eval_key]
        self.assertIsNot(callable_train_first, callable_eval_first)
        _assert_close_with_strict_warning(
            out_dyn_1["energy"],
            out_cmp_1["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval energy mismatch on first compiled call",
        )
        _assert_close_with_strict_warning(
            out_dyn_1["force"],
            out_cmp_1["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval force mismatch on first compiled call",
        )
        _assert_close_with_strict_warning(
            out_dyn_1["virial"],
            out_cmp_1["virial"],
            atol=1.0e-5,
            rtol=1.0e-5,
            msg="eval virial mismatch on first compiled call",
        )

        # === Step 4. Reuse the traced eval graph on a larger batch. ===
        out_dyn_2 = model_dyn(coord_2, atype_2, box=box_2)
        out_cmp_2 = model_cmp(coord_2, atype_2, box=box_2)
        self.assertEqual(out_dyn_2["energy"].shape, (2, 1))
        self.assertEqual(out_cmp_2["energy"].shape, (2, 1))
        self.assertIs(
            model_cmp.compiled_core_compute_cache[eval_key], callable_eval_first
        )
        _assert_close_with_strict_warning(
            out_dyn_2["energy"],
            out_cmp_2["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval energy mismatch after batch-size growth",
        )
        _assert_close_with_strict_warning(
            out_dyn_2["force"],
            out_cmp_2["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="eval force mismatch after batch-size growth",
        )
        _assert_close_with_strict_warning(
            out_dyn_2["virial"],
            out_cmp_2["virial"],
            atol=1.0e-5,
            rtol=1.0e-5,
            msg="eval virial mismatch after batch-size growth",
        )

        # === Step 5. Flip back to train and reuse the existing training slot. ===
        model_cmp.train()
        model_cmp(coord_1, atype_1, box=box_1)
        self.assertIs(
            model_cmp.compiled_core_compute_cache[train_key], callable_train_first
        )
        self.assertIs(
            model_cmp.compiled_core_compute_cache[eval_key], callable_eval_first
        )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_charge_spin_condition_matches_compile(self) -> None:
        """Charge/spin conditions should work through the compiled energy path."""
        coord, atype, box, _, _, _ = self._make_tiny_frame()
        params = self._build_model_params(use_compile=False)
        params["descriptor"]["add_chg_spin_ebd"] = True
        params["descriptor"]["default_chg_spin"] = [0.0, 1.0]

        model_dyn = get_sezm_model(params)
        self._randomize_params(model_dyn)
        params_cmp = self._build_model_params(use_compile=True)
        params_cmp["descriptor"]["add_chg_spin_ebd"] = True
        params_cmp["descriptor"]["default_chg_spin"] = [0.0, 1.0]
        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_cmp = get_sezm_model(params_cmp)
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.eval()
        model_cmp.eval()

        charge_spin = torch.tensor(
            [[0.0, 1.0]], dtype=torch.float32, device=self.device
        )
        out_dyn = model_dyn(coord, atype, box=box, charge_spin=charge_spin)
        out_cmp = model_cmp(coord, atype, box=box, charge_spin=charge_spin)
        out_default = model_cmp(coord, atype, box=box)
        out_shifted = model_cmp(
            coord,
            atype,
            box=box,
            charge_spin=torch.tensor(
                [[1.0, 1.0]], dtype=torch.float32, device=self.device
            ),
        )

        _assert_close_with_strict_warning(
            out_dyn["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="charge/spin energy mismatch",
        )
        _assert_close_with_strict_warning(
            out_dyn["force"],
            out_cmp["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="charge/spin force mismatch",
        )
        _assert_close_with_strict_warning(
            out_dyn["virial"],
            out_cmp["virial"],
            atol=1.0e-5,
            rtol=1.0e-5,
            msg="charge/spin virial mismatch",
        )
        _assert_close_with_strict_warning(
            out_default["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="default charge/spin energy mismatch",
        )
        self.assertFalse(
            torch.allclose(out_shifted["atom_energy"], out_cmp["atom_energy"])
        )

    def test_fixed_edge_geometry_matches_standard_cache(self) -> None:
        """Sparse edge geometry should match the standard descriptor cache."""
        coord, atype, box, _, _, _ = self._make_tiny_frame()
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.train()
        descriptor = model.atomic_model.descriptor

        cc, bb, fp, ap, _ = model._input_type_cast(
            coord, box=box, fparam=None, aparam=None
        )
        del fp, ap
        if cc.ndim == 2:
            cc = cc.view(coord.shape[0], atype.shape[1], 3)
        extended_coord, extended_atype, nlist, mapping = (
            model.build_extended_neighbor_list(cc, atype, bb)
        )
        atype_loc = extended_atype[:, : nlist.shape[1]]
        type_ebed = descriptor.type_embedding(atype_loc).reshape(
            -1, descriptor.channels
        )
        pair_keep_mask = torch.ones_like(
            nlist, dtype=torch.bool, device=extended_coord.device
        )

        cache_std = build_edge_cache(
            type_ebed=type_ebed,
            extended_coord=extended_coord.to(descriptor.compute_dtype),
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            eps=descriptor.eps,
            deg_norm_floor=descriptor.deg_norm_floor,
            edge_envelope=descriptor.edge_envelope,
            radial_basis=descriptor.radial_basis,
            n_radial=descriptor.radial_basis.n_radial,
            random_gamma=False,
            wigner_calc=descriptor.wigner_calc,
        )

        edge_index, edge_vec, edge_mask, _ = model.build_edge_list_from_nlist(
            extended_coord=extended_coord,
            nlist=nlist,
            mapping=mapping,
        )
        cache_sparse = build_edge_cache_from_edges(
            type_ebed=type_ebed,
            atype_flat=atype_loc.reshape(-1),
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            compute_dtype=descriptor.compute_dtype,
            eps=descriptor.eps,
            deg_norm_floor=descriptor.deg_norm_floor,
            inner_clamp=descriptor.inner_clamp,
            bridging_switch=descriptor.bridging_switch,
            edge_envelope=descriptor.edge_envelope,
            radial_basis=descriptor.radial_basis,
            has_exclude_types=False,
            edge_type_keep_mask=descriptor._edge_type_keep_mask,
            random_gamma=False,
            wigner_calc=descriptor.wigner_calc,
        )

        # build_edge_list_from_nlist appends masked dummy edges;
        # compare only the real edges before the padded tail.
        n_real = cache_std.src.shape[0]
        self.assertEqual(edge_mask.shape[0] - n_real, 2)
        self.assertFalse(edge_mask[n_real:].any().item())
        self.assertTrue(torch.equal(cache_std.src, cache_sparse.src[:n_real]))
        self.assertTrue(torch.equal(cache_std.dst, cache_sparse.dst[:n_real]))
        torch.testing.assert_close(cache_std.edge_vec, cache_sparse.edge_vec[:n_real])
        torch.testing.assert_close(cache_std.edge_rbf, cache_sparse.edge_rbf[:n_real])
        torch.testing.assert_close(cache_std.edge_env, cache_sparse.edge_env[:n_real])
        torch.testing.assert_close(cache_std.D_full, cache_sparse.D_full[:n_real])
        torch.testing.assert_close(cache_std.Dt_full, cache_sparse.Dt_full[:n_real])

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_eval_compile_policy(self) -> None:
        """Eval should stay eager by default and compile only with env override."""
        model = get_sezm_model(self._build_model_params(use_compile=True))
        self.assertTrue(model.use_compile)

        model.train()
        self.assertTrue(model.should_use_compile())

        model.eval()
        self.assertFalse(model.should_use_compile())

        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_eval = get_sezm_model(self._build_model_params(use_compile=True))
        model_eval.eval()
        self.assertTrue(model_eval.should_use_compile())

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_forward_backward_double_backward_matches_compile(self) -> None:
        """
        Check forward, backward, double backward, and short training consistency.

        Forward: energy/force outputs should match.
        Backward: d(energy)/d(params) should match.
        Double backward: d(force_loss)/d(params) should match.
        Training: three SGD steps and a larger follow-up batch should still match.
        """
        coord, atype, box, energy, force, virial = self._make_tiny_frame()
        coord_2, atype_2, box_2, _, _, _ = self._make_tiny_frame(nframe=2)

        # === Step 1. Build paired models with shared random weights ===
        model_dyn = get_sezm_model(self._build_model_params(use_compile=False))
        self._randomize_params(model_dyn)
        model_cmp = get_sezm_model(self._build_model_params(use_compile=True))
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.train()
        model_cmp.train()

        # === Step 2. Forward output consistency ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        _assert_close_with_strict_warning(
            out_dyn["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="train energy mismatch on first compiled call",
        )
        _assert_close_with_strict_warning(
            out_dyn["force"],
            out_cmp["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="train force mismatch on first compiled call",
        )

        # === Step 3. Backward on energy ===
        model_dyn.zero_grad(set_to_none=True)
        model_cmp.zero_grad(set_to_none=True)
        loss_dyn = out_dyn["energy"].sum()
        loss_cmp = out_cmp["energy"].sum()
        loss_dyn.backward()
        loss_cmp.backward()
        grads_dyn = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_dyn.named_parameters()
        }
        grads_cmp = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_cmp.named_parameters()
        }
        # Inductor Triton kernels use different reduction order vs eager,
        # so float32 gradients can differ by ~1e-3 on GPU.
        grad_atol = 1.0e-5 if self.device == torch.device("cpu") else 2.0e-3
        grad_rtol = 1.0e-5 if self.device == torch.device("cpu") else 3.0e-3
        self.assertEqual(set(grads_dyn.keys()), set(grads_cmp.keys()))
        for name in grads_dyn.keys():
            _assert_close_with_strict_warning(
                grads_dyn[name],
                grads_cmp[name],
                atol=grad_atol,
                rtol=grad_rtol,
                msg=f"energy-grad mismatch at {name}",
            )

        # === Step 5. Reuse the compiled training graph for three optimizer steps ===
        params_dyn = self._train_steps(
            model_dyn, coord, atype, box, energy, force, virial
        )
        params_cmp = self._train_steps(
            model_cmp, coord, atype, box, energy, force, virial
        )
        self.assertEqual(set(params_dyn.keys()), set(params_cmp.keys()))
        for name in params_dyn.keys():
            _assert_close_with_strict_warning(
                params_dyn[name],
                params_cmp[name],
                strict_atol=1.0e-7,
                strict_rtol=1.0e-7,
                atol=1.0e-7,
                rtol=1.0e-7,
                msg=f"trained parameter mismatch at {name}",
            )

        # === Step 6. The traced training graph should also handle a larger batch ===
        out_dyn = model_dyn(coord_2, atype_2, box=box_2)
        out_cmp = model_cmp(coord_2, atype_2, box=box_2)
        self.assertEqual(out_dyn["energy"].shape, (2, 1))
        self.assertEqual(out_cmp["energy"].shape, (2, 1))
        _assert_close_with_strict_warning(
            out_dyn["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="train energy mismatch after batch-size growth",
        )

        # === Step 4. Double backward via force loss ===
        model_dyn.zero_grad(set_to_none=True)
        model_cmp.zero_grad(set_to_none=True)
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        loss_dyn = torch.sum(out_dyn["force"] * out_dyn["force"])
        loss_cmp = torch.sum(out_cmp["force"] * out_cmp["force"])
        loss_dyn.backward()
        loss_cmp.backward()
        grads_dyn = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_dyn.named_parameters()
        }
        grads_cmp = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_cmp.named_parameters()
        }
        self.assertEqual(set(grads_dyn.keys()), set(grads_cmp.keys()))
        for name in grads_dyn.keys():
            _assert_close_with_strict_warning(
                grads_dyn[name],
                grads_cmp[name],
                atol=grad_atol,
                rtol=grad_rtol,
                msg=f"force-grad mismatch at {name}",
            )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_cartesian_forward_backward_matches_compile(self) -> None:
        """The Cartesian path (Wigner-D skipped) matches eager and compiled runs."""
        coord, atype, box, _, _, _ = self._make_tiny_frame()
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, edge_cartesian=True)
        )
        self._randomize_params(model_dyn)
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, edge_cartesian=True)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.train()
        model_cmp.train()

        # === Step 1. Forward output consistency ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        _assert_close_with_strict_warning(
            out_dyn["energy"],
            out_cmp["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="cartesian energy mismatch on first compiled call",
        )
        _assert_close_with_strict_warning(
            out_dyn["force"],
            out_cmp["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="cartesian force mismatch on first compiled call",
        )

        # === Step 2. Energy-gradient consistency ===
        model_dyn.zero_grad(set_to_none=True)
        model_cmp.zero_grad(set_to_none=True)
        out_dyn["energy"].sum().backward()
        out_cmp["energy"].sum().backward()
        grad_atol = 1.0e-5 if self.device == torch.device("cpu") else 2.0e-3
        grad_rtol = 1.0e-5 if self.device == torch.device("cpu") else 3.0e-3
        grads_dyn = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_dyn.named_parameters()
        }
        grads_cmp = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_cmp.named_parameters()
        }
        self.assertEqual(set(grads_dyn.keys()), set(grads_cmp.keys()))
        for name in grads_dyn.keys():
            _assert_close_with_strict_warning(
                grads_dyn[name],
                grads_cmp[name],
                atol=grad_atol,
                rtol=grad_rtol,
                msg=f"cartesian energy-grad mismatch at {name}",
            )

    def _assert_multitask_compile_matches_eager(
        self,
        *,
        case_film_embd: bool,
    ) -> None:
        """
        Multi-task + compile: two SeZM branches sharing descriptor and
        fitting (with per-task case embedding) should each compile correctly
        and produce outputs matching their eager counterparts.
        """
        from deepmd.pt.train.training import (
            get_model_for_wrapper,
            prepare_model_for_loss,
        )
        from deepmd.pt.train.wrapper import (
            ModelWrapper,
        )
        from deepmd.pt.utils.multi_task import (
            preprocess_shared_params,
        )

        # === Step 1. Build a multi-task model config with shared descriptor
        # + shared fitting (case_embd=2) seeded from the compile fixture. ===
        def _make_mt_cfg(use_compile: bool) -> dict:
            single = self._build_model_params(use_compile=use_compile)
            fitting_shared = dict(single["fitting_net"])
            fitting_shared["type"] = "dpa4_ener"
            fitting_shared["dim_case_embd"] = 2
            fitting_shared["case_film_embd"] = case_film_embd
            return {
                "use_compile": use_compile,
                "shared_dict": {
                    "type_map": single["type_map"],
                    "descriptor": single["descriptor"],
                    "shared_fit": fitting_shared,
                },
                "model_dict": {
                    "water_1": {
                        "type": "SeZM",
                        "type_map": "type_map",
                        "descriptor": "descriptor",
                        "fitting_net": "shared_fit",
                    },
                    "water_2": {
                        "type": "SeZM",
                        "type_map": "type_map",
                        "descriptor": "descriptor",
                        "fitting_net": "shared_fit",
                    },
                },
            }

        def _build_wrapper(use_compile: bool) -> ModelWrapper:
            mt_cfg = _make_mt_cfg(use_compile)
            # ``preprocess_shared_params`` cascades the top-level
            # ``use_compile`` into every branch before unrolling the
            # shared_dict, mirroring the real training flow.
            mt_cfg, shared_links = preprocess_shared_params(mt_cfg)
            loss_params = {
                "water_1": {"type": "ener"},
                "water_2": {"type": "ener"},
            }
            models = get_model_for_wrapper(mt_cfg)
            prepare_model_for_loss(models, loss_params)
            wrapper = ModelWrapper(models)
            wrapper.share_params(shared_links, {"water_1": 0.5, "water_2": 0.5})
            return wrapper

        wrapper_eager = _build_wrapper(use_compile=False)
        self._randomize_params(wrapper_eager)
        wrapper_cmp = _build_wrapper(use_compile=True)
        # Mirror eager weights so the only remaining difference between the
        # two wrappers is the compile path.
        wrapper_cmp.load_state_dict(wrapper_eager.state_dict())

        # Sanity: descriptor and fitting parameters are shared across branches
        # inside each wrapper.
        for w in (wrapper_eager, wrapper_cmp):
            d1 = w.model["water_1"].get_descriptor()
            d2 = w.model["water_2"].get_descriptor()
            self.assertEqual(
                next(d1.parameters()).data_ptr(),
                next(d2.parameters()).data_ptr(),
            )
            f1 = w.model["water_1"].atomic_model.fitting_net
            f2 = w.model["water_2"].atomic_model.fitting_net
            self.assertEqual(
                next(f1.filter_layers.parameters()).data_ptr(),
                next(f2.filter_layers.parameters()).data_ptr(),
            )
            # Per-task case embeddings remain distinct.
            self.assertFalse(torch.equal(f1.case_embd, f2.case_embd))
            expected_in_dim = f1.dim_descrpt + (0 if case_film_embd else 2)
            self.assertEqual(f1.filter_layers.networks[0].in_dim, expected_in_dim)
            self.assertEqual(f1.case_film_embd, case_film_embd)

        # === Step 2. Run compile + eager forward on each branch. ===
        coord, atype, box, _, _, _ = self._make_tiny_frame()
        for branch in ("water_1", "water_2"):
            m_eager = wrapper_eager.model[branch]
            m_cmp = wrapper_cmp.model[branch]
            m_eager.train()
            m_cmp.train()
            out_e = m_eager(coord, atype, box=box)
            out_c = m_cmp(coord, atype, box=box)
            energy_atol = 1.0e-6 if self.device == torch.device("cpu") else 5.0e-6
            energy_rtol = 1.0e-6 if self.device == torch.device("cpu") else 5.0e-4
            _assert_close_with_strict_warning(
                out_e["energy"],
                out_c["energy"],
                atol=energy_atol,
                rtol=energy_rtol,
                msg=f"multitask energy mismatch at {branch}",
            )
            _assert_close_with_strict_warning(
                out_e["force"],
                out_c["force"],
                atol=1.0e-6,
                rtol=1.0e-6,
                msg=f"multitask force mismatch at {branch}",
            )

        # === Step 3. Each branch keeps its own per-instance cache dict, but
        # branches that share descriptor + fitting (same Python-object
        # identity after share_params) reuse a single compiled callable via
        # the module-level ``_SEZM_COMPILE_CACHE``.  This avoids the
        # N x compile-cache OOM / N DDP graph boundary cost on multitask
        # runs.  Step 2 ran every branch in training mode with the default
        # ``do_atomic_virial=False`` and no coordinate correction, so each
        # per-branch cache should hold exactly that one slot, and the
        # compiled callable at that slot must be the *same* object. ===
        cache1 = wrapper_cmp.model["water_1"].compiled_core_compute_cache
        cache2 = wrapper_cmp.model["water_2"].compiled_core_compute_cache
        self.assertIsNot(cache1, cache2)
        train_key = (True, False)
        self.assertIn(train_key, cache1)
        self.assertIn(train_key, cache2)
        c1 = cache1[train_key]
        c2 = cache2[train_key]
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertIs(c1, c2)

        # === Step 4. Per-task case embedding must differentiate outputs. ===
        out_e1 = wrapper_eager.model["water_1"](coord, atype, box=box)
        out_e2 = wrapper_eager.model["water_2"](coord, atype, box=box)
        self.assertFalse(
            torch.allclose(out_e1["energy"], out_e2["energy"], atol=1.0e-8)
        )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_multitask_compile_matches_eager(self) -> None:
        """Legacy case embedding concatenation should match through compile."""
        self._assert_multitask_compile_matches_eager(case_film_embd=False)


class TestSeZMModelProperty(unittest.TestCase):
    """Test DPA4/SeZM invariant property fitting."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    @staticmethod
    def _randomize_params(model: torch.nn.Module, seed: int = 1234) -> None:
        """Fill all trainable tensors with deterministic small values."""
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(torch.randn_like(param) * 0.1)

    def _build_model_params(self, *, use_compile: bool, intensive: bool) -> dict:
        return {
            "type": "SeZM",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 1,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "s2_activation": [False, True],
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "foo",
                "task_dim": 3,
                "intensive": intensive,
                "neuron": [8],
                "activation_function": "tanh",
                "resnet_dt": True,
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": use_compile,
        }

    def _make_tiny_frame(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(2025)
        box = 5.0 * torch.eye(
            3, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=self.device
        )
        coord = (
            torch.rand(
                [1, 5, 3],
                dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                device=self.device,
                generator=generator,
            )
            @ box
        )
        atype = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.long, device=self.device)
        return coord, atype, box.reshape(1, 9)

    def test_forward_shapes_and_reduction(self) -> None:
        """Property outputs should use public property keys and reductions."""
        for intensive in (False, True):
            model = get_sezm_model(
                self._build_model_params(use_compile=False, intensive=intensive)
            ).to(self.device)
            self.assertIsInstance(model, SeZMPropertyModel)
            self.assertEqual(model.get_var_name(), "foo")
            self.assertEqual(model.get_task_dim(), 3)
            self.assertEqual(model.get_intensive(), intensive)

            coord, atype, box = self._make_tiny_frame()
            ret = model(coord, atype, box=box)
            self.assertEqual(ret["atom_foo"].shape, (1, 5, 3))
            self.assertEqual(ret["foo"].shape, (1, 3))
            self.assertEqual(ret["mask"].shape, (1, 5))
            self.assertNotIn("force", ret)
            self.assertNotIn("virial", ret)
            if intensive:
                expected = ret["atom_foo"].mean(dim=1)
            else:
                expected = ret["atom_foo"].sum(dim=1)
            torch.testing.assert_close(ret["foo"], expected)

    def test_property_loss_and_serialization(self) -> None:
        """PropertyLoss metadata and model serialization should round-trip."""
        from deepmd.pt.model.model.model import (
            BaseModel,
        )

        model = get_sezm_model(
            self._build_model_params(use_compile=False, intensive=True)
        ).to(self.device)
        loss = PropertyLoss(
            task_dim=model.get_task_dim(),
            var_name=model.get_var_name(),
            intensive=model.get_intensive(),
        )
        self.assertEqual(loss.var_name, "foo")

        data = model.serialize()
        self.assertEqual(data["type"], "SeZMProperty")
        model2 = BaseModel.deserialize(data).to(self.device)
        self.assertIsInstance(model2, SeZMPropertyModel)

        coord, atype, box = self._make_tiny_frame()
        ret = model2(coord, atype, box=box)
        self.assertEqual(ret["foo"].shape, (1, 3))

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_matches_eager_and_backpropagates(self) -> None:
        """Compiled property forward should match eager and keep gradients."""
        eager = get_sezm_model(
            self._build_model_params(use_compile=False, intensive=False)
        ).to(self.device)
        compiled = get_sezm_model(
            self._build_model_params(use_compile=True, intensive=False)
        ).to(self.device)
        self._randomize_params(eager)
        compiled.load_state_dict(eager.state_dict())
        eager.train()
        compiled.train()

        coord, atype, box = self._make_tiny_frame()
        ret_eager = eager(coord, atype, box=box)
        ret_compiled = compiled(coord, atype, box=box)
        _assert_close_with_strict_warning(
            ret_compiled["foo"],
            ret_eager["foo"],
            atol=1.0e-5,
            rtol=1.0e-5,
            msg="compiled property mismatch",
        )
        self.assertIn((True, False), compiled.compiled_core_compute_cache)

        loss = ret_compiled["foo"].sum()
        loss.backward()
        grad_found = any(
            param.grad is not None and torch.count_nonzero(param.grad).item() > 0
            for param in compiled.parameters()
        )
        self.assertTrue(grad_found)


class TestInterPotential(unittest.TestCase):
    """Test InterPotential ZBL analytical pair potential."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def _pair_edges(
        self, r: float, atype_pair: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two directed edges (i->j and j->i) for one pair at distance r."""
        edge_vec = torch.tensor(
            [[r, 0.0, 0.0], [-r, 0.0, 0.0]],
            dtype=torch.float64,
            device=self.device,
        )
        edge_index = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.long, device=self.device
        )
        atype_flat = torch.tensor(atype_pair, dtype=torch.long, device=self.device)
        edge_mask = torch.tensor([True, True], device=self.device)
        return edge_vec, edge_index, atype_flat, edge_mask

    def test_zbl_known_value_OO(self) -> None:
        """ZBL energy for an O-O pair matches the analytic reference."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)

        import math

        z_o = 8.0
        a_bohr = 0.5291772109
        ke = 14.3996
        a_screen = 0.88534 * a_bohr / (z_o**0.23 + z_o**0.23)
        r = 1.0
        x = r / a_screen
        phi = (
            0.18175 * math.exp(-3.1998 * x)
            + 0.50986 * math.exp(-0.94229 * x)
            + 0.28022 * math.exp(-0.4029 * x)
            + 0.028171 * math.exp(-0.20162 * x)
        )
        expected = ke * z_o * z_o / r * phi

        total_e = pot(*self._pair_edges(r, [0, 0]), n_node=2).sum().item()
        self.assertAlmostEqual(total_e, expected, places=5)

    def test_zbl_known_value_OH(self) -> None:
        """ZBL energy for an O-H pair matches the analytic reference."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)
        import math

        z_o, z_h = 8.0, 1.0
        a_bohr = 0.5291772109
        ke = 14.3996
        a_screen = 0.88534 * a_bohr / (z_o**0.23 + z_h**0.23)
        r = 0.8
        x = r / a_screen
        phi = (
            0.18175 * math.exp(-3.1998 * x)
            + 0.50986 * math.exp(-0.94229 * x)
            + 0.28022 * math.exp(-0.4029 * x)
            + 0.028171 * math.exp(-0.20162 * x)
        )
        expected = ke * z_o * z_h / r * phi

        total_e = pot(*self._pair_edges(r, [0, 1]), n_node=2).sum().item()
        self.assertAlmostEqual(total_e, expected, places=5)

    def test_zbl_gradient_exists(self) -> None:
        """ZBL produces finite gradients w.r.t. the edge vectors."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)
        edge_vec, edge_index, atype_flat, edge_mask = self._pair_edges(1.0, [0, 1])
        edge_vec = edge_vec.detach().requires_grad_(True)

        pot(edge_vec, edge_index, atype_flat, edge_mask, n_node=2).sum().backward()
        self.assertIsNotNone(edge_vec.grad)
        self.assertTrue(torch.isfinite(edge_vec.grad).all())

    def test_virtual_spin_types_masked(self) -> None:
        """Edges touching a virtual spin type (>= real_type_count) contribute 0."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)
        # Node 2 is a virtual spin atom (type 2 >= real_type_count=2).
        edge_vec = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
            dtype=torch.float64,
            device=self.device,
        )
        # Edges: (0<->1) real-real, (0<->2) touch virtual node 2.
        edge_index = torch.tensor(
            [[1, 0, 2, 0], [0, 1, 0, 2]], dtype=torch.long, device=self.device
        )
        atype_flat = torch.tensor([0, 1, 2], dtype=torch.long, device=self.device)
        edge_mask = torch.tensor([True, True, True, True], device=self.device)

        with_virtual = pot(
            edge_vec, edge_index, atype_flat, edge_mask, n_node=3, real_type_count=2
        )
        # Only the real-real pair survives.
        real_only = pot(
            edge_vec[:2],
            edge_index[:, :2],
            atype_flat,
            edge_mask[:2],
            n_node=3,
            real_type_count=2,
        )
        torch.testing.assert_close(with_virtual, real_only)

    def test_unknown_element_raises(self) -> None:
        """Test that unknown element raises ValueError."""
        with self.assertRaises(ValueError):
            InterPotential(type_map=["O", "Xx"])


class TestSeZMEdgeForceScatter(unittest.TestCase):
    """Validate the edge-force-scatter force / virial assembly.

    Force, global virial and per-atom virial all come from a single
    ``autograd.grad`` truncated at the per-edge displacement vectors
    (``edge_energy_deriv``), then scattered back onto atoms.  These eager,
    float64 finite-difference checks pin the conservative-force guarantee
    ``F = -dE/dx`` and the PBC-correct virial ``W = -dE/deps``, and confirm
    the half-split per-atom virial sums back to the global virial.  The ZBL
    cases additionally drive ``InterPotential`` (edge form) through the
    same single backward.
    """

    def setUp(self) -> None:
        self.device = env.DEVICE

    def _build_model(self, *, bridging_method: str = "none") -> SeZMModel:
        """Build a tiny float64 SeZM model with randomized parameters."""
        params = {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "SeZM",
                "sel": [12, 12],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 1,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "s2_activation": [False, True],
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
            "use_compile": False,
            "bridging_method": bridging_method,
            "bridging_r_inner": 0.8,
            "bridging_r_outer": 1.2,
        }
        model = get_sezm_model(params)
        torch.manual_seed(1234)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        model.eval()
        return model

    def _frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Small periodic frame with dense neighbours inside ``rcut``."""
        coord = torch.tensor(
            [
                [
                    [0.10, 0.05, 0.00],
                    [1.05, 0.30, 0.10],
                    [0.20, 1.40, 0.35],
                    [1.60, 1.15, 0.20],
                    [2.20, 0.10, 1.05],
                ]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        atype = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.int64, device=self.device)
        box = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device=self.device,
        )
        return coord, atype, box

    def _energy(
        self,
        model: SeZMModel,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        return model(coord, atype, box=box)["energy"].squeeze()

    def _check_force_fd(self, bridging_method: str, *, periodic: bool = True) -> None:
        model = self._build_model(bridging_method=bridging_method)
        coord, atype, box = self._frame()
        # box=None exercises the non-periodic (open-boundary / cluster) path:
        # the edge-force scatter is PBC-agnostic because it differentiates the
        # real per-edge displacement, so the same assembly must hold.
        if not periodic:
            box = None
        force = model(coord, atype, box=box)["force"]

        eps = 1.0e-5
        nloc = coord.shape[1]
        fd_force = torch.zeros_like(force)
        for a in range(nloc):
            for d in range(3):
                cp = coord.clone()
                cp[0, a, d] += eps
                cm = coord.clone()
                cm[0, a, d] -= eps
                e_plus = self._energy(model, cp, atype, box)
                e_minus = self._energy(model, cm, atype, box)
                fd_force[0, a, d] = -(e_plus - e_minus) / (2 * eps)
        boundary = "periodic" if periodic else "non-periodic"
        torch.testing.assert_close(
            force,
            fd_force,
            atol=1.0e-6,
            rtol=1.0e-4,
            msg=f"edge-scatter force != finite difference "
            f"({bridging_method}, {boundary})",
        )

    def test_force_matches_finite_difference(self) -> None:
        """F = -dE/dx for the pure descriptor path."""
        self._check_force_fd("none")

    def test_force_matches_finite_difference_zbl(self) -> None:
        """F = -dE/dx with ZBL bridging routed through the edge ZBL form."""
        self._check_force_fd("ZBL")

    def test_force_matches_finite_difference_nonperiodic(self) -> None:
        """F = -dE/dx for a non-periodic (box=None) cluster."""
        self._check_force_fd("none", periodic=False)

    def test_virial_matches_strain_finite_difference(self) -> None:
        """W = -dE/deps under a random symmetric strain (PBC-correct virial)."""
        model = self._build_model(bridging_method="none")
        coord, atype, box = self._frame()
        virial = model(coord, atype, box=box)["virial"].view(3, 3)

        torch.manual_seed(0)
        s = torch.randn(3, 3, dtype=torch.float64, device=self.device)
        strain = 1.0e-4 * (s + s.transpose(0, 1))
        eye = torch.eye(3, dtype=torch.float64, device=self.device)

        def deformed_energy(sign: float) -> torch.Tensor:
            m = (eye + sign * strain).transpose(0, 1)
            coord_d = coord @ m
            box_d = (box.view(1, 3, 3) @ m).reshape(1, 9)
            return self._energy(model, coord_d, atype, box_d)

        e_plus = deformed_energy(1.0)
        e_minus = deformed_energy(-1.0)
        # dE/dt|_0 = -<strain, W>, central difference over t = +/-1.
        lhs = (strain * virial).sum()
        rhs = -(e_plus - e_minus) / 2.0
        torch.testing.assert_close(lhs, rhs, atol=1.0e-8, rtol=1.0e-4)

    def test_atom_virial_sums_to_global_virial(self) -> None:
        """Half-split per-atom virial reduces to the global virial."""
        for bridging_method in ("none", "ZBL"):
            model = self._build_model(bridging_method=bridging_method)
            coord, atype, box = self._frame()
            out = model(coord, atype, box=box, do_atomic_virial=True)
            torch.testing.assert_close(
                out["atom_virial"].sum(dim=1),
                out["virial"],
                atol=1.0e-10,
                rtol=1.0e-6,
                msg=f"atom_virial sum != global virial ({bridging_method})",
            )


class TestSeZMNativeSpinModel(unittest.TestCase):
    """Validate the native (virtual-atom-free) spin SeZM model.

    The spin vector enters the descriptor as an equivariant feature, so the
    magnetic force is the negative spin gradient of the energy. float64
    finite-difference checks pin ``force_mag = -dE/dspin`` and the conservative
    ``force = -dE/dx``; a joint rotation of geometry and spin confirms SO(3)
    equivariance of energy, force and magnetic force.
    """

    def setUp(self) -> None:
        self.device = env.DEVICE

    def _build_model(self, *, use_compile: bool = False) -> SeZMNativeSpinModel:
        """Build a tiny float64 native-spin model with randomized parameters."""
        params = {
            "type": "dpa4",
            "type_map": ["Ni", "O"],
            "spin": {"use_spin": [True, False], "scheme": "native"},
            "descriptor": {
                "type": "dpa4",
                "sel": [12, 12],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "random_gamma": False,
                "l_schedule": [1, 0],
                "mmax": 1,
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
            "use_compile": use_compile,
        }
        model = get_model(params)
        # Perturb away from the near-identity initialization so the spin
        # embedding measurably shapes the output.
        torch.manual_seed(1234)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        model.eval()
        return model

    def _frame(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Periodic frame; only Ni (type 0) atoms carry spin."""
        coord = torch.tensor(
            [
                [
                    [0.10, 0.05, 0.00],
                    [1.05, 0.30, 0.10],
                    [0.20, 1.40, 0.35],
                    [1.60, 1.15, 0.20],
                    [2.20, 0.10, 1.05],
                ]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        atype = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.int64, device=self.device)
        spin = torch.zeros(1, 5, 3, dtype=torch.float64, device=self.device)
        is_mag = atype[0] == 0
        torch.manual_seed(99)
        spin[0, is_mag] = torch.randn(
            int(is_mag.sum()), 3, dtype=torch.float64, device=self.device
        )
        box = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device=self.device,
        )
        return coord, atype, spin, box

    @staticmethod
    def _proper_rotation(device: torch.device) -> torch.Tensor:
        """A deterministic proper rotation matrix (det = +1)."""
        torch.manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device=device))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    def test_finite_difference_forces(self) -> None:
        """Force = -dE/dx and force_mag = -dE/dspin to finite-difference accuracy.

        The same frame validates both endpoints of the single backward, and the
        per-type spin gate (non-magnetic atoms carry exactly zero magnetic force).
        """
        model = self._build_model()
        coord, atype, spin, box = self._frame()
        out = model(coord, atype, spin, box=box)
        force, force_mag = out["force"], out["force_mag"]

        def energy(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return model(c, atype, s, box=box)["energy"].squeeze()

        eps = 1.0e-5
        nloc = coord.shape[1]
        fd_force = torch.zeros_like(force)
        fd_mag = torch.zeros_like(force_mag)
        for a in range(nloc):
            for d in range(3):
                cp, cm = coord.clone(), coord.clone()
                cp[0, a, d] += eps
                cm[0, a, d] -= eps
                fd_force[0, a, d] = -(energy(cp, spin) - energy(cm, spin)) / (2 * eps)
                sp, sm = spin.clone(), spin.clone()
                sp[0, a, d] += eps
                sm[0, a, d] -= eps
                fd_mag[0, a, d] = -(energy(coord, sp) - energy(coord, sm)) / (2 * eps)
        torch.testing.assert_close(
            force, fd_force, atol=1.0e-6, rtol=1.0e-4, msg="force != -dE/dx"
        )
        torch.testing.assert_close(
            fd_mag, force_mag, atol=1.0e-6, rtol=1.0e-4, msg="force_mag != -dE/dspin"
        )

        torch.testing.assert_close(out["mask_mag"], (atype == 0).reshape(1, -1, 1))
        self.assertEqual(force_mag[0, atype[0] == 1].abs().max().item(), 0.0)

    def test_joint_rotation_equivariance(self) -> None:
        """Energy is invariant and force / force_mag rotate under joint rotation."""
        model = self._build_model()
        coord, atype, spin, box = self._frame()
        out = model(coord, atype, spin, box=box)

        rot = self._proper_rotation(self.device)
        coord_r = torch.einsum("ij,nkj->nki", rot, coord)
        spin_r = torch.einsum("ij,nkj->nki", rot, spin)
        box_r = (box.view(1, 3, 3) @ rot.transpose(0, 1)).reshape(1, 9)
        out_r = model(coord_r, atype, spin_r, box=box_r)

        torch.testing.assert_close(out_r["energy"], out["energy"], atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(
            out_r["force"],
            torch.einsum("ij,nkj->nki", rot, out["force"]),
            atol=1e-8,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            out_r["force_mag"],
            torch.einsum("ij,nkj->nki", rot, out["force_mag"]),
            atol=1e-8,
            rtol=1e-6,
        )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_export_matches_forward(self) -> None:
        """The traced ``.pt2`` export reduces to the public forward.

        The native scheme reuses the energy edge ABI plus the owned-atom spins,
        so the C++ backend builds the edge schema exactly as for a non-spin
        model. ``make_fx`` unfolds the single ``autograd.grad(energy, [edge_vec,
        spin])``; the extended conservative force and the zero-padded magnetic
        force reduce the LAMMPS way (``communicate_extended_output``) back to the
        per-local-atom public force, while ``mask_mag`` is per-local-atom.
        """
        model = self._build_model()
        coord, atype, spin, box = self._frame()
        nloc = coord.shape[1]
        out = model(coord, atype, spin, box=box)
        ext_coord, ext_atype, ext_spin, nlist, mapping = self._extended_spin_inputs(
            model, coord, atype, spin, box
        )
        # Guard the probe: the frame must carry ghosts (nall > nloc) so the
        # magnetic-force path through ghost-image neighbours is actually
        # exercised; otherwise the reduction would be trivial.
        self.assertGreater(ext_coord.shape[1], nloc)

        edge = edge_schema_from_extended(ext_coord, ext_atype, nlist, mapping)
        edge_inputs = (
            edge.coord,
            edge.atype,
            edge.edge_index,
            edge.edge_vec,
            edge.edge_scatter_index,
            edge.edge_mask,
            ext_spin[:, :nloc],
            None,
            None,
            None,
        )
        traced = model.forward_common_lower_exportable(*edge_inputs)
        model_ret = traced(*edge_inputs)

        torch.testing.assert_close(model_ret["energy_redu"], out["energy"])
        torch.testing.assert_close(
            self._reduce_extended(
                model_ret["energy_derv_r"].squeeze(-2), mapping, nloc
            ),
            out["force"],
            atol=1e-9,
            rtol=1e-7,
        )
        torch.testing.assert_close(
            self._reduce_extended(
                model_ret["energy_derv_r_mag"].squeeze(-2), mapping, nloc
            ),
            out["force_mag"],
            atol=1e-9,
            rtol=1e-7,
        )
        torch.testing.assert_close(model_ret["mask_mag"], out["mask_mag"])

    def test_serialization_roundtrip(self) -> None:
        """Serialized native-spin model restores identical predictions."""
        model = self._build_model()
        coord, atype, spin, box = self._frame()
        out = model(coord, atype, spin, box=box)

        restored = SeZMNativeSpinModel.deserialize(model.serialize())
        restored.eval()
        self.assertTrue(restored.has_spin())
        self.assertEqual(restored.get_type_map(), ["Ni", "O"])
        out2 = restored(coord, atype, spin, box=box)
        for key in ["energy", "force", "force_mag"]:
            torch.testing.assert_close(
                out2[key], out[key], atol=1e-10, rtol=1e-8, msg=f"{key} mismatch"
            )

    def test_ener_spin_loss_smoke(self) -> None:
        """The standard ``ener_spin`` loss runs unchanged on the native model."""
        model = self._build_model()
        coord, atype, spin, box = self._frame()
        nloc = coord.shape[1]
        loss_fn = EnergySpinLoss(
            starter_learning_rate=1.0e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
        )
        input_dict = {"coord": coord, "atype": atype, "spin": spin, "box": box}
        label = {
            "energy": torch.zeros(1, 1, dtype=torch.float64, device=self.device),
            "force": torch.zeros(1, nloc, 3, dtype=torch.float64, device=self.device),
            "force_mag": torch.zeros(
                1, nloc, 3, dtype=torch.float64, device=self.device
            ),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_force_mag": 1.0,
        }
        _, loss, more = loss_fn(
            input_dict, model, label, natoms=nloc, learning_rate=1.0e-3
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("rmse_fm", more)

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_matches_eager(self) -> None:
        """The compiled native-spin path matches eager force and magnetic force."""
        coord, atype, spin, box = self._frame()
        model_eager = self._build_model(use_compile=False)
        model_cmp = self._build_model(use_compile=True)
        model_cmp.load_state_dict(model_eager.state_dict())
        model_eager.train()
        model_cmp.train()

        out_e = model_eager(coord, atype, spin, box=box)
        out_c = model_cmp(coord, atype, spin, box=box)
        self.assertIn((True, False), model_cmp.compiled_core_compute_cache)
        for key in ["energy", "force", "force_mag"]:
            _assert_close_with_strict_warning(
                out_c[key],
                out_e[key],
                atol=1.0e-6,
                rtol=1.0e-6,
                msg=f"native-spin compile mismatch on {key}",
            )

    @staticmethod
    def _extended_spin_inputs(
        model: SeZMNativeSpinModel,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Build the 5-tuple lower inputs ``(ext_coord, ext_atype, ext_spin, nlist, mapping)``."""
        extended_coord, extended_atype, mapping, nlist = (
            extend_input_and_build_neighbor_list(
                coord,
                atype,
                model.get_rcut(),
                model.get_sel(),
                mixed_types=model.mixed_types(),
                box=box,
            )
        )
        extended_spin = torch.gather(spin, 1, mapping.unsqueeze(-1).expand(-1, -1, 3))
        return extended_coord, extended_atype, extended_spin, nlist, mapping

    @staticmethod
    def _reduce_extended(
        extended: torch.Tensor, mapping: torch.Tensor, nloc: int
    ) -> torch.Tensor:
        """Scatter-sum an extended ``(nf, nall, 3)`` tensor onto local owners."""
        reduced = torch.zeros(
            extended.shape[0], nloc, 3, dtype=extended.dtype, device=extended.device
        )
        return reduced.scatter_reduce(
            1, mapping.unsqueeze(-1).expand(-1, -1, 3), extended, reduce="sum"
        )

    def test_allow_missing_label_relaxes_spin_data_requirement(self) -> None:
        """``allow_missing_label`` relaxes the spin data requirement to optional with a
        zero default, and the flag is excluded from serialization.

        ``use_spin`` is given as an element symbol here, so the test also covers the
        symbol form being expanded against ``type_map`` into a per-type boolean list.
        """
        from deepmd.pt.train.training import (
            get_additional_data_requirement,
        )

        def build(allow_missing_label: bool | None) -> SeZMNativeSpinModel:
            params = {
                "type": "dpa4",
                "type_map": ["Ni", "O"],
                # Element-symbol use_spin: expanded against type_map to [True, False].
                "spin": {"use_spin": ["Ni"], "scheme": "native"},
                "descriptor": {
                    "type": "dpa4",
                    "sel": [2, 2],
                    "rcut": 3.0,
                    "channels": 4,
                    "n_radial": 3,
                    "l_schedule": [1, 0],
                    "mmax": 1,
                    "use_env_seed": False,
                    "random_gamma": False,
                    "precision": "float64",
                },
                "fitting_net": {"neuron": [4], "precision": "float64", "seed": 1},
            }
            if allow_missing_label is not None:
                params["spin"]["allow_missing_label"] = allow_missing_label
            return get_model(params)

        for allow_missing_label, expected_must in (
            (None, True),
            (False, True),
            (True, False),
        ):
            model = build(allow_missing_label)
            # The symbol ``["Ni"]`` expands to the per-type mask over ``["Ni", "O"]``.
            self.assertEqual(model.spin.use_spin.tolist(), [True, False])
            self.assertEqual(model.spin.allow_missing_label, bool(allow_missing_label))
            spin_req = {
                item.key: item for item in get_additional_data_requirement(model)
            }["spin"]
            self.assertEqual(spin_req.must, expected_must)
            self.assertEqual(spin_req.default, 0.0)

        self.assertNotIn("allow_missing_label", build(True).spin.serialize())


class TestSeZMModelBridging(unittest.TestCase):
    """Test SeZM model with ZBL bridging enabled."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(self, *, bridging_method: str = "none") -> dict:
        return {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "focus_compete": False,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": False,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 0,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": True,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": False,
            "bridging_method": bridging_method,
            "bridging_r_inner": 0.8,
            "bridging_r_outer": 1.2,
        }

    def test_bridging_none_unchanged(self) -> None:
        """Test that bridging_method='none' produces no inter_potential."""
        model = get_sezm_model(self._build_model_params(bridging_method="none"))
        self.assertIsNone(model.inter_potential)
        self.assertEqual(model.bridging_method, "NONE")

    def test_bridging_zbl_creates_potential(self) -> None:
        """Test that bridging_method='ZBL' creates InterPotential and InnerClamp."""
        model = get_sezm_model(self._build_model_params(bridging_method="ZBL"))
        self.assertIsNotNone(model.inter_potential)
        self.assertEqual(model.bridging_method, "ZBL")
        self.assertIsNotNone(model.atomic_model.descriptor.inner_clamp)

    def test_zbl_adds_energy(self) -> None:
        """Test that ZBL bridging adds energy to the model output."""
        model_plain = get_sezm_model(self._build_model_params(bridging_method="none"))
        model_zbl = get_sezm_model(self._build_model_params(bridging_method="ZBL"))

        sd = model_plain.state_dict()
        model_zbl.load_state_dict(sd, strict=False)

        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        atype = torch.tensor([[0, 1, 0]], dtype=torch.int32, device=self.device)
        box = torch.tensor(
            [[10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]],
            dtype=torch.float32,
            device=self.device,
        )

        model_plain.eval()
        model_zbl.eval()

        out_plain = model_plain(coord, atype, box=box)
        out_zbl = model_zbl(coord, atype, box=box)

        energy_diff = (out_zbl["energy"] - out_plain["energy"]).item()
        self.assertGreater(
            energy_diff,
            0.0,
            "ZBL bridging should add positive (repulsive) energy",
        )


class TestSeZMModelModes(unittest.TestCase):
    """Targeted regression tests for SeZM `ener` / `dens` mode routing."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(
        self,
        *,
        use_compile: bool = False,
        bridging_method: str = "none",
    ) -> dict:
        return {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "focus_compete": False,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": False,
                "l_schedule": [1, 1],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 0,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": use_compile,
            "bridging_method": bridging_method,
        }

    def _tiny_system(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.2, 0.0],
                    [0.2, 1.0, 0.3],
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        atype = torch.tensor([[0, 1, 0]], device=self.device, dtype=torch.int32)
        box = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            device=self.device,
            dtype=torch.float32,
        )
        force = torch.tensor(
            [
                [
                    [0.2, -0.1, 0.0],
                    [-0.3, 0.4, 0.1],
                    [0.1, 0.2, -0.2],
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        noise_mask = torch.tensor(
            [[True, False, True]],
            device=self.device,
            dtype=torch.bool,
        )
        return coord, atype, box, force, noise_mask

    def _dens_stat_samples(self) -> list[dict[str, torch.Tensor | np.float32]]:
        """Build a tiny SeZM `dens` statistics set with force labels."""
        return [
            {
                "atype": torch.tensor(
                    [[0, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 1, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[10.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
            {
                "atype": torch.tensor(
                    [[0, 0]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 2, 0]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[8.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
            {
                "atype": torch.tensor(
                    [[1, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 0, 2]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[12.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[8.0, 10.0, 12.0], [8.0, 10.0, 12.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
        ]

    def _expected_dens_force_rmsd(
        self,
        sampled: list[dict[str, torch.Tensor | np.float32]],
    ) -> float:
        """Compute the expected global direct-force RMSD."""
        force_square_sum = 0.0
        force_atom_count = 0
        for sample in sampled:
            force = sample["force"].detach().cpu().numpy()
            force_square_sum += float(np.square(force).sum())
            force_atom_count += int(force.shape[0] * force.shape[1])
        return float(np.sqrt(force_square_sum / force_atom_count))

    def test_training_setup_routes_mode_without_rebuilding_energy_head(self) -> None:
        """Training setup should route SeZM mode without rebuilding the energy head."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        energy_param_before = (
            next(model.atomic_model.fitting_net.parameters()).detach().clone()
        )
        prepare_model_for_loss(model, {"type": "dens"})
        self.assertEqual(model.get_active_mode(), "dens")
        self.assertIsNotNone(model.atomic_model.dens_fitting_net)
        prepare_model_for_loss(model, {"type": "ener"})
        coord, atype, box, _, _ = self._tiny_system()
        loss_module = EnergyStdLoss(
            starter_learning_rate=1.0e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
        )
        _, loss, _ = loss_module(
            {
                "coord": coord,
                "atype": atype,
                "box": box,
            },
            model,
            {
                "energy": torch.zeros((1, 1), device=self.device, dtype=torch.float32),
                "find_energy": 1.0,
            },
            natoms=atype.shape[1],
            learning_rate=1.0e-3,
        )
        energy_param_after = next(model.atomic_model.fitting_net.parameters()).detach()
        torch.testing.assert_close(energy_param_after, energy_param_before)
        self.assertEqual(model.get_active_mode(), "ener")
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_loading_handles_optional_dens_head(self) -> None:
        """Checkpoint loading should respect whether `dens` weights exist."""
        params = self._build_model_params(use_compile=False)
        model = get_sezm_model(params)
        state_without_dens = {
            key: value
            for key, value in model.state_dict().items()
            if "dens_fitting_net" not in key
        }
        fresh_model = get_sezm_model(params)
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        fresh_model.load_state_dict(state_without_dens, strict=True)
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        self.assertEqual(fresh_model.get_active_mode(), "ener")
        coord, atype, box, _, _ = self._tiny_system()
        out = fresh_model(coord, atype, box=box)
        self.assertIn("energy", out)
        self.assertIn("force", out)
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.set_active_mode("dens")
        dens_state = model.state_dict()
        fresh_model = get_sezm_model(self._build_model_params(use_compile=False))
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        fresh_model.load_state_dict(dens_state, strict=True)
        self.assertIsNotNone(fresh_model.atomic_model.dens_fitting_net)
        self.assertEqual(fresh_model.get_active_mode(), "dens")

    def test_dens_forward_returns_direct_force_outputs(self) -> None:
        """`dens` mode should expose direct-force outputs without virial branches."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.set_active_mode("dens")
        coord, atype, box, force, noise_mask = self._tiny_system()
        out = model(
            coord,
            atype,
            box=box,
            force_input=force,
            noise_mask=noise_mask,
        )
        self.assertIn("energy", out)
        self.assertIn("atom_energy", out)
        self.assertIn("force", out)
        self.assertNotIn("virial", out)
        self.assertEqual(out["force"].shape, force.shape)

    def test_dens_loss_forward_smoke(self) -> None:
        """`DeNSLoss` should build noisy inputs and return a finite training loss."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        prepare_model_for_loss(model, {"type": "dens"})
        loss_module = DeNSLoss(
            starter_learning_rate=1.0e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            dens_prob=1.0,
            dens_std=0.025,
            dens_corrupt_ratio=0.5,
            dens_denoising_pos_coefficient=10.0,
            loss_func="mae",
        )
        coord, atype, box, force, _ = self._tiny_system()
        label = {
            "energy": torch.zeros((1, 1), device=self.device, dtype=torch.float32),
            "force": force,
            "find_energy": 1.0,
            "find_force": 1.0,
        }
        model_pred, loss, more_loss = loss_module(
            {
                "coord": coord,
                "atype": atype,
                "box": box,
            },
            model,
            label,
            natoms=atype.shape[1],
            learning_rate=1.0e-3,
        )
        self.assertEqual(model.get_active_mode(), "dens")
        self.assertIn("force", model_pred)
        self.assertTrue(torch.isfinite(loss))

    def test_dens_stat_roundtrip(self) -> None:
        """`dens` statistics should roundtrip the global direct-force RMSD."""
        sampled = self._dens_stat_samples()
        expected_force_rmsd = self._expected_dens_force_rmsd(sampled)

        model = get_sezm_model(self._build_model_params(use_compile=False))
        prepare_model_for_loss(model, {"type": "dens"})

        with tempfile.TemporaryDirectory() as tmpdir:
            h5file = Path(tmpdir) / "sezm_stat.hdf5"
            with h5py.File(h5file, "w"):
                pass

            stat_path = DPPath(str(h5file), "a")
            try:
                model.atomic_model.compute_or_load_stat(
                    lambda: sampled,
                    stat_file_path=stat_path,
                )
                self.assertAlmostEqual(
                    model.atomic_model.dens_force_rmsd.item(),
                    expected_force_rmsd,
                    places=7,
                )
                self.assertEqual(model.get_active_mode(), "dens")

                stored_force_rmsd = (stat_path / "O H" / "rmsd_dforce").load_numpy()
                self.assertAlmostEqual(
                    float(np.asarray(stored_force_rmsd).reshape(-1)[0]),
                    expected_force_rmsd,
                    places=7,
                )

                fresh_model = get_sezm_model(
                    self._build_model_params(use_compile=False)
                )
                prepare_model_for_loss(fresh_model, {"type": "dens"})

                def raise_error() -> None:
                    raise RuntimeError("statistics should be restored from file")

                fresh_model.atomic_model.compute_or_load_stat(
                    raise_error,
                    stat_file_path=stat_path,
                )
                self.assertAlmostEqual(
                    fresh_model.atomic_model.dens_force_rmsd.item(),
                    expected_force_rmsd,
                    places=7,
                )
                self.assertEqual(fresh_model.get_active_mode(), "dens")
            finally:
                stat_path.root.close()


# =============================================================================
# LoRA fine-tune tests
# =============================================================================


class _LoRATestCase(unittest.TestCase):
    """Shared device / seeding base for LoRA tests."""

    def setUp(self) -> None:
        self.device = env.DEVICE


class TestLoRASO3Adapter(_LoRATestCase):
    """Unit tests for :class:`LoRASO3`."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(17)

    def _build_base_and_lora(
        self,
        *,
        rank: int = 4,
        lmax: int = 2,
        in_channels: int = 4,
        out_channels: int = 5,
        n_focus: int = 1,
        mlp_bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[SO3Linear, LoRASO3]:
        base = SO3Linear(
            lmax=lmax,
            in_channels=in_channels,
            out_channels=out_channels,
            n_focus=n_focus,
            dtype=dtype,
            mlp_bias=mlp_bias,
            trainable=True,
            seed=101,
        )
        lora = LoRASO3(base, rank=rank, alpha=float(rank))
        return base, lora

    def _random_input(self, lora: LoRASO3) -> torch.Tensor:
        n_dim = (lora.lmax + 1) ** 2
        return torch.randn(
            3,
            n_dim,
            lora.n_focus,
            lora.in_channels,
            device=self.device,
            dtype=lora.dtype,
        )

    def test_merge_into_base_matches_forward(self) -> None:
        """Numerical parity between LoRASO3 forward and its merged base."""
        _, lora = self._build_base_and_lora()
        torch.nn.init.normal_(lora.B_by_l, std=0.05)
        x = self._random_input(lora)
        out_lora = lora(x)
        merged = lora.merge_into_base()
        out_merged = merged(x)
        torch.testing.assert_close(out_lora, out_merged, atol=1e-6, rtol=1e-5)
        self.assertIs(type(merged), SO3Linear)


class TestLoRASO2Adapter(_LoRATestCase):
    """Unit tests for :class:`LoRASO2`."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(23)

    def _build_base_and_lora(
        self,
        *,
        rank: int = 4,
        lmax: int = 2,
        mmax: int = 2,
        in_channels: int = 4,
        out_channels: int = 5,
        n_focus: int = 1,
        mlp_bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[SO2Linear, LoRASO2]:
        base = SO2Linear(
            lmax=lmax,
            mmax=mmax,
            in_channels=in_channels,
            out_channels=out_channels,
            n_focus=n_focus,
            dtype=dtype,
            mlp_bias=mlp_bias,
            seed=202,
            trainable=True,
        )
        lora = LoRASO2(base, rank=rank, alpha=float(rank))
        return base, lora

    def _random_input(self, lora: LoRASO2) -> torch.Tensor:
        # Focus-major ``(F, E, D_m, C)`` contract; E=3 edges.
        return torch.randn(
            lora.n_focus,
            3,
            lora.reduced_dim,
            lora.in_channels,
            device=self.device,
            dtype=lora.dtype,
        )

    def _randomize_lora_B(self, lora: LoRASO2) -> None:
        torch.nn.init.normal_(lora.B_m0, std=0.05)
        for b in lora.B_m:
            torch.nn.init.normal_(b, std=0.05)

    def test_merge_into_base_matches_forward(self) -> None:
        """Numerical parity between LoRASO2 forward and its merged base."""
        _, lora = self._build_base_and_lora()
        self._randomize_lora_B(lora)
        x = self._random_input(lora)
        out_lora = lora(x)
        merged = lora.merge_into_base()
        out_merged = merged(x)
        torch.testing.assert_close(out_lora, out_merged, atol=1e-6, rtol=1e-5)
        self.assertIs(type(merged), SO2Linear)

    def test_z_rotation_equivariance(self) -> None:
        """Rotating x by the m-major z-block rotation commutes with LoRASO2 forward."""
        lmax, mmax = 2, 1
        _, lora = self._build_base_and_lora(
            rank=3, lmax=lmax, mmax=mmax, in_channels=6, out_channels=4, n_focus=1
        )
        self._randomize_lora_B(lora)
        batch = 8
        dtype = lora.dtype
        x = torch.randn(
            lora.n_focus,
            batch,
            lora.reduced_dim,
            lora.in_channels,
            device=self.device,
            dtype=dtype,
        )
        angles = torch.rand(batch, device=self.device, dtype=dtype) * 2 * math.pi
        z_mat = _build_m_major_z_rotation(angles, lmax, mmax, self.device)
        x_rot = torch.einsum("eij,fejc->feic", z_mat, x)
        lhs = lora(x_rot)
        rhs = torch.einsum("eij,fejc->feic", z_mat, lora(x))
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


class TestApplyLoRAToSeZM(_LoRATestCase):
    """Tests for the full SeZM LoRA injection policy."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(31)
        self.model = get_sezm_model(_build_lora_sezm_model_params())
        apply_lora_to_sezm(self.model, rank=4, alpha=4.0)

    def test_so3_and_so2_are_subclassed(self) -> None:
        """Every SO3Linear / SO2Linear submodule is now a LoRA subclass."""
        n_lora_so3 = 0
        n_lora_so2 = 0
        for mod in self.model.modules():
            if type(mod) is SO3Linear:
                self.fail("Found a bare SO3Linear; apply_lora_to_sezm missed it.")
            if type(mod) is SO2Linear:
                self.fail("Found a bare SO2Linear; apply_lora_to_sezm missed it.")
            if isinstance(mod, LoRASO3):
                n_lora_so3 += 1
            elif isinstance(mod, LoRASO2):
                n_lora_so2 += 1
        self.assertGreater(n_lora_so3, 0)
        self.assertGreater(n_lora_so2, 0)

    def test_lora_base_weights_are_frozen(self) -> None:
        """Base weight matrices inside every LoRA wrapper stay frozen.

        Bias-like parameters (``bias`` / ``bias0``) remain trainable by the
        leaf-name rule "any leaf containing 'bias' is unfrozen"; this test
        asserts the large weight matrices specifically.
        """
        for mod in self.model.modules():
            if isinstance(mod, LoRASO3):
                self.assertFalse(mod.weight.requires_grad)
            elif isinstance(mod, LoRASO2):
                self.assertFalse(mod.weight_m0.requires_grad)
                for w in mod.weight_m:
                    self.assertFalse(w.requires_grad)

    def test_lora_adapter_params_are_trainable(self) -> None:
        """LoRA A/B parameters are trainable everywhere."""
        for mod in self.model.modules():
            if isinstance(mod, LoRASO3):
                self.assertTrue(mod.A_by_l.requires_grad)
                self.assertTrue(mod.B_by_l.requires_grad)
            elif isinstance(mod, LoRASO2):
                self.assertTrue(mod.A_m0.requires_grad)
                self.assertTrue(mod.B_m0.requires_grad)
                for a, b in zip(mod.A_m, mod.B_m, strict=True):
                    self.assertTrue(a.requires_grad)
                    self.assertTrue(b.requires_grad)

    def test_full_unfreezes(self) -> None:
        """fitting_net / radial_embedding / env_seed_embedding are fully trainable."""
        fitting = self.model.atomic_model.fitting_net
        self.assertIsNotNone(fitting)
        for p in fitting.parameters():
            self.assertTrue(p.requires_grad)
        radial = self.model.atomic_model.descriptor.radial_embedding
        for p in radial.parameters():
            self.assertTrue(p.requires_grad)
        env_seed = self.model.atomic_model.descriptor.env_seed_embedding
        self.assertIsNotNone(env_seed)
        # All non-type-embed params inside env_seed must be trainable.
        for name, p in env_seed.named_parameters():
            if name.endswith("adam_type_embedding"):
                continue
            self.assertTrue(
                p.requires_grad, msg=f"env_seed param {name} should be trainable"
            )

    def test_override_freezes_type_embed_and_radial_freqs(self) -> None:
        """``adam_type_embedding`` and ``adam_freqs`` stay frozen."""
        frozen_leaves = {"adam_type_embedding", "adam_freqs"}
        hit = dict.fromkeys(frozen_leaves, 0)
        for name, p in self.model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in frozen_leaves:
                self.assertFalse(
                    p.requires_grad,
                    msg=f"{name} should stay frozen after LoRA injection",
                )
                hit[leaf] += 1
        for leaf, count in hit.items():
            self.assertGreater(
                count,
                0,
                msg=f"No parameter with leaf {leaf} found; test coverage gap.",
            )

    def test_override_freezes_gated_activation(self) -> None:
        """Every parameter inside a GatedActivation is frozen."""
        found = False
        for mod in self.model.modules():
            if isinstance(mod, GatedActivation):
                for p in mod.parameters():
                    self.assertFalse(p.requires_grad)
                    found = True
        self.assertTrue(found, msg="Expected at least one GatedActivation in SeZM.")


class TestBuildMergedStateDict(_LoRATestCase):
    """Tests for the non-destructive merged-state-dict helper."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(41)
        self.model = get_sezm_model(_build_lora_sezm_model_params())
        apply_lora_to_sezm(self.model, rank=4, alpha=4.0)
        # Randomize every B so LoRA delta is non-trivial.
        for mod in self.model.modules():
            if isinstance(mod, LoRASO3):
                torch.nn.init.normal_(mod.B_by_l, std=0.05)
            elif isinstance(mod, LoRASO2):
                torch.nn.init.normal_(mod.B_m0, std=0.05)
                for b in mod.B_m:
                    torch.nn.init.normal_(b, std=0.05)

    def test_keys_match_plain_sezm(self) -> None:
        """Merged state dict has the same keys as a never-LoRA'ed sibling model."""
        plain_model = get_sezm_model(_build_lora_sezm_model_params())
        plain_keys = set(plain_model.state_dict().keys())
        merged = build_merged_state_dict(self.model)
        merged_keys = set(merged.keys())
        self.assertEqual(merged_keys, plain_keys)
        # Explicitly assert that no LoRA-only key survived.
        for key in merged_keys:
            leaf = key.rsplit(".", 1)[-1]
            self.assertNotIn(
                leaf,
                {"A_by_l", "B_by_l", "A_m0", "B_m0"},
                msg=f"LoRA-only leaf {leaf} should not appear in merged state",
            )
            parts = key.split(".")
            i = len(parts) - 1
            while i > 0 and parts[i].isdigit():
                i -= 1
            self.assertNotIn(parts[i], {"A_m", "B_m"})

    def test_weight_values_include_delta(self) -> None:
        """Every LoRA weight key in the merged state equals ``W + ΔW``."""
        merged = build_merged_state_dict(self.model)
        # Keys live under `atomic_model.descriptor....` inside SeZMModel; helper
        # walks self.model.named_modules() so prefix is "" at the top.
        for name, mod in self.model.named_modules():
            prefix = name + "." if name else ""
            if isinstance(mod, LoRASO3):
                expected = (
                    mod.weight.detach()
                    + torch.einsum("lor,lri->lio", mod.B_by_l, mod.A_by_l).detach()
                    * mod.scaling
                )
                torch.testing.assert_close(
                    merged[prefix + "weight"], expected, atol=1e-6, rtol=1e-5
                )
            elif isinstance(mod, LoRASO2):
                expected_m0 = (
                    mod.weight_m0.detach()
                    + torch.einsum("ri,or->io", mod.A_m0, mod.B_m0).detach()
                    * mod.scaling
                )
                torch.testing.assert_close(
                    merged[prefix + "weight_m0"], expected_m0, atol=1e-6, rtol=1e-5
                )
                for m_idx, w in enumerate(mod.weight_m):
                    expected_m = (
                        w.detach()
                        + torch.einsum(
                            "ri,or->io", mod.A_m[m_idx], mod.B_m[m_idx]
                        ).detach()
                        * mod.scaling
                    )
                    torch.testing.assert_close(
                        merged[prefix + f"weight_m.{m_idx}"],
                        expected_m,
                        atol=1e-6,
                        rtol=1e-5,
                    )


class TestSeZMModelLoRACompile(unittest.TestCase):
    """LoRA + ``torch.compile`` end-to-end consistency test.

    Runs the SeZM ``ener`` path with ``use_compile=True`` against the eager
    reference on the same LoRA-injected model (randomized ``B`` so the LoRA
    delta is non-trivial) and checks forward / first-order / second-order
    consistency, mirroring the style of
    :meth:`TestSeZMModelCompile.test_forward_backward_double_backward_matches_compile`.
    """

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _tiny_system(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a compact two-frame, three-atom system for LoRA compile tests."""
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.1, 0.3, 0.0], [0.2, 1.5, 0.4]],
                [[0.1, 0.2, 0.3], [0.9, 1.0, 0.1], [2.0, 0.5, 1.2]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        atype = torch.tensor(
            [[0, 1, 0], [1, 0, 1]], dtype=torch.int32, device=self.device
        )
        box = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return coord, atype, box

    @staticmethod
    def _build_matched_lora_models() -> tuple[SeZMModel, SeZMModel]:
        """Build eager + compile SeZM twins that share LoRA-augmented weights."""
        params_eager = _build_lora_sezm_model_params(use_compile=False)
        model_eager = get_sezm_model(params_eager)
        apply_lora_to_sezm(model_eager, rank=2, alpha=4.0)
        # Randomize every LoRA B so the LoRA delta is non-trivial across both
        # branches; randomize A similarly so the low-rank term has full rank.
        for mod in model_eager.modules():
            if isinstance(mod, LoRASO3):
                torch.nn.init.normal_(mod.A_by_l, std=0.05)
                torch.nn.init.normal_(mod.B_by_l, std=0.05)
            elif isinstance(mod, LoRASO2):
                torch.nn.init.normal_(mod.A_m0, std=0.05)
                torch.nn.init.normal_(mod.B_m0, std=0.05)
                for a, b in zip(mod.A_m, mod.B_m, strict=True):
                    torch.nn.init.normal_(a, std=0.05)
                    torch.nn.init.normal_(b, std=0.05)

        params_compile = _build_lora_sezm_model_params(use_compile=True)
        model_compile = get_sezm_model(params_compile)
        apply_lora_to_sezm(model_compile, rank=2, alpha=4.0)
        # After injection both models share the same named-parameter layout;
        # copying the eager state_dict also copies the randomized LoRA A/B.
        model_compile.load_state_dict(model_eager.state_dict())
        return model_eager, model_compile

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_forward_and_backward_match_eager(self) -> None:
        """Forward / first-order / second-order outputs agree with eager."""
        coord, atype, box = self._tiny_system()
        model_eager, model_compile = self._build_matched_lora_models()
        model_eager.train()
        model_compile.train()

        # === Forward ===
        out_eager = model_eager(coord, atype, box=box)
        out_compile = model_compile(coord, atype, box=box)
        energy_atol = 1.0e-6 if self.device == torch.device("cpu") else 1.0e-4
        energy_rtol = 1.0e-6 if self.device == torch.device("cpu") else 1.0e-4
        _assert_close_with_strict_warning(
            out_eager["energy"],
            out_compile["energy"],
            atol=energy_atol,
            rtol=energy_rtol,
            msg="LoRA energy mismatch",
        )
        _assert_close_with_strict_warning(
            out_eager["force"],
            out_compile["force"],
            atol=2.0e-4,
            rtol=1.0e-5,
            msg="LoRA force mismatch",
        )

        # === First-order backward (d energy / d params) ===
        model_eager.zero_grad(set_to_none=True)
        model_compile.zero_grad(set_to_none=True)
        out_eager["energy"].sum().backward()
        out_compile["energy"].sum().backward()
        grad_atol = 1.0e-5 if self.device == torch.device("cpu") else 2.0e-3
        grad_rtol = 1.0e-5 if self.device == torch.device("cpu") else 3.0e-3
        force_grad_atol = 1.0e-2
        force_grad_rtol = 1.0e-4
        grads_eager = {
            name: (
                torch.zeros_like(param)
                if param.grad is None
                else param.grad.detach().clone()
            )
            for name, param in model_eager.named_parameters()
        }
        grads_compile = {
            name: (
                torch.zeros_like(param)
                if param.grad is None
                else param.grad.detach().clone()
            )
            for name, param in model_compile.named_parameters()
        }
        self.assertEqual(set(grads_eager.keys()), set(grads_compile.keys()))
        for name in grads_eager.keys():
            _assert_close_with_strict_warning(
                grads_eager[name],
                grads_compile[name],
                atol=grad_atol,
                rtol=grad_rtol,
                msg=f"energy-grad mismatch at {name}",
            )

        # === Second-order backward via force loss (d force^2 / d params) ===
        model_eager.zero_grad(set_to_none=True)
        model_compile.zero_grad(set_to_none=True)
        out_eager = model_eager(coord, atype, box=box)
        out_compile = model_compile(coord, atype, box=box)
        torch.sum(out_eager["force"] * out_eager["force"]).backward()
        torch.sum(out_compile["force"] * out_compile["force"]).backward()
        grads_eager_2 = {
            name: (
                torch.zeros_like(param)
                if param.grad is None
                else param.grad.detach().clone()
            )
            for name, param in model_eager.named_parameters()
        }
        grads_compile_2 = {
            name: (
                torch.zeros_like(param)
                if param.grad is None
                else param.grad.detach().clone()
            )
            for name, param in model_compile.named_parameters()
        }
        for name in grads_eager_2.keys():
            _assert_close_with_strict_warning(
                grads_eager_2[name],
                grads_compile_2[name],
                atol=force_grad_atol,
                rtol=force_grad_rtol,
                msg=f"force-grad-sq mismatch at {name}",
            )
