# SPDX-License-Identifier: LGPL-3.0-or-later
"""The pt_expt DPA4 training paths: bindings, and gradients against the dense path.

The accelerated training paths are two mutually independent layers over the
same block. The Triton layer (``DP_TRITON_TRAIN``) replaces individual stages
-- the rotations, the block-diagonal GEMM, the radial mixer, the gated
activation, the rotate-mix front end, the segmented attention softmax and the
flash aggregation -- each carrying its own analytic backward and second order.
The CUDA layer (``DP_CUDA_TRAIN``) replaces the whole value stream up to the
attention aggregation with one operator, and supersedes the Triton stages it
covers while leaving the attention span to them.

Two things are asserted: that each gate binds exactly the paths it owns, and
that a training step through them reproduces the dense reference's loss and
coordinate gradient. The gates are read at construction to decide the
bindings, so every case builds its own descriptor.
"""

from __future__ import (
    annotations,
)

import numpy as np
import pytest
import torch

try:
    # Loads ``libdeepmd_op_pt.so``, which registers the hand-written operators.
    import deepmd.pt.cxx_op  # noqa: F401
except ImportError:
    pass

from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.descriptor.dpa4_nn.activation import (
    GatedActivation,
)
from deepmd.pt_expt.descriptor.dpa4_nn.grid_net import (
    S2GridNet,
    SO3GridNet,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
    DynamicRadialDegreeMixer,
    SO2Convolution,
    SO2Linear,
    active_triton_level,
)
from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    op_available as cuda_value_available,
)
from deepmd.pt_expt.kernels.triton.sezm.grid_pair import (
    GRID_PAIR_TRITON_AVAILABLE,
)
from deepmd.pt_expt.kernels.triton.sezm.segment_softmax import (
    SEGMENT_SOFTMAX_TRITON_AVAILABLE,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_block_gemm import (
    SO2_BLOCK_GEMM_TRITON_AVAILABLE,
    slices_supported,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    SO2_VALUE_PATH_TRITON_AVAILABLE,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_train_enabled,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
)

TRAIN_GATES = ("DP_TRITON_TRAIN", "DP_CUDA_TRAIN")
INFER_GATES = ("DP_TRITON_INFER", "DP_CUDA_INFER", "DP_CUTILE_INFER", "DP_CUTE_INFER")


def _make_descriptor(ntypes: int, sel: list[int], rcut: float) -> DescrptDPA4:
    """Build a small DPA4 descriptor in the deployed layout."""
    return DescrptDPA4(
        ntypes=ntypes,
        sel=sel,
        rcut=rcut,
        channels=32,
        n_radial=8,
        lmax=2,
        mmax=1,
        n_blocks=2,
        mixing_layers=3,
        radial_so2_mode="degree_channel",
        radial_so2_rank=1,
        n_atten_head=1,
        grid_branch=[1, 1, 1],
        s2_activation=[False, True],
        random_gamma=False,
        precision="float32",
        seed=7,
    )


def _clear_gates(monkeypatch) -> None:
    """Silence every accelerated gate so a case only sets what it needs."""
    for name in TRAIN_GATES + INFER_GATES:
        monkeypatch.setenv(name, "0")


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_cuda_train_gate_accepts_shared_truthy_values(monkeypatch, value: str) -> None:
    """The CUDA training gate accepts the module's common truthy vocabulary."""
    monkeypatch.setenv("DP_CUDA_TRAIN", value)
    assert cuda_train_enabled()


@pytest.mark.parametrize(
    ("gate_name", "training"),
    [("DP_TRITON_TRAIN", True), ("DP_TRITON_INFER", False)],
    ids=("training", "inference"),
)
@pytest.mark.parametrize("enabled", [0, 1])
def test_triton_mode_gate_binds_each_stage(
    monkeypatch, gate_name: str, training: bool, enabled: int
) -> None:
    """Each Triton gate binds every supported stage for only its own mode."""
    _clear_gates(monkeypatch)
    monkeypatch.setenv(gate_name, str(enabled))
    requested = bool(enabled)
    train_level = enabled if training else 0
    infer_level = enabled if not training else 0

    descriptor = _make_descriptor(2, [20], 4.0)
    convolutions = [
        module for module in descriptor.modules() if isinstance(module, SO2Convolution)
    ]
    assert convolutions

    for conv in convolutions:
        assert conv.triton_train_level == train_level
        assert conv.triton_infer_level == infer_level
        # Rotation and flash wrappers retain their eager implementations when
        # Triton is unavailable, so their binding follows the mode gates alone.
        assert (conv.triton_l_1_rotate_to_local is not None) is requested
        assert (conv.triton_l_1_rotate_back is not None) is requested
        assert (conv.flash_attention is not None) is requested
        assert conv.flash_attention_supports_training is (requested and training)
        # Segment softmax has no wrapper-level fallback and binds only when its
        # own Triton implementation is importable.
        assert (conv.triton_l_1_segment_softmax is not None) is (
            requested and SEGMENT_SOFTMAX_TRITON_AVAILABLE
        )
        # The rotate-mix front end is bound by a profitability bound on the
        # hidden width, which this narrow block sits below.
        assert conv.hidden_channels < 128
        assert conv.triton_l_1_rotate_mix is None
        # The CUDA gate is off, so the value stream stays on the stages.
        assert conv.cuda_train_value is None

    for module in descriptor.modules():
        if isinstance(module, SO2Linear):
            # The fused GEMM additionally needs every |m| block width to align
            # to its BN=64 tile, which a narrow block does not satisfy.
            aligned = slices_supported(module._block_diag_slices)
            assert (module.triton_l_1_block_diag_gemm is not None) is (
                requested and SO2_BLOCK_GEMM_TRITON_AVAILABLE and aligned
            )
        if isinstance(module, DynamicRadialDegreeMixer):
            # The callable contains its eager fallback, so construction binds it
            # whenever either mode requests the stage.
            assert (module.triton_l_1_radial_mix is not None) is requested
        if isinstance(module, GatedActivation):
            assert module.triton_train_level == train_level
            assert module.triton_infer_level == infer_level
            # The fused activation is bounded by the register footprint of one
            # focus stream's degrees.
            footprint_ok = module.channels <= 32 or (
                module.channels <= 64 and module.lmax <= 3
            )
            assert (module._fused_gated_act is not None) is (
                requested and footprint_ok and module.layout == "fndc"
            )

    # A shared binding is only a construction-time capability. Runtime dispatch
    # follows the active module mode, so the opposite gate remains disabled.
    for mode, active_level in ((training, enabled), (not training, 0)):
        descriptor.train(mode)
        for module in descriptor.modules():
            if isinstance(
                module, (SO2Convolution, SO2Linear, DynamicRadialDegreeMixer)
            ):
                assert active_triton_level(module) == active_level
            if isinstance(module, SO2Convolution):
                assert module.rotation_kernel_active() is bool(active_level)
            if isinstance(module, GatedActivation):
                level = (
                    module.triton_train_level
                    if module.training
                    else module.triton_infer_level
                )
                assert level == active_level


def test_cuda_train_gate_binds_the_value_stream(monkeypatch) -> None:
    """``DP_CUDA_TRAIN`` binds the fused value path without the Triton gate."""
    if not cuda_value_available():
        pytest.skip("the DPA4 CUDA training operators are unavailable")
    _clear_gates(monkeypatch)
    monkeypatch.setenv("DP_CUDA_TRAIN", "1")

    descriptor = _make_descriptor(2, [20], 4.0)
    convolutions = [
        module for module in descriptor.modules() if isinstance(module, SO2Convolution)
    ]
    assert convolutions
    for conv in convolutions:
        assert conv.cuda_train_value is not None
        # The two layers are independent: the CUDA value stream does not
        # switch on any Triton stage, and the attention span stays dense
        # until the Triton gate asks for it.
        assert conv.triton_train_level == 0
        assert conv.triton_l_1_segment_softmax is None
        assert conv.flash_attention_supports_training is False


def test_cuda_triton_train_reuses_packed_wigner_runs(monkeypatch) -> None:
    """The composed training path does not construct dense Wigner matrices."""
    if not cuda_value_available():
        pytest.skip("the DPA4 CUDA training operators are unavailable")
    if not SO2_VALUE_PATH_TRITON_AVAILABLE:
        pytest.skip("Triton is unavailable")
    _clear_gates(monkeypatch)
    monkeypatch.setenv("DP_CUDA_TRAIN", "1")
    monkeypatch.setenv("DP_TRITON_TRAIN", "1")

    descriptor = _make_descriptor(2, [20], 4.0).train()
    assert descriptor.cuda_train_covers_all_blocks
    assert not descriptor._build_full_wigner()

    for block in descriptor.blocks:
        conv = block.so2_conv
        assert conv.cuda_train_value is not None
        assert conv.flash_attention is not None
        assert conv.flash_attention_supports_training


@pytest.mark.parametrize("gate_name", ["DP_TRITON_TRAIN", "DP_TRITON_INFER"])
def test_grid_pair_train_follows_its_gate_and_slot_bound(
    monkeypatch, gate_name: str
) -> None:
    """Grid-pair training ignores the inference gate and its narrow layouts."""
    _clear_gates(monkeypatch)
    monkeypatch.setenv(gate_name, "1")

    descriptor = _make_descriptor(2, [20], 4.0)
    grid_nets = [
        module
        for module in descriptor.modules()
        if isinstance(module, (S2GridNet, SO3GridNet))
    ]
    assert grid_nets
    for net in grid_nets:
        slots = int(net.projector.to_grid_mat.shape[1])
        # Below the crossover the dense section is small enough that the
        # operator's dispatch costs more than its kernels save.
        expected = (
            gate_name == "DP_TRITON_TRAIN"
            and GRID_PAIR_TRITON_AVAILABLE
            and slots >= 75
        )
        assert (net.triton_train_l_1_grid_pair is not None) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestDPA4TrainPathParity(TestCaseSingleFrameWithNlist):
    """Loss and coordinate gradient of a training step, fused against dense."""

    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            self.coord_ext, dtype=torch.float32, device=self.device, requires_grad=True
        )
        atype = torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=self.device)
        return coord, atype, nlist

    def _step(self, descriptor: DescrptDPA4) -> tuple[np.ndarray, np.ndarray]:
        """One training step: a scalar objective and its coordinate gradient.

        The objective is second order in the descriptor output so that the
        gradient exercises the same double-differentiation a force loss does,
        which is what the operators' analytic second order serves.
        """
        coord, atype, nlist = self._inputs()
        output = descriptor(coord, atype, nlist)[0]
        objective = (output**2).sum()
        gradient = torch.autograd.grad(objective, coord)[0]
        return (
            objective.detach().cpu().numpy(),
            gradient.detach().cpu().numpy(),
        )

    @pytest.mark.parametrize("path", ["triton", "cuda", "cuda-triton"])
    def test_training_step_matches_the_dense_path(self, monkeypatch, path) -> None:
        if path in ("triton", "cuda-triton") and not SO2_VALUE_PATH_TRITON_AVAILABLE:
            pytest.skip("Triton is unavailable")
        if path in ("cuda", "cuda-triton") and not cuda_value_available():
            pytest.skip("the DPA4 CUDA training operators are unavailable")

        _clear_gates(monkeypatch)
        data = _make_descriptor(self.nt, self.sel_mix, self.rcut).serialize()
        dense = DescrptDPA4.deserialize(data).to(self.device).train()
        dense_objective, dense_gradient = self._step(dense)

        # The accelerated descriptor is deserialized from the same weights, so
        # the two runs differ only in dispatch.
        monkeypatch.setenv(
            "DP_TRITON_TRAIN", "1" if path in ("triton", "cuda-triton") else "0"
        )
        monkeypatch.setenv(
            "DP_CUDA_TRAIN", "1" if path in ("cuda", "cuda-triton") else "0"
        )
        fused = DescrptDPA4.deserialize(data).to(self.device).train()
        conv = next(
            module for module in fused.modules() if isinstance(module, SO2Convolution)
        )
        if path in ("cuda", "cuda-triton"):
            assert conv.cuda_train_value is not None
        if path in ("triton", "cuda-triton"):
            assert conv.triton_l_1_segment_softmax is not None
        fused_objective, fused_gradient = self._step(fused)

        np.testing.assert_allclose(
            fused_objective, dense_objective, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(fused_gradient, dense_gradient, rtol=2e-4, atol=2e-5)
