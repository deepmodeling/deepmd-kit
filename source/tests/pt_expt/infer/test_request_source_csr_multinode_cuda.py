# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small native regressions of optional source CSR, without model assets."""

import pytest
import torch

# Register the native library before evaluating the collection-time skip.
import deepmd.pt  # noqa: F401
from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
    build_compression_artifacts,
    ensure_registered,
    op_available,
)
from deepmd.pt_expt.utils.canonical_graph import (
    canonical_graph_from_pairs,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="CUDA and the compiled DPA4C operator are required",
)


@_GPU
@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes"),
    [(8, 2, 0), (64, 3, 4), (128, 3, 4)],
)
def test_multinode_frame_outputs_ignore_optional_source_csr(
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    channels: int,
    lmax: int,
    radial_modes: int,
) -> None:
    """A real/full -> placeholder -> real/full sequence changes no public value.

    Each frame has a non-collinear triangle and an isolated fourth node.
    All six directed triangle edges are retained, with no neighbor-search
    ordering variability. Multiple frames use disjoint global node indices.
    This is the ordinary N>1/E>2 case, unlike the two-slot N=1 boundary.
    The source-attributed atom virial is intentionally not compared for the
    placeholder call, which explicitly declines that output. Both full calls
    must nevertheless produce identical atom virials.
    """
    from ..descriptor.test_dpa4c_cuda import (
        _build_descriptor,
        _energy_fitting,
        _fused_arguments,
    )

    ensure_registered()
    descriptor = _build_descriptor(channels, lmax=lmax, radial_modes=radial_modes)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = _energy_fitting(descriptor)
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force

    position = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.2, 0.3], [8.0, 8.0, 8.0]],
        dtype=torch.float32,
        device="cuda",
    )
    center_one = torch.tensor([1, 0, 2, 1, 0, 2], device="cuda")
    source_one = torch.tensor([0, 2, 1, 2, 1, 0], device="cuda")
    center = torch.cat([center_one + 4 * frame for frame in range(frames)])
    source = torch.cat([source_one + 4 * frame for frame in range(frames)])
    edge_vec = (position[source_one] - position[center_one]).repeat(frames, 1)
    n_node = torch.full((frames,), 4, dtype=torch.int64, device="cuda")
    atype = torch.tensor([0, 1, 0, 1], device="cuda").repeat(frames)
    full = canonical_graph_from_pairs(center, source, edge_vec, n_node, 4 * frames)
    lean = canonical_graph_from_pairs(
        center, source, edge_vec, n_node, 4 * frames, with_source_csr=False
    )

    # A non-empty normal-size CSR, not the ambiguous N=1/two-slot shape.
    assert full.source_row_ptr.numel() == 4 * frames + 1
    assert full.source_order.numel() == 6 * frames
    assert int(full.destination_row_ptr[-1]) == 6 * frames
    assert lean.source_row_ptr.numel() == lean.source_order.numel() == 2
    for name in ("n_node", "n_local", "source", "edge_vec", "destination_row_ptr"):
        assert torch.equal(getattr(full, name), getattr(lean, name))

    results = []
    with torch.no_grad():
        for graph in (full, lean, full):
            arguments = _fused_arguments(
                monkeypatch,
                descriptor,
                fitting,
                graph,
                atype,
                fitting.bias_atom_e[:, 0],
            )
            results.append(tuple(value.clone() for value in operator(*arguments)))

    for index in (0, 1, 2, 3, 6):  # energy, atom energy, force, virial, mask
        assert torch.isfinite(results[0][index]).all()
        assert torch.equal(results[0][index], results[1][index])
        assert torch.equal(results[0][index], results[2][index])
    # Restoring the full CSR after a declined request also restores this output.
    assert torch.isfinite(results[0][4]).all()
    assert torch.equal(results[0][4], results[2][4])
    assert results[0][0].shape == (frames, 1)
    assert results[0][2].shape == (4 * frames, 3)
