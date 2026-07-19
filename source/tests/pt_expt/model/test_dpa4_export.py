# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-level freeze test for the DPA4/SeZM energy model.

A DPA4 model is a message-passing GNN (``has_message_passing() == True``),
but no lower path implements cross-rank ghost-feature exchange: the dense
``call`` never forwards ``comm_dict`` to the interaction blocks, and the
NeighborGraph route raises on it. Consequently
``has_message_passing_across_ranks()`` is False, and
``deserialize_to_file`` produces a .pt2 archive with a SINGLE compiled
artifact (no ``forward_lower_with_comm`` sidecar) — multi-rank inference
must fail fast at the C++ dispatch instead of silently skipping the
exchange.

Task 8 parametrizes the freeze over ``lower_kind``: ``"auto"`` now resolves
to ``"graph"`` (``_resolve_lower_kind`` sees ``model_uses_graph_lower() is
True``; ``canonical_model_eligible`` is dpa1-specific so DPA4 never takes the
``"dpa1_canonical"`` branch), while ``"nlist"`` stays reachable for
back-compat. This test verifies, per ``lower_kind``:
  1. The .pt2 archive is produced and the with-comm artifact is ABSENT
     (neither lower kind implements cross-rank ghost exchange for DPA4).
  2. ``metadata.json`` carries the correct ``type_map``/``rcut``,
     ``has_message_passing: true``, ``has_comm_artifact: false``, and
     ``lower_input_kind == expected_input_kind``; the graph kind additionally
     carries ``graph_edge_dtype``.
  3. The regular artifact loads via ``aoti_load_package``.
  4. The loaded artifact reproduces the eager model: the ``"nlist"`` kind
     against ``forward_common_lower`` (dense ABI, fp64 AOTI parity, rtol
     1e-10); the ``"graph"`` kind against ``forward_common_lower_graph``
     (NeighborGraph ABI, same tolerance).
"""

from __future__ import (
    annotations,
)

import json
import os
import zipfile

import numpy as np
import pytest
import torch

# Note: registration of the deepmd_export::border_op opaque wrapper (needed by
# the with-comm artifact) happens inside ``deserialize_to_file`` via
# ``ensure_comm_registered()``; no explicit comm import is required here.
from deepmd.pt_expt.model.ener_model import (
    _translate_energy_keys,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils import (
    env as _env,
)
from deepmd.pt_expt.utils.serialization import (
    _make_sample_inputs,
    build_synthetic_graph_inputs,
    deserialize_to_file,
)

from ...common.dpmodel.test_dpa4_call_graph import (
    _jitter_zero_arrays,
)


def _to_artifact_device(*tensors: torch.Tensor | None) -> tuple:
    """Move sample tensors to the AOTI artifact's compile device.

    ``deserialize_to_file`` runs ``move_to_device_pass(exported, _env.DEVICE)``
    before AOTI compile, so a CUDA box produces a CUDA-only artifact; feeding
    it CPU tensors triggers an illegal-memory-access at the AOTI boundary.
    The eager reference stays on CPU untouched -- only the artifact call
    needs the move. ``None`` placeholders (unset fparam/aparam/charge_spin)
    pass through unchanged.
    """
    return tuple(t if t is None else t.to(_env.DEVICE) for t in tensors)


# Small fp64 DPA4 config (channels 16, n_radial 8, lmax 2, mmax 1,
# n_blocks 2) — large enough to exercise the SO(2)/SO(3) + attention +
# embedding paths that previously specialized ``nloc`` during export, but
# small enough to keep the AOTInductor compile time bounded.
_DPA4_CONFIG = {
    "type": "dpa4",
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    },
}


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
@pytest.mark.parametrize(
    "lower_kind,expected_input_kind",
    [
        ("auto", "graph"),  # default now resolves to the graph lower
        ("nlist", "nlist"),  # dense kind stays reachable for back-compat
    ],
)
def test_dpa4_freeze_to_pt2(tmp_path, lower_kind, expected_input_kind) -> None:
    """End-to-end: DPA4 model freezes to a single-artifact .pt2 (no
    with-comm sidecar) under both lower kinds, and the regular artifact
    reproduces the matching eager forward (dense ``forward_common_lower``
    for ``"nlist"``, ``forward_common_lower_graph`` for ``"graph"``).
    """
    model = get_model(_DPA4_CONFIG)
    model.to("cpu")
    model.eval()

    # 1. Serialize → deserialize_to_file (compiles and packs both artifacts).
    #
    # DPA4 deliberately zero-initializes several residual output
    # projections (see ``_jitter_zero_arrays``'s docstring), so a fresh,
    # untrained model is architecturally edge-INDEPENDENT: force/virial are
    # ~0 regardless of edge handling. That would make the AOTI-vs-eager
    # parity below vacuous (it couldn't catch an inductor miscompile of the
    # edge scatter/index_add path) for either lower kind, so the jitter is
    # applied uniformly to both parametrizations.
    data = {"model": model.serialize()}
    _jitter_zero_arrays(data["model"], np.random.default_rng(99))
    # Rebuild the eager reference model from the SAME jittered dict that
    # gets frozen below, so the AOTI artifact and the eager reference
    # compared against it are the same (edge-sensitive) model.
    model = BaseModel.deserialize(data["model"]).to("cpu")
    model.eval()
    pt2_path = str(tmp_path / f"test_dpa4_{expected_input_kind}.pt2")
    deserialize_to_file(pt2_path, data, lower_kind=lower_kind)
    assert os.path.exists(pt2_path)

    # 2. ZIP layout + metadata sanity. PyTorch's strict layout puts our
    #    sidecars under ``model/extra/`` (PT2_EXTRA_PREFIX).
    with zipfile.ZipFile(pt2_path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        assert "model/extra/forward_lower_with_comm.pt2" not in names, (
            f"with-comm artifact present but no lower path implements "
            f"cross-rank exchange; names={sorted(names)}"
        )
    assert meta["type_map"] == _DPA4_CONFIG["type_map"]
    assert meta["rcut"] == model.get_rcut()
    # DPA4 is a message-passing GNN descriptor, but no lower path
    # implements the cross-rank exchange (see module docstring).
    assert meta["has_message_passing"] is True
    assert meta["has_comm_artifact"] is False
    assert meta["lower_input_kind"] == expected_input_kind

    # 3. The regular artifact loads.
    from torch._inductor import (
        aoti_load_package,
    )

    regular = aoti_load_package(pt2_path)

    if expected_input_kind == "nlist":
        # 4a. Dense-ABI eager reference vs. AOTI artifact parity on
        # forward_common_lower.
        # _make_sample_inputs creates tensors on _env.DEVICE (CUDA on a GPU
        # box); the eager reference model lives on CPU, so make explicit CPU
        # copies for it. The artifact call below gets separate _env.DEVICE
        # copies via _to_artifact_device.
        sample = _make_sample_inputs(model, nframes=1, has_spin=False)
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam, charge_spin = tuple(
            t if t is None else t.to("cpu") for t in sample
        )

        eager_out = model.forward_common_lower(
            ext_coord.detach().requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=False,
            charge_spin=charge_spin,
        )

        # Anti-vacuity guard: fresh DPA4 is edge-independent (forces ~0);
        # confirm the jitter above made the eager reference force
        # non-trivial, else the AOTI parity below would pass trivially.
        f_ref = eager_out["energy_derv_r"].detach().cpu().numpy()
        assert np.abs(f_ref).max() > 1e-6, (
            f"eager reference force is near-zero ({np.abs(f_ref).max():.3e}); "
            f"jitter not effective -- AOTI parity check would be vacuous"
        )

        # The artifact is compiled for _env.DEVICE (move_to_device_pass in
        # deserialize_to_file); move inputs there while the eager reference
        # above stays on CPU.
        (
            d_ext_coord,
            d_ext_atype,
            d_nlist_t,
            d_mapping_t,
            d_fparam,
            d_aparam,
            d_charge_spin,
        ) = _to_artifact_device(
            ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam, charge_spin
        )
        artifact_out = regular(
            d_ext_coord,
            d_ext_atype,
            d_nlist_t,
            d_mapping_t,
            d_fparam,
            d_aparam,
            d_charge_spin,
        )

        # The AOTI artifact returns the internal forward_common_lower keys;
        # compare every key it produces against the eager reference (fp64
        # AOTI tolerance).
        compared = 0
        for key, val in artifact_out.items():
            if key not in eager_out or eager_out[key] is None or val is None:
                continue
            np.testing.assert_allclose(
                val.detach().cpu().numpy(),
                eager_out[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"artifact vs eager forward_common_lower differs: {key}",
            )
            compared += 1
        # Guard against a vacuous pass (no overlapping keys compared).
        assert compared > 0, (
            f"no overlapping output keys compared; artifact keys="
            f"{sorted(artifact_out)}, eager keys={sorted(eager_out)}"
        )
        # The energy output must be among the compared keys.
        assert "energy_redu" in artifact_out or "energy" in artifact_out
    else:
        # 4b. NeighborGraph-ABI eager reference vs. AOTI artifact parity on
        # forward_common_lower_graph. Metadata carries the edge dtype the
        # graph artifact was frozen with.
        assert meta["graph_edge_dtype"] in ("float32", "float64")

        sample = build_synthetic_graph_inputs(
            model,
            e_max=None,
            nframes=1,
            nloc=6,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs = sample

        eager_internal = model.forward_common_lower_graph(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            do,
            drp,
            so,
            srp,
            destination_sorted=True,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
        )
        # The graph AOTI artifact was frozen via forward_lower_graph_exportable,
        # which translates the internal fitting keys ("energy_redu",
        # "energy_derv_r", ...) to the public forward_lower convention
        # ("energy", "force", ...) -- see ener_model.py:441. Apply the same
        # translation to the eager reference so the two key sets line up.
        eager_out = _translate_energy_keys(
            eager_internal,
            do_grad_r=model.do_grad_r("energy"),
            do_grad_c=model.do_grad_c("energy"),
            do_atomic_virial=True,
            local=True,
        )

        # Anti-vacuity guard: fresh DPA4 is edge-independent (forces ~0);
        # confirm the jitter above made the eager reference force
        # non-trivial, else the AOTI parity below would pass trivially.
        f_ref = eager_out["force"].detach().cpu().numpy()
        assert np.abs(f_ref).max() > 1e-6, (
            f"eager reference force is near-zero ({np.abs(f_ref).max():.3e}); "
            f"jitter not effective -- AOTI parity check would be vacuous"
        )

        # The artifact is compiled for _env.DEVICE (move_to_device_pass in
        # deserialize_to_file); move inputs there while the eager reference
        # above stays on CPU.
        (
            d_atype,
            d_n_node,
            d_n_local,
            d_ei,
            d_ev,
            d_em,
            d_do,
            d_drp,
            d_so,
            d_srp,
            d_fp,
            d_ap,
            d_cs,
        ) = _to_artifact_device(
            atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs
        )
        artifact_out = regular(
            d_atype,
            d_n_node,
            d_n_local,
            d_ei,
            d_ev,
            d_em,
            d_do,
            d_drp,
            d_so,
            d_srp,
            d_fp,
            d_ap,
            d_cs,
        )

        # Compare every key the artifact produces against the (translated)
        # eager reference.
        compared = 0
        for key, val in artifact_out.items():
            if key not in eager_out or eager_out[key] is None or val is None:
                continue
            np.testing.assert_allclose(
                val.detach().cpu().numpy(),
                eager_out[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=(
                    f"artifact vs eager forward_common_lower_graph differs: {key}"
                ),
            )
            compared += 1
        assert compared > 0, (
            f"no overlapping output keys compared; artifact keys="
            f"{sorted(artifact_out)}, eager keys={sorted(eager_out)}"
        )
        assert "energy_redu" in artifact_out or "energy" in artifact_out
