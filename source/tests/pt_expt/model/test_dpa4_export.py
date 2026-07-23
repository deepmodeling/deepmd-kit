# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-level freeze test for the DPA4/SeZM energy model.

A DPA4 model is a message-passing GNN (``has_message_passing() == True``),
and cross-rank ghost-feature exchange is implemented ONLY on the
NeighborGraph lower: it carries a real per-layer ``border_op`` exchange
(``has_message_passing_across_ranks()`` is True). The dense (nlist) lower's
``call`` adapter still raises on ``comm_dict`` (``dense_lower_supports_comm()``
is False), so ``deserialize_to_file`` embeds the ``forward_lower_with_comm``
sidecar for the ``"graph"`` kind only; the ``"nlist"`` kind stays a SINGLE
compiled artifact and multi-rank inference on it must fail fast at the C++
dispatch instead of silently skipping the exchange.

Task 8 parametrizes the freeze over ``lower_kind``: ``"auto"`` now resolves
to ``"graph"`` (``_resolve_lower_kind`` sees ``model_uses_graph_lower() is
True``; ``canonical_model_eligible`` is dpa1-specific so DPA4 never takes the
``"dpa1_canonical"`` branch), while ``"nlist"`` stays reachable for
back-compat. This test verifies, per ``lower_kind``:
  1. The .pt2 archive is produced and the with-comm artifact is PRESENT for
     the ``"graph"`` kind (cross-rank exchange) and ABSENT for the
     ``"nlist"`` kind (dense lower is comm-less).
  2. ``metadata.json`` carries the correct ``type_map``/``rcut``,
     ``has_message_passing: true``, kind-conditional ``has_comm_artifact``,
     and ``lower_input_kind == expected_input_kind``; the graph kind
     additionally carries ``graph_edge_dtype``.
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
from deepmd.pt_expt.utils import env as _env
from deepmd.pt_expt.utils.serialization import (
    _make_sample_inputs,
    build_synthetic_graph_inputs,
    deserialize_to_file,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)
from .test_dpa4_native_spin import (
    NATIVE_SPIN_CONFIG,
    _build_native_spin_model_cpu,
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
    """End-to-end: DPA4 model freezes to a .pt2 archive with kind-conditional
    artifact layout (with-comm sidecar embedded for the ``"graph"`` kind's
    multi-rank exchange, single-artifact for the ``"nlist"`` kind's comm-less
    dense lower), and the regular artifact reproduces the matching eager
    forward (dense ``forward_common_lower`` for ``"nlist"``,
    ``forward_common_lower_graph`` for ``"graph"``).
    """
    model = get_model(_DPA4_CONFIG)
    model.to("cpu")
    model.eval()

    # 1. Serialize → deserialize_to_file (compiles and packs both artifacts).
    #
    # DPA4 deliberately zero-initializes several residual output
    # projections (see ``jitter_zero_arrays``'s docstring in
    # ``dpa4_fixtures.py``), so a fresh, untrained model is architecturally
    # edge-INDEPENDENT: force/virial are ~0 regardless of edge handling.
    # That would make the AOTI-vs-eager parity below vacuous (it couldn't
    # catch an inductor miscompile of the edge scatter/index_add path) for
    # either lower kind, so the jitter is applied uniformly to both
    # parametrizations.
    data = {"model": model.serialize()}
    data["model"] = jitter_zero_arrays(data["model"], np.random.default_rng(99))
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
        if expected_input_kind == "graph":
            assert "model/extra/forward_lower_with_comm.pt2" in names, (
                "graph kind must embed the with-comm sidecar (multi-rank)"
            )
            assert meta["has_comm_artifact"] is True
        else:
            assert "model/extra/forward_lower_with_comm.pt2" not in names, (
                "nlist kind must stay single-artifact (dense lower is comm-less)"
            )
            assert meta["has_comm_artifact"] is False
    assert meta["type_map"] == _DPA4_CONFIG["type_map"]
    assert meta["rcut"] == model.get_rcut()
    # DPA4 is a message-passing GNN descriptor; only the graph lower
    # implements the cross-rank exchange (see module docstring).
    assert meta["has_message_passing"] is True
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


# =============================================================================
# Task 6: graph-kind ``.pt2`` freeze for the NATIVE-spin DPA4 wrapper
# (``NativeSpinEnergyModel``, type ``native_spin``) -- spin rides the
# NeighborGraph lower ONLY (no dense/nlist lower, no with-comm sidecar: see
# ``_needs_with_comm_artifact``'s native-spin first rule). The VIRTUAL-atom
# spin scheme (``SpinModel``, type ``spin_ener``) has no graph-lower
# implementation at all and must keep raising ``NotImplementedError``.
# =============================================================================

# Minimal virtual-atom (deepspin) spin config: a non-GNN se_e2_a backbone is
# enough to prove the graph-form rejection still fires for "spin_ener" --
# the rejection happens before any tracing/AOTI compile, so this stays fast.
_VIRTUAL_SPIN_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [4, 4],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [4, 4],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [4, 4],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
    },
    "spin": {
        "use_spin": [True, False],
        "virtual_scale": [0.3140],
    },
}


def _freeze_native_spin(model_file) -> None:
    """Freeze a jittered native-spin DPA4 model to a graph-kind ``.pt2``.

    Mirrors ``test_dpa4_freeze_to_pt2``'s serialize -> ``deserialize_to_file``
    sequence, but native spin has ONLY a graph lower (no dense/nlist lower --
    see ``NativeSpinEnergyModel``'s module docstring), so ``lower_kind="graph"``
    is explicit rather than parametrized. ``_build_native_spin_model_cpu``
    already jitters DPA4's zero-initialized residual projections (else the
    model is architecturally spin-independent -- see that helper's
    docstring), so no extra jitter step is needed here.
    """
    model = _build_native_spin_model_cpu()
    data = {"model": model.serialize()}
    deserialize_to_file(str(model_file), data, lower_kind="graph")


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
def test_native_spin_graph_freeze(tmp_path) -> None:
    """Native-spin DPA4 freezes to a graph-kind .pt2: metadata + no sidecar."""
    model_file = tmp_path / "dpa4_spin_graph.pt2"
    _freeze_native_spin(model_file)

    with zipfile.ZipFile(model_file) as z:
        names = z.namelist()
        # AOTInductor also embeds its OWN internal per-kernel files whose
        # names end with "metadata.json" (e.g.
        # "<hash>.wrapper_metadata.json", "<hash>.kernel_metadata.json"
        # under "data/aotinductor/model/") -- read the exact PyTorch
        # PT2_EXTRA_PREFIX path for OUR sidecar, not a fragile
        # ``endswith("metadata.json")`` scan (mirrors
        # ``test_dpa4_freeze_to_pt2`` above).
        md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))

    assert md["type_map"] == NATIVE_SPIN_CONFIG["type_map"]
    assert md["lower_input_kind"] == "graph"
    assert md["is_spin"] is True
    assert md["has_comm_artifact"] is False
    assert md["has_message_passing"] is True
    assert md["ntypes_spin"] == 1  # use_spin=[True, False]
    assert md["use_spin"] == [True, False]
    assert "force_mag" in md["output_keys"]
    for key in ("atom_energy", "energy", "force", "virial"):
        assert key in md["output_keys"]
    assert not any(n.endswith("forward_lower_with_comm.pt2") for n in names)


def test_virtual_spin_graph_freeze_still_rejected(tmp_path) -> None:
    """spin_ener (virtual) graph freeze keeps raising ``NotImplementedError``.

    The virtual-atom scheme doubles the atom count (real + virtual) and has
    no graph-lower implementation; only the native scheme
    (``native_spin``) is graph-eligible (see the module docstring
    above).
    """
    model = get_model(_VIRTUAL_SPIN_CONFIG)
    model.to("cpu")
    model.eval()
    data = {"model": model.serialize()}
    assert data["model"]["type"] == "spin_ener"

    with pytest.raises(NotImplementedError, match="graph-form"):
        deserialize_to_file(
            str(tmp_path / "virtual_spin_graph.pt2"), data, lower_kind="graph"
        )


# =============================================================================
# Task 7: DeepEval graph fast path for the NATIVE-spin ``.pt2`` -- the frozen
# artifact from Task 6 above must be evaluable through the public DeepPot API
# (``deepmd.pt_expt.infer.deep_eval.DeepEval._eval_model_graph_spin``), NOT
# just constructible.  Compares against the EAGER pt_expt
# ``NativeSpinEnergyModel.forward`` on the SAME weights/system (rtol=atol=1e-10,
# CPU fp64 -- project convention for same-math weight-copied parity).
# =============================================================================

_SPIN_EVAL_NATOMS = 6
_SPIN_EVAL_ATYPES = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)  # Ni,Ni,Ni,O,O,O
_SPIN_EVAL_COORDS = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ],
    dtype=np.float64,
).reshape(1, _SPIN_EVAL_NATOMS, 3)
_SPIN_EVAL_CELL = (np.eye(3, dtype=np.float64) * 6.0).reshape(1, 9)
# Deliberately NOT pre-masked by type (mirrors TestNativeSpinEnergyModelPtExpt):
# the model's own descriptor gating must zero the non-spin (type 1) rows.
_SPIN_EVAL_SPINS = np.array(
    [
        [0.11, 0.05, -0.02],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.01, -0.01, 0.02],
        [-0.02, 0.03, -0.01],
        [0.015, 0.02, -0.03],
    ],
    dtype=np.float64,
).reshape(1, _SPIN_EVAL_NATOMS, 3)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
def test_deep_eval_graph_spin_parity(tmp_path) -> None:
    """DeepEval on a graph-kind native-spin ``.pt2`` matches eager ``forward``.

    Exercises ``DeepEval._eval_model_spin``'s ``lower_input_kind == "graph"``
    branch end to end through the public ``DeepPot`` API: energy, force,
    force_mag and (global) virial must reproduce the eager
    ``NativeSpinEnergyModel.forward`` on the identical weights and system.
    ``atomic=False`` is used deliberately -- the graph-spin ABI has no owner
    site for ``mask_mag`` (see ``_graph_spin_output_key``'s docstring), so
    only the four outputs that route through the exported forward are
    compared here.
    """
    from deepmd.infer import (
        DeepPot,
    )

    model = _build_native_spin_model_cpu()

    coord_t = torch.tensor(_SPIN_EVAL_COORDS, dtype=torch.float64)
    atype_t = torch.tensor(
        _SPIN_EVAL_ATYPES.reshape(1, _SPIN_EVAL_NATOMS), dtype=torch.int64
    )
    spin_t = torch.tensor(_SPIN_EVAL_SPINS, dtype=torch.float64)
    box_t = torch.tensor(_SPIN_EVAL_CELL, dtype=torch.float64)
    ref = model.forward(coord_t, atype_t, spin_t, box=box_t)

    # Anti-vacuity: a bare (non-jittered) DPA4 zero-initializes residual
    # projections, which would make force_mag identically zero and the
    # parity check below vacuous by construction (see
    # ``_build_native_spin_model_cpu``'s docstring for why jitter fixes
    # this).
    fm_max = ref["force_mag"].abs().max().item()
    assert fm_max > 1e-6, (
        "expected the jittered model's force_mag to be non-trivial; got "
        f"max |force_mag| = {fm_max:.3e} (jitter not effective -- the "
        "parity check below would be vacuous)"
    )

    model_file = tmp_path / "dpa4_spin_graph_eval.pt2"
    data = {"model": model.serialize()}
    deserialize_to_file(str(model_file), data, lower_kind="graph")

    dp = DeepPot(str(model_file))
    assert dp.has_spin
    e, f, v, fm, _mm = dp.eval(
        _SPIN_EVAL_COORDS,
        _SPIN_EVAL_CELL,
        _SPIN_EVAL_ATYPES,
        atomic=False,
        spin=_SPIN_EVAL_SPINS,
    )

    np.testing.assert_allclose(
        e.reshape(-1),
        ref["energy"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="energy",
    )
    np.testing.assert_allclose(
        f.reshape(-1),
        ref["force"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="force",
    )
    np.testing.assert_allclose(
        fm.reshape(-1),
        ref["force_mag"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="force_mag",
    )
    np.testing.assert_allclose(
        v.reshape(-1),
        ref["virial"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="virial",
    )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
def test_native_spin_chg_spin_graph_freeze(tmp_path) -> None:
    """COMBINED native-spin + charge-spin FiLM freezes to a graph .pt2.

    Review 3638047227: the combined public configuration must export. The
    charge_spin slot rides the ABI tail (slot 13); metadata carries the
    chg-spin fields the C++/DeepEval loaders key on.
    """
    from deepmd.pt_expt.descriptor.dpa4 import (
        DescrptDPA4,
    )
    from deepmd.pt_expt.model.get_model import (
        get_model as pt_expt_get_model,
    )

    from ...dpa4_fixtures import (
        jitter_zero_arrays,
    )
    from .test_dpa4_native_spin import (
        COMBINED_CHG_SPIN_CONFIG,
    )

    import copy

    cpu = torch.device("cpu")
    model = pt_expt_get_model(copy.deepcopy(COMBINED_CHG_SPIN_CONFIG))
    ds = model.atomic_model.descriptor
    jittered = jitter_zero_arrays(ds.serialize(), np.random.default_rng(21))
    model.atomic_model.descriptor = DescrptDPA4.deserialize(jittered).to(cpu)
    model = model.to(cpu).eval()
    data = {"model": model.serialize()}
    model_file = tmp_path / "dpa4_spin_chg_graph.pt2"
    deserialize_to_file(str(model_file), data, lower_kind="graph")

    with zipfile.ZipFile(model_file) as z:
        md = json.loads(z.read("model/extra/metadata.json").decode("utf-8"))
    assert md["is_spin"] is True
    assert md["lower_input_kind"] == "graph"
    assert md["has_chg_spin_ebd"] is True
    assert md["dim_chg_spin"] == 2
    assert "force_mag" in md["output_keys"]

    # DeepEval through the compiled artifact: the slot-13 charge_spin is
    # LIVE (an integer-valued change moves the energy) and matches the eager
    # forward with the same conditioning at the cross-artifact tolerance.
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(str(model_file))
    assert dp.has_spin
    cs1 = np.array([[1.0, 2.0]])
    e1, f1, _v1, fm1, _mm1 = dp.eval(
        _SPIN_EVAL_COORDS,
        _SPIN_EVAL_CELL,
        _SPIN_EVAL_ATYPES,
        atomic=False,
        spin=_SPIN_EVAL_SPINS,
        charge_spin=cs1,
    )
    e0, _f0, _v0, _fm0, _mm0 = dp.eval(
        _SPIN_EVAL_COORDS,
        _SPIN_EVAL_CELL,
        _SPIN_EVAL_ATYPES,
        atomic=False,
        spin=_SPIN_EVAL_SPINS,
        charge_spin=np.array([[0.0, 0.0]]),
    )
    de = float(np.abs(np.asarray(e1) - np.asarray(e0)).max())
    assert de > 1e-10, f"charge_spin slot dead through the artifact: {de:.3e}"

    coord_t = torch.tensor(_SPIN_EVAL_COORDS, dtype=torch.float64).reshape(1, -1, 3)
    atype_t = torch.tensor(_SPIN_EVAL_ATYPES, dtype=torch.int64).reshape(1, -1)
    box_t = torch.tensor(_SPIN_EVAL_CELL, dtype=torch.float64).reshape(1, 9)
    spin_t = torch.tensor(_SPIN_EVAL_SPINS, dtype=torch.float64).reshape(1, -1, 3)
    ref = model.forward(
        coord_t,
        atype_t,
        spin_t,
        box=box_t,
        charge_spin=torch.tensor(cs1, dtype=torch.float64),
    )
    np.testing.assert_allclose(
        np.asarray(e1).reshape(-1),
        ref["energy"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="combined energy vs eager",
    )
    np.testing.assert_allclose(
        np.asarray(fm1).reshape(-1),
        ref["force_mag"].detach().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="combined force_mag vs eager",
    )
