# SPDX-License-Identifier: LGPL-3.0-or-later

from types import (
    SimpleNamespace,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    _resolve_lower_kind,
    _trace_and_export,
    build_synthetic_graph_inputs,
)

#: The compact canonical ABI and the fused spin backward are CUDA-only routes,
#: so they follow the configured backend device rather than the mere presence
#: of CUDA hardware: a run pinned to the CPU takes the generic graph lower.
_GPU = pytest.mark.skipif(
    env.DEVICE.type != "cuda",
    reason="the compact canonical and fused spin routes are CUDA only",
)


def _config() -> dict:
    return {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 16,
            "lmax": 4,
            "n_radial": 4,
            "precision": "float64",
            "seed": 17,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
        },
    }


def _compressed_config(channels: int = 8) -> dict:
    config = _config()
    descriptor = config["descriptor"]
    descriptor["channels"] = channels
    descriptor["lmax"] = 2
    descriptor["n_radial"] = 8
    descriptor["precision"] = "float32"
    fitting = config["fitting_net"]
    fitting["neuron"] = [32, 32]
    fitting["activation_function"] = "silu"
    fitting["precision"] = "float32"
    # The fused fitting operator has no per-layer timestep, and the shipped
    # DPA4C grades do not use one either, so the compact canonical path is
    # only reachable without it.
    fitting["resnet_dt"] = False
    return config


def _run_graph(
    model: torch.nn.Module,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    sample = build_synthetic_graph_inputs(
        model,
        e_max=None,
        nframes=2,
        nloc=7,
        dtype=dtype,
        device=env.DEVICE,
    )
    (
        atype,
        n_node,
        n_local,
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
    ) = sample
    return model.forward_common_lower_graph(
        atype,
        n_node,
        n_local,
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


def test_graph_lower_energy_force_are_finite() -> None:
    model = get_model(_config()).to(env.DEVICE).eval()
    assert model.get_descriptor().uses_graph_lower()
    result = _run_graph(model)
    for key in (
        "energy",
        "energy_redu",
        "energy_derv_r",
        "energy_derv_c_redu",
    ):
        assert key in result
        assert torch.isfinite(result[key]).all(), key


def test_phantom_atoms_leave_energy_and_force_unchanged() -> None:
    """A mixed-nloc batch's padding must not perturb its real atoms.

    Frames of unequal atom count only share a batch when the shorter ones are
    padded to a rectangular shape, with the padded slots marked ``atype = -1``.
    Such a phantom atom stands for no physical site: the graph builders drop it
    from the edge set and the atomic model zeroes its output, so the padded
    batch has to reproduce, frame by frame, what the frames give on their own.
    """
    torch.manual_seed(1234)
    model = get_model(_config()).to(env.DEVICE)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    model.eval()

    rng = np.random.default_rng(0)
    nlocs = (4, 7, 3)
    pad_nloc = max(nlocs)
    box = (np.eye(3) * 10.0).reshape(9)
    coords = [rng.uniform(0.0, 6.0, (nloc, 3)) for nloc in nlocs]
    atypes = [rng.integers(0, 2, nloc) for nloc in nlocs]

    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    for index, (frame_coord, frame_atype) in enumerate(
        zip(coords, atypes, strict=True)
    ):
        padded_coord[index, : len(frame_atype)] = frame_coord
        padded_atype[index, : len(frame_atype)] = frame_atype

    def run(coord: np.ndarray, atype: np.ndarray) -> dict[str, torch.Tensor]:
        nframes = coord.shape[0]
        return model(
            torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
            torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
            box=torch.tensor(
                np.tile(box, (nframes, 1)), dtype=torch.float64, device=env.DEVICE
            ),
        )

    batched = run(padded_coord, padded_atype)
    for index, (frame_coord, frame_atype) in enumerate(
        zip(coords, atypes, strict=True)
    ):
        alone = run(frame_coord[None], frame_atype[None].astype(np.int64))
        torch.testing.assert_close(
            batched["energy"].reshape(-1)[index],
            alone["energy"].reshape(-1)[0],
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        torch.testing.assert_close(
            batched["force"][index, : len(frame_atype)],
            alone["force"][0],
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    # Padded slots receive no gradient at all.
    phantom = torch.tensor(padded_atype, device=env.DEVICE) < 0
    assert bool(phantom.any()), "fixture must exercise padding"
    assert bool(torch.all(batched["force"][phantom] == 0.0))


def test_padding_never_reaches_the_network() -> None:
    """The node axis the graph lower receives holds the real atoms alone.

    The equivalence test above holds whether or not the phantom atoms are
    dropped, since the atomic model masks their output either way. What is
    asserted here is that they are dropped: the padded slots must cost no
    descriptor, no fitting-net evaluation and no gradient, which is the whole
    point of packing frames of unequal atom count into one batch.
    """
    model = get_model(_config()).to(env.DEVICE).eval()

    seen: list[int] = []
    lower = model.forward_common_lower_graph

    def spy(atype, *args, **kwargs):
        seen.append(int(atype.shape[0]))
        return lower(atype, *args, **kwargs)

    model.forward_common_lower_graph = spy

    rng = np.random.default_rng(5)
    nlocs = (4, 7, 3)
    pad_nloc = max(nlocs)
    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    for index, nloc in enumerate(nlocs):
        padded_coord[index, :nloc] = rng.uniform(0.0, 6.0, (nloc, 3))
        padded_atype[index, :nloc] = rng.integers(0, 2, nloc)

    out = model(
        torch.tensor(padded_coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(padded_atype, dtype=torch.long, device=env.DEVICE),
        box=torch.tensor(
            np.tile((np.eye(3) * 10.0).reshape(9), (len(nlocs), 1)),
            dtype=torch.float64,
            device=env.DEVICE,
        ),
    )

    assert seen == [sum(nlocs)], (
        f"the lower saw {seen} nodes; the batch holds {sum(nlocs)} real atoms "
        f"padded to {len(nlocs) * pad_nloc} slots"
    )
    # The public output still carries the padded shape the callers expect.
    assert out["force"].shape == (len(nlocs), pad_nloc, 3)


def test_compiled_lower_accepts_a_compacted_node_axis() -> None:
    """The compiled artifact must not carry ``N == nframes * nloc`` as a guard.

    Its trace is taken on a uniform system, where the flat node axis happens to
    be the product of the frame count and the atom count. Dropping the padding
    breaks that relation, so this exercises the compiled lower on a batch where
    it no longer holds, and holds the result against the eager graph path,
    which takes the same compaction.
    """
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    torch.manual_seed(0)
    config = _config()
    config["descriptor"]["channels"] = 8
    config["fitting_net"]["neuron"] = [8, 8]
    model = get_model(config).to(env.DEVICE).train()
    compiled = _CompiledModel(model, _get_model_structure_key(model))

    rng = np.random.default_rng(0)
    nlocs = (4, 7, 3)
    pad_nloc = max(nlocs)
    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    for index, nloc in enumerate(nlocs):
        padded_coord[index, :nloc] = rng.uniform(0.0, 6.0, (nloc, 3))
        padded_atype[index, :nloc] = rng.integers(0, 2, nloc)

    args = (
        torch.tensor(padded_coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(padded_atype, dtype=torch.long, device=env.DEVICE),
        torch.tensor(
            np.tile((np.eye(3) * 10.0).reshape(1, 3, 3), (len(nlocs), 1, 1)),
            dtype=torch.float64,
            device=env.DEVICE,
        ),
    )
    got = compiled(*args)
    expected = model(*args)

    assert got["force"].shape == (len(nlocs), pad_nloc, 3)
    torch.testing.assert_close(got["energy"], expected["energy"])
    torch.testing.assert_close(got["force"], expected["force"])
    phantom = args[1] < 0
    assert bool(torch.all(got["force"][phantom] == 0.0))


def test_ragged_and_padded_batches_agree() -> None:
    """The two layouts are two spellings of one batch, so they must agree.

    This is the invariant the whole flat-node-axis path rests on: concatenating
    the frames rather than padding them to a common width changes how the batch
    is stored, and nothing about the physics it describes.
    """
    torch.manual_seed(11)
    model = get_model(_config()).to(env.DEVICE)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    model.eval()

    rng = np.random.default_rng(0)
    nlocs = (4, 7, 3)
    pad_nloc, boxlen = max(nlocs), 10.0
    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    flat_coord, flat_atype = [], []
    for index, nloc in enumerate(nlocs):
        coord = rng.uniform(0.0, 6.0, (nloc, 3))
        atype = rng.integers(0, 2, nloc)
        padded_coord[index, :nloc] = coord
        padded_atype[index, :nloc] = atype
        flat_coord.append(coord)
        flat_atype.append(atype)

    padded = model(
        torch.tensor(padded_coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(padded_atype, dtype=torch.long, device=env.DEVICE),
        box=torch.tensor(
            np.tile((np.eye(3) * boxlen).reshape(9), (len(nlocs), 1)),
            dtype=torch.float64,
            device=env.DEVICE,
        ),
    )
    ragged = model.forward_ragged(
        torch.tensor(
            np.concatenate(flat_coord), dtype=torch.float64, device=env.DEVICE
        ),
        torch.tensor(np.concatenate(flat_atype), dtype=torch.long, device=env.DEVICE),
        torch.tensor(nlocs, dtype=torch.long, device=env.DEVICE),
        torch.tensor(
            np.tile(np.eye(3)[None] * boxlen, (len(nlocs), 1, 1)),
            dtype=torch.float64,
            device=env.DEVICE,
        ),
    )

    torch.testing.assert_close(ragged["energy"], padded["energy"])
    offset = 0
    for index, nloc in enumerate(nlocs):
        torch.testing.assert_close(
            ragged["force"][offset : offset + nloc],
            padded["force"][index, :nloc],
        )
        offset += nloc
    assert offset == ragged["force"].shape[0], "the ragged axis holds real atoms only"


def test_graph_force_loss_trains_descriptor() -> None:
    model = get_model(_config()).to(env.DEVICE).train()
    result = _run_graph(model)
    loss = result["energy_redu"].square().mean()
    loss = loss + result["energy_derv_r"].square().mean()
    loss.backward()
    descriptor = model.get_descriptor()
    gradients = {
        name: parameter.grad for name, parameter in descriptor.named_parameters()
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_graph_export() -> None:
    model = get_model(_config()).to("cpu").eval()
    exported, _metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="graph",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)


def test_compressed_graph_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    model = get_model(_compressed_config()).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    exported, metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="graph",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)
    assert metadata["graph_edge_dtype"] == "float32"


def test_charge_state_with_comm_is_rejected_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent compiled lowers cannot share one mutable charge state."""
    from deepmd.pt_expt.utils import (
        serialization,
    )

    descriptor = object()
    monkeypatch.setattr(
        serialization,
        "_trace_and_export",
        lambda *args, **kwargs: (
            object(),
            {"has_comm_artifact": True},
            {},
            [],
        ),
    )
    monkeypatch.setattr(
        serialization,
        "_charge_state_descriptor",
        lambda *args, **kwargs: descriptor,
    )
    monkeypatch.setattr(
        torch._inductor,
        "aoti_compile_and_package",
        lambda *args, **kwargs: pytest.fail("compilation must not start"),
    )

    with pytest.raises(ValueError, match="independent constants"):
        serialization._deserialize_to_file_pt2(
            "unused.pt2",
            {},
            lower_kind="graph",
        )


@_GPU
@pytest.mark.parametrize("channels", [8, 64])
def test_compact_canonical_graph_export(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = get_model(_compressed_config(channels)).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    exported, metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)
    assert metadata["lower_input_kind"] == "dpa4c_canonical"
    assert metadata["graph_edge_dtype"] == "float32"
    assert metadata["canonical_index_dtype"] == "uint32"


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_auto_lower_kind_selects_compact_canonical(channels: int) -> None:
    model = get_model(_compressed_config(channels)).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    data = {"model": model.serialize()}
    assert _resolve_lower_kind("model.pt2", data, "auto") == "dpa4c_canonical"


def test_the_graph_lower_conditions_each_frame_on_its_own_charge_state() -> None:
    """The frame condition must survive the model seam and stay per frame.

    The atomic model forwards ``charge_spin`` only to descriptors that
    declare the capability, and the graph lower flattens every frame onto one
    node axis, so a batch of mixed charge states exercises both the seam and
    the node-to-frame map.
    """
    config = _config()
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [0.0, 1.0]
    model = get_model(config).to(env.DEVICE).eval()
    descriptor = model.get_descriptor()
    assert descriptor.supports_charge_spin()

    # The condition output projection is zero initialized, so an untrained
    # model would be inert and the comparison below would hold vacuously.
    head = descriptor.charge_spin_embedding.network.layers[-1]
    generator = torch.Generator(device=head.w.device).manual_seed(5)
    with torch.no_grad():
        head.w.copy_(
            torch.randn(
                head.w.shape,
                dtype=head.w.dtype,
                device=head.w.device,
                generator=generator,
            )
            * 0.5
        )

    sample = build_synthetic_graph_inputs(
        model,
        e_max=None,
        nframes=2,
        nloc=7,
        dtype=torch.float64,
        device=env.DEVICE,
    )

    def energy(second_state: list[float]) -> torch.Tensor:
        return model.forward_common_lower_graph(
            *sample[:10],
            destination_sorted=True,
            do_atomic_virial=True,
            fparam=sample[10],
            aparam=sample[11],
            charge_spin=torch.tensor(
                [[0.0, 1.0], second_state],
                dtype=torch.float64,
                device=env.DEVICE,
            ),
        )["energy_redu"]

    neutral, mixed = energy([0.0, 1.0]), energy([2.0, 3.0])
    torch.testing.assert_close(neutral[0], mixed[0], atol=0.0, rtol=0.0)
    assert float((mixed[1] - neutral[1]).detach().abs().max()) > 0.0


def test_an_uncompressed_export_keeps_the_charge_state_as_a_runtime_input() -> None:
    """An uncompressed artifact conditions at run time, not at export time.

    Whether a deployed model accepts a charge state is a property of
    compression rather than of the export format: the graph lower carries a
    conditioning slot with a dynamic frame axis, and only the fold of the
    compact canonical path removes it.
    """
    config = _compressed_config()
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [2.0, 3.0]
    model = get_model(config).to("cpu").eval()
    exported, metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="graph",
        do_atomic_virial=True,
    )
    assert metadata["dim_chg_spin"] == 2
    assert metadata["default_chg_spin"] == [2.0, 3.0]
    placeholders = [
        node.name
        for node in exported.graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    assert placeholders[-1].startswith("charge_spin")


@_GPU
def test_a_baked_charge_state_reaches_the_compact_canonical_lower() -> None:
    """Compression must remove the runtime condition, not just satisfy it.

    The compact canonical argument list carries no conditioning slot, and
    evaluation rejects an artifact that claims to need one. A charge-
    conditioned model reaches that lower only because compression folds the
    charge state into the frozen tables and the snapshot then reports a zero
    runtime condition width.
    """
    config = _compressed_config()
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [2.0, 3.0]
    model = get_model(config).to("cpu").eval()
    assert model.get_dim_chg_spin() == 2
    assert _resolve_lower_kind("model.pt2", {"model": model.serialize()}, "auto") == (
        "graph"
    )

    descriptor = model.get_descriptor()
    descriptor.enable_compression(min_nbor_dist=0.5)
    assert model.get_dim_chg_spin() == 0
    assert _resolve_lower_kind("model.pt2", {"model": model.serialize()}, "auto") == (
        "dpa4c_canonical"
    )


def test_compact_canonical_eligibility_rejects_other_descriptors() -> None:
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        canonical_model_eligible,
    )

    model = SimpleNamespace(
        atomic_model=SimpleNamespace(
            descriptor=SimpleNamespace(compress=True),
            fitting_net=object(),
        )
    )
    assert not canonical_model_eligible(model)


@pytest.mark.parametrize("channels", [8, 64])
def test_compressed_level_two_matches_autograd(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    model = get_model(_compressed_config(channels)).to(env.DEVICE).eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    # A compressed model is deployed on single-precision edge vectors. The
    # level-two composition assembles force and virial in that precision
    # throughout, whereas the autograd lower returns them in the precision of
    # its edge leaf, so a double-precision sample would compare two different
    # element types rather than two code paths.
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    reference = _run_graph(model, dtype=torch.float32)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    actual = _run_graph(model, dtype=torch.float32)
    for key in (
        "energy",
        "energy_redu",
        "energy_derv_r",
        "energy_derv_c",
        "energy_derv_c_redu",
    ):
        torch.testing.assert_close(actual[key], reference[key])


def _spin_config(channels: int = 16) -> dict:
    config = _compressed_config(channels)
    config["descriptor"]["use_spin"] = [True, False]
    return config


def _spin_sample(model: torch.nn.Module) -> tuple:
    """Return a canonical graph, its flat types and a per-node moment."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        attach_edge_csr,
        build_neighbor_graph,
    )

    nodes = 24
    generator = torch.Generator(device=env.DEVICE).manual_seed(31)
    coord = 5.0 * torch.rand(
        1, nodes, 3, dtype=torch.float32, device=env.DEVICE, generator=generator
    )
    atype = torch.arange(nodes, device=env.DEVICE).reshape(1, -1) % 2
    graph = attach_edge_csr(
        build_neighbor_graph(coord, atype, None, model.get_rcut()), nodes
    )
    spin = torch.randn(
        nodes, 3, dtype=torch.float32, device=env.DEVICE, generator=generator
    )
    return graph, atype.reshape(-1), spin


@_GPU
@pytest.mark.parametrize("gate", [0.8, 0.0])
def test_compressed_spin_lowers_match_autograd(
    monkeypatch: pytest.MonkeyPatch,
    gate: float,
) -> None:
    """Both fused lowers reproduce the autograd magnetic force.

    The magnetic force is assembled from two halves that no other output
    exercises: the on-site gradient closes inside the node kernel, and the
    neighbour half is emitted per edge and reduced onto source nodes. Checking
    it against the autograd lower covers the fused generic composition and the
    compact canonical deployment path in the same comparison.

    The operator assembles the spin invariants without reading the branch
    gate, which the portable path applies to the calibrated block; compression
    carries it as a factor on the inverse deviation of those columns. A
    non-unit value is what makes this comparison check that fold, and the
    closed gate pins the case an affine mean-and-deviation fold could not
    express -- a trained gate may legitimately reach zero.
    """
    import numpy as np

    model = get_model(_spin_config()).to(env.DEVICE).eval()
    descriptor = model.get_descriptor()
    descriptor.spin.set_spin_reference(np.array([1.7, 1.0, 1.0]))
    with torch.no_grad():
        descriptor.spin.spin_gate.fill_(gate)
    graph, atype, spin = _spin_sample(model)
    descriptor.enable_compression(min_nbor_dist=0.5)

    def lower() -> dict[str, torch.Tensor]:
        return model.forward_common_lower_graph(
            atype,
            graph.n_node,
            graph.n_node.clone(),
            graph.edge_index,
            graph.edge_vec,
            graph.edge_mask,
            destination_order=graph.destination_order,
            destination_row_ptr=graph.destination_row_ptr,
            source_order=graph.source_order,
            source_row_ptr=graph.source_row_ptr,
            destination_sorted=bool(graph.destination_sorted),
            spin=spin,
        )

    monkeypatch.setenv("DP_CUDA_INFER", "1")
    reference = lower()
    assert "energy_derv_r_mag" in reference
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    fused = lower()
    for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_r_mag"):
        torch.testing.assert_close(fused[key], reference[key], atol=8e-6, rtol=1e-4)

    physical = int(graph.destination_row_ptr[-1])
    canonical = model.forward_lower_canonical_graph(
        atype,
        graph.n_node,
        graph.n_node.clone(),
        graph.edge_index[0][:physical].to(torch.uint32).contiguous(),
        graph.edge_vec[:physical].contiguous(),
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order[:physical].to(torch.uint32).contiguous(),
        do_atomic_virial=False,
        spin=spin,
    )
    torch.testing.assert_close(
        canonical["force_mag"].reshape(-1, 3),
        reference["energy_derv_r_mag"].reshape(-1, 3),
        atol=8e-6,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        canonical["force"].reshape(-1, 3),
        reference["energy_derv_r"].reshape(-1, 3),
        atol=8e-6,
        rtol=1e-4,
    )
