# SPDX-License-Identifier: LGPL-3.0-or-later
"""Embedding extraction of the pt_expt backend (``DeepPot.eval_embedding``).

The embeddings are the per-atom descriptor, the per-atom activation after the
last fitting hidden layer (atomic feature) and its masked atom-sum over every
frame (structural feature), evaluated by one forward on the neighbor graph.
"""

import copy
import os
from unittest import (
    mock,
)

import h5py
import numpy as np
import pytest
import torch

from deepmd.dpmodel.model.base_model import BaseModel as DPBaseModel
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.entrypoints.embedding import (
    embedding,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt.train.wrapper import ModelWrapper as PtModelWrapper
from deepmd.pt_expt.kernels.dpa4c import (
    graph_compress,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

# A small fp64 DPA4 conditioned on the frame charge state, with a frame and an
# atomic parameter and an excluded atom type, so that every conditioning input
# and the exclusion convention of the structural feature are exercised. The
# charge state and the frame parameter have defaults.
_DPA4_CONFIG = {
    "type": "dpa4",
    "type_map": ["O", "H"],
    "atom_exclude_types": [1],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "numb_fparam": 1,
        "default_fparam": [0.1],
        "numb_aparam": 1,
        "precision": "float64",
        "seed": 1,
    },
}

_DPA4C_CONFIG = {
    "type_map": ["O", "H", "He"],
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
_DPA4C_BRIDGED_CONFIG = {
    **_DPA4C_CONFIG,
    "atom_exclude_types": [2],
    "bridging_method": "zbl",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8,
}

# Two frames of a water-like box. The second frame pads its last slot with a
# virtual atom (type -1), as a batch of unequal atom counts does.
_ATYPES = np.array([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, -1]], dtype=np.int32)
_COORDS = np.array(
    [
        [
            [1.0, 1.0, 1.0],
            [3.2, 1.4, 1.1],
            [1.3, 1.8, 1.0],
            [0.4, 1.2, 1.6],
            [3.6, 2.0, 1.3],
            [3.4, 0.7, 1.7],
        ],
        [
            [1.1, 0.9, 1.0],
            [3.0, 1.5, 1.2],
            [1.4, 1.7, 0.9],
            [0.5, 1.1, 1.7],
            [3.7, 2.1, 1.2],
            [0.0, 0.0, 0.0],
        ],
    ]
)
_CELLS = np.tile((np.eye(3) * 6.0).reshape(1, 9), (2, 1))
_FPARAM = np.array([[0.3], [-0.7]])
_APARAM = np.linspace(-0.5, 0.6, _ATYPES.size).reshape(*_ATYPES.shape, 1)
# One charge state per frame, neither of them the default of the model.
_CHARGE_SPIN = np.array([[1.0, 2.0], [-1.0, 2.0]])


def _normalize_model(model: dict) -> dict:
    config = {
        "model": copy.deepcopy(model),
        "training": {"training_data": {"systems": ["dummy"]}, "numb_steps": 1},
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }
    return normalize(update_deepmd_input(config, warning=False))["model"]


def _perturb(model: torch.nn.Module, seed: int, scale: float = 0.05) -> None:
    """Move every weight off its initialization, zero-initialized ones included.

    The noise has standard deviation ``scale``, small enough to keep the
    activations of order one.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
                parameter.add_(scale * noise.to(parameter.device))


def _save_checkpoint(model: torch.nn.Module, model_params: dict, path: str) -> str:
    """Write a pt_expt training checkpoint."""
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(model_params))
    torch.save({"model": wrapper.state_dict()}, path)
    return path


def _assert_embeddings_close(
    actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...], **tol: float
) -> None:
    for name, aa, ee in zip(
        ("descriptor", "atomic_feature", "structural_feature"),
        actual,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(aa, ee, err_msg=name, **tol)


@pytest.fixture(scope="module")
def dpa4_models(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One DPA4 model as a pt checkpoint, a pt_expt checkpoint and a .pte."""
    tmp_path = tmp_path_factory.mktemp("dpa4_embedding")
    model_params = _normalize_model(_DPA4_CONFIG)
    pt_model = pt_get_model(copy.deepcopy(model_params)).to(torch.float64)
    _perturb(pt_model, seed=3)
    pt_model.eval()
    pt_path = str(tmp_path / "dpa4_pt.pt")
    wrapper = PtModelWrapper(pt_model, model_params=copy.deepcopy(model_params))
    torch.save({"model": wrapper.state_dict()}, pt_path)

    # The pt_expt model takes the weights of the pt one through its
    # serialization, which every dpmodel backend then reads.
    model = BaseModel.deserialize(pt_model.serialize())
    pt_expt_path = _save_checkpoint(
        model, model_params, str(tmp_path / "dpa4_pt_expt.pt")
    )
    data = {"model": model.serialize()}
    pte_path = str(tmp_path / "dpa4.pte")
    deserialize_to_file(pte_path, copy.deepcopy(data))
    return {"pt": pt_path, "pt_expt": pt_expt_path, "pte": pte_path, "data": data}


def test_dpa4_matches_pt_backend(dpa4_models: dict) -> None:
    """The graph forward reproduces the embeddings of the pt backend.

    The pt backend evaluates its own implementation of the model, so the two
    backends agree on the descriptor, on the fitting hidden activation under a
    charge state and a frame and an atomic parameter per frame, on the zero rows
    of a virtual atom and on the exclusion of a type from the structural feature
    only.
    """
    kwargs = {
        "fparam": _FPARAM,
        "aparam": _APARAM,
        "charge_spin": _CHARGE_SPIN,
        "mixed_type": True,
        "dtype": "fp64",
    }
    expected = DeepPot(dpa4_models["pt"]).eval_embedding(
        _COORDS, _CELLS, _ATYPES, **kwargs
    )
    checkpoint = DeepPot(dpa4_models["pt_expt"])
    actual = checkpoint.eval_embedding(_COORDS, _CELLS, _ATYPES, **kwargs)
    _assert_embeddings_close(actual, expected, rtol=1e-10, atol=1e-12)
    # The comparison must see the excluded type and the virtual atom.
    descriptor, atomic_feature, structural_feature = actual
    assert np.abs(atomic_feature[0, _ATYPES[0] == 1]).max() > 1e-3
    np.testing.assert_allclose(
        structural_feature, atomic_feature[:, :2].sum(axis=1), rtol=1e-12
    )
    assert not descriptor[1, -1].any() and not atomic_feature[1, -1].any()
    # The default output precision is float32.
    assert all(
        out.dtype == np.float32
        for out in checkpoint.eval_embedding(
            _COORDS,
            _CELLS,
            _ATYPES,
            fparam=_FPARAM,
            aparam=_APARAM,
            charge_spin=_CHARGE_SPIN,
            mixed_type=True,
        )
    )


def test_dpa4_frozen_model_and_dpmodel(dpa4_models: dict) -> None:
    """A frozen model and the NumPy dpmodel give the embeddings of the checkpoint.

    The frozen model evaluates one frame per batch, so its frame and atomic
    parameters are split along with the frames.
    """
    kwargs = {
        "fparam": _FPARAM,
        "aparam": _APARAM,
        "charge_spin": _CHARGE_SPIN,
        "mixed_type": True,
        "dtype": "fp64",
    }
    expected = DeepPot(dpa4_models["pt_expt"]).eval_embedding(
        _COORDS, _CELLS, _ATYPES, **kwargs
    )
    frozen = DeepPot(
        dpa4_models["pte"], auto_batch_size=_ATYPES.shape[1]
    ).eval_embedding(_COORDS, _CELLS, _ATYPES, **kwargs)
    _assert_embeddings_close(frozen, expected, rtol=1e-12, atol=1e-12)

    # The dpmodel atomic model takes the same graph in NumPy.
    model = DPBaseModel.deserialize(copy.deepcopy(dpa4_models["data"]["model"]))
    graph = build_neighbor_graph(
        _COORDS, _ATYPES, _CELLS, model.get_rcut(), canonicalize=True
    )
    ret = model.atomic_model.forward_common_embedding_graph(
        graph,
        _ATYPES.reshape(-1),
        fparam=_FPARAM,
        aparam=_APARAM.reshape(-1, 1),
        charge_spin=_CHARGE_SPIN,
    )
    nframes, natoms = _ATYPES.shape
    numpy_embeddings = (
        ret["descriptor"].reshape(nframes, natoms, -1),
        ret["atomic_feature"].reshape(nframes, natoms, -1),
        ret["structural_feature"],
    )
    _assert_embeddings_close(numpy_embeddings, expected, rtol=1e-10, atol=1e-12)


def test_dpa4_command_reads_the_conditions_of_the_system(
    dpa4_models: dict, tmp_path: "os.PathLike[str]"
) -> None:
    """``dp embed`` evaluates every frame under the conditions its system holds.

    Two mixed-type systems pad their second frame with a virtual atom. One
    stores the frame and atomic parameters and the charge state of every
    frame; the other stores the atomic parameters alone and so takes the
    default frame parameter and charge state of the model. For each, the
    command writes the embeddings that the Python interface returns for the
    same inputs, and the types of the virtual atoms.
    """
    nframes, natoms = _ATYPES.shape
    stored = {
        "conditioned": {"fparam": _FPARAM, "charge_spin": _CHARGE_SPIN},
        "defaults": {},
    }
    for name, conditions in stored.items():
        system = os.path.join(tmp_path, "systems", name)
        set_dir = os.path.join(system, "set.000")
        os.makedirs(set_dir)
        np.savetxt(
            os.path.join(system, "type.raw"), np.zeros(natoms, dtype=int), fmt="%d"
        )
        with open(os.path.join(system, "type_map.raw"), "w") as stream:
            stream.write("\n".join(_DPA4_CONFIG["type_map"]) + "\n")
        arrays = {
            "coord": _COORDS.reshape(nframes, -1),
            "box": _CELLS,
            "real_atom_types": _ATYPES,
            "aparam": _APARAM.reshape(nframes, -1),
            **conditions,
        }
        for key, value in arrays.items():
            np.save(os.path.join(set_dir, f"{key}.npy"), value)
    output = os.path.join(tmp_path, "embedding.hdf5")
    embedding(
        model=dpa4_models["pt_expt"],
        system=os.path.join(tmp_path, "systems"),
        datafile=None,
        output=output,
        dtype="fp64",
    )

    dp = DeepPot(dpa4_models["pt_expt"])
    written = {}
    with h5py.File(output, "r") as stream:
        for name, conditions in stored.items():
            group = stream[name]
            np.testing.assert_array_equal(group["atom_types"][:], _ATYPES)
            written[name] = tuple(
                group[key][:]
                for key in ("descriptor", "atomic_feature", "structural_feature")
            )
            expected = dp.eval_embedding(
                _COORDS,
                _CELLS,
                _ATYPES,
                aparam=_APARAM,
                mixed_type=True,
                dtype="fp64",
                **conditions,
            )
            _assert_embeddings_close(written[name], expected, rtol=1e-12, atol=1e-12)
    # The conditions the first system stores move its embeddings away from
    # those of the defaults.
    conditioned, defaults = written["conditioned"][1], written["defaults"][1]
    assert np.abs(conditioned - defaults).max() > 1e-2 * np.abs(defaults).max()


_DPA1_DESCRIPTOR = {
    "type": "dpa1",
    "sel": 30,
    "rcut": 4.0,
    "rcut_smth": 0.5,
    "neuron": [4, 8],
    "axis_neuron": 2,
    "attn_layer": 0,
    "precision": "float64",
    "seed": 1,
}
_DPA2_DESCRIPTOR = {
    "type": "dpa2",
    "repinit": {
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "nsel": 30,
        "neuron": [4, 8],
        "axis_neuron": 2,
        "tebd_input_mode": "concat",
    },
    "repformer": {
        "rcut": 3.0,
        "rcut_smth": 0.5,
        "nsel": 20,
        "nlayers": 2,
        "g1_dim": 8,
        "g2_dim": 4,
        "attn2_hidden": 4,
        "attn2_nhead": 1,
        "attn1_hidden": 4,
        "attn1_nhead": 1,
        "axis_neuron": 2,
    },
    "precision": "float64",
    "seed": 1,
}


@pytest.mark.parametrize(
    ("descriptor", "fitting_type"),
    [
        pytest.param(_DPA2_DESCRIPTOR, "ener", id="dpa2-energy"),
        pytest.param(_DPA1_DESCRIPTOR, "dipole", id="dpa1-dipole"),
        pytest.param(_DPA1_DESCRIPTOR, "polar", id="dpa1-polar"),
    ],
)
def test_graph_descriptors_and_tensor_fittings_match_pt_backend(
    descriptor: dict, fitting_type: str, tmp_path: "os.PathLike[str]"
) -> None:
    """Other graph descriptors and the tensor fittings reproduce the pt backend.

    The pt backend leaves the rows of a virtual atom of these models
    unmasked, so the per-atom comparison covers the real atoms.
    """
    model_params = _normalize_model(
        {
            "type_map": ["O", "H"],
            "descriptor": descriptor,
            "fitting_net": {
                "type": fitting_type,
                "neuron": [8, 8],
                "precision": "float64",
                "seed": 1,
            },
        }
    )
    pt_model = pt_get_model(copy.deepcopy(model_params)).to(torch.float64)
    _perturb(pt_model, seed=3)
    pt_model.eval()
    pt_path = os.path.join(tmp_path, "pt.pt")
    wrapper = PtModelWrapper(pt_model, model_params=copy.deepcopy(model_params))
    torch.save({"model": wrapper.state_dict()}, pt_path)
    pt_expt_path = _save_checkpoint(
        BaseModel.deserialize(pt_model.serialize()),
        model_params,
        os.path.join(tmp_path, "pt_expt.pt"),
    )
    kwargs = {"mixed_type": True, "dtype": "fp64"}
    expected = DeepEval(pt_path).eval_embedding(_COORDS, _CELLS, _ATYPES, **kwargs)
    actual = DeepEval(pt_expt_path).eval_embedding(_COORDS, _CELLS, _ATYPES, **kwargs)
    real = _ATYPES >= 0
    for name, aa, ee in zip(
        ("descriptor", "atomic_feature"), actual[:2], expected[:2], strict=True
    ):
        np.testing.assert_allclose(
            aa[real], ee[real], rtol=1e-10, atol=1e-12, err_msg=name
        )
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-10, atol=1e-12)


class TestDPA4CEmbedding:
    """DPA4C embeddings against the energy forward of the same model."""

    @pytest.fixture(autouse=True)
    def _models(self, tmp_path: "os.PathLike[str]") -> None:
        self.plain_params = _normalize_model(_DPA4C_CONFIG)
        self.plain = get_model(copy.deepcopy(self.plain_params)).eval()
        _perturb(self.plain, seed=11)
        self.plain_path = _save_checkpoint(
            self.plain, self.plain_params, os.path.join(tmp_path, "plain.pt")
        )
        self.bridged_params = _normalize_model(_DPA4C_BRIDGED_CONFIG)
        bridged = get_model(copy.deepcopy(self.bridged_params)).eval()
        bridged.atomic_model.models[0].load_state_dict(
            self.plain.atomic_model.state_dict()
        )
        self.bridged_path = _save_checkpoint(
            bridged, self.bridged_params, os.path.join(tmp_path, "bridged.pt")
        )
        # A jittered cubic lattice holding every type, whose pairs all lie
        # beyond the bridging window.
        grid = np.stack(np.meshgrid(*(np.arange(2),) * 3, indexing="ij"), axis=-1)
        jitter = np.random.default_rng(1).uniform(-0.3, 0.3, (8, 3))
        self.coord = ((grid.reshape(8, 3) + 0.5) * 3.0 + jitter).reshape(1, 8, 3)
        self.atype = np.array([0, 2, 1, 1, 0, 2, 1, 0], dtype=np.int32)
        self.cell = _CELLS[:1]

    def test_atomic_feature_reproduces_the_atomic_energy(self) -> None:
        """The output layer maps the atomic feature to the atomic energy.

        The fitting output of an atom is its output layer applied to the
        atomic feature plus a per-type bias, so the residual of the atomic
        energy depends on the atom type alone. The fused fitting operator is
        made available, since an embedding forward keeps the reference
        network wherever that operator would serve the energy.
        """
        dp = DeepPot(self.plain_path)
        _, _, _, atom_energy, _ = dp.eval(
            self.coord, self.cell, self.atype, atomic=True
        )
        with (
            mock.patch(
                "deepmd.pt_expt.fitting.ener_fitting.fused_fitting_available",
                return_value=True,
            ),
            mock.patch(
                "deepmd.pt_expt.fitting.ener_fitting.fitting_eligible",
                return_value=True,
            ),
        ):
            _, atomic_feature, structural_feature = dp.eval_embedding(
                self.coord, self.cell, self.atype, dtype="fp64"
            )
        output_layer = self.plain.atomic_model.fitting_net.nets[()].layers[-1]
        weight = output_layer.w.detach().cpu().numpy()
        bias = output_layer.b.detach().cpu().numpy()
        residual = atom_energy[0, :, 0] - (atomic_feature[0] @ weight + bias)[:, 0]
        for itype in np.unique(self.atype):
            same = residual[self.atype == itype]
            np.testing.assert_allclose(same, same[0], rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(
            structural_feature, atomic_feature.sum(axis=1), rtol=1e-12
        )

    def test_bridged_model_takes_the_learned_child(self) -> None:
        """A bridged composition reports the embeddings of its learned child.

        Beyond the bridging window the learned child evaluates the plain
        model, so the per-atom embeddings agree, and the atom type the
        composition excludes leaves the structural feature only.
        """
        plain = DeepPot(self.plain_path).eval_embedding(
            self.coord, self.cell, self.atype, dtype="fp64"
        )
        bridged = DeepPot(self.bridged_path).eval_embedding(
            self.coord, self.cell, self.atype, dtype="fp64"
        )
        np.testing.assert_allclose(bridged[0], plain[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bridged[1], plain[1], rtol=1e-12, atol=1e-12)
        counted = self.atype != 2
        np.testing.assert_allclose(
            bridged[2], plain[1][:, counted].sum(axis=1), rtol=1e-12, atol=1e-12
        )
        assert np.abs(plain[1][0, ~counted]).max() > 1e-3


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
@pytest.mark.skipif(
    not graph_compress.op_available(),
    reason="the compressed DPA4C operator is not built",
)
def test_compressed_charge_state(
    tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compressed archive serves every charge state to its embeddings.

    Compression folds the default charge state into the tables of the
    descriptor, which a runtime state rebuilds. The fused descriptor of the
    frozen archive reads those tables, and its embeddings follow the
    uncompressed checkpoint at both states, within the compression error,
    which the separation of the two states exceeds.
    """
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 4.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 8,
            "precision": "float32",
            "seed": 11,
            "add_chg_spin_ebd": True,
            "default_chg_spin": [0.0, 1.0],
        },
        "fitting_net": {"neuron": [16, 16], "precision": "float32", "seed": 11},
    }
    model_params = _normalize_model(config)
    model = get_model(copy.deepcopy(model_params)).eval()
    # The charge-state head starts with a zero projection, which would make
    # the two states indistinguishable; this opening separates them far beyond
    # the compression error.
    _perturb(model.atomic_model.descriptor.charge_spin_embedding, seed=5, scale=0.3)
    reference = DeepPot(
        _save_checkpoint(model, model_params, os.path.join(tmp_path, "ref.pt"))
    )
    model.atomic_model.descriptor.enable_compression(1.4, table_stride_1=0.01)
    frozen_path = os.path.join(tmp_path, "compressed.pt2")
    deserialize_to_file(frozen_path, {"model": model.serialize()}, lower_kind="graph")
    frozen = DeepPot(frozen_path)

    atype = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    states = {"default": np.array([[0.0, 1.0]]), "other": np.array([[1.0, 2.0]])}
    expected = {
        name: reference.eval_embedding(
            _COORDS[:1], _CELLS[:1], atype, charge_spin=charge_spin, dtype="fp64"
        )
        for name, charge_spin in states.items()
    }
    # The fused operators serve an inference call on every device at this
    # level.
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    with mock.patch.object(
        graph_compress,
        "dpa4c_graph_compress",
        wraps=graph_compress.dpa4c_graph_compress,
    ) as kernel:
        for name, charge_spin in states.items():
            actual = frozen.eval_embedding(
                _COORDS[:1], _CELLS[:1], atype, charge_spin=charge_spin, dtype="fp64"
            )
            _assert_embeddings_close(actual, expected[name], rtol=1e-4, atol=1e-5)
    assert kernel.call_count == len(states)
    # The folded tables hold one state, which all frames of a call share.
    with pytest.raises(ValueError, match="single state"):
        frozen.eval_embedding(
            np.tile(_COORDS[:1], (2, 1, 1)),
            np.tile(_CELLS[:1], (2, 1)),
            atype,
            charge_spin=np.concatenate(list(states.values())),
        )
    assert (
        np.abs(expected["other"][1] - expected["default"][1]).max()
        > 1e-2 * np.abs(expected["default"][1]).max()
    )


def test_rejects_a_model_without_graph_forward(tmp_path: "os.PathLike[str]") -> None:
    """A model that is not evaluated on the neighbor graph has no embedding."""
    model_params = _normalize_model(
        {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [8, 8],
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "neuron": [4, 8],
                "axis_neuron": 4,
                "seed": 1,
            },
            "fitting_net": {"neuron": [8], "seed": 1},
        }
    )
    model = get_model(copy.deepcopy(model_params)).eval()
    dp = DeepPot(_save_checkpoint(model, model_params, os.path.join(tmp_path, "se.pt")))
    with pytest.raises(NotImplementedError, match="neighbor-graph"):
        dp.eval_embedding(_COORDS[:1], _CELLS[:1], _ATYPES[0])
