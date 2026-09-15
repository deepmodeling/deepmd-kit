# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated-atom reference of a DPA4 model on the pt_expt graph route.

The reference node of every type is conditioned as the neutral ground-state
atom: zero charge with the ground-state multiplicity for the charge/spin
conditioning and a spin vector of one Bohr magneton per unpaired electron for
the native spin. An isolated atom under these conditions contributes exactly
its bias, and any cluster differs from the unreferenced model by the
isolated-atom network output of its atoms.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.utils.vacuum_reference import (
    reference_charge_spin,
    reference_spin,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)

TYPE_MAP = ["O", "H"]
BIAS = np.array([[-3.0], [0.5]])
CONFIG = {
    "type": "dpa4",
    "type_map": TYPE_MAP,
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
        "use_spin": [True, False],
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0, 1],
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    },
}


def make_model(vacuum_ref: bool) -> EnergyModel:
    config = {
        **CONFIG,
        "fitting_net": {**CONFIG["fitting_net"], "vacuum_ref": vacuum_ref},
    }
    model = get_model(config)
    data = jitter_zero_arrays(
        model.atomic_model.descriptor.serialize(), np.random.default_rng(3)
    )
    model.atomic_model.descriptor = DescrptDPA4.deserialize(data)
    fitting = model.atomic_model.fitting_net
    with torch.no_grad():
        fitting.bias_atom_e.copy_(
            torch.as_tensor(BIAS, dtype=fitting.bias_atom_e.dtype)
        )
    return model.to(env.DEVICE).eval()


def atom_energies(
    model: EnergyModel,
    coord: np.ndarray,
    atype: np.ndarray,
    charge_spin: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    def tensor(array: np.ndarray, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=env.DEVICE)

    ret = model.call_common(
        tensor(coord),
        tensor(atype, torch.long),
        None,
        charge_spin=tensor(charge_spin),
        spin=tensor(spin),
    )
    return ret["energy"][..., 0].detach().cpu().numpy()


def test_isolated_atoms_and_reference_subtraction() -> None:
    rng = np.random.default_rng(0)
    coord = rng.normal(size=(2, 6, 3)) * 1.2
    atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
    charge_spin = np.array([[0.0, 1.0], [1.0, 2.0]])
    spin = rng.normal(size=(2, 6, 3)) * (atype == 0)[..., None]
    iso_coord = np.zeros((2, 1, 3))
    iso_atype = np.array([[0], [1]])
    iso_charge_spin = reference_charge_spin(TYPE_MAP)
    iso_spin = reference_spin(TYPE_MAP)[:, None, :]
    vac = make_model(True)
    ref = make_model(False)

    # an isolated neutral ground-state atom contributes exactly its bias
    np.testing.assert_allclose(
        atom_energies(vac, iso_coord, iso_atype, iso_charge_spin, iso_spin),
        BIAS[:, 0][:, None],
        rtol=1e-10,
        atol=1e-10,
    )
    # the reference removes the isolated-atom network output of every atom,
    # whatever the charge, spin and geometry of the cluster
    e_vac = atom_energies(vac, coord, atype, charge_spin, spin)
    e_ref = atom_energies(ref, coord, atype, charge_spin, spin)
    iso_ref = atom_energies(ref, iso_coord, iso_atype, iso_charge_spin, iso_spin)[:, 0]
    expected = e_ref - (iso_ref - BIAS[:, 0])[atype]
    np.testing.assert_allclose(e_vac, expected, rtol=1e-10, atol=1e-10)
    # the conditioning is essential: a differently conditioned isolated atom
    # carries a learned deviation from the bias
    other = atom_energies(
        vac,
        iso_coord,
        iso_atype,
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.zeros((2, 1, 3)),
    )
    assert np.abs(other[0, 0] - BIAS[0, 0]) > 1e-8


def test_fold_reproduces_the_referenced_model() -> None:
    """Folding the reference into the bias keeps every energy and drops the rows."""
    rng = np.random.default_rng(1)
    coord = rng.normal(size=(2, 6, 3)) * 1.2
    atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
    charge_spin = np.array([[0.0, 1.0], [1.0, 2.0]])
    spin = rng.normal(size=(2, 6, 3)) * (atype == 0)[..., None]
    iso_coord = np.zeros((2, 1, 3))
    iso_atype = np.array([[0], [1]])
    iso_charge_spin = reference_charge_spin(TYPE_MAP)
    iso_spin = reference_spin(TYPE_MAP)[:, None, :]
    model = make_model(True)
    e_cluster = atom_energies(model, coord, atype, charge_spin, spin)
    e_iso = atom_energies(model, iso_coord, iso_atype, iso_charge_spin, iso_spin)

    model.fold_vacuum_reference()
    assert not model.atomic_model.fitting_net.vacuum_ref
    np.testing.assert_allclose(
        atom_energies(model, coord, atype, charge_spin, spin),
        e_cluster,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        atom_energies(model, iso_coord, iso_atype, iso_charge_spin, iso_spin),
        e_iso,
        rtol=1e-10,
        atol=1e-10,
    )


def test_dense_and_graph_routes_share_the_vacuum_descriptor() -> None:
    """The reference rows appended to a graph equal the single-atom-frame descriptor."""
    model = make_model(vacuum_ref=True)
    am = model.atomic_model
    rng = np.random.default_rng(5)
    coord = torch.as_tensor(
        rng.normal(size=(1, 4, 3)) * 1.2, dtype=torch.float64, device=env.DEVICE
    )
    atype = torch.as_tensor([[0, 1, 1, 0]], dtype=torch.long, device=env.DEVICE)
    graph = build_neighbor_graph(
        coord, atype, None, CONFIG["descriptor"]["rcut"], with_csr=True
    )
    charge_spin = torch.as_tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
    spin = torch.as_tensor(
        rng.normal(size=(4, 3)), dtype=torch.float64, device=env.DEVICE
    )
    graph, atype_all, charge_spin, spin = am.append_vacuum_frames(
        graph, atype.reshape(-1), charge_spin, spin
    )
    gg, _ = am.descriptor.call_graph(
        graph,
        atype_all,
        type_embedding=am.descriptor.graph_type_embedding_table(),
        spin=spin,
        charge_spin=charge_spin,
    )
    np.testing.assert_allclose(
        gg[4:].detach().cpu().numpy(),
        am.vacuum_descriptor().detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def freeze_model(
    tmp_path, numb_fparam: int, suffix: str = ".pt2"
) -> tuple[EnergyModel, DeepPot]:
    """Freeze a vacuum-referenced model without charge/spin conditioning."""
    import copy

    from deepmd.pt_expt.entrypoints.main import (
        freeze,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )

    config = copy.deepcopy(CONFIG)
    for key in ("use_spin", "add_chg_spin_ebd", "default_chg_spin"):
        config["descriptor"].pop(key)
    config["fitting_net"].update({"vacuum_ref": True, "numb_fparam": numb_fparam})
    model = get_model(config)
    data = jitter_zero_arrays(
        model.atomic_model.descriptor.serialize(), np.random.default_rng(3)
    )
    model.atomic_model.descriptor = DescrptDPA4.deserialize(data)
    fitting = model.atomic_model.fitting_net
    with torch.no_grad():
        fitting.bias_atom_e.copy_(
            torch.as_tensor(BIAS, dtype=fitting.bias_atom_e.dtype)
        )
    model = model.to(env.DEVICE).eval()
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(config))
    ckpt = tmp_path / "vacuum.pt"
    torch.save({"model": wrapper.state_dict()}, ckpt)
    frozen = tmp_path / f"vacuum_frozen{suffix}"
    freeze(model=str(ckpt), output=str(frozen))
    return model, DeepPot(str(frozen))


def eager_energies(
    model: EnergyModel, coord: np.ndarray, atype: np.ndarray, fparam=None
) -> np.ndarray:
    kwargs = {}
    if fparam is not None:
        kwargs["fparam"] = torch.as_tensor(
            fparam, dtype=torch.float64, device=env.DEVICE
        )
    ret = model.call_common(
        torch.as_tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.as_tensor(atype[None], dtype=torch.long, device=env.DEVICE),
        None,
        **kwargs,
    )
    return ret["energy"][..., 0].detach().cpu().numpy().reshape(-1)


@pytest.mark.parametrize("suffix", [".pt2", ".pte"])
def test_freeze_folds_the_reference(tmp_path, suffix) -> None:
    """A frozen model reproduces the referenced model without reference atoms."""
    model, dp = freeze_model(tmp_path, numb_fparam=0, suffix=suffix)
    assert dp.deep_eval.get_ntypes() == 2
    rng = np.random.default_rng(2)
    coord = rng.normal(size=(1, 6, 3)) * 1.2
    atype = np.array([0, 1, 1, 0, 1, 0])
    for itype in range(2):
        _, _, _, atom_energy, _ = dp.eval(
            np.zeros((1, 1, 3)), None, np.array([itype]), atomic=True
        )
        np.testing.assert_allclose(
            atom_energy.reshape(-1), BIAS[itype], rtol=1e-8, atol=1e-8
        )
    _, _, _, atom_energy, _ = dp.eval(coord, None, atype, atomic=True)
    np.testing.assert_allclose(
        atom_energy.reshape(-1),
        eager_energies(model, coord, atype),
        rtol=1e-8,
        atol=1e-8,
    )


def test_freeze_keeps_the_reference_with_fparam(tmp_path) -> None:
    """With frame parameters the frozen model references from the stored vacuum table."""
    model, dp = freeze_model(tmp_path, numb_fparam=1)
    rng = np.random.default_rng(2)
    coord = rng.normal(size=(1, 6, 3)) * 1.2
    atype = np.array([0, 1, 1, 0, 1, 0])
    for fparam in (np.array([[0.3]]), np.array([[-1.1]])):
        for itype in range(2):
            _, _, _, atom_energy, _ = dp.eval(
                np.zeros((1, 1, 3)), None, np.array([itype]), atomic=True, fparam=fparam
            )
            np.testing.assert_allclose(
                atom_energy.reshape(-1), BIAS[itype], rtol=1e-8, atol=1e-8
            )
        _, _, _, atom_energy, _ = dp.eval(
            coord, None, atype, atomic=True, fparam=fparam
        )
        np.testing.assert_allclose(
            atom_energy.reshape(-1),
            eager_energies(model, coord, atype, fparam),
            rtol=1e-8,
            atol=1e-8,
        )


def test_reference_rows_follow_a_padded_node_axis() -> None:
    """Padding nodes after the real frames leave the reference conditioning intact."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_neighbor_graph,
    )

    def tensor(array: np.ndarray, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=env.DEVICE)

    rng = np.random.default_rng(4)
    coord = tensor(rng.normal(size=(2, 6, 3)) * 1.2)
    atype = tensor(np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]]), torch.long)
    charge_spin = tensor(np.array([[0.0, 1.0], [1.0, 2.0]]))
    spin = tensor(rng.normal(size=(2, 6, 3)) * (atype.cpu().numpy() == 0)[..., None])
    model = make_model(True)
    atomic_model = model.atomic_model
    graph = build_neighbor_graph(coord, atype, None, CONFIG["descriptor"]["rcut"])
    atype_flat = atype.reshape(-1)
    spin_flat = spin.reshape(-1, 3)
    reference = atomic_model.forward_common_atomic_graph(
        graph, atype_flat, charge_spin=charge_spin, spin=spin_flat
    )["energy"]

    n_pad = 3
    atype_padded = torch.cat(
        [atype_flat, torch.full((n_pad,), -1, dtype=torch.long, device=env.DEVICE)]
    )
    spin_padded = torch.cat(
        [spin_flat, torch.zeros((n_pad, 3), dtype=torch.float64, device=env.DEVICE)]
    )
    padded = atomic_model.forward_common_atomic_graph(
        graph, atype_padded, charge_spin=charge_spin, spin=spin_padded
    )["energy"]
    np.testing.assert_allclose(
        padded[: atype_flat.shape[0]].detach().cpu().numpy(),
        reference.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.all(padded[atype_flat.shape[0] :] == 0.0)


def test_archive_carries_the_live_model(tmp_path) -> None:
    """The archive keeps the unfolded model, so a re-export resolves the reference once."""
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
        serialize_from_file,
    )

    _, dp = freeze_model(tmp_path, numb_fparam=0)
    data = serialize_from_file(str(tmp_path / "vacuum_frozen.pt2"))
    model = BaseModel.deserialize(data["model"])
    fitting = model.atomic_model.fitting_net
    assert fitting.needs_vacuum_descriptor()
    np.testing.assert_allclose(fitting.bias_atom_e.detach().cpu().numpy(), BIAS)
    again = tmp_path / "vacuum_again.pt2"
    deserialize_to_file(str(again), data)
    dp_again = DeepPot(str(again))
    for itype in range(2):
        for evaluator in (dp, dp_again):
            _, _, _, atom_energy, _ = evaluator.eval(
                np.zeros((1, 1, 3)), None, np.array([itype]), atomic=True
            )
            np.testing.assert_allclose(
                atom_energy.reshape(-1), BIAS[itype], rtol=1e-8, atol=1e-8
            )
