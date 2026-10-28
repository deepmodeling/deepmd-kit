# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freezing and evaluating non-energy models.

DPA4C is graph-only, so its dipole, polarizability, DOS and property models
freeze to the graph lower with the internal output keys and are evaluated
from the frozen archive and from a training checkpoint alike. The dense
polarizability case pins the ``polar`` request alias, which the pt_expt
inference previously answered with NaN, and ``eval_full`` pins the
component-major layout of the per-atom tensor derivatives.
"""

import os
import shutil
import tempfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.entrypoints.test import test as dp_test
from deepmd.infer import (
    DeepEval,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_polar import (
    DeepPolar,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.pt_expt.entrypoints.main import (
    freeze,
)
from deepmd.pt_expt.infer.deep_eval import (
    _fold_output,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_has_energy_output,
    model_uses_graph_lower,
)
from deepmd.pt_expt.model.make_model import (
    _fused_energy_force_graph,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

WATER_TENSOR = Path(__file__).parents[2] / "pt" / "water_tensor"


def _params(kind: str, descriptor: dict | None = None, **fitting: object) -> dict:
    return {
        "type_map": ["O", "H"],
        "atom_exclude_types": [1],
        "descriptor": descriptor
        or {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 16,
            "lmax": 2,
            "n_radial": 4,
            "precision": "float64",
            "seed": 17,
            "output_rot_mat": True,
        },
        "fitting_net": {
            "type": kind,
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
            **fitting,
        },
    }


def _frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two periodic water-like frames: three O and six H."""
    rng = np.random.default_rng(7)
    cell = 4.0 * np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    coord = rng.uniform(0.0, 4.0, (2, 9, 3))
    atype = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    return coord, np.tile(cell.reshape(1, 9), (2, 1)), atype


def _reference(
    model: torch.nn.Module, kind: str, coord: np.ndarray, cell: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic and global tensors of the eager model forward."""
    ret = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(np.tile(_frames()[2], (coord.shape[0], 1)), device=env.DEVICE),
        box=None
        if cell is None
        else torch.tensor(cell, dtype=torch.float64, device=env.DEVICE),
    )
    nf = coord.shape[0]
    return (
        ret[kind].detach().cpu().numpy().reshape(nf, 9, -1),
        ret[f"global_{kind}"].detach().cpu().numpy().reshape(nf, -1),
    )


def _save_checkpoint(model: torch.nn.Module, params: dict, path: str) -> None:
    """Save a checkpoint in the layout produced by pt_expt training."""
    wrapper = ModelWrapper(model, model_params=params)
    torch.save({"model": wrapper.state_dict()}, path)


@pytest.fixture(scope="module", params=["dipole", "polar"])
def frozen(request: pytest.FixtureRequest):
    kind = request.param
    tmpdir = tempfile.mkdtemp()
    params = _params(kind)
    model = get_model(params).to(env.DEVICE).eval()
    ckpt = os.path.join(tmpdir, "model.ckpt.pt")
    _save_checkpoint(model, params, ckpt)
    # A graph .pte skips the AOTInductor compile of the default .pt2.
    pte = os.path.join(tmpdir, "frozen_model.pte")
    freeze(model=ckpt, output=pte)
    yield kind, model, ckpt, pte
    shutil.rmtree(tmpdir)


@pytest.mark.parametrize("artifact", ["pte", "ckpt"])
@pytest.mark.parametrize("periodic", [True, False])
def test_eval_matches_forward(frozen, artifact: str, periodic: bool) -> None:
    kind, model, ckpt, pte = frozen
    dp = DeepEval(pte if artifact == "pte" else ckpt)
    assert isinstance(dp, DeepDipole if kind == "dipole" else DeepPolar)
    assert dp.get_sel_type() == [0]
    metadata = dp.deep_eval.metadata
    assert metadata["lower_input_kind"] == "graph"
    assert metadata["graph_output_keys"] == "internal"

    coord, cell, atype = _frames()
    cell = cell if periodic else None
    atomic, total = _reference(model, kind, coord, cell)
    np.testing.assert_allclose(
        dp.eval(coord, cell, atype, atomic=True), atomic, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        dp.eval(coord, cell, atype, atomic=False), total, atol=1e-10, rtol=1e-10
    )


def test_dp_test_reports_finite_errors(frozen) -> None:
    kind, _, _, pte = frozen
    label = "atomic_dipole" if kind == "dipole" else "atomic_polarizability"
    detail = os.path.join(os.path.dirname(pte), "detail")
    dp_test(
        model=pte,
        system=str(WATER_TENSOR / kind / "atomic_system"),
        datafile=None,
        set_prefix="set",
        numb_test=2,
        rand_seed=None,
        shuffle_test=False,
        detail_file=detail,
        atomic=True,
    )
    (detail_file,) = Path(detail).parent.glob("detail*.out")
    # Columns hold the reference block followed by the prediction block.
    out = np.loadtxt(detail_file)
    assert out.size > 0
    assert np.all(np.isfinite(out)), label


def test_default_freeze_is_a_graph_pt2() -> None:
    """The CLI default for a graph-only dipole model is an AOTInductor .pt2."""
    tmpdir = tempfile.mkdtemp()
    try:
        params = _params("dipole")
        model = get_model(params).to(env.DEVICE).eval()
        ckpt = os.path.join(tmpdir, "model.ckpt.pt")
        _save_checkpoint(model, params, ckpt)
        freeze(model=ckpt, output=os.path.join(tmpdir, "frozen_model"))
        pt2 = os.path.join(tmpdir, "frozen_model.pt2")
        assert os.path.exists(pt2)
        dp = DeepEval(pt2)
        assert isinstance(dp, DeepDipole)
        coord, cell, atype = _frames()
        atomic, total = _reference(model, "dipole", coord, cell)
        np.testing.assert_allclose(
            dp.eval(coord, cell, atype, atomic=True), atomic, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            dp.eval(coord, cell, atype, atomic=False), total, atol=1e-10, rtol=1e-10
        )
    finally:
        shutil.rmtree(tmpdir)


def test_dense_polar_request_alias() -> None:
    """``DeepPolar`` requests ``polar``; the model emits ``polarizability``."""
    model = get_model(_params("polar", SE_E2_A)).to(env.DEVICE).eval()
    tmpdir = tempfile.mkdtemp()
    try:
        pte = os.path.join(tmpdir, "polar.pte")
        deserialize_to_file(pte, {"model": model.serialize()})
        dp = DeepEval(pte)
        assert isinstance(dp, DeepPolar)
        assert dp.deep_eval.metadata["lower_input_kind"] == "nlist"
        coord, cell, atype = _frames()
        atomic, total = _reference(model, "polar", coord, cell)
        np.testing.assert_allclose(
            dp.eval(coord, cell, atype, atomic=True), atomic, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            dp.eval(coord, cell, atype, atomic=False), total, atol=1e-10, rtol=1e-10
        )
    finally:
        shutil.rmtree(tmpdir)


SE_E2_A = {
    "type": "se_e2_a",
    "sel": [8, 16],
    "rcut_smth": 0.5,
    "rcut": 3.0,
    "neuron": [4, 8],
    "axis_neuron": 4,
    "precision": "float64",
    "seed": 17,
}


@pytest.mark.parametrize("route", ["graph-pte", "graph-ckpt", "dense-pte"])
def test_eval_full_derivatives_are_component_major(route: str) -> None:
    """``eval_full`` returns force and atomic virial as ``(nf, 3, natoms, *)``.

    The model lays the derivatives of a vector output out atom-major. The
    force of component ``c`` on atom ``a`` must equal minus the derivative
    of the global dipole component ``c`` with respect to that atom, which a
    central difference checks independently of any output layout.
    """
    params = _params("dipole", SE_E2_A if route == "dense-pte" else None)
    model = get_model(params).to(env.DEVICE).eval()
    tmpdir = tempfile.mkdtemp()
    try:
        if route == "dense-pte":
            path = os.path.join(tmpdir, "dipole.pte")
            deserialize_to_file(
                path, {"model": model.serialize()}, do_atomic_virial=True
            )
        else:
            path = os.path.join(tmpdir, "model.ckpt.pt")
            _save_checkpoint(model, params, path)
            if route == "graph-pte":
                pte = os.path.join(tmpdir, "frozen_model.pte")
                freeze(model=path, output=pte)
                path = pte
        dp = DeepEval(path)
        coord, cell, atype = _frames()
        nf, natoms = coord.shape[:2]
        _, force, virial, _, atom_virial = dp.eval_full(coord, cell, atype, atomic=True)
        assert force.shape == (nf, 3, natoms, 3)
        assert atom_virial.shape == (nf, 3, natoms, 9)

        step = 1e-5
        for atom, axis in ((0, 0), (1, 2), (2, 1)):
            shift = np.zeros_like(coord)
            shift[:, atom, axis] = step
            plus = dp.eval(coord + shift, cell, atype, atomic=False)
            minus = dp.eval(coord - shift, cell, atype, atomic=False)
            np.testing.assert_allclose(
                force[:, :, atom, axis],
                -(plus - minus) / (2.0 * step),
                atol=1e-6,
                rtol=1e-6,
            )
        np.testing.assert_allclose(
            atom_virial.sum(axis=2), virial, atol=1e-10, rtol=1e-10
        )
        ret = model(
            torch.tensor(
                coord, dtype=torch.float64, device=env.DEVICE, requires_grad=True
            ),
            torch.tensor(np.tile(atype, (nf, 1)), device=env.DEVICE),
            box=torch.tensor(cell, dtype=torch.float64, device=env.DEVICE),
        )
        np.testing.assert_allclose(
            virial,
            ret["virial"].detach().cpu().numpy().reshape(nf, 3, 9),
            atol=1e-10,
            rtol=1e-10,
        )
    finally:
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize(
    ("kind", "fitting"),
    [
        ("dos", {"numb_dos": 5}),
        ("property", {"task_dim": 3, "property_name": "foo"}),
        # A property that is itself named ``polar`` must not be renamed by
        # the ``polar`` request alias of DeepPolar.
        ("property", {"task_dim": 3, "property_name": "polar"}),
        # A property named ``energy`` is still a property: it exports the
        # common lower, not the energy model's.
        ("property", {"task_dim": 3, "property_name": "energy", "intensive": True}),
    ],
)
def test_dos_and_property_freeze_and_eval(kind: str, fitting: dict) -> None:
    """Graph-only DOS and property models export the common lower too."""
    params = _params(kind, **fitting)
    model = get_model(params).to(env.DEVICE).eval()
    tmpdir = tempfile.mkdtemp()
    try:
        ckpt = os.path.join(tmpdir, "model.ckpt.pt")
        _save_checkpoint(model, params, ckpt)
        pte = os.path.join(tmpdir, "frozen_model.pte")
        freeze(model=ckpt, output=pte)
        coord, cell, atype = _frames()
        nf = coord.shape[0]
        ret = model(
            torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
            torch.tensor(np.tile(atype, (nf, 1)), device=env.DEVICE),
            box=torch.tensor(cell, dtype=torch.float64, device=env.DEVICE),
        )
        name = "dos" if kind == "dos" else model.get_var_name()
        for path in (pte, ckpt):
            dp = DeepEval(path)
            assert dp.deep_eval.metadata["graph_output_keys"] == "internal"
            if kind == "property":
                assert isinstance(dp, DeepProperty)
            if kind == "dos":
                assert dp.get_numb_dos() == fitting["numb_dos"]
            total, atomic = dp.eval(coord, cell, atype, atomic=True)
            np.testing.assert_allclose(
                total,
                ret[name].detach().cpu().numpy().reshape(total.shape),
                atol=1e-10,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                atomic,
                ret[f"atom_{name}"].detach().cpu().numpy().reshape(atomic.shape),
                atol=1e-10,
                rtol=1e-10,
            )
        if kind == "dos":
            # An archive without model.json carries no dpmodel; the DOS width
            # then comes from the fitting output metadata.
            dp = DeepEval(pte)
            dp.deep_eval._dpmodel = None
            assert dp.get_numb_dos() == fitting["numb_dos"]
    finally:
        shutil.rmtree(tmpdir)


def test_dense_property_named_polar_keeps_its_key() -> None:
    """The ``polar`` alias applies only when the exact key is absent."""
    params = _params("property", SE_E2_A, task_dim=3, property_name="polar")
    model = get_model(params).to(env.DEVICE).eval()
    tmpdir = tempfile.mkdtemp()
    try:
        pte = os.path.join(tmpdir, "property.pte")
        deserialize_to_file(pte, {"model": model.serialize()})
        dp = DeepEval(pte)
        assert dp.deep_eval.metadata["lower_input_kind"] == "nlist"
        coord, cell, atype = _frames()
        nf = coord.shape[0]
        ret = model(
            torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
            torch.tensor(np.tile(atype, (nf, 1)), device=env.DEVICE),
            box=torch.tensor(cell, dtype=torch.float64, device=env.DEVICE),
        )
        total, atomic = dp.eval(coord, cell, atype, atomic=True)
        np.testing.assert_allclose(
            total,
            ret["polar"].detach().cpu().numpy().reshape(total.shape),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            atomic,
            ret["atom_polar"].detach().cpu().numpy().reshape(atomic.shape),
            atol=1e-10,
            rtol=1e-10,
        )
    finally:
        shutil.rmtree(tmpdir)


def test_property_named_energy_is_not_an_energy_model() -> None:
    """Energy-only paths tell a property named ``energy`` from an energy model."""
    descriptor = {**_params("dipole")["descriptor"], "output_rot_mat": False}
    energy = get_model(_params("ener", descriptor)).eval()
    prop = get_model(
        _params(
            "property",
            descriptor,
            task_dim=1,
            property_name="energy",
            intensive=True,
        )
    ).eval()
    assert model_has_energy_output(energy)
    assert not model_has_energy_output(prop)
    assert not model_uses_graph_lower(prop)
    # The fused operators reduce by sum, not the mean of an intensive property.
    assert _fused_energy_force_graph(prop, None, None, None, None) is None


@pytest.mark.parametrize("category", ["derv_r", "derv_c"])
def test_fold_output_without_atoms(category: str) -> None:
    """An empty frame folds to the API shape instead of failing to reshape."""
    width, cat = (
        (3, OutputVariableCategory.DERV_R)
        if category == "derv_r"
        else (9, OutputVariableCategory.DERV_C)
    )
    odef = OutputVariableDef("dipole_derv", [3, width], category=cat.value)
    shape = [2, 3, 0, width]
    out = _fold_output(np.empty((2, 0, 3, width)), odef, shape, 2, 0)
    assert out.shape == tuple(shape)
