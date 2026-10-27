# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freezing and evaluating dipole and polarizability models.

DPA4C is graph-only, so its tensorial models freeze to the graph lower with
the internal output keys and are evaluated through ``DeepDipole`` /
``DeepPolar`` from the frozen archive and from a training checkpoint alike.
The dense polarizability case pins the ``polar`` request alias, which the
pt_expt inference previously answered with NaN.
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
from deepmd.pt_expt.entrypoints.main import (
    freeze,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
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


def _params(kind: str, descriptor: dict | None = None) -> dict:
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
    descriptor = {
        "type": "se_e2_a",
        "sel": [8, 16],
        "rcut_smth": 0.5,
        "rcut": 3.0,
        "neuron": [4, 8],
        "axis_neuron": 4,
        "precision": "float64",
        "seed": 17,
    }
    model = get_model(_params("polar", descriptor)).to(env.DEVICE).eval()
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
