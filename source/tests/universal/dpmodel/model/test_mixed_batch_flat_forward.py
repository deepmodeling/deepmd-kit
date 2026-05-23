# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the dpmodel mixed-nloc flat forward interface."""

from itertools import (
    pairwise,
)

import numpy as np
import pytest

from deepmd.dpmodel.descriptor import (
    DescrptDPA3,
)
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.dpmodel.fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model import (
    EnergyModel,
)

from ....seed import (
    GLOBAL_SEED,
)


def _build_model(with_params: bool = False) -> EnergyModel:
    repflow = RepFlowArgs(
        n_dim=8,
        e_dim=4,
        a_dim=4,
        nlayers=1,
        e_rcut=1.5,
        e_rcut_smth=0.2,
        e_sel=4,
        a_rcut=1.0,
        a_rcut_smth=0.1,
        a_sel=3,
        axis_neuron=2,
        update_angle=True,
        update_style="res_residual",
        update_residual_init="const",
        a_compress_rate=0,
        n_multi_edge_message=1,
        smooth_edge_update=True,
    )
    descriptor = DescrptDPA3(
        2,
        repflow=repflow,
        precision="float64",
        seed=GLOBAL_SEED,
        type_map=["O", "H"],
    )
    fitting_kwargs = {
        "ntypes": 2,
        "dim_descrpt": descriptor.get_dim_out(),
        "neuron": [8],
        "mixed_types": descriptor.mixed_types(),
        "type_map": ["O", "H"],
        "precision": "float64",
        "seed": GLOBAL_SEED,
    }
    if with_params:
        fitting_kwargs.update(
            {
                "numb_fparam": 2,
                "numb_aparam": 2,
                "dim_case_embd": 2,
                "default_fparam": [1.0, 1.0],
            }
        )
    fitting = EnergyFittingNet(**fitting_kwargs)
    return EnergyModel(descriptor, fitting, type_map=["O", "H"])


def _mixed_batch() -> tuple[np.ndarray, ...]:
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    atype = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    batch = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    ptr = np.array([0, 2, 5], dtype=np.int64)
    box = np.tile(np.eye(3).reshape(1, 9), (2, 1)).astype(np.float64) * 10.0
    return coord, atype, batch, ptr, box


def _regular_outputs(
    model: EnergyModel,
    coord: np.ndarray,
    atype: np.ndarray,
    ptr: np.ndarray,
    box: np.ndarray,
    fparam: np.ndarray | None = None,
    aparam: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    regular = []
    for frame_idx, (start, end) in enumerate(pairwise(ptr)):
        nloc = end - start
        frame_aparam = (
            aparam[start:end].reshape(1, nloc, *aparam.shape[1:])
            if aparam is not None
            else None
        )
        regular.append(
            model.call(
                coord[start:end].reshape(1, nloc * 3),
                atype[start:end].reshape(1, nloc),
                box=box[frame_idx : frame_idx + 1],
                fparam=fparam[frame_idx : frame_idx + 1]
                if fparam is not None
                else None,
                aparam=frame_aparam,
            )
        )
    return regular


def test_dpmodel_dpa3_flat_call_matches_regular_per_frame_outputs() -> None:
    model = _build_model(with_params=True)
    coord, atype, batch, ptr, box = _mixed_batch()
    fparam = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    aparam = np.arange(10, dtype=np.float64).reshape(5, 2) / 10.0

    flat = model.call(
        coord,
        atype,
        box=box,
        fparam=fparam,
        aparam=aparam,
        batch=batch,
        ptr=ptr,
        extended_atype=atype,
        extended_batch=batch,
        extended_image=np.zeros_like(coord, dtype=np.int64),
        extended_ptr=ptr,
        mapping=np.arange(coord.shape[0], dtype=np.int64),
        central_ext_index=np.arange(coord.shape[0], dtype=np.int64),
        nlist=np.full((coord.shape[0], 0), -1, dtype=np.int64),
        nlist_ext=np.full((coord.shape[0], 0), -1, dtype=np.int64),
        a_nlist=np.full((coord.shape[0], 0), -1, dtype=np.int64),
        a_nlist_ext=np.full((coord.shape[0], 0), -1, dtype=np.int64),
        nlist_mask=np.zeros((coord.shape[0], 0), dtype=bool),
        a_nlist_mask=np.zeros((coord.shape[0], 0), dtype=bool),
    )
    regular = _regular_outputs(model, coord, atype, ptr, box, fparam, aparam)

    np.testing.assert_allclose(
        flat["energy"],
        np.concatenate([item["energy"] for item in regular], axis=0),
    )
    np.testing.assert_allclose(
        flat["atom_energy"],
        np.concatenate([item["atom_energy"].reshape(-1, 1) for item in regular]),
    )
    np.testing.assert_array_equal(
        flat["mask"],
        np.concatenate([item["mask"].reshape(-1) for item in regular]),
    )


def test_dpmodel_flat_call_requires_batch_and_ptr() -> None:
    model = _build_model()
    coord, atype, batch, _, box = _mixed_batch()

    with pytest.raises(ValueError, match="Both batch and ptr"):
        model.call(coord, atype, box=box, batch=batch)


def test_dpmodel_flat_call_validates_ptr() -> None:
    model = _build_model()
    coord, atype, batch, _, box = _mixed_batch()

    with pytest.raises(ValueError, match="end at the number of atoms"):
        model.call(
            coord,
            atype,
            box=box,
            batch=batch,
            ptr=np.array([0, 2, 4], dtype=np.int64),
        )


def test_dpmodel_flat_call_rejects_hessian() -> None:
    model = _build_model()
    model.enable_hessian()
    coord, atype, batch, ptr, box = _mixed_batch()

    with pytest.raises(NotImplementedError, match="Hessian"):
        model.call(coord, atype, box=box, batch=batch, ptr=ptr)
