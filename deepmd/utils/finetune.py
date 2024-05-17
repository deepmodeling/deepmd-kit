# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import logging
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    from deepmd.infer.deep_eval import (
        DeepEval,
    )
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )

log = logging.getLogger(__name__)


def change_energy_bias_lower(
    data: DeepmdDataSystem,
    dp: DeepEval,
    origin_type_map: list[str],
    full_type_map: list[str],
    bias_atom_e: np.ndarray,
    bias_adjust_mode="change-by-statistic",
    ntest=10,
):
    """Change the energy bias according to the input data and the pretrained model.

    Parameters
    ----------
    data : DeepmdDataSystem
        The training data.
    dp : str
        The DeepEval object.
    origin_type_map : list
        The original type_map in dataset, they are targets to change the energy bias.
    full_type_map : str
        The full type_map in pretrained model
    bias_atom_e : np.ndarray
        The old energy bias in the pretrained model.
    bias_adjust_mode : str
        The mode for changing energy bias : ['change-by-statistic', 'set-by-statistic']
        'change-by-statistic' : perform predictions on energies of target dataset,
                and do least sqaure on the errors to obtain the target shift as bias.
        'set-by-statistic' : directly use the statistic energy bias in the target dataset.
    ntest : int
        The number of test samples in a system to change the energy bias.
    """
    type_numbs = []
    energy_ground_truth = []
    energy_predict = []
    sorter = np.argsort(full_type_map)
    idx_type_map = sorter[
        np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
    ]
    mixed_type = data.mixed_type
    numb_type = len(full_type_map)
    for sys in data.data_systems:
        test_data = sys.get_test()
        nframes = test_data["box"].shape[0]
        numb_test = min(nframes, ntest)
        if mixed_type:
            atype = test_data["type"][:numb_test].reshape([numb_test, -1])
        else:
            atype = test_data["type"][0]
        assert np.array(
            [i in idx_type_map for i in list(set(atype.reshape(-1)))]
        ).all(), "Some types are not in 'type_map'!"
        energy_ground_truth.append(
            test_data["energy"][:numb_test].reshape([numb_test, 1])
        )
        if mixed_type:
            type_numbs.append(
                np.array(
                    [(atype == i).sum(axis=-1) for i in idx_type_map],
                    dtype=np.int32,
                ).T
            )
        else:
            type_numbs.append(
                np.tile(
                    np.bincount(atype, minlength=numb_type)[idx_type_map],
                    (numb_test, 1),
                )
            )
        if bias_adjust_mode == "change-by-statistic":
            coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
            if sys.pbc:
                box = test_data["box"][:numb_test]
            else:
                box = None
            if dp.get_dim_fparam() > 0:
                fparam = test_data["fparam"][:numb_test]
            else:
                fparam = None
            if dp.get_dim_aparam() > 0:
                aparam = test_data["aparam"][:numb_test]
            else:
                aparam = None
            ret = dp.eval(
                coord,
                box,
                atype,
                mixed_type=mixed_type,
                fparam=fparam,
                aparam=aparam,
            )
            energy_predict.append(ret[0].reshape([numb_test, 1]))
    type_numbs = np.concatenate(type_numbs)
    energy_ground_truth = np.concatenate(energy_ground_truth)
    old_bias = bias_atom_e[idx_type_map]
    if bias_adjust_mode == "change-by-statistic":
        energy_predict = np.concatenate(energy_predict)
        bias_diff = energy_ground_truth - energy_predict
        delta_bias = np.linalg.lstsq(type_numbs, bias_diff, rcond=None)[0]
        unbias_e = energy_predict + type_numbs @ delta_bias
        atom_numbs = type_numbs.sum(-1)
        rmse_ae = np.sqrt(
            np.mean(
                np.square((unbias_e.ravel() - energy_ground_truth.ravel()) / atom_numbs)
            )
        )
        bias_atom_e[idx_type_map] += delta_bias.reshape(-1)
        log.info(
            f"RMSE of atomic energy after linear regression is: {rmse_ae} eV/atom."
        )
    elif bias_adjust_mode == "set-by-statistic":
        statistic_bias = np.linalg.lstsq(type_numbs, energy_ground_truth, rcond=None)[0]
        bias_atom_e[idx_type_map] = statistic_bias.reshape(-1)
    else:
        raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)
    log.info(
        f"Change energy bias of {origin_type_map!s} from {old_bias!s} to {bias_atom_e[idx_type_map]!s}."
    )
    return bias_atom_e
