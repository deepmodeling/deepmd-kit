# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
)

import numpy as np

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def change_energy_bias_lower(
    data: DeepmdDataSystem,
    dp: DeepEval,
    origin_type_map: List[str],
    full_type_map: List[str],
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


class FinetuneRuleItem:
    def __init__(
        self,
        p_type_map: List[str],
        type_map: List[str],
        model_branch: str = "Default",
        random_fitting: bool = False,
        resuming: bool = False,
    ):
        """
        The rules for fine-tuning the model from pretrained model.

        Parameters
        ----------
        p_type_map
            The type map from the pretrained model.
        type_map
            The newly defined type map.
        model_branch
            From which branch the model should be fine-tuned.
        random_fitting
            If true, the fitting net will be randomly initialized instead of inherit from the pretrained model.
        resuming
            If true, the model will just resume from model_branch without fine-tuning.
        """
        self.p_type_map = p_type_map
        self.type_map = type_map
        self.model_branch = model_branch
        self.random_fitting = random_fitting
        self.resuming = resuming
        missing_type = [i for i in type_map if i not in p_type_map]
        assert not missing_type, (
            "Only support for smaller type map when finetuning or resuming! "
            f"While these types are not in the pretrained model: {missing_type}."
        )
        self.update_type = not (self.p_type_map == self.type_map)

    def get_index_mapping(self):
        """Returns the mapping index of newly defined types to those in the pretrained model."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)

    def get_model_branch(self):
        """Returns the chosen model branch."""
        return self.model_branch

    def get_random_fitting(self):
        """Returns whether to use random fitting."""
        return self.random_fitting

    def get_resuming(self):
        """Returns whether to only do resuming."""
        return self.resuming

    def get_update_type(self):
        """Returns whether to update the type related params when loading from pretrained model with redundant types."""
        return self.update_type

    def get_pretrained_tmap(self):
        """Returns the type map in the pretrained model."""
        return self.p_type_map

    def get_finetune_tmap(self):
        """Returns the type map in the fine-tuned model."""
        return self.type_map


def get_index_between_two_maps(
    large_map: List[str],
    small_map: List[str],
):
    """Returns the mapping index of types in small_map to those in the large_map.

    Parameters
    ----------
    large_map : List[str]
        The larger list of atom type names.
    small_map : List[str]
        The smaller list of atom type names.

    Returns
    -------
    slimmed_index: List[int]
        The indices in the larger type list that correspond to the types in the smaller type list.
    """
    missing_type = [i for i in small_map if i not in large_map]
    assert not missing_type, (
        "Only support for smaller type map when doing type slimming!"
        f"While these types are not in the pretrained model: {missing_type}."
    )
    return [large_map.index(i) for i in small_map]


def map_atom_exclude_types(
    atom_exclude_types: List[int],
    slim_index: List[int],
):
    """Return the slimmed atom_exclude_types according to slim_index.

    Parameters
    ----------
    atom_exclude_types : List[int]
        Exclude the atomic contribution of the given types.
    slim_index : List[int]
        The indices in the larger type list that correspond to the types in the smaller type list.

    Returns
    -------
    slimmed_atom_exclude_types: List[int]
        Slimmed atom_exclude_types that only keeps the types in the smaller type list.

    """
    atom_exclude_types = [
        slim_index.index(i) for i in atom_exclude_types if i in slim_index
    ]
    return atom_exclude_types


def map_pair_exclude_types(
    pair_exclude_types: List[Tuple[int, int]],
    slim_index: List[int],
):
    """Return the slimmed atom_exclude_types according to slim_index.

    Parameters
    ----------
    pair_exclude_types : List[Tuple[int, int]]
        Exclude the pair of atoms of the given types from computing the output
        of the atomic model.
    slim_index : List[int]
        The indices in the larger type list that correspond to the types in the smaller type list.

    Returns
    -------
    slimmed_pair_exclude_typess: List[Tuple[int, int]]
        Slimmed pair_exclude_types that only keeps the types in the smaller type list.

    """
    slimmed_pair_exclude_types = [
        (slim_index.index(pair[0]), slim_index.index(pair[1]))
        for pair in pair_exclude_types
        if pair[0] in slim_index and pair[1] in slim_index
    ]
    return slimmed_pair_exclude_types
