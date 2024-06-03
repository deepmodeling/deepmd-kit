# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Tuple,
)


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
