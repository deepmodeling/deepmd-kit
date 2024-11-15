# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

log = logging.getLogger(__name__)


class FinetuneRuleItem:
    def __init__(
        self,
        p_type_map: list[str],
        type_map: list[str],
        model_branch: str = "Default",
        random_fitting: bool = False,
        resuming: bool = False,
    ) -> None:
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
        self.update_type = self.p_type_map != self.type_map

    def get_index_mapping(self):
        """Returns the mapping index of newly defined types to those in the pretrained model."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[0]

    def get_has_new_type(self):
        """Returns whether there are unseen types in the new type_map."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[1]

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
    old_map: list[str],
    new_map: list[str],
):
    """Returns the mapping index of types in new_map to those in the old_map.

    Parameters
    ----------
    old_map : list[str]
        The old list of atom type names.
    new_map : list[str]
        The new list of atom type names.

    Returns
    -------
    index_map: list[int]
        List contains `len(new_map)` indices, where `index_map[i]` is the index of `new_map[i]` in `old_map`.
        If `new_map[i]` is not in the `old_map`, the index will be `i - len(new_map)`.
    has_new_type: bool
        Whether there are unseen types in the new type_map.
        If True, some type related params in the model, such as statistics, need to be extended
        to have a length of `len(old_map) + len(new_map)` in the type related dimension.
        Then positive indices from 0 to `len(old_map) - 1` will select old params of types in `old_map`,
        while negative indices from `-len(new_map)` to -1 will select new params of types in `new_map`.
    """
    missing_type = [i for i in new_map if i not in old_map]
    has_new_type = False
    if len(missing_type) > 0:
        has_new_type = True
        log.warning(
            f"These types are not in the pretrained model and related params will be randomly initialized: {missing_type}."
        )
    index_map = []
    for ii, t in enumerate(new_map):
        index_map.append(old_map.index(t) if t in old_map else ii - len(new_map))
    return index_map, has_new_type


def map_atom_exclude_types(
    atom_exclude_types: list[int],
    remap_index: list[int],
):
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    atom_exclude_types : list[int]
        Exclude the atomic contribution of the given types.
    remap_index : list[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_atom_exclude_types: list[int]
        Remapped atom_exclude_types that only keeps the types in the new type list.

    """
    remapped_atom_exclude_types = [
        remap_index.index(i) for i in atom_exclude_types if i in remap_index
    ]
    return remapped_atom_exclude_types


def map_pair_exclude_types(
    pair_exclude_types: list[tuple[int, int]],
    remap_index: list[int],
):
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    pair_exclude_types : list[tuple[int, int]]
        Exclude the pair of atoms of the given types from computing the output
        of the atomic model.
    remap_index : list[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_pair_exclude_typess: list[tuple[int, int]]
        Remapped pair_exclude_types that only keeps the types in the new type list.

    """
    remapped_pair_exclude_typess = [
        (remap_index.index(pair[0]), remap_index.index(pair[1]))
        for pair in pair_exclude_types
        if pair[0] in remap_index and pair[1] in remap_index
    ]
    return remapped_pair_exclude_typess
