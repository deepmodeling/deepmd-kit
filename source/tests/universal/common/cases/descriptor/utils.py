# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.utils import (
    PairExcludeMask,
)

from .....seed import (
    GLOBAL_SEED,
)
from ..cases import (
    TestCaseSingleFrameWithNlist,
)


class DescriptorTestCase(TestCaseSingleFrameWithNlist):
    """Common test case for descriptor."""

    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "type_map": ["O", "H"],
        }

    def test_forward_consistency(self):
        ret = []
        for module in self.modules_to_test:
            module = self.forward_wrapper(module)
            ret.append(
                module(
                    self.coord_ext,
                    self.atype_ext,
                    self.nlist,
                    mapping=self.mapping,
                )
            )
        for kk, vv in enumerate(ret[0]):
            subret = []
            for rr in ret:
                if rr is not None:
                    subret.append(rr[kk])
            if len(subret):
                for ii, rr in enumerate(subret[1:]):
                    if subret[0] is None:
                        assert rr is None
                    else:
                        np.testing.assert_allclose(
                            subret[0],
                            rr,
                            err_msg=f"compare {kk} output between 0 and {ii}",
                        )

    def test_exclude_types(
        self,
    ):
        coord_ext_device = self.coord_ext
        atype_ext_device = self.atype_ext
        nlist_device = self.nlist
        mapping_device = self.mapping
        dd = self.forward_wrapper(self.module)
        # only equal when set_davg_zero is True
        serialize_dict = self.module.serialize()

        for em in [[[0, 1]], [[1, 1]]]:
            ex_pair = PairExcludeMask(self.nt, em)
            pair_mask = ex_pair.build_type_exclude_mask(nlist_device, atype_ext_device)
            # exclude neighbors in the nlist
            nlist_exclude = np.where(pair_mask == 1, nlist_device, -1)
            rd_ex, _, _, _, sw_ex = dd(
                coord_ext_device,
                atype_ext_device,
                nlist_exclude,
                mapping=mapping_device,
            )

            # normal nlist but use exclude_types params
            serialize_dict_em = deepcopy(serialize_dict)
            if "list" not in serialize_dict_em:
                serialize_dict_em.update({"exclude_types": em})
            else:
                # for hybrid
                for sd in serialize_dict_em["list"]:
                    sd.update({"exclude_types": em})
            dd0 = self.forward_wrapper(self.module.deserialize(serialize_dict_em))
            rd0, _, _, _, sw0 = dd0(
                coord_ext_device,
                atype_ext_device,
                nlist_device,
                mapping=mapping_device,
            )
            np.testing.assert_allclose(rd0, rd_ex)

    def test_change_type_map(self):
        if (
            not self.module.mixed_types()
            or getattr(self.module, "sel_no_mixed_types", None) is not None
        ):
            # skip if not mixed_types
            return
        coord_ext_device = self.coord_ext
        atype_ext_device = self.atype_ext
        nlist_device = self.nlist
        mapping_device = self.mapping
        # type_map for data and exclude_types
        original_type_map = ["O", "H"]
        full_type_map_test = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
        ]  # 18 elements
        rng = np.random.default_rng(GLOBAL_SEED)
        for old_tm, new_tm, em, econf in itertools.product(
            [
                full_type_map_test[:],  # 18 elements
                full_type_map_test[
                    :16
                ],  # 16 elements, double of tebd default first dim
                full_type_map_test[:8],  # 8 elements, tebd default first dim
                ["H", "O"],  # slimmed types
            ],  # old_type_map
            [
                full_type_map_test[:],  # 18 elements
                full_type_map_test[
                    :16
                ],  # 16 elements, double of tebd default first dim
                full_type_map_test[:8],  # 8 elements, tebd default first dim
                ["H", "O"],  # slimmed types
            ],  # new_type_map
            [[], [[0, 1]], [[1, 1]]],  # exclude_types for original_type_map
            [False, True],  # use_econf_tebd
        ):
            # use shuffled type_map
            rng.shuffle(old_tm)
            rng.shuffle(new_tm)
            old_tm_index = np.array(
                [old_tm.index(i) for i in original_type_map], dtype=np.int32
            )
            new_tm_index = np.array(
                [new_tm.index(i) for i in original_type_map], dtype=np.int32
            )
            old_tm_em = remap_exclude_types(em, original_type_map, old_tm)
            old_tm_input = update_input_type_map(self.input_dict, old_tm)
            old_tm_input = update_input_use_econf_tebd(old_tm_input, econf)
            old_tm_input = update_input_exclude_types(old_tm_input, old_tm_em)
            old_tm_module = self.module_class(**old_tm_input)
            old_tm_dd = self.forward_wrapper(old_tm_module)
            rd_old_tm, _, _, _, sw_old_tm = old_tm_dd(
                coord_ext_device,
                old_tm_index[atype_ext_device],
                nlist_device,
                mapping=mapping_device,
            )
            old_tm_module.change_type_map(new_tm)
            new_tm_dd = self.forward_wrapper(old_tm_module)
            rd_new_tm, _, _, _, sw_new_tm = new_tm_dd(
                coord_ext_device,
                new_tm_index[atype_ext_device],
                nlist_device,
                mapping=mapping_device,
            )
            np.testing.assert_allclose(rd_old_tm, rd_new_tm)


def update_input_type_map(input_dict, type_map):
    updated_input_dict = deepcopy(input_dict)
    if "list" not in updated_input_dict:
        updated_input_dict["type_map"] = type_map
        updated_input_dict["ntypes"] = len(type_map)
    else:
        # for hybrid
        for sd in updated_input_dict["list"]:
            sd["type_map"] = type_map
            sd["ntypes"] = len(type_map)
    return updated_input_dict


def update_input_use_econf_tebd(input_dict, use_econf_tebd):
    updated_input_dict = deepcopy(input_dict)
    if "list" not in updated_input_dict:
        updated_input_dict["use_econf_tebd"] = use_econf_tebd
    else:
        # for hybrid
        for sd in updated_input_dict["list"]:
            sd["use_econf_tebd"] = use_econf_tebd
    return updated_input_dict


def update_input_exclude_types(input_dict, exclude_types):
    updated_input_dict = deepcopy(input_dict)
    if "list" not in updated_input_dict:
        updated_input_dict["exclude_types"] = exclude_types
    else:
        # for hybrid
        for sd in updated_input_dict["list"]:
            sd["exclude_types"] = exclude_types
    return updated_input_dict


def remap_exclude_types(exclude_types, ori_tm, new_tm):
    assert set(ori_tm).issubset(set(new_tm))
    new_ori_index = [new_tm.index(i) for i in ori_tm]
    updated_em = [
        (new_ori_index[pair[0]], new_ori_index[pair[1]]) for pair in exclude_types
    ]
    return updated_em
