# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from copy import (
    deepcopy,
)
from random import (
    shuffle,
)

import numpy as np

from deepmd.dpmodel.utils import (
    PairExcludeMask,
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

    def test_slim_type_map(self):
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
        for ltm, stm, em, econf in itertools.product(
            [
                deepcopy(full_type_map_test),  # 18 elements
                deepcopy(
                    full_type_map_test[:16]
                ),  # 16 elements, double of tebd default first dim
                deepcopy(full_type_map_test[:8]),  # 8 elements, tebd default first dim
            ],  # large_type_map
            [
                deepcopy(
                    full_type_map_test[:16]
                ),  # 16 elements, double of tebd default first dim
                deepcopy(full_type_map_test[:8]),  # 8 elements, tebd default first dim
                ["H", "O"],  # slimmed types
            ],  # small_type_map
            [[], [[0, 1]], [[1, 1]]],  # exclude_types for original_type_map
            [False, True],  # use_econf_tebd
        ):
            if len(ltm) < len(stm):
                continue
            # use shuffled type_map
            shuffle(ltm)
            shuffle(stm)
            ltm_index = np.array(
                [ltm.index(i) for i in original_type_map], dtype=np.int32
            )
            stm_index = np.array(
                [stm.index(i) for i in original_type_map], dtype=np.int32
            )
            ltm_em = remap_exclude_types(em, original_type_map, ltm)
            ltm_input = update_input_type_map(self.input_dict, ltm)
            ltm_input = update_input_use_econf_tebd(ltm_input, econf)
            ltm_input = update_input_exclude_types(ltm_input, ltm_em)
            ltm_module = self.module_class(**ltm_input)
            ltm_dd = self.forward_wrapper(ltm_module)
            rd_ltm, _, _, _, sw_ltm = ltm_dd(
                coord_ext_device,
                ltm_index[atype_ext_device],
                nlist_device,
                mapping=mapping_device,
            )
            ltm_module.slim_type_map(stm)
            stm_dd = self.forward_wrapper(ltm_module)
            rd_stm, _, _, _, sw_stm = stm_dd(
                coord_ext_device,
                stm_index[atype_ext_device],
                nlist_device,
                mapping=mapping_device,
            )
            np.testing.assert_allclose(rd_ltm, rd_stm)


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
