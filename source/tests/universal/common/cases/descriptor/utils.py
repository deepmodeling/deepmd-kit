# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.utils import (
    PairExcludeMask,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
)

from .....seed import (
    GLOBAL_SEED,
)
from ..cases import (
    TestCaseSingleFrameWithNlist,
)


class DescriptorTestCase(TestCaseSingleFrameWithNlist):
    """Common test case for descriptor."""

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_forward_consistency(self) -> None:
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
    ) -> None:
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

    def test_change_type_map(self) -> None:
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

    def test_change_type_map_extend_stat(self) -> None:
        if (
            not self.module.mixed_types()
            or getattr(self.module, "sel_no_mixed_types", None) is not None
        ):
            # skip if not mixed_types
            return
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
        for small_tm, large_tm in itertools.product(
            [
                full_type_map_test[:8],  # 8 elements, tebd default first dim
                ["H", "O"],  # slimmed types
            ],  # small_tm
            [
                full_type_map_test[:],  # 18 elements
                full_type_map_test[
                    :16
                ],  # 16 elements, double of tebd default first dim
                full_type_map_test[:8],  # 8 elements, tebd default first dim
            ],  # large_tm
        ):
            # use shuffled type_map
            rng.shuffle(small_tm)
            rng.shuffle(large_tm)
            small_tm_input = update_input_type_map(self.input_dict, small_tm)
            small_tm_module = self.module_class(**small_tm_input)

            large_tm_input = update_input_type_map(self.input_dict, large_tm)
            large_tm_module = self.module_class(**large_tm_input)

            # set random stat
            mean_small_tm, std_small_tm = small_tm_module.get_stat_mean_and_stddev()
            mean_large_tm, std_large_tm = large_tm_module.get_stat_mean_and_stddev()
            if "list" not in self.input_dict:
                mean_rand_small_tm, std_rand_small_tm = self.get_rand_stat(
                    rng, mean_small_tm, std_small_tm
                )
                mean_rand_large_tm, std_rand_large_tm = self.get_rand_stat(
                    rng, mean_large_tm, std_large_tm
                )
            else:
                # for hybrid
                mean_rand_small_tm, std_rand_small_tm = [], []
                mean_rand_large_tm, std_rand_large_tm = [], []
                for ii in range(len(mean_small_tm)):
                    mean_rand_item_small_tm, std_rand_item_small_tm = (
                        self.get_rand_stat(rng, mean_small_tm[ii], std_small_tm[ii])
                    )
                    mean_rand_small_tm.append(mean_rand_item_small_tm)
                    std_rand_small_tm.append(std_rand_item_small_tm)
                    mean_rand_item_large_tm, std_rand_item_large_tm = (
                        self.get_rand_stat(rng, mean_large_tm[ii], std_large_tm[ii])
                    )
                    mean_rand_large_tm.append(mean_rand_item_large_tm)
                    std_rand_large_tm.append(std_rand_item_large_tm)

            small_tm_module.set_stat_mean_and_stddev(
                mean_rand_small_tm, std_rand_small_tm
            )
            large_tm_module.set_stat_mean_and_stddev(
                mean_rand_large_tm, std_rand_large_tm
            )

            # extend the type map
            small_tm_module.change_type_map(
                large_tm, model_with_new_type_stat=large_tm_module
            )

            # check the stat
            mean_result, std_result = small_tm_module.get_stat_mean_and_stddev()
            type_index_map = get_index_between_two_maps(small_tm, large_tm)[0]

            if "list" not in self.input_dict:
                self.check_expect_stat(
                    type_index_map, mean_rand_small_tm, mean_rand_large_tm, mean_result
                )
                self.check_expect_stat(
                    type_index_map, std_rand_small_tm, std_rand_large_tm, std_result
                )
            else:
                # for hybrid
                for ii in range(len(mean_small_tm)):
                    self.check_expect_stat(
                        type_index_map,
                        mean_rand_small_tm[ii],
                        mean_rand_large_tm[ii],
                        mean_result[ii],
                    )
                    self.check_expect_stat(
                        type_index_map,
                        std_rand_small_tm[ii],
                        std_rand_large_tm[ii],
                        std_result[ii],
                    )

    def get_rand_stat(self, rng, mean, std):
        if not isinstance(mean, list):
            mean_rand, std_rand = self.get_rand_stat_item(rng, mean, std)
        else:
            mean_rand, std_rand = [], []
            for ii in range(len(mean)):
                mean_rand_item, std_rand_item = self.get_rand_stat_item(
                    rng, mean[ii], std[ii]
                )
                mean_rand.append(mean_rand_item)
                std_rand.append(std_rand_item)
        return mean_rand, std_rand

    def get_rand_stat_item(self, rng, mean, std):
        mean = self.convert_to_numpy(mean)
        std = self.convert_to_numpy(std)
        mean_rand = rng.random(size=mean.shape)
        std_rand = rng.random(size=std.shape)
        mean_rand = self.convert_from_numpy(mean_rand)
        std_rand = self.convert_from_numpy(std_rand)
        return mean_rand, std_rand

    def check_expect_stat(
        self, type_index_map, stat_small, stat_large, stat_result
    ) -> None:
        if not isinstance(stat_small, list):
            self.check_expect_stat_item(
                type_index_map, stat_small, stat_large, stat_result
            )
        else:
            for ii in range(len(stat_small)):
                self.check_expect_stat_item(
                    type_index_map, stat_small[ii], stat_large[ii], stat_result[ii]
                )

    def check_expect_stat_item(
        self, type_index_map, stat_small, stat_large, stat_result
    ) -> None:
        stat_small = self.convert_to_numpy(stat_small)
        stat_large = self.convert_to_numpy(stat_large)
        stat_result = self.convert_to_numpy(stat_result)
        full_stat = np.concatenate([stat_small, stat_large], axis=0)
        expected_stat = full_stat[type_index_map]
        np.testing.assert_allclose(expected_stat, stat_result)


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
