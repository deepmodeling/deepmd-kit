# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.utils import (
    AtomExcludeMask,
)

from .....seed import (
    GLOBAL_SEED,
)
from ..cases import (
    TestCaseSingleFrameWithNlist,
)


class FittingTestCase(TestCaseSingleFrameWithNlist):
    """Common test case for descriptor."""

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "dim_descrpt": self.dim_descrpt,
            "mixed_types": self.mixed_types,
            "type_map": ["O", "H"],
        }

    def test_forward_consistency(self) -> None:
        serialize_dict = self.module.serialize()
        # set random bias
        rng = np.random.default_rng()
        serialize_dict["@variables"]["bias_atom_e"] = rng.random(
            size=serialize_dict["@variables"]["bias_atom_e"].shape
        )
        self.module = self.module.deserialize(serialize_dict)
        ret = []
        for module in self.modules_to_test:
            module = self.forward_wrapper(module)
            ret.append(
                module(
                    self.mock_descriptor,
                    self.atype_ext[:, : self.nloc],
                    gr=self.mock_gr,
                )
            )
        for kk in ret[0]:
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
        atype_device = self.atype_ext[:, : self.nloc]
        serialize_dict = self.module.serialize()
        # set random bias
        rng = np.random.default_rng()
        serialize_dict["@variables"]["bias_atom_e"] = rng.random(
            size=serialize_dict["@variables"]["bias_atom_e"].shape
        )
        self.module = self.module.deserialize(serialize_dict)
        ff = self.forward_wrapper(self.module)
        var_name = self.module.var_name
        if var_name == "polar":
            var_name = "polarizability"

        for em in [[0], [1]]:
            ex_pair = AtomExcludeMask(self.nt, em)
            atom_mask = ex_pair.build_type_exclude_mask(atype_device)
            # exclude neighbors in the output
            rd = ff(
                self.mock_descriptor,
                self.atype_ext[:, : self.nloc],
                gr=self.mock_gr,
            )[var_name]
            for _ in range(len(rd.shape) - len(atom_mask.shape)):
                atom_mask = atom_mask[..., None]
            rd = rd * atom_mask

            # normal nlist but use exclude_types params
            serialize_dict_em = deepcopy(serialize_dict)
            serialize_dict_em.update({"exclude_types": em})
            ff_ex = self.forward_wrapper(self.module.deserialize(serialize_dict_em))
            rd_ex = ff_ex(
                self.mock_descriptor,
                self.atype_ext[:, : self.nloc],
                gr=self.mock_gr,
            )[var_name]
            np.testing.assert_allclose(rd, rd_ex)

    def test_change_type_map(self) -> None:
        if not self.module.mixed_types:
            # skip if not mixed_types
            return
        atype_device = self.atype_ext[:, : self.nloc]
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
        for old_tm, new_tm, em in itertools.product(
            [
                full_type_map_test[:8],  # 8 elements
                ["H", "O"],  # slimmed types
            ],  # large_type_map
            [
                full_type_map_test[:8],  # 8 elements
                ["H", "O"],  # slimmed types
            ],  # small_type_map
            [
                [],
                [0],
                [1],
            ],  # exclude_types for original_type_map
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
            old_tm_input = deepcopy(self.input_dict)
            old_tm_input["type_map"] = old_tm
            old_tm_input["ntypes"] = len(old_tm)
            old_tm_input["exclude_types"] = old_tm_em
            old_tm_module = self.module_class(**old_tm_input)
            serialize_dict = old_tm_module.serialize()
            # set random bias
            serialize_dict["@variables"]["bias_atom_e"] = rng.random(
                size=serialize_dict["@variables"]["bias_atom_e"].shape
            )
            old_tm_module = old_tm_module.deserialize(serialize_dict)
            var_name = old_tm_module.var_name
            if var_name == "polar":
                var_name = "polarizability"
            old_tm_ff = self.forward_wrapper(old_tm_module)
            rd_old_tm = old_tm_ff(
                self.mock_descriptor,
                old_tm_index[atype_device],
                gr=self.mock_gr,
            )[var_name]
            old_tm_module.change_type_map(new_tm)
            new_tm_ff = self.forward_wrapper(old_tm_module)
            rd_new_tm = new_tm_ff(
                self.mock_descriptor,
                new_tm_index[atype_device],
                gr=self.mock_gr,
            )[var_name]
            np.testing.assert_allclose(rd_old_tm, rd_new_tm)


def remap_exclude_types(exclude_types, ori_tm, new_tm):
    assert set(ori_tm).issubset(set(new_tm))
    updated_em = [new_tm.index(i) for i in ori_tm if ori_tm.index(i) in exclude_types]
    return updated_em
