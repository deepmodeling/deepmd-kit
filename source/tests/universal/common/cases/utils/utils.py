# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from copy import (
    deepcopy,
)

import numpy as np

from .....seed import (
    GLOBAL_SEED,
)
from ..cases import (
    TestCaseSingleFrameWithNlist,
)


class TypeEmbdTestCase(TestCaseSingleFrameWithNlist):
    """Common test case for type embedding network."""

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "neuron": [8],
            "type_map": ["O", "H"],
            "use_econf_tebd": False,
        }
        self.module_input = {}

    def test_change_type_map(self) -> None:
        atype_ext_device = self.atype_ext
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
        for old_tm, new_tm, neuron, act, econf in itertools.product(
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
            [[8], [8, 16, 32]],  # neuron
            ["Linear", "tanh"],  # activation_function
            [False, True],  # use_econf_tebd
        ):
            do_resnet = neuron[0] in [
                len(old_tm),
                len(old_tm) * 2,
                len(new_tm),
                len(new_tm) * 2,
            ]
            if do_resnet and act != "Linear":
                # `activation_function` must be "Linear" when performing type changing on resnet structure
                continue
            # use shuffled type_map
            rng.shuffle(old_tm)
            rng.shuffle(new_tm)
            old_tm_index = np.array(
                [old_tm.index(i) for i in original_type_map], dtype=np.int32
            )
            new_tm_index = np.array(
                [new_tm.index(i) for i in original_type_map], dtype=np.int32
            )
            old_tm_input = deepcopy(self.input_dict)
            old_tm_input["type_map"] = old_tm
            old_tm_input["ntypes"] = len(old_tm)
            old_tm_input["neuron"] = neuron
            old_tm_input["activation_function"] = act
            old_tm_input["use_econf_tebd"] = econf
            old_tm_module = self.module_class(**old_tm_input)
            old_tm_dd = self.forward_wrapper(old_tm_module)

            rd_old_tm = old_tm_dd(**self.module_input)[old_tm_index[atype_ext_device]]
            old_tm_module.change_type_map(new_tm)
            new_tm_dd = self.forward_wrapper(old_tm_module)
            rd_new_tm = new_tm_dd(**self.module_input)[new_tm_index[atype_ext_device]]
            np.testing.assert_allclose(rd_old_tm, rd_new_tm)
