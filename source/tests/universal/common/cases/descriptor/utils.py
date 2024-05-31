# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.utils import (
    PairExcludeMask,
)

from ..cases import (
    TestCaseSingleFrameWithNlist,
)


class DescriptorTestCase(TestCaseSingleFrameWithNlist):
    """Common test case for atomic model."""

    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
        }

    def test_forward(self):
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
