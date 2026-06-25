# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
from deepmd.dpmodel.utils.nlist import extend_input_and_build_neighbor_list
from deepmd.pt_expt.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1PT
from deepmd.pt_expt.utils import env


def _small_extended():
    rng = np.random.default_rng(7)
    coord = rng.normal(size=(1, 5, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, 4.0, [30], mixed_types=True, box=None
    )
    return ext_coord, ext_atype, nlist, mapping


class TestDpa1GraphSerialize(unittest.TestCase):
    def _make(self):
        return DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[30], ntypes=2, attn_layer=0)

    def test_roundtrip_forward_identical(self) -> None:
        """dpa1(attn_layer=0) serialize->deserialize gives an identical forward."""
        dd = self._make()
        dd2 = DescrptDPA1.deserialize(dd.serialize())
        ext_coord, ext_atype, nlist, mapping = _small_extended()
        np.testing.assert_allclose(
            dd2.call(ext_coord, ext_atype, nlist, mapping)[0],
            dd.call(ext_coord, ext_atype, nlist, mapping)[0],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_dpmodel_to_pt_expt_interop(self) -> None:
        """Dpmodel dpa1 serialize -> pt_expt deserialize -> identical descriptor
        (cross-backend checkpoint interop, graph-routed attn_layer=0 forward).
        """
        dd = self._make()
        dd_pt = DescrptDPA1PT.deserialize(dd.serialize()).to(env.DEVICE)
        ext_coord, ext_atype, nlist, mapping = _small_extended()
        ref = dd.call(ext_coord, ext_atype, nlist, mapping)[0]
        got = dd_pt(
            torch.from_numpy(ext_coord).to(env.DEVICE),
            torch.from_numpy(ext_atype).to(env.DEVICE),
            torch.from_numpy(nlist).to(env.DEVICE),
            torch.from_numpy(mapping).to(env.DEVICE),
        )[0]
        np.testing.assert_allclose(
            got.detach().cpu().numpy(),
            ref,
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
