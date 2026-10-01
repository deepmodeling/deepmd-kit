# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pt_expt DOS-model inference via the DeepDOS interface.

A pt_expt DOS model configured with ``numb_dos`` > 0 must report that width
after export, otherwise ``DeepDOS.eval`` reshapes its output to width zero.
"""

import os
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.infer import (
    DeepEval,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    DOSFittingNet,
)
from deepmd.pt_expt.model import (
    DOSModel,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_deep_eval_metadata_only import (
    _strip_extra_model_json,
)


class TestDeepEvalDOS(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.numb_dos = 2
        ds = DescrptSeA(4.0, 0.5, [8, 6])
        ft = DOSFittingNet(
            2,
            ds.get_dim_out(),
            numb_dos=cls.numb_dos,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        cls.model = DOSModel(ds, ft, type_map=["foo", "bar"]).to(torch.float64)
        cls.model.eval()
        cls.dps = {}
        cls._tmpdir = tempfile.TemporaryDirectory()
        for suffix in (".pte", ".pt2"):
            full = os.path.join(cls._tmpdir.name, "full" + suffix)
            meta_only = os.path.join(cls._tmpdir.name, "meta_only" + suffix)
            deserialize_to_file(full, {"model": cls.model.serialize()})
            _strip_extra_model_json(Path(full), Path(meta_only))
            cls.dps[suffix] = DeepEval(full)
            cls.dps["meta_only" + suffix] = DeepEval(meta_only)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_model_get_numb_dos(self) -> None:
        self.assertEqual(self.model.get_numb_dos(), self.numb_dos)

    def test_get_numb_dos(self) -> None:
        for suffix, dp in self.dps.items():
            with self.subTest(suffix=suffix):
                # the metadata-only archives have no dpmodel to ask
                self.assertEqual(
                    dp.deep_eval._dpmodel is None, suffix.startswith("meta_only")
                )
                self.assertIs(dp.deep_eval.model_type, DeepDOS)
                self.assertEqual(dp.deep_eval.get_numb_dos(), self.numb_dos)

    def test_eval_shape(self) -> None:
        coords = np.arange(12, dtype=np.float64).reshape(1, 4, 3) * 0.3
        cells = (np.eye(3) * 10.0).reshape(1, 9)
        atypes = np.array([[0, 1, 0, 1]], dtype=np.int32)
        for suffix, dp in self.dps.items():
            with self.subTest(suffix=suffix):
                dos, atom_dos = dp.eval(coords, cells, atypes, atomic=True)
                self.assertEqual(dos.shape, (1, self.numb_dos))
                self.assertEqual(atom_dos.shape, (1, 4, self.numb_dos))

    def test_non_dos_archive_has_no_numb_dos(self) -> None:
        from deepmd.pt_expt.descriptor.se_e2_a import (
            DescrptSeA,
        )
        from deepmd.pt_expt.fitting import (
            EnergyFittingNet,
        )
        from deepmd.pt_expt.model import (
            EnergyModel,
        )

        ds = DescrptSeA(4.0, 0.5, [8, 6])
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=ds.mixed_types())
        model = EnergyModel(ds, ft, type_map=["foo", "bar"]).to(torch.float64).eval()
        path = os.path.join(self._tmpdir.name, "energy.pte")
        deserialize_to_file(path, {"model": model.serialize()})
        self.assertNotIn("numb_dos", DeepEval(path).deep_eval.metadata)


if __name__ == "__main__":
    unittest.main()
