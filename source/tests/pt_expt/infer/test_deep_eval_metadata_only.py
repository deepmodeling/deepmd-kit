# SPDX-License-Identifier: LGPL-3.0-or-later
"""Metadata-only loading tests for the pt_expt DeepEval.

Exercises the "no ``model.json``" fallback path added to
:class:`deepmd.pt_expt.infer.deep_eval.DeepEval`: pt_expt ``.pte`` /
``.pt2`` archives are loadable when they only ship ``extra/metadata.json``
(matching the contract the C++ ``DeepPotPTExpt`` reader enforces).

Strategy
--------
1. Build a tiny pt_expt SeA energy model and freeze it to a regular
   ``.pte`` (the fast path; ``.pt2`` AOTInductor compilation is too
   heavy for a routine unit test).
2. Read back that ``.pte`` and record the reference outputs.
3. Copy all archive entries except ``extra/model.json`` into a
   metadata-only variant.
4. Load the metadata-only archive via ``DeepPot`` and assert that the
   metadata-level accessors and the numeric ``eval`` result are
   **bitwise identical** to the reference.
5. Verify that the dpmodel-only hooks (``eval_descriptor``,
   ``eval_typeebd``, ``eval_fitting_last_layer``) raise
   :class:`NotImplementedError` in metadata-only mode, since they
   inherently need the deserialised dpmodel instance.
"""

from __future__ import (
    annotations,
)

import tempfile
import unittest
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)


def _strip_extra_model_json(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` dropping any ``extra/model.json`` entry.

    ``torch.export.save`` lays the archive out as
    ``<tmp_prefix>/extra/{model,metadata,model_def_script}.json``; the
    tmp prefix is chosen at save time so we match by suffix rather than
    by an exact path.  Every other entry (including the AOTI-compiled
    binaries for ``.pt2``) is copied through unmodified.
    """
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            if info.filename.endswith("extra/model.json"):
                continue
            zout.writestr(info, zin.read(info.filename))


class TestDeepEvalMetadataOnlyPte(unittest.TestCase):
    """End-to-end parity between full and metadata-only ``.pte`` archives."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

        # ----- build a tiny fp64 SeA energy model -----
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [6, 6]
        cls.type_map = ["O", "H"]
        cls.ntypes = len(cls.type_map)

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.ntypes,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            seed=7,
        )
        model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = model.to(torch.float64).eval()
        cls.model_data = {"model": cls.model.serialize()}

        # ----- freeze to .pte (full + metadata-only variants) -----
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_root = Path(cls._tmpdir.name)
        cls.full_path = tmp_root / "full.pte"
        cls.meta_only_path = tmp_root / "meta_only.pte"
        deserialize_to_file(str(cls.full_path), cls.model_data)
        _strip_extra_model_json(cls.full_path, cls.meta_only_path)

        cls.dp_full = DeepPot(str(cls.full_path))
        cls.dp_meta = DeepPot(str(cls.meta_only_path))

        # ----- a deterministic sample for numeric parity -----
        rng = np.random.default_rng(42)
        cls.natoms = 5
        cls.coord = rng.random((1, cls.natoms, 3), dtype=np.float64) * 6.0
        cls.cell = (np.eye(3, dtype=np.float64) * 12.0).reshape(1, 9)
        cls.atype = np.array([0, 1, 0, 1, 0], dtype=np.int32)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    # ----- archive layout sanity ------------------------------------

    def test_meta_only_archive_has_no_extra_model_json(self) -> None:
        with zipfile.ZipFile(self.meta_only_path, "r") as zf:
            names = zf.namelist()
        self.assertFalse(
            any(n.endswith("extra/model.json") for n in names),
            msg="extra/model.json must be absent in the metadata-only archive",
        )
        self.assertTrue(
            any(n.endswith("extra/metadata.json") for n in names),
            msg="extra/metadata.json is mandatory and must survive zip surgery",
        )

    # ----- metadata-level parity ------------------------------------

    def test_metadata_level_accessors_match(self) -> None:
        """All metadata-level queries agree between the two archives."""
        full = self.dp_full.deep_eval
        meta = self.dp_meta.deep_eval
        self.assertEqual(full.get_rcut(), meta.get_rcut())
        self.assertEqual(full.get_ntypes(), meta.get_ntypes())
        self.assertEqual(full.get_type_map(), meta.get_type_map())
        self.assertEqual(full.get_dim_fparam(), meta.get_dim_fparam())
        self.assertEqual(full.get_dim_aparam(), meta.get_dim_aparam())
        self.assertEqual(full.get_sel_type(), meta.get_sel_type())
        self.assertEqual(full.get_has_spin(), meta.get_has_spin())
        self.assertEqual(full.get_use_spin(), meta.get_use_spin())
        self.assertIs(full.model_type, meta.model_type)

    def test_internal_attributes_match(self) -> None:
        """The hot-path attributes hoisted in both init paths must agree."""
        full = self.dp_full.deep_eval
        meta = self.dp_meta.deep_eval
        self.assertEqual(list(full._sel), list(meta._sel))
        self.assertEqual(bool(full._mixed_types), bool(meta._mixed_types))
        self.assertEqual(full._rcut, meta._rcut)
        self.assertEqual(list(full._type_map), list(meta._type_map))

    def test_dpmodel_presence(self) -> None:
        """``_dpmodel`` is the single signal that separates the two modes."""
        self.assertIsNotNone(self.dp_full.deep_eval._dpmodel)
        self.assertIsNone(self.dp_meta.deep_eval._dpmodel)

    # ----- numeric parity -------------------------------------------

    def test_eval_numeric_parity(self) -> None:
        """``DeepPot.eval`` must be bitwise identical across the two archives."""
        e_full, f_full, v_full = self.dp_full.eval(
            self.coord, self.cell, self.atype, atomic=False
        )[:3]
        e_meta, f_meta, v_meta = self.dp_meta.eval(
            self.coord, self.cell, self.atype, atomic=False
        )[:3]
        np.testing.assert_array_equal(
            e_meta, e_full, err_msg="energy mismatch between full / meta-only"
        )
        np.testing.assert_array_equal(
            f_meta, f_full, err_msg="force mismatch between full / meta-only"
        )
        np.testing.assert_array_equal(
            v_meta, v_full, err_msg="virial mismatch between full / meta-only"
        )

    def test_eval_atomic_parity(self) -> None:
        """Atomic outputs (atom_energy / atom_virial) match as well."""
        full_out = self.dp_full.eval(self.coord, self.cell, self.atype, atomic=True)
        meta_out = self.dp_meta.eval(self.coord, self.cell, self.atype, atomic=True)
        self.assertEqual(len(full_out), len(meta_out))
        for ref, test in zip(full_out, meta_out, strict=True):
            np.testing.assert_array_equal(test, ref)

    # ----- dpmodel-only hooks must degrade to NotImplementedError ---

    def test_eval_descriptor_requires_dpmodel(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dp_meta.deep_eval.eval_descriptor(self.coord, self.cell, self.atype)

    def test_eval_fitting_last_layer_requires_dpmodel(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dp_meta.deep_eval.eval_fitting_last_layer(
                self.coord, self.cell, self.atype
            )

    def test_eval_typeebd_requires_dpmodel(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dp_meta.deep_eval.eval_typeebd()


class TestDeepEvalMetadataOnlyGuards(unittest.TestCase):
    """Error-path coverage that is independent of the .pte fixture."""

    def test_missing_metadata_json_is_rejected(self) -> None:
        """A ``.pte`` stripped of ``metadata.json`` must raise on load.

        Metadata is the minimum contract — unlike ``model.json`` it
        must always be present.
        """
        torch.manual_seed(0)
        ds = DescrptSeA(4.0, 0.5, [6, 6])
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=ds.mixed_types(), seed=1)
        model = EnergyModel(ds, ft, type_map=["a", "b"]).to(torch.float64).eval()
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full.pte"
            broken = Path(tmp) / "no_metadata.pte"
            deserialize_to_file(str(full), {"model": model.serialize()})
            with (
                zipfile.ZipFile(full, "r") as zin,
                zipfile.ZipFile(broken, "w") as zout,
            ):
                for info in zin.infolist():
                    if info.filename.endswith("extra/metadata.json"):
                        continue
                    zout.writestr(info, zin.read(info.filename))
            with self.assertRaises(ValueError):
                DeepPot(str(broken))


if __name__ == "__main__":
    unittest.main()
