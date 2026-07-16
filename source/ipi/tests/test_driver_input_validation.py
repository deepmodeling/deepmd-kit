# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)


class TestDPIPIInputValidation(unittest.TestCase):
    """Test driver input validation without requiring a socket or model backend."""

    def test_unknown_atom_name_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coord_file = tmp_path / "unknown_atom.xyz"
            config_file = tmp_path / "config.json"
            missing_model = tmp_path / "model-that-must-not-be-loaded.pb"

            coord_file.write_text(
                "1\nunknown atom name regression test\nXx 0.0 0.0 0.0\n",
                encoding="utf-8",
            )
            config_file.write_text(
                json.dumps(
                    {
                        "verbose": False,
                        "use_unix": True,
                        "port": 31415,
                        "host": "unused",
                        # The deliberately missing model proves that atom-name
                        # validation happens before model initialization. The
                        # old operator[] behavior reached model loading after
                        # silently converting Xx to type 0.
                        "graph_file": str(missing_model),
                        "coord_file": str(coord_file),
                        "atom_type": {"O": 0},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                ["dp_ipi", str(config_file)],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        self.assertEqual(
            result.stderr,
            "dp_ipi: Unknown atom name 'Xx' in coordinate file: "
            "no matching entry in atom_type.\n",
        )

    def test_missing_atom_type_is_rejected_before_model_loading(self) -> None:
        """Report a missing atom-type map instead of leaking a JSON exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coord_file = tmp_path / "coord.xyz"
            config_file = tmp_path / "config.json"
            missing_model = tmp_path / "model-that-must-not-be-loaded.pb"

            coord_file.write_text(
                "1\nmissing atom map\nO 0.0 0.0 0.0\n", encoding="utf-8"
            )
            config_file.write_text(
                json.dumps(
                    {
                        "verbose": False,
                        "use_unix": True,
                        "port": 31415,
                        "host": "unused",
                        "graph_file": str(missing_model),
                        "coord_file": str(coord_file),
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                ["dp_ipi", str(config_file)],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        self.assertIn("dp_ipi: invalid atom_type configuration:", result.stderr)
