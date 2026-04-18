# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from source.lmp.tests.model_convert import ensure_converted_pb


class TestEnsureConvertedPb(unittest.TestCase):
    def test_skips_up_to_date_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "model.pbtxt"
            output = tmp_path / "model.pb"
            source.write_text("source")
            output.write_text("converted")
            source.touch()
            output.touch()

            with patch(
                "source.lmp.tests.model_convert.sp.check_output"
            ) as convert_mock:
                ensure_converted_pb(source, output)

            convert_mock.assert_not_called()

    def test_rebuilds_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "model.pbtxt"
            output = tmp_path / "model.pb"
            output.write_text("old")
            output.touch()
            source.write_text("new")
            source.touch()

            def fake_convert(cmd: list[str]) -> bytes:
                Path(cmd[-1]).write_text("converted")
                return b""

            with patch(
                "source.lmp.tests.model_convert.sp.check_output",
                side_effect=fake_convert,
            ) as convert_mock:
                ensure_converted_pb(source, output)

            convert_mock.assert_called_once()
            self.assertEqual(output.read_text(), "converted")


if __name__ == "__main__":
    unittest.main()
