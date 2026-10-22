# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import socket
import struct
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)


@unittest.skipIf(
    os.environ.get("ENABLE_PYTORCH", "1") != "1",
    reason="Skip test because PyTorch support is not enabled.",
)
class TestDPIPIProtocolValidation(unittest.TestCase):
    def test_posdata_atom_count_must_match_configuration(self) -> None:
        tests_path = Path(__file__).parent.parent.parent / "tests"
        model_file = tests_path / "infer" / "deeppot_sea.pth"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coord_file = tmp_path / "coord.xyz"
            config_file = tmp_path / "config.json"
            socket_name = f"dpipi_count_{os.getpid()}_{tmp_path.name}"
            socket_path = Path("/tmp") / f"ipi_{socket_name}"
            coord_file.write_text(
                "1\nPOSDATA atom-count regression test\nO 0.0 0.0 0.0\n",
                encoding="utf-8",
            )
            config_file.write_text(
                json.dumps(
                    {
                        "verbose": False,
                        "use_unix": True,
                        "port": 31415,
                        "host": socket_name,
                        "graph_file": str(model_file),
                        "coord_file": str(coord_file),
                        "atom_type": {"O": 0},
                    }
                ),
                encoding="utf-8",
            )

            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            process = None
            try:
                server.bind(str(socket_path))
                server.listen(1)
                server.settimeout(30)
                process = subprocess.Popen(
                    ["dp_ipi", str(config_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    if process.poll() is None:
                        process.kill()
                    stdout, stderr = process.communicate(timeout=5)
                    self.fail(
                        "dp_ipi did not connect within 30 seconds "
                        f"(return code {process.returncode}).\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                with connection:
                    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                    connection.sendall(b"POSDATA     ")
                    connection.sendall(struct.pack("=9d", *identity))
                    connection.sendall(struct.pack("=9d", *identity))
                    connection.sendall(struct.pack("=i", 2))
                stdout, stderr = process.communicate(timeout=60)
                self.assertEqual(
                    process.returncode,
                    1,
                    msg=f"stdout:\n{stdout}\nstderr:\n{stderr}",
                )
                self.assertIn(
                    "dp_ipi: POSDATA atom count 2 does not match configured "
                    "atom count 1.\n",
                    stderr,
                )
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                server.close()
                socket_path.unlink(missing_ok=True)
