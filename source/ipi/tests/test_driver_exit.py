# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import socket
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
class TestDPIPIExit(unittest.TestCase):
    def test_exit_header_stops_driver_cleanly(self) -> None:
        tests_path = Path(__file__).parent.parent.parent / "tests"
        model_file = tests_path / "infer" / "deeppot_sea.pth"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coord_file = tmp_path / "coord.xyz"
            config_file = tmp_path / "config.json"
            socket_name = f"dpipi_exit_{os.getpid()}_{tmp_path.name}"
            socket_path = Path("/tmp") / f"ipi_{socket_name}"
            coord_file.write_text(
                "1\nEXIT protocol regression test\nO 0.0 0.0 0.0\n",
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
                connection, _ = server.accept()
                with connection:
                    connection.sendall(b"EXIT        ")
                stdout, stderr = process.communicate(timeout=60)
                self.assertEqual(
                    process.returncode,
                    0,
                    msg=f"stdout:\n{stdout}\nstderr:\n{stderr}",
                )
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                server.close()
                socket_path.unlink(missing_ok=True)
