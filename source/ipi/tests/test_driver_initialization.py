# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate framing of the standard i-PI initialization message."""

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
class TestDPIPIInitialization(unittest.TestCase):
    def exchange(self, payload: bytes, length: int | None = None) -> None:
        model = Path(__file__).resolve().parents[2] / "tests/infer/deeppot_sea.pth"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinate = root / "coord.xyz"
            coordinate.write_text("1\nINIT protocol test\nO 0 0 0\n")
            name = f"dpipi_init_{os.getpid()}_{root.name}"
            address = Path("/tmp") / f"ipi_{name}"
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "verbose": False,
                        "use_unix": True,
                        "port": 31415,
                        "host": name,
                        "graph_file": str(model),
                        "coord_file": str(coordinate),
                        "atom_type": {"O": 0},
                    }
                )
            )
            process = None
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(address))
                server.listen(1)
                server.settimeout(30)
                process = subprocess.Popen(
                    ["dp_ipi", str(config)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(10)
                    size = len(payload) if length is None else length
                    connection.sendall(b"INIT        " + struct.pack("=ii", 0, size))
                    connection.sendall(payload)
                    if size >= 0:
                        connection.sendall(b"STATUS      ")
                        response = b""
                        while len(response) < 12:
                            part = connection.recv(12 - len(response))
                            self.assertTrue(part, "driver closed before STATUS reply")
                            response += part
                        self.assertEqual(response, b"READY       ")
                        connection.sendall(b"EXIT        ")
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(
                    process.returncode,
                    0 if size >= 0 else 1,
                    msg=f"stdout:\n{stdout}\nstderr:\n{stderr}",
                )
                if size < 0:
                    self.assertIn(
                        "dp_ipi: INIT payload length must be nonnegative.", stderr
                    )
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                server.close()
                address.unlink(missing_ok=True)

    def test_empty_initialization_preserves_next_header(self) -> None:
        self.exchange(b"")

    def test_initialization_consumes_replica_id_and_payload(self) -> None:
        self.exchange(b'{"temperature": 300}')

    def test_initialization_accepts_payload_larger_than_read_buffer(self) -> None:
        self.exchange(b"x" * 4097)

    def test_negative_initialization_length_is_rejected(self) -> None:
        self.exchange(b"", length=-1)
