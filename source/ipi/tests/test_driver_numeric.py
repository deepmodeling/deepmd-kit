# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare i-PI wire results with the direct inference API."""

import json
import os
import socket
import struct
import subprocess
from pathlib import (
    Path,
)

import numpy as np
import pytest
from scipy.constants import (
    physical_constants,
)

# Independent CODATA values; the driver uses older rounded conversion constants.
BOHR = physical_constants["Bohr radius"][0] / 1e-10
HARTREE = physical_constants["Hartree energy in eV"][0]


def receive(connection: socket.socket, size: int) -> bytes:
    """Read exactly one protocol field, including fragmented socket replies."""
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise EOFError(f"Driver disconnected after {len(data)} of {size} bytes")
        data.extend(part)
    return bytes(data)


@pytest.mark.skipif(
    os.environ.get("ENABLE_PYTORCH", "1") != "1",
    reason="PyTorch support is not enabled.",
)
@pytest.mark.parametrize("use_unix", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_socket_energy_force_virial_parity(tmp_path, monkeypatch, use_unix, permuted):
    """Preserve units and ordering across repeated triclinic POSDATA requests."""
    # Keep each direct-inference and socket-driver process within this test's
    # small CPU budget, independent of the host's visible core count.
    monkeypatch.setenv("DP_INTRA_OP_PARALLELISM_THREADS", "1")
    monkeypatch.setenv("DP_INTER_OP_PARALLELISM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "6")

    from deepmd.infer import (
        DeepPot,
    )

    model = Path(__file__).resolve().parents[2] / "tests/infer/deeppot_sea.pth"
    potential = DeepPot(str(model))
    coordinates = np.array(
        [
            [12.83, 2.56, 2.18],
            [12.09, 2.87, 2.74],
            [0.25, 3.32, 1.68],
            [3.36, 3.00, 1.81],
            [3.51, 2.51, 2.60],
            [4.27, 3.22, 1.56],
        ]
    )
    types = np.array([0, 1, 1, 0, 1, 1])
    if permuted:
        order = [2, 3, 5, 0, 4, 1]
        coordinates, types = coordinates[order], types[order]
    box = np.array([[13.0, 0.0, 0.0], [1.0, 12.0, 0.0], [0.3, 0.7, 11.0]])
    names = np.array(["O", "H"])[types]
    coordinate_file = tmp_path / "coord.xyz"
    coordinate_file.write_text(
        "6\nSocket numerical parity\n" + "".join(f"{name} 0 0 0\n" for name in names)
    )
    server = socket.socket(socket.AF_UNIX if use_unix else socket.AF_INET)
    name = f"dpipi_parity_{os.getpid()}_{tmp_path.name}"
    address = Path("/tmp") / f"ipi_{name}"
    process = None
    try:
        server.bind(str(address) if use_unix else ("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(30)
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "verbose": False,
                    "use_unix": use_unix,
                    "port": 0 if use_unix else server.getsockname()[1],
                    "host": name if use_unix else "127.0.0.1",
                    "graph_file": str(model),
                    "coord_file": str(coordinate_file),
                    "atom_type": {"O": 0, "H": 1},
                }
            )
        )
        with (tmp_path / "driver.log").open("w+") as log:
            process = subprocess.Popen(
                ["dp_ipi", str(config)], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                connection, _ = server.accept()
            except TimeoutError:
                log.seek(0)
                pytest.fail("Driver did not connect:\n" + log.read())
            with connection:
                connection.settimeout(30)
                shifted = coordinates.copy()
                shifted[0] += box[0] - box[1]
                # A second cell must replace the previous cell, not reuse it.
                strain = np.array(
                    [[1.01, 0.0, 0.0], [0.02, 0.99, 0.0], [0.0, -0.01, 1.02]]
                )
                frames = [
                    (coordinates, box),
                    (shifted, box),
                    (coordinates @ strain, box @ strain),
                ]
                for positions, cell in frames:
                    energy, force, virial = potential.eval(
                        positions.reshape(1, -1), cell.reshape(1, -1), types
                    )
                    connection.sendall(b"STATUS      ")
                    assert receive(connection, 12) == b"READY       "
                    h = cell.T / BOHR
                    connection.sendall(
                        b"POSDATA     "
                        + np.asarray(h, dtype="=f8").tobytes()
                        + np.asarray(np.linalg.inv(h), dtype="=f8").tobytes()
                        + struct.pack("=i", len(types))
                        + np.asarray(positions / BOHR, dtype="=f8").tobytes()
                    )
                    connection.sendall(b"STATUS      ")
                    assert receive(connection, 12) == b"HAVEDATA    "
                    connection.sendall(b"GETFORCE    ")
                    assert receive(connection, 12) == b"FORCEREADY  "
                    actual_energy = struct.unpack("=d", receive(connection, 8))[0]
                    count = struct.unpack("=i", receive(connection, 4))[0]
                    assert count == len(types)
                    actual_force = np.frombuffer(
                        receive(connection, count * 3 * 8), dtype="=f8"
                    ).reshape(count, 3)
                    actual_virial = (
                        np.frombuffer(receive(connection, 9 * 8), dtype="=f8")
                        .reshape(3, 3)
                        .T
                    )
                    extra_size = struct.unpack("=i", receive(connection, 4))[0]
                    assert extra_size >= 0
                    receive(connection, extra_size)
                    # Accommodate the driver's documented legacy rounding, not
                    # arbitrary force, energy or cell-layout discrepancies.
                    np.testing.assert_allclose(
                        actual_energy, energy.item() / HARTREE, rtol=2e-7, atol=1e-9
                    )
                    np.testing.assert_allclose(
                        actual_force,
                        force.reshape(-1, 3) * BOHR / HARTREE,
                        rtol=2e-7,
                        atol=1e-9,
                    )
                    np.testing.assert_allclose(
                        actual_virial,
                        virial.reshape(3, 3) / HARTREE,
                        rtol=2e-7,
                        atol=1e-9,
                    )
                connection.sendall(b"EXIT        ")
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.seek(0)
                pytest.fail("Driver did not exit after EXIT:\n" + log.read())
            log.seek(0)
            assert process.returncode == 0, log.read()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        server.close()
        if use_unix:
            address.unlink(missing_ok=True)
