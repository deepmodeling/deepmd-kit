# SPDX-License-Identifier: LGPL-3.0-or-later
"""Subprocess-only export and inference for the official small DPA4C fixture.

Keep PyTorch and the LAMMPS plugin in separate processes, as in the existing
``test_lammps_model_devi_pt2`` oracle. No generated archives are checked in.
"""

import argparse
import json
import sys
import zipfile
from pathlib import (
    Path,
)

import numpy as np


def export_models(directory: Path) -> None:
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    # Same standalone-helper import as source/tests/infer/gen_dpa4c_spin.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from dpa4_fixtures import (
        activate_dpa4c_condition_head,
        conditioned_dpa4c_config,
        jitter_zero_arrays,
    )

    directory.mkdir(parents=True, exist_ok=True)
    for name, seed, state in (
        ("m0", 19, (0.0, 1.0)),
        ("m1", 29, (0.0, 1.0)),
        ("other_state", 19, (2.0, 3.0)),
    ):
        config = conditioned_dpa4c_config(state)
        config["fitting_net"]["seed"] = seed
        model = get_model(config).to("cpu").eval()
        # Existing DPA4 fixtures activate zero residual projections so force
        # and virial comparisons cannot pass on edge-independent models.
        model = (
            BaseModel.deserialize(
                jitter_zero_arrays(model.serialize(), np.random.default_rng(seed))
            )
            .to("cuda")
            .eval()
        )
        activate_dpa4c_condition_head(model.get_descriptor())
        model.get_descriptor().enable_compression(min_nbor_dist=0.5)
        path = directory / f"{name}.pt2"
        deserialize_to_file(
            str(path),
            {"model": model.serialize()},
            lower_kind="dpa4c_canonical",
            do_atomic_virial=True,
        )
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("model/extra/metadata.json"))
        assert metadata["lower_input_kind"] == "dpa4c_canonical"
        assert metadata["graph_edge_dtype"] == "float32"
        assert metadata["do_atomic_virial"] is True
        assert metadata["default_chg_spin"] == list(state)
        assert "charge_state_constants" in metadata


def evaluate(request: Path, result: Path) -> None:
    from deepmd.infer import (
        DeepPot,
    )

    data = json.loads(request.read_text())
    coords = np.asarray(data["coords"])
    cells = np.tile(np.eye(3).reshape(1, 9) * 13.0, (len(coords), 1))
    atom_types = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
    outputs = []
    for path in data["models"]:
        model = DeepPot(path)
        energy, force, virial, _, atom_virial = model.eval(
            coords, cells, atom_types, atomic=True
        )
        outputs.append(
            {
                "energy": energy.reshape(-1).tolist(),
                "force": force.reshape(-1, 6, 3).tolist(),
                "virial": virial.reshape(-1, 9).tolist(),
                "atom_virial": atom_virial.reshape(-1, 6, 9).tolist(),
            }
        )
    result.write_text(json.dumps(outputs))


def check_c_api(library: Path, model: Path, state: list, atomic_virial: bool) -> None:
    """Exercise the public caller-owned buffer contract in isolation."""
    import ctypes

    api = ctypes.CDLL(str(library))
    api.DP_NewDeepPot.argtypes = [ctypes.c_char_p]
    api.DP_NewDeepPot.restype = ctypes.c_void_p
    api.DP_DeleteDeepPot.argtypes = [ctypes.c_void_p]
    api.DP_DeleteDeepPot.restype = None
    api.DP_DeepPotCheckOK.argtypes = [ctypes.c_void_p]
    api.DP_DeepPotCheckOK.restype = ctypes.c_void_p
    api.DP_DeleteChar.argtypes = [ctypes.c_void_p]
    api.DP_DeleteChar.restype = None
    api.DP_DeepPotGetDefaultChgSpin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    api.DP_DeepPotGetDefaultChgSpin.restype = ctypes.c_int
    api.DP_DeepPotHasAtomicVirial.argtypes = [ctypes.c_void_p]
    api.DP_DeepPotHasAtomicVirial.restype = ctypes.c_bool

    def error(handle):
        message = api.DP_DeepPotCheckOK(handle)
        try:
            return ctypes.string_at(message).decode() if message else ""
        finally:
            if message:
                api.DP_DeleteChar(message)

    handle = api.DP_NewDeepPot(str(model).encode())
    assert handle
    try:
        assert not error(handle)
        assert api.DP_DeepPotHasAtomicVirial(handle) is atomic_virial
        assert not error(handle)
        getter = api.DP_DeepPotGetDefaultChgSpin
        size = getter(handle, None, 0)
        assert size == len(state)
        assert not error(handle)
        sentinel = -12345.0
        buffer = (ctypes.c_double * (size + 1))(*([sentinel] * (size + 1)))
        assert getter(handle, buffer, size) == size
        assert list(buffer) == [*state, sentinel]
        assert not error(handle)
        # A failed query must not partially overwrite caller-owned storage.
        for pointer, capacity in (
            (buffer, size - 1),
            (buffer, -1),
            (None, 1),
            (None, -1),
        ):
            for index in range(size + 1):
                buffer[index] = sentinel
            before = list(buffer)
            assert getter(handle, pointer, capacity) == -1
            assert list(buffer) == before
            assert error(handle), "C API must expose the buffer error through CheckOK"
    finally:
        api.DP_DeleteDeepPot(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("export").add_argument("directory", type=Path)
    oracle = sub.add_parser("oracle")
    oracle.add_argument("request", type=Path)
    oracle.add_argument("result", type=Path)
    capi = sub.add_parser("capi")
    capi.add_argument("library", type=Path)
    capi.add_argument("model", type=Path)
    capi.add_argument("state", type=json.loads)
    capi.add_argument("atomic_virial", type=json.loads)
    args = parser.parse_args()
    if args.action == "export":
        export_models(args.directory)
    elif args.action == "oracle":
        evaluate(args.request, args.result)
    else:
        check_c_api(args.library, args.model, args.state, args.atomic_virial)
