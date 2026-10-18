#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the DPA4C long-range .pt2 test models and reference values.

A latent-charge long-range module sums over every pair of atoms of a complete
non-periodic system, so the C++ interface hands the model what the pair sum
needs beyond the neighbor list: the graph lower of an uncompressed model
receives the pair list the interface builds from the positions of the atoms,
and the compact canonical lower of a compressed model reads the positions
themselves. One archive covers each lower:

- ``deeppot_dpa4c_long_range.pt2``: uncompressed, fp64, graph lower, which
  compiles on the CPU;
- ``deeppot_dpa4c_long_range_compressed.pt2``: compressed, fp32, compact
  canonical lower, whose operators are CUDA only; it is generated for the
  CUDA variant of the C++ suite (``DP_VARIANT=cuda``).

Both models condition the descriptor on the charge state and constrain the
latent charges to the total charge that state names, so a state other than
the default changes the long-range energy as well as the descriptor. The
references are those of the Python interface on each archive, without a cell,
at the default state and at ``[1, 2]``; each ``.expected`` sidecar holds the
sections ``default`` and ``explicit``. The six atoms form two groups about
9 Å apart, beyond the cutoff of the descriptor, which only the long-range
module couples.
"""

import copy
import json
import os
import sys
import zipfile

import numpy as np

# The gen scripts run standalone, outside pytest's package machinery, so the
# relative imports the test suite uses do not apply here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpa4_fixtures import (
    jitter_zero_arrays,
)
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

LONG_RANGE = {
    "type": "latent_charge",
    "channels": 1,
    "charge_net": {"neuron": [8, 8], "seed": 9},
    "kernel": {"type": "les", "alpha": 0.8},
    "charge_constraint": {"source": "charge_spin"},
}

# The uncompressed model is fp64, so that its references hold to the
# precision of the double C++ interface.
CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 4,
        "precision": "float64",
        "seed": 7,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
    },
    "fitting_net": {"neuron": [8, 8], "precision": "float64", "seed": 7},
    "long_range": LONG_RANGE,
}

# Compression takes an fp32 descriptor and, for the compact canonical lower,
# a fitting that the fused fitting operator evaluates.
COMPRESSED_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 8,
        "precision": "float32",
        "seed": 7,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
    },
    "fitting_net": {
        "neuron": [32, 32],
        "activation_function": "silu",
        "resnet_dt": False,
        "precision": "float32",
        "seed": 7,
    },
    "long_range": LONG_RANGE,
}

# Jitters the zero-initialized arrays, among them the charge-state head of the
# descriptor, so that the charge state reaches the descriptor as well as the
# constraint.
_JITTER_SEED = 20261018

_ATYPES = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
_COORDS = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ],
    dtype=np.float64,
).reshape(1, 6, 3)
_OTHER_CHG_SPIN = np.array([[1.0, 2.0]], dtype=np.float64)


def _model_dict(config: dict) -> dict:
    """Build the dpmodel of a configuration and jitter it in place of its zeros."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(config))
    return jitter_zero_arrays(model.serialize(), np.random.default_rng(_JITTER_SEED))


def _check_metadata(pt2_path: str, lower: str) -> None:
    """Assert the archive declares its lower and the long-range input it needs."""
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
    assert md["lower_input_kind"] == lower, md["lower_input_kind"]
    if lower == "graph":
        assert md["topologies"] == ["all_pairs"]
        assert md["dim_chg_spin"] == 2
    else:
        # The compressed descriptor folds the charge state; the archive
        # carries the rebuild of every constant that holds it.
        assert md["dim_chg_spin"] == 0
        assert len(md["charge_state_constants"]) == 5


def _write_ref(pt2_path: str, ref_path: str) -> None:
    """Evaluate the reference cases on the archive and write the C++ sidecar."""
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    sections = {}
    energies = {}
    for name, charge_spin in (("default", None), ("explicit", _OTHER_CHG_SPIN)):
        condition = {} if charge_spin is None else {"charge_spin": charge_spin}
        e, f, v, ae, av = dp.eval(_COORDS, None, _ATYPES, atomic=True, **condition)
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(f))
        energies[name] = float(e[0, 0])
        print(f"// {name} total energy: {e[0, 0]:.18e}")  # noqa: T201
        sections[name] = {
            "expected_e": ae[0, :, 0],
            "expected_f": f[0],
            "expected_v": av[0],
        }
    # The C++ cases compare the two states, which needs a real difference.
    assert abs(energies["explicit"] - energies["default"]) > 1e-3, (
        "the two charge states must differ, or a call that ignored the "
        "requested state would still match"
    )
    write_expected_ref(
        ref_path,
        sections=sections,
        source_script="source/tests/infer/gen_dpa4c_long_range.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201


def _export_compressed(pt2_path: str) -> None:
    """Compress the fp32 model and export it for the CUDA device."""
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = BaseModel.deserialize(_model_dict(COMPRESSED_CONFIG))
    model.atomic_model.enable_compression(0.5)
    data = {
        "model": model.serialize(),
        "model_def_script": COMPRESSED_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    deserialize_to_file(pt2_path, data, do_atomic_virial=True, lower_kind="auto")


def main() -> None:
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4c_long_range.pt2")
    data = {
        "model": _model_dict(CONFIG),
        "model_def_script": CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    print(f"Exporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    deserialize_to_file(pt2_path, data, do_atomic_virial=True, lower_kind="graph")
    _check_metadata(pt2_path, "graph")
    _write_ref(pt2_path, os.path.join(base_dir, "deeppot_dpa4c_long_range.expected"))

    if os.environ.get("DP_VARIANT") == "cuda":
        compressed_path = os.path.join(
            base_dir, "deeppot_dpa4c_long_range_compressed.pt2"
        )
        _export_compressed(compressed_path)
        _check_metadata(compressed_path, "dpa4c_canonical")
        _write_ref(
            compressed_path,
            os.path.join(base_dir, "deeppot_dpa4c_long_range_compressed.expected"),
        )

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
