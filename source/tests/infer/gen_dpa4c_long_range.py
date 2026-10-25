#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the DPA4C long-range .pt2 test models and reference values.

A latent-charge long-range module sums over every pair of atoms of a complete
non-periodic system, and over the images of the cell of a periodic one when
it serves periodic systems, so the C++ interface hands the model what the sum
needs beyond the neighbor list: the graph lower of an uncompressed model
receives the pair list, or the mesh of the cell, that the interface builds
from the positions of the atoms, and the compact canonical lower of a
compressed model reads the positions and the cell themselves. One archive
covers each lower and each kind of module:

- ``deeppot_dpa4c_long_range.pt2`` and
  ``deeppot_dpa4c_long_range_periodic.pt2``: uncompressed, fp64, graph lower,
  which compiles on the CPU;
- ``deeppot_dpa4c_long_range_compressed.pt2`` and
  ``deeppot_dpa4c_long_range_periodic_compressed.pt2``: compressed, fp32,
  compact canonical lower, whose operators are CUDA only; they are generated
  for the CUDA variant of the C++ suite (``DP_VARIANT=cuda``).

Every model conditions the descriptor on the charge state and constrains the
latent charges to the total charge that state names, so a state other than
the default changes the long-range energy as well as the descriptor. The
references are those of the Python interface on each archive. For a
non-periodic module they are taken without a cell, at the default state and
at ``[1, 2]`` (sections ``default`` and ``explicit``); for a periodic module,
in the cell :data:`BOX` and without a cell, at the neutral default state
(sections ``periodic`` and ``cluster``). The six atoms form two groups about
9 Å apart, beyond the cutoff of the descriptor, which only the long-range
module couples; in the cell, the first group meets its images across the
boundary.
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
# A triclinic cell, row-major with the lattice vectors as rows.
BOX = np.array(
    [[13.0, 0.0, 0.0], [0.6, 12.5, 0.0], [-0.4, 0.8, 13.5]], dtype=np.float64
).reshape(1, 9)


def _periodic(config: dict) -> dict:
    """The configuration with a long-range module that serves periodic systems."""
    config = copy.deepcopy(config)
    config["long_range"]["periodic"] = True
    return config


def _model_dict(config: dict) -> dict:
    """Build the dpmodel of a configuration and jitter it in place of its zeros."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(config))
    return jitter_zero_arrays(model.serialize(), np.random.default_rng(_JITTER_SEED))


def _check_metadata(pt2_path: str, lower: str, periodic: bool = False) -> None:
    """Assert the archive declares its lower and the long-range input it needs."""
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
    assert md["lower_input_kind"] == lower, md["lower_input_kind"]
    expected = ["all_pairs", "pme_mesh"] if periodic else ["all_pairs"]
    assert md["topologies"] == expected, md["topologies"]
    if lower == "graph":
        assert md["dim_chg_spin"] == 2
    else:
        # The compressed descriptor folds the charge state; the archive
        # carries the rebuild of every constant that holds it: the four
        # charge-state artifacts of the descriptor and the total charge of
        # the constraint.
        assert md["dim_chg_spin"] == 0
        assert len(md["charge_state_constants"]) == 5
        # The fused long-range operator serves the charge network of one
        # channel and reads the source CSR for the per-atom virial alone.
        assert md["canonical_requires_source_csr"] is False


def _write_ref(pt2_path: str, ref_path: str, periodic: bool = False) -> None:
    """Evaluate the reference cases on the archive and write the C++ sidecar."""
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    if periodic:
        cases = (("periodic", BOX, None), ("cluster", None, None))
    else:
        cases = (("default", None, None), ("explicit", None, _OTHER_CHG_SPIN))
    sections = {}
    energies = {}
    for name, box, charge_spin in cases:
        condition = {} if charge_spin is None else {"charge_spin": charge_spin}
        e, f, v, ae, av = dp.eval(_COORDS, box, _ATYPES, atomic=True, **condition)
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(f))
        energies[name] = float(e[0, 0])
        print(f"// {name} total energy: {e[0, 0]:.18e}")  # noqa: T201
        sections[name] = {
            "expected_e": ae[0, :, 0],
            "expected_f": f[0],
            "expected_v": av[0],
        }
    # The C++ cases compare the two cases, which needs a real difference.
    first, second = (name for name, _, _ in cases)
    assert abs(energies[second] - energies[first]) > 1e-3, (
        "the two cases must differ, or a call that ignored the cell or the "
        "requested state would still match"
    )
    write_expected_ref(
        ref_path,
        sections=sections,
        source_script="source/tests/infer/gen_dpa4c_long_range.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201


def _export_compressed(pt2_path: str, config: dict) -> None:
    """Compress the fp32 model and export it for the CUDA device."""
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = BaseModel.deserialize(_model_dict(config))
    model.atomic_model.enable_compression(0.5)
    data = {
        "model": model.serialize(),
        "model_def_script": config,
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
    for suffix, periodic in (("", False), ("_periodic", True)):
        config = _periodic(CONFIG) if periodic else CONFIG
        stem = os.path.join(base_dir, f"deeppot_dpa4c_long_range{suffix}")
        data = {
            "model": _model_dict(config),
            "model_def_script": config,
            "backend": "dpmodel",
            "software": "deepmd-kit",
            "version": "3.0.0",
        }
        print(f"Exporting to {stem}.pt2 (lower_kind='graph') ...")  # noqa: T201
        deserialize_to_file(
            f"{stem}.pt2", data, do_atomic_virial=True, lower_kind="graph"
        )
        _check_metadata(f"{stem}.pt2", "graph", periodic)
        _write_ref(f"{stem}.pt2", f"{stem}.expected", periodic)

        if os.environ.get("DP_VARIANT") == "cuda":
            compressed = f"{stem}_compressed"
            compressed_config = (
                _periodic(COMPRESSED_CONFIG) if periodic else COMPRESSED_CONFIG
            )
            _export_compressed(f"{compressed}.pt2", compressed_config)
            _check_metadata(f"{compressed}.pt2", "dpa4c_canonical", periodic)
            _write_ref(f"{compressed}.pt2", f"{compressed}.expected", periodic)

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
