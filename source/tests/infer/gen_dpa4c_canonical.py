#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the compressed DPA4C .pt2 test models of the compact canonical lower.

A compressed DPA4C frozen with the compact canonical lower is served by two
entry points: the host ones, which build the compact canonical graph from
coordinates, and the device-resident ``compute_canonical_graph_gpu``, which
LAMMPS + Kokkos calls with the graph and the outputs already on the model
device. Two archives reach it: an energy model, which ``DeepPotPTExpt``
serves, and a native-spin model, which ``NativeSpinPTExpt`` serves. The C++
cases built on them (``dpa4c_canonical_test_common.h``) evaluate the graph
of the host entry points through the device-resident one, including the
replay of repeated evaluations of a graph without halo nodes and a charge
state applied at run time, and check the host entry points against the
references written here.

Both descriptors are charge-conditioned, so each archive carries the rebuild
of its frozen tables that ``set_charge_spin`` runs. The compressed operators
are fp32 CUDA kernels, so the fixtures are generated only where the suite
has a GPU.
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

from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

# Small fp32 DPA4C, the configurations the compressed operators specialize on:
# eight channels and degree two, the profile of the smallest released grade.
CANONICAL_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "basis_type": "bessel",
        "n_radial": 8,
        "precision": "float32",
        "seed": 11,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
    },
    "fitting_net": {
        "type": "ener",
        "neuron": [16, 16],
        "resnet_dt": False,
        "activation_function": "silu",
        "precision": "float32",
        "seed": 11,
    },
}

# The same descriptor with a native spin branch on its first type, which
# carries a magnetic moment, while the second carries none.
SPIN_CONFIG = copy.deepcopy(CANONICAL_CONFIG)
SPIN_CONFIG["type_map"] = ["Ni", "O"]
SPIN_CONFIG["spin"] = {"use_spin": [True, False], "scheme": "native"}

# Jittered simple cubic lattice of 27 atoms with a spacing of 3 Å in a 9 Å
# box: every atom has its six nearest neighbours within the 4 Å cutoff, and
# the box exceeds twice the cutoff, so the single-rank graph has one edge per
# neighbouring pair and no halo node. The lattice sits half a spacing inside
# the box, so that every jittered atom lies in it, as the host neighbour list
# of the C++ suite expects.
_SPACING = 3.0
_CELLS = 3
_NATOMS = _CELLS**3
_GRID = np.stack(
    np.meshgrid(*(np.arange(_CELLS),) * 3, indexing="ij"), axis=-1
).reshape(_NATOMS, 3)
_RNG = np.random.default_rng(20261016)
_COORDS = ((_GRID + 0.5) * _SPACING + _RNG.uniform(-0.2, 0.2, (_NATOMS, 3))).reshape(
    1, _NATOMS, 3
)
_ATYPES = (np.arange(_NATOMS) % 2).astype(np.int32)
_CELL = (np.eye(3) * (_SPACING * _CELLS)).reshape(1, 9)
# The moments of the spin model, on every atom: the gating of the model, not
# the input, keeps the second type non-magnetic.
_SPINS = _RNG.uniform(-0.5, 0.5, (1, _NATOMS, 3))

# The state the archive is frozen with, and one a caller applies at run time.
_DEFAULT_CHG_SPIN = np.array([[0.0, 1.0]])
_OTHER_CHG_SPIN = np.array([[1.0, 2.0]])

# Seed of the charge-state head of the energy model, whose output projection
# starts at zero.
_HEAD_SEED = 5

# Seed and scale of the zero-initialized parameters of the spin model, which
# include its spin gate: a closed gate would make every magnetic force vanish.
_SPIN_REDRAW_SEED = 7
_SPIN_REDRAW_SCALE = 0.3


def _compress(model):
    """Compress the descriptor of ``model`` as a deployment does."""
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=0.01)
    return model


def _build_model():
    """Build, charge-activate and compress the energy model."""
    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    model = get_model(copy.deepcopy(CANONICAL_CONFIG)).eval()
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    # The projection of the charge-state head is zero initialized, which
    # would make the state unobservable and every comparison between states
    # vacuous.
    head = descriptor.charge_spin_embedding.network.layers[-1]
    generator = torch.Generator(device=head.w.device).manual_seed(_HEAD_SEED)
    with torch.no_grad():
        head.w.copy_(
            0.5
            * torch.randn(
                head.w.shape,
                dtype=head.w.dtype,
                device=head.w.device,
                generator=generator,
            )
        )
    return _compress(model)


def _build_spin_model():
    """Build the native-spin model, open its zero parameters and compress it."""
    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    model = get_model(copy.deepcopy(SPIN_CONFIG)).eval()
    generator = torch.Generator(device="cpu").manual_seed(_SPIN_REDRAW_SEED)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() == 0 or bool((parameter != 0).any()):
                continue
            values = _SPIN_REDRAW_SCALE * torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            )
            parameter.copy_(values.to(dtype=parameter.dtype, device=parameter.device))
    return _compress(model)


def _check_metadata(pt2_path: str, spin: bool) -> None:
    """Assert the archive is the compact canonical lower with a charge state."""
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        names = zf.namelist()
    print("\n// metadata:")  # noqa: T201
    print(  # noqa: T201
        json.dumps(
            {
                k: md.get(k)
                for k in (
                    "type_map",
                    "lower_input_kind",
                    "do_atomic_virial",
                    "has_chg_spin_ebd",
                    "default_chg_spin",
                    "spin_scheme",
                    "has_comm_artifact",
                )
            },
            indent=2,
        )
    )
    assert md["lower_input_kind"] == "dpa4c_canonical"
    assert md["do_atomic_virial"] is True
    # The compact canonical lower takes no condition per call; the state is
    # frozen into its tables, which ``set_charge_spin`` rebuilds. Without the
    # rebuild a runtime charge state cannot be served, and the C++ cases built
    # on it would not reach the replay.
    assert md.get("charge_state_constants"), (
        "the archive carries no charge-state rebuild"
    )
    assert any(name.endswith("charge_state.pt2") for name in names)
    if spin:
        # The dispatch serves a native-spin archive without a with-comm
        # artifact through NativeSpinPTExpt, whose device entry is the one
        # the spin cases reach.
        assert md["spin_scheme"] == "native"
        assert md.get("has_comm_artifact") is False
        assert "force_mag" in md["output_keys"]


def _write_ref(pt2_path: str, ref_path: str, spin: bool) -> None:
    """Evaluate the archive under both states and write the C++ sidecar."""
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    assert dp.has_spin is spin
    # The system itself, which the C++ cases read back rather than restate.
    system = {
        "coord": _COORDS[0],
        "atype": _ATYPES.astype(np.float64),
        "box": _CELL[0],
    }
    if spin:
        system["spin"] = _SPINS[0]
    sections = {"system": system}
    energies = {}
    for name, charge_spin in (
        ("default_state", _DEFAULT_CHG_SPIN),
        ("other_state", _OTHER_CHG_SPIN),
    ):
        if spin:
            e, f, v, ae, av, fm, _ = dp.eval(
                _COORDS,
                _CELL,
                _ATYPES,
                atomic=True,
                spin=_SPINS,
                charge_spin=charge_spin,
            )
        else:
            e, f, v, ae, av = dp.eval(
                _COORDS, _CELL, _ATYPES, atomic=True, charge_spin=charge_spin
            )
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(f))
        energies[name] = float(e[0, 0])
        print(f"// {name} total energy: {e[0, 0]:.18e}")  # noqa: T201
        sections[name] = {
            "expected_e": ae[0, :, 0],
            "expected_f": f[0],
            "expected_tot_v": v[0],
            "expected_atom_v": av[0],
        }
        if spin:
            magnetic = fm[0].reshape(_NATOMS, 3)
            # A closed spin branch yields no magnetic force at all, which
            # would leave the magnetic outputs of the C++ cases unchecked;
            # the gating of the model zeroes those of the second type.
            assert np.abs(magnetic[_ATYPES == 0]).max() > 1e-6, (
                "the magnetic force vanishes on the magnetic atoms"
            )
            assert np.abs(magnetic[_ATYPES == 1]).max() == 0.0, (
                "the magnetic force does not vanish on the non-magnetic atoms"
            )
            sections[name]["expected_fm"] = fm[0]
    # The C++ cases compare evaluations under the two states with each other.
    assert abs(energies["other_state"] - energies["default_state"]) > 1e-4, (
        "the two charge states must differ, or a replay that kept the "
        "constants of the first state would still match"
    )
    write_expected_ref(
        ref_path,
        sections=sections,
        source_script="source/tests/infer/gen_dpa4c_canonical.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201


def main():
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    for stem, build, spin in (
        ("deeppot_dpa4c_canonical", _build_model, False),
        ("deeppot_dpa4c_canonical_spin", _build_spin_model, True),
    ):
        pt2_path = os.path.join(base_dir, f"{stem}.pt2")
        print(f"Exporting to {pt2_path} (lower_kind='dpa4c_canonical') ...")  # noqa: T201
        pt_expt_deserialize_to_file(
            pt2_path,
            {"model": build().serialize()},
            lower_kind="dpa4c_canonical",
            do_atomic_virial=True,
        )
        print("Export done.")  # noqa: T201
        _check_metadata(pt2_path, spin)
        _write_ref(pt2_path, os.path.join(base_dir, f"{stem}.expected"), spin)
    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
