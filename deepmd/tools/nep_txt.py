# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export a trained NEP energy model to a GPUMD ``nep.txt`` potential file.

The exporter consumes the backend-agnostic ``serialize()`` dictionary of a
standard energy model whose descriptor is :class:`DescrptNep` and whose fitting
network is a single-hidden-layer ``tanh`` energy network, and writes a
GPUMD-compatible ``nep.txt`` (NEP5 format).

NEP5 is used because GPUMD's NEP4 ANN only carries a single global bias, whereas
a DeePMD-kit energy model holds a per-element energy baseline (the fitting
output-layer bias, ``bias_atom_e``, and the atomic-model ``out_bias``). NEP5
stores a per-element ("typewise") bias, which represents this baseline exactly.
The descriptor itself is identical between NEP4 and NEP5.
"""

import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)

# GPUMD's built-in element table: the first 94 elements (H..Pu) in atomic-number
# order. See ``src/utilities/common.cuh`` (NUM_ELEMENTS = 94) and
# ``src/force/nep.cu`` (ELEMENTS). A nep.txt that names any other element, or more
# than 94 columns, overruns GPUMD's fixed-size arrays at load time.
GPUMD_ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
)
GPUMD_ELEMENT_SET = frozenset(GPUMD_ELEMENTS)

# Mapping between DeePMD-kit and GPUMD conventions (see the module docstring of
# ``nep.py`` and GPUMD ``src/force/nep.cu``):
#   * hidden layer:  GPUMD tanh(sum_d w0[n,d] x[d] - b0[n])
#                    DeePMD  tanh(sum_d w0_dp[d,n] x[d] + b0_dp[n])
#     => w0[n,d] = w0_dp[d,n],  b0[n] = -b0_dp[n]
#   * output:        GPUMD energy = sum_n w1[n] h[n] - w1[N] - b1
#                    DeePMD energy = sum_n w1_dp[n] h[n] + B,
#                    B = b_out + bias_atom_e + out_bias  (per element)
#     => w1[n] = w1_dp[n],  typewise bias w1[N] = -B,  common bias b1 = 0
#   * input scaling: GPUMD x[d] = q[d] * q_scaler[d];  DeePMD x = q / dstd
#     => q_scaler = 1 / dstd  (requires davg == 0)


def serialize_to_nep_txt(
    model: dict, filename: str, elements: list[str] | None = None
) -> None:
    """Write a GPUMD ``nep.txt`` from a serialized NEP energy model.

    Parameters
    ----------
    model : dict
        The dictionary returned by ``EnergyModel.serialize()`` (identical across
        the DP, PyTorch, and JAX backends).
    filename : str
        Destination path of the ``nep.txt`` file.
    elements : list[str], optional
        A subset of the model ``type_map`` to export, in the desired ``nep.txt``
        column order. The descriptor coefficients and per-element ANN are sliced
        accordingly. Defaults to every ``type_map`` entry that GPUMD supports:
        GPUMD only knows the first 94 elements (H..Pu), so a full periodic-table
        model is automatically narrowed to that set (the dropped trailing types
        do not occur in the training data).

    Raises
    ------
    ValueError
        If the descriptor is not ``nep``, the fitting network is incompatible
        with the GPUMD ANN (not a single ``tanh`` hidden layer, or uses a
        time-step), the descriptor mean ``davg`` is non-zero, a requested element
        is not in the model ``type_map``, or a requested element is outside
        GPUMD's supported table (H..Pu).
    """
    # The backend serialization hooks wrap the model under a "model" key; the
    # in-memory ``model.serialize()`` returns it directly. Accept both.
    if "descriptor" not in model and isinstance(model.get("model"), dict):
        model = model["model"]
    descriptor = model["descriptor"]
    if descriptor.get("type") != "nep":
        raise ValueError("nep.txt export requires a 'nep' descriptor")
    fitting = model["fitting"]

    # === Step 1. Validate the fitting network against the GPUMD ANN ===
    neuron = fitting["neuron"]
    if len(neuron) != 1:
        raise ValueError(
            "nep.txt export requires a single-hidden-layer fitting network "
            f"(got neuron={neuron})"
        )
    if fitting["activation_function"] != "tanh":
        raise ValueError("nep.txt export requires a 'tanh' fitting activation")
    num_neurons = neuron[0]

    # === Step 1b. Resolve the exported element set (subset of the type_map) ===
    full_elements = list(descriptor.get("type_map") or model["type_map"])
    n_full = len(full_elements)
    if elements is None:
        # GPUMD only knows H..Pu, so a larger type_map (e.g. the full periodic
        # table used to match an LMDB dataset) is narrowed to the supported set;
        # the dropped trailing elements are not present in the training data.
        elements = [e for e in full_elements if e in GPUMD_ELEMENT_SET]
        if len(elements) != n_full:
            log.warning(
                "model type_map has %d elements; GPUMD supports only H..Pu, "
                "exporting %d and dropping %s",
                n_full,
                len(elements),
                [e for e in full_elements if e not in GPUMD_ELEMENT_SET],
            )
    missing = [e for e in elements if e not in full_elements]
    if missing:
        raise ValueError(f"elements {missing} are not in the model type_map")
    unsupported = [e for e in elements if e not in GPUMD_ELEMENT_SET]
    if unsupported:
        raise ValueError(
            f"GPUMD does not support elements {unsupported}; its table is H..Pu"
        )
    type_index = [full_elements.index(e) for e in elements]
    ntypes = len(elements)
    n_max_radial = descriptor["n_max_radial"]
    n_max_angular = descriptor["n_max_angular"]
    basis_size_radial = descriptor["basis_size_radial"]
    basis_size_angular = descriptor["basis_size_angular"]
    l_max = descriptor["l_max"]
    l_max_4body = descriptor["l_max_4body"]
    l_max_5body = descriptor["l_max_5body"]

    # === Step 2. Descriptor coefficients and scaler ===
    # Slice the (nt, nt, n, k) coefficient tensors to the exported type pairs.
    sel_pairs = np.ix_(type_index, type_index)
    c_radial = np.asarray(descriptor["radial_coeff"]["@variables"]["coeff"])[sel_pairs]
    c_angular = np.asarray(descriptor["angular_coeff"]["@variables"]["coeff"])[
        sel_pairs
    ]
    davg = np.asarray(descriptor["@variables"]["davg"])
    dstd = np.asarray(descriptor["@variables"]["dstd"])
    if not np.allclose(davg, 0.0):
        raise ValueError("nep.txt export requires davg == 0 (NEP does not shift)")
    q_scaler = 1.0 / dstd

    # === Step 3. Per-element ANN parameters and energy baseline ===
    nets = fitting["nets"]["networks"]
    bias_atom_e = np.asarray(fitting["@variables"]["bias_atom_e"]).reshape(n_full, -1)
    out_bias = np.asarray(model["@variables"]["out_bias"]).reshape(-1, n_full, 1)

    ann: list[float] = []
    for t in type_index:
        layers = nets[t]["layers"]
        hidden, output = layers[0]["@variables"], layers[1]["@variables"]
        if hidden.get("idt") is not None:
            raise ValueError("nep.txt export does not support resnet time-steps")
        w0 = np.asarray(hidden["w"])  # (dim, num_neurons)
        b0 = np.asarray(hidden["b"])  # (num_neurons,)
        w1 = np.asarray(output["w"])  # (num_neurons, 1)
        b_out = (
            0.0 if output["b"] is None else float(np.asarray(output["b"]).ravel()[0])
        )
        baseline = b_out + float(bias_atom_e[t, 0]) + float(out_bias[0, t, 0])
        # GPUMD layout: w0[n][d], then b0, then w1, then the typewise bias.
        ann.extend(w0.T.reshape(-1).tolist())
        ann.extend((-b0).reshape(-1).tolist())
        ann.extend(w1[:, 0].tolist())
        ann.append(-baseline)
    ann.append(0.0)  # common bias b1

    # === Step 4. Flatten descriptor coefficients in GPUMD order [n][k][t1][t2] ===
    def _flatten_coeff(coeff: np.ndarray, n_desc: int, k_max: int) -> list[float]:
        coeff = np.asarray(coeff)
        # (nt, nt, n_desc, k_max) -> (n_desc, k_max, nt, nt) -> flat
        return np.transpose(coeff, (2, 3, 0, 1)).reshape(-1).tolist()

    params = (
        ann
        + _flatten_coeff(c_radial, n_max_radial + 1, basis_size_radial + 1)
        + _flatten_coeff(c_angular, n_max_angular + 1, basis_size_angular + 1)
        + q_scaler.reshape(-1).tolist()
    )

    # === Step 5. Write the nep.txt ===
    # GPUMD's MN is the per-atom neighbor capacity (a single count for all
    # types), which it further inflates by 25% and caps at 819. The deepmd
    # ``sel`` is resolved per type, so the largest single entry is the correct
    # estimate; summing across types would massively overcount the capacity.
    max_neighbors = int(np.max(descriptor["sel"]))
    with open(filename, "w") as f:
        f.write(f"nep5 {ntypes} {' '.join(elements)}\n")
        f.write(
            f"cutoff {descriptor['rcut_radial']:g} {descriptor['rcut_angular']:g} "
            f"{max_neighbors} {max_neighbors}\n"
        )
        f.write(f"n_max {n_max_radial} {n_max_angular}\n")
        f.write(f"basis_size {basis_size_radial} {basis_size_angular}\n")
        f.write(f"l_max {l_max} {l_max_4body} {l_max_5body}\n")
        f.write(f"ANN {num_neurons} 0\n")
        # Full precision is written so the export is faithful to the trained
        # model; GPUMD truncates to single precision at inference.
        for value in params:
            f.write(f"{value:.15e}\n")


def _load_pt_checkpoint(input_file: str) -> dict:
    """Reconstruct a serialized model from a pt_expt training checkpoint (.pt)."""
    from copy import (
        deepcopy,
    )

    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    checkpoint = torch.load(input_file, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model_params = state["_extra_state"]["model_params"]
    if "model_dict" in model_params:
        raise ValueError("multi-task checkpoints are not supported for nep.txt export")
    model = get_model(deepcopy(model_params))
    prefix = "model.Default."
    sub_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    model.load_state_dict(sub_state)
    return model.serialize()


def convert_to_nep_txt(
    input_file: str, output_file: str, elements: list[str] | None = None
) -> None:
    """Export a trained NEP model to a GPUMD ``nep.txt``.

    The model is loaded from either a PyTorch training checkpoint (``.pt``), an
    exported PyTorch model (``.pte``/``.pt2``), or a JAX checkpoint
    (``.jax``/``.savedmodel``).

    Parameters
    ----------
    input_file : str
        Path to the trained model checkpoint.
    output_file : str
        Destination path of the ``nep.txt`` file.
    elements : list[str], optional
        A subset of the model ``type_map`` to export; see
        :func:`serialize_to_nep_txt`. Required when the model has more than 94
        element types, as GPUMD only supports the first 94 elements (H..Pu).
    """
    if input_file.endswith(".pt"):
        # A pt_expt training checkpoint stores a state dict, not an exported
        # graph, so the backend serialization hook (which reads .pte/.pt2) does
        # not apply.
        model = _load_pt_checkpoint(input_file)
    else:
        from deepmd.backend.backend import (
            Backend,
        )

        backend = Backend.detect_backend_by_model(input_file)()
        model = backend.serialize_hook(input_file)
    serialize_to_nep_txt(model, output_file, elements=elements)


def main(args: list[str] | None = None) -> None:
    """Command-line entry point for ``nep.txt`` export."""
    parser = argparse.ArgumentParser(
        description="Export a trained NEP model to a GPUMD nep.txt potential file."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="trained NEP checkpoint (PyTorch .pt/.pte/.pt2 or JAX .jax/.savedmodel)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output nep.txt path",
    )
    parser.add_argument(
        "-e",
        "--elements",
        default=None,
        help="comma-separated subset of the model type_map to export, in column "
        "order (e.g. 'H,C,N,O'); required when the model has more than 94 types, "
        "since GPUMD only supports the first 94 elements (H..Pu)",
    )
    parsed = parser.parse_args(args)
    elements = parsed.elements.split(",") if parsed.elements else None
    convert_to_nep_txt(parsed.input, parsed.output, elements=elements)


if __name__ == "__main__":
    main()
