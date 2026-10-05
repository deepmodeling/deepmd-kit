# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spin-free checkpoint transfer for the PyTorch and pt_expt native-spin models."""

from functools import (
    singledispatch,
)

import torch

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    GLUFittingNet,
)
from deepmd.dpmodel.utils.network import (
    FittingNet,
    NativeLayer,
)
from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.utils.finetune import (
    _learned_part_path,
)


def get_native_spin_finetune_state(
    model: torch.nn.Module,
    pretrained_model: torch.nn.Module,
    *,
    random_fitting: bool = False,
) -> dict[str, torch.Tensor]:
    """Build native-spin state from a spin-free learned atomic model.

    Type maps must already agree. The learned parts are matched independently
    of analytical bridging. Output biases and scales belong to the target's
    subsequent statistics pass; spin-only state keeps its constructor values.
    Descriptor-specific transfer rules preserve the geometric function while
    leaving a nonzero gradient into the initially closed spin routes.

    Parameters
    ----------
    model
        Fresh native-spin target, with its input configuration and spin mask.
    pretrained_model
        Loaded spin-free source after type-map alignment.
    random_fitting
        Whether to retain the target fitting network instead of transferring it.

    Returns
    -------
    dict[str, torch.Tensor]
        Complete state dict for a strict load into the target model.

    Raises
    ------
    ValueError
        If the learned parts, descriptor families, type maps or tensor shapes
        are incompatible with spin-free transfer, or widening the fitting input
        changes an implicit residual connection.
    """
    path = _learned_part_path(model)
    source_path = _learned_part_path(pretrained_model)
    if path is None or source_path is None:
        raise ValueError(
            "Spin-free transfer requires one learned atomic model in each model."
        )
    if model.get_type_map() != pretrained_model.get_type_map():
        raise ValueError("Spin-free transfer requires aligned type maps.")
    learned = model.get_submodule(path.removesuffix("."))
    source = pretrained_model.get_submodule(source_path.removesuffix("."))
    descriptor = learned.descriptor
    if type(descriptor) is not type(source.descriptor):
        raise ValueError(
            "Spin-free transfer requires the same descriptor family: "
            f"source={type(source.descriptor).__name__}, "
            f"target={type(descriptor).__name__}."
        )
    if descriptor.use_spin is None or source.descriptor.use_spin is not None:
        raise ValueError(
            "Spin-free transfer requires a spin-free source descriptor and a "
            "native-spin target descriptor."
        )

    expanded = _expanded_spin_state(
        descriptor, source, learned, random_fitting=random_fitting
    )
    state = model.state_dict()
    for key, value in source.state_dict().items():
        target_key = path + key
        if target_key not in state or key.rsplit(".", 1)[-1] in {
            "out_bias",
            "out_std",
            "bias_atom_e",
        }:
            continue
        if random_fitting and not key.startswith("descriptor."):
            continue
        if key in expanded:
            state[target_key] = expanded[key]
        else:
            _check_shape(key, value.shape, state[target_key].shape)
            state[target_key] = value.detach().clone()
    return state


@singledispatch
def _expanded_spin_state(
    descriptor: object,
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    random_fitting: bool,
) -> dict[str, torch.Tensor]:
    """Expand state under the descriptor's native-spin layout contract."""
    raise ValueError(
        f"Spin-free transfer is not registered for {type(descriptor).__name__}."
    )


@_expanded_spin_state.register(DescrptSeZM)
@_expanded_spin_state.register(DescrptDPA4)
def _expanded_dpa4_state(
    descriptor: object,
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    random_fitting: bool,
) -> dict[str, torch.Tensor]:
    # DPA4 injects zero-initialized additive spin routes without widening the
    # geometric descriptor or its fitting input.
    return {}


@_expanded_spin_state.register(DescrptDPA4C)
def _expanded_dpa4c_state(
    descriptor: DescrptDPA4C,
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    random_fitting: bool,
) -> dict[str, torch.Tensor]:
    # The descriptor is [readout, spin, masses, type embedding]. Any frame or
    # atomic fitting parameters follow this descriptor and shift with its tail.
    start = descriptor.readout.get_dim_out()
    width = descriptor.spin.get_dim_out()
    expanded = (
        {}
        if random_fitting
        else _expanded_fitting_input_state(
            source.fitting_net, target.fitting_net, start=start, width=width
        )
    )
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key in ("descriptor.mean", "descriptor.stddev"):
        expanded[key] = _insert_spin_block(
            key,
            source_state[key],
            target_state[key],
            axis=0,
            start=start,
            width=width,
        )
    # Ordered spin scale/shift columns follow every geometric FiLM head.
    key = "descriptor.pair_film.network.layers.1.w"
    expanded[key] = _insert_spin_block(
        key,
        source_state[key],
        target_state[key],
        axis=1,
        start=source_state[key].shape[1],
        width=2 * descriptor.spin_channels,
    )
    return expanded


@singledispatch
def _fitting_input_projection(network: object) -> NativeLayer:
    """Return the affine layer that reads the fitting input."""
    raise ValueError(
        "Spin-free input expansion is not registered for fitting network "
        f"{type(network).__name__}."
    )


@_fitting_input_projection.register(FittingNet)
def _mlp_input_projection(network: FittingNet) -> NativeLayer:
    return network.layers[0]


@_fitting_input_projection.register(GLUFittingNet)
def _glu_input_projection(network: GLUFittingNet) -> NativeLayer:
    return network.hidden_layers[0] if network.hidden_layers else network.output_layer


def _expanded_fitting_input_state(
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    start: int,
    width: int,
) -> dict[str, torch.Tensor]:
    """Expand fitting input projections without changing their residual semantics."""
    if source.nets.ndim != target.nets.ndim:
        raise ValueError("Spin-free transfer requires the same fitting type layout.")
    expanded: dict[int, torch.Tensor] = {}
    for index in range(target.nets.ntypes**target.nets.ndim):
        source_net, target_net = source.nets[index], target.nets[index]
        if type(source_net) is not type(target_net):
            raise ValueError(
                "Spin-free transfer requires matching fitting network types: "
                f"source={type(source_net).__name__}, "
                f"target={type(target_net).__name__}."
            )
        source_layer = _fitting_input_projection(source_net)
        target_layer = _fitting_input_projection(target_net)
        # NativeLayer derives its residual and timestep from the input/output
        # widths. An inserted input block changes an identity or duplicated
        # residual, which cannot in general be absorbed into the nonlinear
        # branch's weights. Reject that structural change before loading state.
        for layer in (source_layer, target_layer):
            if layer.resnet and layer.dim_out() in (
                layer.dim_in(),
                2 * layer.dim_in(),
            ):
                raise ValueError(
                    "Spin-free transfer cannot preserve the first fitting layer's "
                    "residual connection when its input expands from "
                    f"{source_layer.dim_in()} to {target_layer.dim_in()} "
                    f"with output width {target_layer.dim_out()}. "
                    "Use a compatible pretrained fitting architecture or select "
                    "finetune_head='RANDOM' to initialize a fresh fitting network."
                )
        # A closed descriptor spin gate preserves the geometric function while
        # fresh fitting rows give that gate a nonzero gradient.
        expanded[id(target_layer)] = _insert_spin_block(
            f"fitting_net.nets[{index}].input_projection.w",
            source_layer.w,
            target_layer.w,
            axis=0,
            start=start,
            width=width,
        )
    # Network collections expose multiple state-dict paths for the same layer.
    # Every alias refers to the same expanded weight, computed once per network.
    return {
        f"fitting_net.{name}.w": expanded[id(module)]
        for name, module in target.named_modules(remove_duplicate=False)
        if id(module) in expanded
    }


def _check_shape(key: str, expected: tuple[int, ...], actual: tuple[int, ...]) -> None:
    if actual != expected:
        raise ValueError(
            f"Incompatible spin-free transfer for {key}: "
            f"expected target shape {tuple(expected)}, got {tuple(actual)}."
        )


def _insert_spin_block(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    axis: int,
    start: int,
    width: int,
) -> torch.Tensor:
    """Copy the source around an inserted block, retaining its target values."""
    expected = list(source.shape)
    expected[axis] += width
    _check_shape(key, tuple(expected), target.shape)
    source = source.detach()
    result = target.detach().clone()
    result.narrow(axis, 0, start).copy_(source.narrow(axis, 0, start))
    tail = source.shape[axis] - start
    result.narrow(axis, start + width, tail).copy_(source.narrow(axis, start, tail))
    return result
