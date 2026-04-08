# SPDX-License-Identifier: LGPL-3.0-or-later
"""Helpers for enhanced per-module export tests.

Provides symbolic tracing, torch.export with dynamic shapes, and .pte
save/load round-trip verification used by descriptor, fitting, and model
test_make_fx / test_forward_lower_exportable methods.
"""

import tempfile

import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)


def export_save_load_and_compare(
    fn,
    inputs: tuple,
    eager_outputs: tuple,
    dynamic_shapes: tuple,
    rtol: float = 0.0,
    atol: float = 1e-12,
):
    """Symbolic trace -> export with dynamic shapes -> .pte save/load -> compare.

    Parameters
    ----------
    fn : callable
        The function to trace (same one used for eager and concrete make_fx).
    inputs : tuple of torch.Tensor
        Input tensors for the function.
    eager_outputs : tuple of torch.Tensor
        Reference outputs from eager execution.
    dynamic_shapes : tuple
        Dynamic shape specs for torch.export.export.
    rtol, atol : float
        Tolerances for np.testing.assert_allclose.

    Returns
    -------
    loaded_module : torch.nn.Module
        The module loaded from the .pte round-trip, for further testing.
    """
    # 1. Symbolic make_fx trace
    traced_sym = make_fx(fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
        *inputs
    )

    # 2. Compare symbolic-traced output vs eager
    sym_outputs = traced_sym(*inputs)
    if not isinstance(sym_outputs, tuple):
        sym_outputs = (sym_outputs,)
    if not isinstance(eager_outputs, tuple):
        eager_outputs = (eager_outputs,)
    for sym_out, eager_out in zip(sym_outputs, eager_outputs, strict=True):
        np.testing.assert_allclose(
            sym_out.detach().cpu().numpy(),
            eager_out.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

    # 3. torch.export.export with dynamic shapes
    exported = torch.export.export(
        traced_sym,
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    # 4. .pte save -> load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pte") as f:
        torch.export.save(exported, f.name)
        loaded = torch.export.load(f.name)

    loaded_module = loaded.module()

    # 5. Compare loaded output vs eager (same shapes)
    loaded_outputs = loaded_module(*inputs)
    if not isinstance(loaded_outputs, tuple):
        loaded_outputs = (loaded_outputs,)
    for loaded_out, eager_out in zip(loaded_outputs, eager_outputs, strict=True):
        np.testing.assert_allclose(
            loaded_out.detach().cpu().numpy(),
            eager_out.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

    # 6. Compare loaded output vs eager (different nframes via nf=1 slice)
    inputs_1f = tuple(t[0:1] if t is not None else None for t in inputs)
    eager_1f = fn(*inputs_1f)
    loaded_1f = loaded_module(*inputs_1f)
    if not isinstance(eager_1f, tuple):
        eager_1f = (eager_1f,)
    if not isinstance(loaded_1f, tuple):
        loaded_1f = (loaded_1f,)
    for eager_out, loaded_out in zip(eager_1f, loaded_1f, strict=True):
        np.testing.assert_allclose(
            eager_out.detach().cpu().numpy(),
            loaded_out.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

    return loaded_module


def model_forward_lower_export_round_trip(
    md_pt,
    ext_coord,
    ext_atype,
    nlist_t,
    mapping_t,
    fparam,
    aparam,
    output_keys: tuple[str, ...],
    rtol: float = 1e-10,
    atol: float = 1e-10,
):
    """Full forward_lower_exportable test: concrete trace + export + symbolic + .pte.

    Performs the complete export pipeline test:
    1. Eager reference via forward_lower
    2. Concrete trace via forward_lower_exportable
    3. torch.export.export (no dynamic shapes)
    4. Compare traced/exported vs eager
    5. Symbolic trace + dynamic shapes + .pte save/load round-trip
    6. Compare loaded vs eager (nf=1 — different shapes)

    Parameters
    ----------
    md_pt : torch.nn.Module
        The model (already deserialized and in eval mode).
    ext_coord, ext_atype, nlist_t, mapping_t : torch.Tensor
        Extended coordinates, atom types, neighbor list, mapping.
    fparam, aparam : torch.Tensor or None
        Frame and atom parameters.
    output_keys : tuple of str
        Output dictionary keys to verify.
    rtol, atol : float
        Tolerances for np.testing.assert_allclose.
    """
    from deepmd.pt_expt.utils.serialization import (
        _build_dynamic_shapes,
    )

    # 1. Eager reference
    ret_eager = md_pt.forward_lower(
        ext_coord.requires_grad_(True),
        ext_atype,
        nlist_t,
        mapping_t,
        fparam=fparam,
        aparam=aparam,
    )

    # 2. Concrete trace
    traced = md_pt.forward_lower_exportable(
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        fparam=fparam,
        aparam=aparam,
    )
    assert isinstance(traced, torch.nn.Module)

    # 3. Basic export (no dynamic shapes)
    exported = torch.export.export(
        traced,
        (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam),
        strict=False,
    )
    assert exported is not None

    # 4. Compare traced and exported vs eager
    ret_traced = traced(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
    ret_exported = exported.module()(
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
    )
    for key in output_keys:
        np.testing.assert_allclose(
            ret_eager[key].detach().cpu().numpy(),
            ret_traced[key].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"traced vs eager: {key}",
        )
        np.testing.assert_allclose(
            ret_eager[key].detach().cpu().numpy(),
            ret_exported[key].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"exported vs eager: {key}",
        )

    # 5. Symbolic trace + dynamic shapes + .pte round-trip
    inputs_2f = tuple(
        torch.cat([t, t], dim=0) if t is not None else None
        for t in (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
    )
    traced_sym = md_pt.forward_lower_exportable(
        inputs_2f[0],
        inputs_2f[1],
        inputs_2f[2],
        inputs_2f[3],
        fparam=inputs_2f[4],
        aparam=inputs_2f[5],
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    dynamic_shapes = _build_dynamic_shapes(*inputs_2f)
    exported_dyn = torch.export.export(
        traced_sym,
        inputs_2f,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    with tempfile.NamedTemporaryFile(suffix=".pte") as f:
        torch.export.save(exported_dyn, f.name)
        loaded = torch.export.load(f.name).module()

    # 6. Compare loaded vs eager (nf=1 — different shapes)
    ret_loaded_1f = loaded(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
    for key in output_keys:
        np.testing.assert_allclose(
            ret_eager[key].detach().cpu().numpy(),
            ret_loaded_1f[key].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"loaded vs eager (nf=1): {key}",
        )


def make_descriptor_dynamic_shapes(has_mapping: bool = False) -> tuple:
    """Build dynamic shapes for descriptor inputs (coord_ext, atype_ext, nlist[, mapping]).

    Note: coord_ext is in flattened form (nframes, nall*3), not (nframes, nall, 3).

    Parameters
    ----------
    has_mapping : bool
        Whether the descriptor takes a mapping argument.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)

    shapes = (
        {0: nframes_dim, 1: 3 * nall_dim},  # coord_ext: (nframes, nall*3)
        {0: nframes_dim, 1: nall_dim},  # atype_ext: (nframes, nall)
        {0: nframes_dim, 1: nloc_dim},  # nlist: (nframes, nloc, nnei)
    )
    if has_mapping:
        shapes = (*shapes, {0: nframes_dim, 1: nall_dim})  # + mapping: (nframes, nall)
    return shapes


def make_fitting_dynamic_shapes(
    has_gr: bool = False,
    has_fparam: bool = False,
    has_aparam: bool = False,
) -> tuple:
    """Build dynamic shapes for fitting inputs (descriptor, atype[, gr][, fparam][, aparam]).

    Only nframes is marked dynamic. Fitting nets tested in isolation may
    specialize on nloc during symbolic tracing, making nloc incompatible
    with dynamic dim specs. In the full model pipeline, nloc comes from
    the descriptor output and remains dynamic; here we only test nframes.

    Parameters
    ----------
    has_gr : bool
        Whether the fitting takes a gr (rotation matrix) argument.
    has_fparam : bool
        Whether the fitting takes fparam.
    has_aparam : bool
        Whether the fitting takes aparam.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)

    shapes: list = [
        {0: nframes_dim},  # descriptor: (nframes, nloc, dim_descrpt)
        {0: nframes_dim},  # atype: (nframes, nloc)
    ]
    if has_gr:
        shapes.append({0: nframes_dim})  # gr: (nframes, nloc, nnei, 3)
    if has_fparam:
        shapes.append({0: nframes_dim})  # fparam: (nframes, nfp)
    if has_aparam:
        shapes.append({0: nframes_dim})  # aparam: (nframes, nloc, nap)
    return tuple(shapes)
