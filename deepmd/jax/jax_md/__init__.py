# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX-MD compatible interface for JAX DeePMD models."""

import inspect
import json
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)

Array = jax.Array
EnergyFn = Callable[..., Array]

_JAX_MD_SENTINEL = object()


def load_model(model: str | Path | Any) -> Any:
    """Load a JAX DeePMD model, or return an already constructed model."""
    if not isinstance(model, str | Path):
        return model

    model_path = str(Path(model).resolve())
    if model_path.endswith(".jax"):
        data = serialize_from_file(model_path)
        jax_model = BaseModel.deserialize(data["model"])
        jax_model.model_def_script = json.dumps(data.get("model_def_script", {}))
        return jax_model
    if model_path.endswith(".hlo"):
        raise NotImplementedError(
            "JAX-MD does not support .hlo models yet. The JAX-MD simulation "
            "helpers require differentiating the energy function, while exported "
            "StableHLO models do not expose a VJP to JAX. Use a .jax checkpoint."
        )
    raise ValueError("JAX-MD interface supports .jax checkpoints.")


def energy_fn(
    model: str | Path | Any,
    atom_types: Sequence[int | str] | Array,
    *,
    box: Array | Sequence[float] | None = None,
    displacement_fn: Callable[..., Array] | None = None,
    fparam: Array | Sequence[float] | None = None,
    aparam: Array | Sequence[float] | None = None,
    charge_spin: Array | Sequence[float] | None = None,
) -> EnergyFn:
    """Create a JAX-MD compatible ``energy_fn(R, neighbor=None, **kwargs)``.

    The returned function accepts a single-frame coordinate array ``R`` with
    shape ``(natoms, 3)`` and returns a scalar total energy.  If a JAX-MD dense
    neighbor list object is passed as ``neighbor``, DeePMD uses it to build the
    lower-interface extended system.  Otherwise DeePMD builds its native JAX
    neighbor list from ``R`` and ``box``.
    """
    jax_model = load_model(model)
    default_atom_types = _normalize_atom_types(jax_model, atom_types)
    default_box = _normalize_box(box)
    default_fparam = fparam
    default_aparam = aparam
    default_charge_spin = charge_spin

    def apply(
        R: Array,
        *,
        neighbor: Any | None = None,
        atom_types: Sequence[int | str] | Array | None = None,
        box: Array | Sequence[float] | None | object = _JAX_MD_SENTINEL,
        fparam: Array | Sequence[float] | None | object = _JAX_MD_SENTINEL,
        aparam: Array | Sequence[float] | None | object = _JAX_MD_SENTINEL,
        charge_spin: Array | Sequence[float] | None | object = _JAX_MD_SENTINEL,
        **kwargs: Any,
    ) -> Array:
        """Evaluate a single-frame total energy in the JAX-MD call convention."""
        coord = _normalize_coord(R)
        atype = (
            default_atom_types
            if atom_types is None
            else _normalize_atom_types(jax_model, atom_types)
        )
        current_box = default_box if box is _JAX_MD_SENTINEL else _normalize_box(box)
        current_fparam = default_fparam if fparam is _JAX_MD_SENTINEL else fparam
        current_aparam = default_aparam if aparam is _JAX_MD_SENTINEL else aparam
        current_charge_spin = (
            default_charge_spin if charge_spin is _JAX_MD_SENTINEL else charge_spin
        )
        fparam_batch = _normalize_fparam(jax_model, current_fparam, coord.dtype)
        aparam_batch = _normalize_aparam(
            jax_model, current_aparam, coord.shape[0], coord.dtype
        )
        charge_spin_batch = _normalize_charge_spin(current_charge_spin, coord.dtype)

        if neighbor is None:
            model_kwargs = {
                "box": None if current_box is None else current_box[None, ...],
                "fparam": fparam_batch,
                "aparam": aparam_batch,
            }
            if charge_spin_batch is not None:
                if not _accepts_keyword(jax_model, "charge_spin"):
                    raise TypeError("This model does not accept charge_spin input.")
                model_kwargs["charge_spin"] = charge_spin_batch
            ret = jax_model(
                coord[None, ...],
                atype[None, ...],
                **model_kwargs,
            )
        else:
            ret = _eval_with_jax_md_neighbor(
                jax_model,
                coord,
                atype,
                neighbor,
                displacement_fn,
                fparam_batch,
                aparam_batch,
                charge_spin_batch,
                kwargs,
            )
        return _extract_energy(ret)

    return apply


def force_fn(energy: EnergyFn) -> EnergyFn:
    """Create a JAX-MD compatible force function from an energy function."""

    def apply(R: Array, **kwargs: Any) -> Array:
        """Evaluate forces by differentiating the supplied energy function."""
        return -jax.grad(lambda coord: energy(coord, **kwargs))(R)

    return apply


def neighbor_list(
    model: str | Path | Any,
    displacement_or_metric: Callable[..., Array],
    box: Array | Sequence[float],
    **kwargs: Any,
) -> Any:
    """Create a dense JAX-MD neighbor-list function using the model cutoff."""
    try:
        from jax_md import (
            partition,
        )
    except ImportError as exc:
        raise ImportError(
            "The JAX-MD neighbor-list helper requires the optional jax-md package."
        ) from exc

    jax_model = load_model(model)
    neighbor_format = kwargs.setdefault("format", partition.NeighborListFormat.Dense)
    if neighbor_format != partition.NeighborListFormat.Dense:
        raise ValueError("Only dense JAX-MD neighbor lists are supported.")
    _validate_displacement_or_metric(displacement_or_metric)
    return partition.neighbor_list(
        displacement_or_metric,
        box,
        r_cutoff=jax_model.get_rcut(),
        **kwargs,
    )


def as_jax_md(
    model: str | Path | Any,
    displacement_or_metric: Callable[..., Array],
    box: Array | Sequence[float],
    atom_types: Sequence[int | str] | Array,
    **kwargs: Any,
) -> tuple[Any, EnergyFn]:
    """Return ``(neighbor_fn, energy_fn)`` in the usual JAX-MD style."""
    jax_model = load_model(model)
    potential = energy_fn(
        jax_model,
        atom_types,
        box=_normalize_box(box),
        displacement_fn=displacement_or_metric,
        fparam=kwargs.pop("fparam", None),
        aparam=kwargs.pop("aparam", None),
        charge_spin=kwargs.pop("charge_spin", None),
    )
    nlist_fn = neighbor_list(jax_model, displacement_or_metric, box, **kwargs)
    return nlist_fn, potential


def _validate_displacement_or_metric(
    displacement_or_metric: Callable[..., Array],
) -> None:
    """Reject scalar metrics where DeePMD needs vector displacements."""
    coord = jnp.zeros((3,), dtype=jnp.float32)
    displacement = jnp.asarray(displacement_or_metric(coord, coord))
    if displacement.shape != coord.shape:
        raise ValueError(
            "Dense neighbor evaluation requires a displacement function returning "
            "vectors with shape (..., 3); scalar metric functions are not supported."
        )


def _normalize_atom_types(model: Any, atom_types: Sequence[int | str] | Array) -> Array:
    """Convert type names or type indexes to a JAX int32 type array."""
    if isinstance(atom_types, jax.Array):
        return atom_types.astype(jnp.int32)
    atom_types_list = list(atom_types)
    if atom_types_list and isinstance(atom_types_list[0], str):
        type_map = {name: idx for idx, name in enumerate(model.get_type_map())}
        atom_types_list = [type_map[str(atom_type)] for atom_type in atom_types_list]
    return jnp.asarray(atom_types_list, dtype=jnp.int32)


def _accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    """Return whether a callable signature accepts a keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _normalize_coord(coord: Array) -> Array:
    """Validate and convert a single-frame coordinate array."""
    coord = jnp.asarray(coord)
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError("JAX-MD DeePMD energy functions require R with shape (N, 3).")
    return coord


def _normalize_box(box: Array | Sequence[float] | None) -> Array | None:
    """Convert supported box representations to a 3-by-3 cell matrix."""
    if box is None:
        return None
    box_array = jnp.asarray(box)
    if box_array.ndim == 0:
        return jnp.eye(3, dtype=box_array.dtype) * box_array
    if box_array.shape == (3,):
        return jnp.diag(box_array)
    if box_array.shape == (9,):
        return box_array.reshape(3, 3)
    if box_array.shape == (3, 3):
        return box_array
    raise ValueError("box must be a scalar, shape (3,), shape (9,), or shape (3, 3).")


def _normalize_fparam(
    model: Any, fparam: Array | Sequence[float] | None, dtype: Any
) -> Array | None:
    """Convert frame parameters to DeePMD's batched JAX input shape."""
    dim_fparam = model.get_dim_fparam()
    if dim_fparam == 0:
        return None
    if fparam is None:
        if model.has_default_fparam():
            default_fparam = model.get_default_fparam()
            if default_fparam is not None:
                return jnp.asarray(default_fparam, dtype=dtype).reshape(1, dim_fparam)
        raise ValueError("This model requires fparam, but none was provided.")
    return jnp.asarray(fparam, dtype=dtype).reshape(1, dim_fparam)


def _normalize_aparam(
    model: Any,
    aparam: Array | Sequence[float] | None,
    natoms: int,
    dtype: Any,
) -> Array | None:
    """Convert atomic parameters to DeePMD's batched JAX input shape."""
    dim_aparam = model.get_dim_aparam()
    if dim_aparam == 0:
        return None
    if aparam is None:
        raise ValueError("This model requires aparam, but none was provided.")
    aparam_array = jnp.asarray(aparam, dtype=dtype)
    if aparam_array.shape == (dim_aparam,):
        aparam_array = jnp.tile(aparam_array[None, :], (natoms, 1))
    return aparam_array.reshape(1, natoms, dim_aparam)


def _normalize_charge_spin(
    charge_spin: Array | Sequence[float] | None, dtype: Any
) -> Array | None:
    """Convert charge-spin parameters to a batched JAX input array."""
    if charge_spin is None:
        return None
    return jnp.asarray(charge_spin, dtype=dtype)[None, ...]


def _eval_with_jax_md_neighbor(
    model: Any,
    coord: Array,
    atype: Array,
    neighbor: Any,
    displacement_fn: Callable[..., Array] | None,
    fparam: Array | None,
    aparam: Array | None,
    charge_spin: Array | None,
    displacement_kwargs: dict[str, Any],
) -> dict[str, Array]:
    """Evaluate a DeePMD model with a precomputed dense JAX-MD neighbor list."""
    if not hasattr(model, "call_lower"):
        raise TypeError("JAX-MD neighbor lists require a DeePMD model with call_lower.")
    extended_coord, extended_atype, nlist, mapping = _jax_md_neighbor_to_lower_inputs(
        coord,
        atype,
        neighbor,
        displacement_fn,
        displacement_kwargs,
    )
    # Model-level ``pair_exclude_types`` is a nlist-BUILD transform (decision
    # #18/A4): ``call_lower`` consumes a pre-excluded nlist and no longer
    # re-applies it. The JAX-MD neighbor list is built without exclusion, so
    # fold it in at this ingestion seam -- otherwise excluded pairs would be
    # silently included (fail-open).
    pair_excl = getattr(getattr(model, "atomic_model", None), "pair_excl", None)
    if pair_excl is not None:
        from deepmd.dpmodel.utils.nlist import (
            apply_pair_exclusion_nlist,
        )

        nlist = apply_pair_exclusion_nlist(nlist, extended_atype, pair_excl)
    return model.call_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam=fparam,
        aparam=aparam,
        charge_spin=charge_spin,
    )


def _jax_md_neighbor_to_lower_inputs(
    coord: Array,
    atype: Array,
    neighbor: Any,
    displacement_fn: Callable[..., Array] | None,
    displacement_kwargs: dict[str, Any],
) -> tuple[Array, Array, Array, Array]:
    """Convert a dense JAX-MD neighbor list to DeePMD lower-interface inputs.

    DeePMD's lower interface expects an extended coordinate array plus a
    neighbor list that points from local atoms into that extended array.  JAX-MD
    dense neighbor lists store original atom indexes instead, so each valid edge
    is materialized as a ghost coordinate with the minimum-image displacement
    supplied by the JAX-MD displacement function.
    """
    if not hasattr(neighbor, "idx"):
        raise TypeError("Expected a JAX-MD neighbor object with an idx attribute.")
    nlist = jnp.asarray(neighbor.idx)
    natoms = coord.shape[0]
    if nlist.ndim != 2 or nlist.shape[0] != natoms:
        raise ValueError(
            "Only dense JAX-MD neighbor lists with shape (N, max_occupancy) are supported."
        )

    valid = (nlist >= 0) & (nlist < natoms)
    safe_nlist = jnp.where(valid, nlist, 0).astype(jnp.int32)
    neighbor_coord = coord[safe_nlist]
    central_coord = jnp.broadcast_to(coord[:, None, :], neighbor_coord.shape)

    if displacement_fn is None:
        ghost_coord = neighbor_coord
    else:
        displacement = jax.vmap(
            jax.vmap(
                lambda central, neighbor: displacement_fn(
                    central, neighbor, **displacement_kwargs
                )
            )
        )(central_coord, neighbor_coord)
        if displacement.shape != neighbor_coord.shape:
            raise ValueError(
                "Dense neighbor evaluation requires a displacement function returning "
                "vectors with shape (..., 3); scalar metric functions are not supported."
            )
        # JAX-MD displacement functions use the Ra - Rb convention.
        ghost_coord = central_coord - displacement

    ghost_coord = jnp.where(valid[..., None], ghost_coord, central_coord)
    ghost_atype = jnp.where(valid, atype[safe_nlist], -1)
    nedge = nlist.size
    ghost_start = natoms
    ghost_indices = jnp.arange(ghost_start, ghost_start + nedge, dtype=jnp.int64)
    lower_nlist = jnp.where(valid, ghost_indices.reshape(nlist.shape), -1)

    extended_coord = jnp.concatenate(
        [coord, ghost_coord.reshape(nedge, 3)],
        axis=0,
    )[None, ...]
    extended_atype = jnp.concatenate(
        [atype, ghost_atype.reshape(nedge)],
        axis=0,
    )[None, ...]
    mapping = jnp.concatenate(
        [
            jnp.arange(natoms, dtype=jnp.int64),
            safe_nlist.reshape(nedge).astype(jnp.int64),
        ],
        axis=0,
    )[None, ...]
    return extended_coord, extended_atype, lower_nlist[None, ...], mapping


def _extract_energy(ret: Any) -> Array:
    """Extract a scalar total energy from a DeePMD model return value."""
    if isinstance(ret, tuple):
        ret = ret[0]
    if "energy" in ret and ret["energy"] is not None:
        return jnp.ravel(ret["energy"])[0]
    raise KeyError("Model output does not contain an energy value.")


__all__ = [
    "as_jax_md",
    "energy_fn",
    "force_fn",
    "load_model",
    "neighbor_list",
]
