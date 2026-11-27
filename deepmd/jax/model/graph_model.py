# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiation of fixed-capacity, frame-segmented neighbor graphs."""

import math

from deepmd.dpmodel.model.edge_transform_output import (
    node_ownership_mask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    edge_force_virial,
    frame_id_from_n_node,
    graph_from_dense_quartet,
    segment_sum,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)


def forward_common_atomic_graph(
    model: BaseModel,
    atype: jnp.ndarray,
    n_node: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_vec: jnp.ndarray,
    edge_mask: jnp.ndarray,
    n_local: jnp.ndarray | None = None,
    fparam: jnp.ndarray | None = None,
    aparam: jnp.ndarray | None = None,
    comm_dict: dict | None = None,
    charge_spin: jnp.ndarray | None = None,
    spin: jnp.ndarray | None = None,
    topologies: dict | None = None,
) -> dict[str, jnp.ndarray]:
    """Differentiate atomic outputs with respect to edge and spin vectors.

    Node and edge capacities are static shapes; counts and validity masks
    remain dynamic. Every frame is independent, so differentiation of the
    sum over frames yields its own edge gradients without a frame Jacobian.
    Energies and forces support higher-order JAX transformations. Explicit
    Hessian output tensors selected by ``enable_hessian()`` are not part of
    this graph lower contract.

    Parameters
    ----------
    model : BaseModel
        Model containing the atomic energy function.
    atype : jnp.ndarray
        Flat atom types with shape (N,); padding atoms have type -1.
    n_node : jnp.ndarray
        Physical node counts per frame with shape (nf,). Real nodes occupy
        a contiguous frame-major prefix of the padded node axis.
    edge_index : jnp.ndarray
        Source and destination node indices with shape (2, E).
    edge_vec : jnp.ndarray
        Source-minus-destination displacements with shape (E, 3), in Angstrom.
    edge_mask : jnp.ndarray
        Valid-edge mask with shape (E,).
    n_local : jnp.ndarray, optional
        Owned-node counts per frame. Other nodes contribute features but
        their atomic energies are excluded from the reduction.
    fparam : jnp.ndarray, optional
        Frame parameters with shape (nf, Df).
    aparam : jnp.ndarray, optional
        Atomic parameters with shape (N, Da).
    comm_dict : dict, optional
        MPI communication metadata; unsupported by this backend.
    charge_spin : jnp.ndarray, optional
        Charge/spin conditioning with shape (nf, 2).
    spin : jnp.ndarray, optional
        Per-node native spin vectors with shape (N, 3).
    topologies : dict, optional
        Additional topology inputs; unsupported by this backend.

    Returns
    -------
    dict[str, jnp.ndarray]
        Internal model outputs on the flat node axis, with frame reductions
        and conservative coordinate, cell, and optional spin derivatives.
    """
    if topologies:
        raise NotImplementedError(
            "JAX graph evaluation does not implement additional topologies."
        )
    if comm_dict is not None:
        raise NotImplementedError(
            "JAX graph evaluation does not implement MPI exchange."
        )
    graph = NeighborGraph(
        n_node=n_node,
        n_local=n_local,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )
    return _evaluate_graph(
        model,
        graph,
        atype,
        edge_index,
        n_node,
        atype.shape[0],
        fparam=fparam,
        aparam=aparam,
        charge_spin=charge_spin,
        spin=spin,
    )


def _evaluate_graph(
    model: BaseModel,
    graph: NeighborGraph,
    atype: jnp.ndarray,
    scatter_index: jnp.ndarray,
    scatter_n_node: jnp.ndarray,
    scatter_capacity: int,
    fparam: jnp.ndarray | None = None,
    aparam: jnp.ndarray | None = None,
    charge_spin: jnp.ndarray | None = None,
    spin: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Evaluate an owner graph and scatter derivatives to its coordinate domain.

    Periodic ghost coordinates have distinct derivative destinations while
    message passing gathers the features of their local owners. The two
    endpoint maps share the same differentiable edge vectors.
    """
    n_node, n_local = graph.n_node, graph.n_local
    edge_vec, edge_index, edge_mask = graph.edge_vec, graph.edge_index, graph.edge_mask
    n_total = atype.shape[0]
    frame_id = frame_id_from_n_node(n_node, n_total=n_total)
    owned = (
        node_ownership_mask(n_node, n_local, n_total)
        if n_local is not None
        else jnp.ones(n_total, dtype=bool)
    )
    output_def = model.atomic_output_def()
    if any(definition.r_hessian for definition in output_def.get_data().values()):
        raise NotImplementedError(
            "Explicit Hessian outputs are not implemented for graph-native JAX "
            "models. Apply JAX higher-order differentiation to the energy or "
            "force function instead."
        )
    result = {}
    for key, definition in output_def.get_data().items():
        if not definition.reducible:
            continue

        def evaluate(
            vectors: jnp.ndarray,
            spins: jnp.ndarray | None,
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            graph = NeighborGraph(
                n_node=n_node,
                n_local=n_local,
                edge_index=edge_index,
                edge_vec=vectors,
                edge_mask=edge_mask,
            )
            atomic = model.atomic_model.forward_common_atomic_graph(
                graph,
                atype,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                spin=spins,
            )
            values = atomic[key].astype(jnp.float64)
            values = values * owned.reshape((n_total,) + (1,) * (values.ndim - 1))
            reduced = segment_sum(values, frame_id, n_node.shape[0])
            if definition.intensive:
                counts = segment_sum(atomic["mask"] * owned, frame_id, n_node.shape[0])
                # Padding frames contain no atoms and retain their zero output.
                counts = jnp.maximum(counts, 1)
                reduced = reduced / counts.reshape((-1,) + (1,) * (values.ndim - 1))
            atomic[f"{key}_redu"] = reduced
            return jnp.sum(reduced, axis=0), atomic

        if not definition.r_differentiable:
            _, atomic = evaluate(edge_vec, spin)
            result.update(atomic)
            continue
        argnums = (0, 1) if spin is not None else 0
        gradients, atomic = jax.jacrev(evaluate, argnums=argnums, has_aux=True)(
            edge_vec, spin
        )
        result.update(atomic)
        if spin is not None:
            gradients, magnetic = gradients
            result[f"{key}_derv_r_mag"] = -jnp.moveaxis(magnetic, -2, 0)
        output_shape = gradients.shape[:-2]
        force, atom_virial, virial = jax.vmap(
            lambda gradient: edge_force_virial(
                gradient,
                edge_vec,
                scatter_index,
                edge_mask,
                scatter_n_node,
                node_capacity=scatter_capacity,
            )
        )(gradients.reshape(math.prod(output_shape), *edge_vec.shape))
        result[f"{key}_derv_r"] = jnp.moveaxis(force, 1, 0).reshape(
            scatter_capacity, *output_shape, 3
        )
        if definition.c_differentiable:
            result[f"{key}_derv_c"] = jnp.moveaxis(atom_virial, 1, 0).reshape(
                scatter_capacity, *output_shape, 9
            )
            result[f"{key}_derv_c_redu"] = jnp.moveaxis(virial, 1, 0).reshape(
                n_node.shape[0], *output_shape, 9
            )
    if spin is not None:
        result["mask_mag"] = model._spin_active_mask(atype)
    return result


def forward_common_atomic(
    model: BaseModel,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray | None = None,
    fparam: jnp.ndarray | None = None,
    aparam: jnp.ndarray | None = None,
    spin: jnp.ndarray | None = None,
    charge_spin: jnp.ndarray | None = None,
    extended_coord_corr: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Evaluate a dense quartet through the same graph derivative convention."""
    nf, nloc, nnei = nlist.shape
    nall = extended_atype.shape[1]
    graph, atype = graph_from_dense_quartet(
        extended_coord, extended_atype, nlist, mapping
    )
    offsets = jnp.arange(nf, dtype=nlist.dtype)[:, None, None] * nall
    src = jnp.maximum(nlist, 0) + offsets
    dst = jnp.broadcast_to(
        jnp.arange(nloc, dtype=nlist.dtype)[None, :, None] + offsets,
        (nf, nloc, nnei),
    )
    scatter_index = jnp.stack((src.reshape(-1), dst.reshape(-1)))
    spins = (
        spin.reshape(nf, -1, 3)[:, :nloc].reshape(-1, 3) if spin is not None else None
    )
    flat_result = _evaluate_graph(
        model,
        graph,
        atype,
        scatter_index,
        jnp.full((nf,), nall, dtype=nlist.dtype),
        nf * nall,
        fparam=fparam,
        aparam=aparam.reshape(nf * nloc, -1) if aparam is not None else None,
        charge_spin=charge_spin,
        spin=spins,
    )
    result = {}
    for key, value in flat_result.items():
        if key.endswith("_redu"):
            result[key] = value
        elif key.endswith("_derv_r_mag"):
            value = value.reshape(nf, nloc, *value.shape[1:])
            padding = ((0, 0), (0, nall - nloc)) + ((0, 0),) * (value.ndim - 2)
            result[key] = jnp.pad(value, padding)
        else:
            count = nall if key.endswith(("_derv_r", "_derv_c")) else nloc
            result[key] = value.reshape(nf, count, *value.shape[1:])
    if extended_coord_corr is not None:
        for key, definition in model.atomic_output_def().get_data().items():
            if not definition.c_differentiable:
                continue
            force = result[f"{key}_derv_r"]
            correction = extended_coord_corr.reshape(
                nf, nall, *((1,) * len(definition.shape)), 3
            )
            correction = (force[..., :, None] * correction[..., None, :]).reshape(
                *force.shape[:-1], 9
            )
            result[f"{key}_derv_c"] += correction
            result[f"{key}_derv_c_redu"] = jnp.sum(result[f"{key}_derv_c"], axis=1)
    return result
