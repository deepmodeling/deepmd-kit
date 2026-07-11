# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import tensorflow as tf

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)
from deepmd.dpmodel.utils.nlist import (
    apply_pair_exclusion_nlist,
    nlist_distinguish_types,
)
from deepmd.tf2.common import (
    to_tensorflow_array,
    to_tf_tensor,
    unwrap_value,
    wrap_value,
)
from deepmd.tf2.env import (
    xp,
)
from deepmd.tf2.transform_output import (
    communicate_extended_output,
)
from deepmd.tf2.utils._dpmodel import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    normalize_coord,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


def _unwrap_tuple(values: tuple[Array, ...]) -> tuple[tf.Tensor, ...]:
    return tuple(to_tf_tensor(value) for value in values)


def _box_has_pbc(box: Array | None) -> bool | None:
    if box is None:
        return False
    last_dim = box.shape[-1]
    return (last_dim != 0) if isinstance(last_dim, int) else None


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[..., dict[str, Any]],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: Array,
    atype: Array,
    box: Array | None,
    fparam: Array | None,
    aparam: Array | None,
    do_atomic_virial: bool = False,
    do_deriv_c: bool = True,
    coord_corr_for_virial: Array | None = None,
    charge_spin: Array | None = None,
    neighbor_list: NeighborList | None = None,
    pass_lower_kwargs: bool = False,
    pair_excl: "PairExcludeMask | None" = None,
) -> dict[str, Array]:
    """Return model prediction from lower interface.

    Parameters
    ----------
    coord
        The coordinates of the atoms.
        shape: nf x (nloc x 3)
    atype
        The type of atoms. shape: nf x nloc
    box
        The simulation box. shape: nf x 9
    fparam
        frame parameter. nf x ndf
    aparam
        atomic parameter. nf x nloc x nda
    do_atomic_virial
        If calculate the atomic virial.
    neighbor_list
        Optional dense-neighbor-list strategy. ``None`` uses the native TF2
        all-pairs builder.
    pass_lower_kwargs
        Pass optional lower-interface keyword arguments. SavedModel export wraps
        the lower with fixed signatures and keeps this disabled; direct TF2 model
        calls enable it.

    Returns
    -------
    ret_dict
        The result dict of type dict[str, Array].
        The keys are defined by the `ModelOutputDef`.

    """
    (
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fp,
        ap,
        cs,
        extended_coord_corr,
        nlist_is_formatted,
    ) = prepare_lower_inputs(
        rcut=rcut,
        sel=sel,
        mixed_types=mixed_types,
        coord=coord,
        atype=atype,
        box=box,
        fparam=fparam,
        aparam=aparam,
        coord_corr_for_virial=coord_corr_for_virial,
        charge_spin=charge_spin,
        neighbor_list=neighbor_list,
        # Model-level pair exclusion is folded into the nlist inside
        # prepare_lower_inputs (single owner), so the compiled-training prepare
        # step gets the same pre-excluded nlist as this upper call.
        pair_excl=pair_excl,
    )
    lower_kwargs: dict[str, Any] = {"fparam": fp, "aparam": ap}
    if pass_lower_kwargs:
        if nlist_is_formatted:
            lower_kwargs["nlist_is_formatted"] = True
        lower_kwargs.update(
            {
                "do_atomic_virial": do_atomic_virial,
                "do_deriv_c": do_deriv_c,
                "charge_spin": cs,
            }
        )
        if extended_coord_corr is not None:
            lower_kwargs["extended_coord_corr"] = extended_coord_corr
    model_predict_lower = call_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        **lower_kwargs,
    )
    model_predict = wrap_value(
        communicate_extended_output(
            unwrap_value(model_predict_lower),
            model_output_def,
            to_tf_tensor(mapping),
            do_atomic_virial=do_atomic_virial,
        )
    )
    return model_predict


def prepare_lower_inputs(
    *,
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    coord: Array,
    atype: Array,
    box: Array | None,
    fparam: Array | None,
    aparam: Array | None,
    coord_corr_for_virial: Array | None = None,
    charge_spin: Array | None = None,
    neighbor_list: NeighborList | None = None,
    pair_excl: "PairExcludeMask | None" = None,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array | None,
    Array | None,
    Array | None,
    Array | None,
    bool,
]:
    """Build lower-interface tensors outside the train-step compiler boundary.

    Model-level ``pair_exclude_types`` is a nlist-BUILD transform (decision
    #18/A4): when ``pair_excl`` is provided it is folded into the freshly built
    nlist here, so EVERY caller (the eager/compiled upper call and the compiled
    training prepare step) gets a pre-excluded nlist and the lower never
    re-applies it.
    """
    cc = to_tensorflow_array(coord)
    atype = to_tensorflow_array(atype)
    bb = to_tensorflow_array(box)
    fp = to_tensorflow_array(fparam)
    ap = to_tensorflow_array(aparam)
    cs = to_tensorflow_array(charge_spin)
    coord_corr = to_tensorflow_array(coord_corr_for_virial)
    del coord, box, fparam, aparam, charge_spin, coord_corr_for_virial
    nframes, nloc = atype.shape[:2]

    def with_pbc() -> tuple[Array, Array, Array, Array]:
        assert bb is not None
        coord_normalized = normalize_coord(
            xp.reshape(cc, (nframes, nloc, 3)),
            xp.reshape(bb, (nframes, 3, 3)),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, bb, rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            sel,
            # types will be distinguished in the lower interface,
            # so it doesn't need to be distinguished here
            distinguish_types=False,
        )
        return extended_coord, extended_atype, nlist, mapping

    def no_pbc() -> tuple[Array, Array, Array, Array]:
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            cc, atype, None, rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            sel,
            # types will be distinguished in the lower interface,
            # so it doesn't need to be distinguished here
            distinguish_types=False,
        )
        return extended_coord, extended_atype, nlist, mapping

    uses_native_nlist_builder = neighbor_list is None
    if neighbor_list is not None:
        # The BUILDER owns model-level pair exclusion (same convention as
        # dpmodel/pt_expt make_model). The keyword is passed only when set, so
        # legacy custom strategies keep working without exclusion and fail
        # loudly (TypeError) with it instead of silently including pairs.
        excl_kwargs = {} if pair_excl is None else {"pair_excl": pair_excl}
        extended_coord, extended_atype, nlist, mapping = neighbor_list.build(
            cc, atype, bb, rcut, sel, **excl_kwargs
        )
    else:
        has_pbc = _box_has_pbc(bb)
        if has_pbc is True:
            extended_coord, extended_atype, nlist, mapping = with_pbc()
        elif has_pbc is False:
            extended_coord, extended_atype, nlist, mapping = no_pbc()
        else:
            assert bb is not None
            (
                extended_coord_tensor,
                extended_atype_tensor,
                nlist_tensor,
                mapping_tensor,
            ) = tf.cond(
                tf.shape(to_tf_tensor(bb))[-1] != 0,
                lambda: _unwrap_tuple(with_pbc()),
                lambda: _unwrap_tuple(no_pbc()),
            )
            extended_coord = to_tensorflow_array(extended_coord_tensor)
            extended_atype = to_tensorflow_array(extended_atype_tensor)
            nlist = to_tensorflow_array(nlist_tensor)
            mapping = to_tensorflow_array(mapping_tensor)
        if pair_excl is not None:
            # Native inline builder: exclude at BUILD time, mirroring
            # DefaultNeighborList.build on the dpmodel path (the custom-builder
            # branch above already excluded inside build()).
            nlist = apply_pair_exclusion_nlist(nlist, extended_atype, pair_excl)
    extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
    if coord_corr is not None:
        coord_corr = xp.reshape(coord_corr, (nframes, nloc, 3))
        mapping_idx = xp.tile(
            xp.reshape(mapping, (nframes, -1, 1)),
            (1, 1, 3),
        )
        extended_coord_corr = xp.take_along_axis(coord_corr, mapping_idx, axis=1)
    else:
        extended_coord_corr = None
    if uses_native_nlist_builder and not mixed_types:
        nlist = nlist_distinguish_types(nlist, extended_atype, sel)
    return (
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fp,
        ap,
        cs,
        extended_coord_corr,
        uses_native_nlist_builder,
    )
