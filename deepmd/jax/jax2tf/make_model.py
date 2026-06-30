# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility wrappers for TensorFlow model-call helpers."""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import tensorflow as tf

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.tf2.common import (
    to_tf_tensor,
    unwrap_value,
    wrap_value,
)
from deepmd.tf2.make_model import (
    model_call_from_call_lower as tf2_model_call_from_call_lower,
)

__all__ = ["model_call_from_call_lower"]


def _wrap_call_lower(call_lower: Callable[..., dict[str, Any]]) -> Callable:
    def wrapped_call_lower(
        extended_coord: Any,
        extended_atype: Any,
        nlist: Any,
        mapping: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return wrap_value(
            call_lower(
                to_tf_tensor(extended_coord),
                to_tf_tensor(extended_atype),
                to_tf_tensor(nlist),
                to_tf_tensor(mapping),
                **{kk: to_tf_tensor(vv) for kk, vv in kwargs.items()},
            )
        )

    return wrapped_call_lower


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[..., dict[str, Any]],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: tf.Tensor,
    atype: tf.Tensor,
    box: tf.Tensor,
    fparam: tf.Tensor,
    aparam: tf.Tensor,
    do_atomic_virial: bool = False,
) -> dict[str, tf.Tensor]:
    return unwrap_value(
        tf2_model_call_from_call_lower(
            call_lower=_wrap_call_lower(call_lower),
            rcut=rcut,
            sel=sel,
            mixed_types=mixed_types,
            model_output_def=model_output_def,
            coord=coord,
            atype=atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
    )
