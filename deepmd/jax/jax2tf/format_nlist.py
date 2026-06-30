# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow graph helpers for JAX/jax2tf SavedModel export.

This module is not a generic TF2 compatibility wrapper. The functions here are
traced while saving the JAX ``.savedmodel`` artifact, before control reaches
``jax2tf.convert``. Keep the implementation in plain TensorFlow ops so
AutoGraph can see the tensor-dependent branches and emit graph control flow.
Routing through ndtensorflow/dpmodel helpers can leave symbolic shape
comparisons as Python ``if`` statements during SavedModel tracing.
"""

import tensorflow as tf


@tf.function(autograph=True)
def format_nlist(
    extended_coord: tf.Tensor,
    nlist: tf.Tensor,
    nsel: int,
    rcut: float,
) -> tf.Tensor:
    """Format neighbor list.

    If nnei == nsel, do nothing;
    If nnei < nsel, pad -1;
    If nnei > nsel, sort by distance and truncate.
    """
    nlist_shape = tf.shape(nlist)
    n_nf, n_nloc, n_nsel = nlist_shape[0], nlist_shape[1], nlist_shape[2]
    extended_coord = tf.reshape(extended_coord, [n_nf, -1, 3])

    if n_nsel < nsel:
        ret = tf.concat(
            [
                nlist,
                tf.fill([n_nf, n_nloc, nsel - n_nsel], tf.cast(-1, nlist.dtype)),
            ],
            axis=-1,
        )
    elif n_nsel > nsel:
        m_real_nei = nlist >= 0
        ret = tf.where(m_real_nei, nlist, tf.zeros_like(nlist))
        coord0 = extended_coord[:, :n_nloc, :]
        index = tf.reshape(ret, [n_nf, n_nloc * n_nsel])
        coord1 = tf.gather(extended_coord, index, batch_dims=1)
        coord1 = tf.reshape(coord1, [n_nf, n_nloc, n_nsel, 3])
        rr2 = tf.reduce_sum(tf.square(coord0[:, :, None, :] - coord1), axis=-1)
        rr2 = tf.where(
            m_real_nei,
            rr2,
            tf.fill(tf.shape(rr2), tf.constant(float("inf"), rr2.dtype)),
        )
        ret_mapping = tf.argsort(rr2, axis=-1)
        rr2 = tf.sort(rr2, axis=-1)
        ret = tf.gather(ret, ret_mapping, batch_dims=2)
        ret = tf.where(
            rr2 > rcut * rcut,
            tf.fill(tf.shape(ret), tf.cast(-1, ret.dtype)),
            ret,
        )
        ret = ret[..., :nsel]
    else:
        ret = nlist
    # Reshape anyway; this tells XLA the shape without dynamic shape.
    ret = tf.reshape(ret, [n_nf, n_nloc, nsel])
    return ret
