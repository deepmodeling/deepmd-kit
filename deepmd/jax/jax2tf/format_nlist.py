# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


@tf.function(autograph=True)
def format_nlist(
    extended_coord: tnp.ndarray,
    nlist: tnp.ndarray,
    nsel: int,
    rcut: float,
):
    """Format neighbor list.

    If nnei == nsel, do nothing;
    If nnei < nsel, pad -1;
    If nnei > nsel, sort by distance and truncate.

    Parameters
    ----------
    extended_coord
        The extended coordinates of the atoms.
        shape: nf x nall x 3
    nlist
        The neighbor list.
        shape: nf x nloc x nnei
    nsel
        The number of selected neighbors.
    rcut
        The cutoff radius.

    Returns
    -------
    nlist
        The formatted neighbor list.
        shape: nf x nloc x nsel
    """
    nlist_shape = tf.shape(nlist)
    n_nf, n_nloc, n_nsel = nlist_shape[0], nlist_shape[1], nlist_shape[2]
    extended_coord = extended_coord.reshape([n_nf, -1, 3])

    if n_nsel < nsel:
        # make a copy before revise
        ret = tnp.concatenate(
            [
                nlist,
                tnp.full([n_nf, n_nloc, nsel - n_nsel], -1, dtype=nlist.dtype),
            ],
            axis=-1,
        )

    elif n_nsel > nsel:
        # make a copy before revise
        m_real_nei = nlist >= 0
        ret = tnp.where(m_real_nei, nlist, 0)
        coord0 = extended_coord[:, :n_nloc, :]
        index = ret.reshape(n_nf, n_nloc * n_nsel, 1)
        index = tnp.repeat(index, 3, axis=2)
        coord1 = tnp.take_along_axis(extended_coord, index, axis=1)
        coord1 = coord1.reshape(n_nf, n_nloc, n_nsel, 3)
        rr2 = tnp.sum(tnp.square(coord0[:, :, None, :] - coord1), axis=-1)
        rr2 = tnp.where(m_real_nei, rr2, float("inf"))
        rr2, ret_mapping = tnp.sort(rr2, axis=-1), tnp.argsort(rr2, axis=-1)
        ret = tnp.take_along_axis(ret, ret_mapping, axis=2)
        ret = tnp.where(rr2 > rcut * rcut, -1, ret)
        ret = ret[..., :nsel]
    else:  # n_nsel == nsel:
        ret = nlist
    # do a reshape any way; this will tell the xla the shape without any dynamic shape
    ret = tnp.reshape(ret, [n_nf, n_nloc, nsel])
    return ret
