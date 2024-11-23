# SPDX-License-Identifier: LGPL-3.0-or-later

import torch


def concat_switch_virtual(
    extended_tensor,
    extended_tensor_virtual,
    nloc: int,
):
    """
    Concat real and virtual extended tensors, and switch all the local ones to the first nloc * 2 atoms.
    - [:, :nloc]: original nloc real atoms.
    - [:, nloc: nloc + nloc]: virtual atoms corresponding to nloc real atoms.
    - [:, nloc + nloc: nloc + nall]: ghost real atoms.
    - [:, nloc + nall: nall + nall]: virtual atoms corresponding to ghost real atoms.
    """
    nframes, nall = extended_tensor.shape[:2]
    out_shape = list(extended_tensor.shape)
    out_shape[1] *= 2
    extended_tensor_updated = torch.zeros(
        out_shape,
        dtype=extended_tensor.dtype,
        device=extended_tensor.device,
    )
    extended_tensor_updated[:, :nloc] = extended_tensor[:, :nloc]
    extended_tensor_updated[:, nloc : nloc + nloc] = extended_tensor_virtual[:, :nloc]
    extended_tensor_updated[:, nloc + nloc : nloc + nall] = extended_tensor[:, nloc:]
    extended_tensor_updated[:, nloc + nall :] = extended_tensor_virtual[:, nloc:]
    return extended_tensor_updated.view(out_shape)
