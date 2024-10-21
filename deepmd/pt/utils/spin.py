from typing import Optional
import torch


def concat_switch_virtual(
    extended_tensor,
    extended_tensor_virtual,
    nloc: int,
    recv_num: Optional[torch.Tensor] = None,
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
    extended_tensor_updated[:, nloc : nloc + nloc] = extended_tensor_virtual[
        :, :nloc
    ]
    extended_tensor_updated[:, nloc + nloc : nloc + nall] = extended_tensor[
        :, nloc:
    ]
    extended_tensor_updated[:, nloc + nall :] = extended_tensor_virtual[:, nloc:]
    # nloc + nloc + nghost + nghost
    if recv_num is not None:
        # recv_num : nswap * 1
        origin_recv_num = torch.div(recv_num, 2).to(torch.int)
        prefix_sum = torch.cumsum(recv_num, dim=0)
        prefix_sum = torch.cat((torch.tensor([0]), prefix_sum))
        # prefix_sum: (nswap+1) * 1
        origin_prefix_sum = torch.cumsum(origin_recv_num, dim=0)
        origin_prefix_sum = torch.cat((torch.tensor([0]), origin_prefix_sum))
        # origin_prefix_sum: (nswap+1) * 1
        for i in range(recv_num.size(0)):
            extended_tensor_updated[
                :,
                nloc + nloc + prefix_sum[i] : nloc
                + nloc
                + prefix_sum[i]
                + origin_recv_num[i],
            ] = extended_tensor[
                :, nloc + origin_prefix_sum[i] : nloc + origin_prefix_sum[i + 1]
            ]
            extended_tensor_updated[
                :,
                nloc + nloc + prefix_sum[i] + origin_recv_num[i] : nloc
                + nloc
                + prefix_sum[i + 1],
            ] = extended_tensor_virtual[
                :, nloc + origin_prefix_sum[i] : nloc + origin_prefix_sum[i + 1]
            ]
    return extended_tensor_updated.view(out_shape)