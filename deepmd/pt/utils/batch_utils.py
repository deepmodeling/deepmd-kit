# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for converting between flat and dense batch formats."""

from typing import Any

import torch


def pack_flat_to_dense(
    coord: torch.Tensor,
    atype: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    aparam: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pack flat mixed-nloc batch into dense padded batch.

    Parameters
    ----------
    coord : torch.Tensor
        Flattened coordinates with shape [total_atoms, 3].
    atype : torch.Tensor
        Flattened atom types with shape [total_atoms].
    batch : torch.Tensor
        Atom-to-frame assignment with shape [total_atoms].
    ptr : torch.Tensor
        Frame boundaries with shape [nframes + 1].
    aparam : torch.Tensor | None
        Optional flattened atomic parameters with shape [total_atoms, da].

    Returns
    -------
    coord_dense : torch.Tensor
        Dense padded coordinates with shape [nframes, max_nloc, 3].
    atype_dense : torch.Tensor
        Dense padded atom types with shape [nframes, max_nloc].
    aparam_dense : torch.Tensor | None
        Dense padded atomic parameters with shape [nframes, max_nloc, da] or None.

    Notes
    -----
    Padding is done with zeros for coord and -1 for atype (invalid type).
    """
    nframes = ptr.numel() - 1
    natoms_per_frame = ptr[1:] - ptr[:-1]  # [nframes]
    max_nloc = int(natoms_per_frame.max().item())

    device = coord.device
    dtype_coord = coord.dtype
    dtype_atype = atype.dtype

    # Initialize dense tensors with padding
    coord_dense = torch.zeros(
        (nframes, max_nloc, 3), dtype=dtype_coord, device=device
    )
    atype_dense = torch.full(
        (nframes, max_nloc), -1, dtype=dtype_atype, device=device
    )
    aparam_dense = None
    if aparam is not None:
        da = aparam.shape[-1]
        aparam_dense = torch.zeros(
            (nframes, max_nloc, da), dtype=aparam.dtype, device=device
        )

    # Fill in actual data
    for i in range(nframes):
        start = int(ptr[i].item())
        end = int(ptr[i + 1].item())
        nloc = end - start
        coord_dense[i, :nloc] = coord[start:end]
        atype_dense[i, :nloc] = atype[start:end]
        if aparam is not None:
            aparam_dense[i, :nloc] = aparam[start:end]

    return coord_dense, atype_dense, aparam_dense


def unpack_dense_to_flat(
    output_dense: dict[str, torch.Tensor],
    ptr: torch.Tensor,
    atomwise_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Unpack dense padded batch output into flat mixed-nloc batch.

    Parameters
    ----------
    output_dense : dict[str, torch.Tensor]
        Dense output dictionary. Atomwise outputs have shape [nframes, max_nloc, ...],
        frame-wise outputs have shape [nframes, ...].
    ptr : torch.Tensor
        Frame boundaries with shape [nframes + 1].
    atomwise_keys : set[str] | None
        Set of keys that are atomwise (need unpacking). If None, will auto-detect
        by checking if the second dimension matches max_nloc.

    Returns
    -------
    output_flat : dict[str, torch.Tensor]
        Flat output dictionary. Atomwise outputs have shape [total_atoms, ...],
        frame-wise outputs keep shape [nframes, ...].

    Notes
    -----
    This function removes padding from atomwise outputs and concatenates them
    along the atom dimension.
    """
    nframes = ptr.numel() - 1
    natoms_per_frame = ptr[1:] - ptr[:-1]  # [nframes]
    max_nloc = int(natoms_per_frame.max().item())

    output_flat = {}

    for key, value in output_dense.items():
        if value is None:
            output_flat[key] = None
            continue

        # Auto-detect atomwise keys if not provided
        is_atomwise = False
        if atomwise_keys is not None:
            is_atomwise = key in atomwise_keys
        elif value.dim() >= 2 and value.shape[1] == max_nloc:
            is_atomwise = True

        if is_atomwise:
            # Unpack atomwise output: [nframes, max_nloc, ...] -> [total_atoms, ...]
            chunks = []
            for i in range(nframes):
                start = int(ptr[i].item())
                end = int(ptr[i + 1].item())
                nloc = end - start
                chunks.append(value[i, :nloc])
            output_flat[key] = torch.cat(chunks, dim=0)
        else:
            # Keep frame-wise output as is
            output_flat[key] = value

    return output_flat


def create_batch_and_ptr(
    natoms_list: list[int], device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batch and ptr tensors from a list of atom counts.

    Parameters
    ----------
    natoms_list : list[int]
        List of atom counts for each frame.
    device : torch.device | None
        Device to create tensors on. If None, uses CPU.

    Returns
    -------
    batch : torch.Tensor
        Atom-to-frame assignment with shape [total_atoms].
    ptr : torch.Tensor
        Frame boundaries with shape [nframes + 1].

    Examples
    --------
    >>> natoms_list = [10, 20, 15]
    >>> batch, ptr = create_batch_and_ptr(natoms_list)
    >>> batch
    tensor([0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2])  # 10 zeros, 20 ones, 15 twos
    >>> ptr
    tensor([0, 10, 30, 45])
    """
    if device is None:
        device = torch.device("cpu")

    nframes = len(natoms_list)
    total_atoms = sum(natoms_list)

    # Create ptr
    ptr = torch.zeros(nframes + 1, dtype=torch.long, device=device)
    ptr[1:] = torch.cumsum(torch.tensor(natoms_list, dtype=torch.long), dim=0)

    # Create batch
    batch = torch.repeat_interleave(
        torch.arange(nframes, dtype=torch.long, device=device),
        torch.tensor(natoms_list, dtype=torch.long, device=device),
    )

    return batch, ptr
