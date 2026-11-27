# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data-parallel reduction of the element rows that a step trains.

An element bank of a mixture of experts holds one row per element slot (see
``element_banks`` of a descriptor or a fitting), and the gradient of the row
of a slot is nonzero only when an atom of one of its types is in the batch.
The rows of the slots that no rank sees in a step are therefore zero on every
rank and stay zero after the reduction. With the periodic table as type map
the banks hold most of the parameters, and the reduction carries the rows of
the elements in the global batch alone; the other rows take their average, zero,
without being sent. The saving grows with the number of elements that the
global batch lacks, and the gradients equal those of the full reduction up to
rounding.

The slots of a step are known on every rank once its batch is read: each rank
marks the types of its atoms, and an all-reduce of one byte per type takes
their union before the forward. The host reads the union, because it fixes the
sizes of the messages. The communication hook of the data-parallel wrapper then
reduces, in every bucket, the parameters without element rows together with the
rows of the union, and writes them back into the bucket.
"""

from collections.abc import (
    Iterable,
    Sequence,
)
from typing import (
    Any,
)

import torch
import torch.distributed as dist

__all__ = ["ElementRowReduction"]


class ElementRowReduction:
    """Reduce the gradients of a data-parallel step over the trained element rows.

    Parameters
    ----------
    components : Iterable[tuple[Sequence[torch.Tensor], Sequence[int]]]
        The element banks of every descriptor and fitting of the trained
        models, with the element slot of every atom type
        (:func:`~deepmd.dpmodel.model.expert_merge.element_banks`). Branches
        that share a component list it repeatedly. Every other parameter is
        reduced whole.
    device : torch.device
        Device of the gradients.
    """

    def __init__(
        self,
        components: Iterable[tuple[Sequence[torch.Tensor], Sequence[int]]],
        device: torch.device,
    ) -> None:
        # Component of every bank, and the bank row of every type per component.
        self._bank_component: dict[int, int] = {}
        rows_of_type = []
        for banks, slots in components:
            if id(banks[0]) in self._bank_component:
                continue
            for bank in banks:
                self._bank_component[id(bank)] = len(rows_of_type)
            # Row ``z - 1`` of a bank belongs to slot ``z``; slot 0 has none.
            rows_of_type.append([slot - 1 for slot in slots])
        self._rows_of_type = torch.tensor(rows_of_type, dtype=torch.int64, device="cpu")
        self._device = device
        self._rows: list[torch.Tensor] = []
        self._select(
            torch.ones(self._rows_of_type.shape[1], dtype=torch.bool, device="cpu")
        )

    def observe(self, atype: torch.Tensor) -> None:
        """
        Select the element rows of the next step from the atom types of its batch.

        Every rank calls this before the forward of a training step; it takes
        part in one collective and reads its result on the host.

        Parameters
        ----------
        atype : torch.Tensor
            Atom types of the batch of this rank; negative types are padding.
        """
        # Padding, of type -1, marks the entry behind those of the types.
        present = torch.zeros(
            self._rows_of_type.shape[1] + 1, dtype=torch.uint8, device=atype.device
        )
        present[atype.reshape(-1).clamp(min=-1)] = 1
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        self._select(present[:-1].cpu().bool())

    def _select(self, present: torch.Tensor) -> None:
        """Keep the bank rows of the present atom types; a type of slot 0 has none."""
        rows = self._rows_of_type[:, present]
        self._rows = [
            torch.unique(row[row >= 0]).to(self._device, non_blocking=True)
            for row in rows
        ]

    def hook(
        self, state: Any, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        """
        Average a gradient bucket across ranks over its trained element rows.

        The parameters without element rows and the selected rows of the banks
        are packed into one tensor, averaged and written back; the other rows
        are zero on every rank, which is their average. A bucket whose rows
        are all selected is averaged in place.

        Parameters
        ----------
        state : Any
            The state registered with the hook; unused.
        bucket : torch.distributed.GradBucket
            A bucket of the data-parallel wrapper, whose gradients are views
            of its buffer.

        Returns
        -------
        torch.futures.Future[torch.Tensor]
            The averaged buffer of the bucket.
        """
        buffer = bucket.buffer()
        parts = [
            (gradient, self._selected_rows(parameter))
            for parameter, gradient in zip(
                bucket.parameters(), bucket.gradients(), strict=True
            )
        ]
        sizes = [
            gradient.numel() if rows is None else rows.numel() * gradient[0].numel()
            for gradient, rows in parts
        ]
        if sum(sizes) == buffer.numel():
            packed = buffer
        else:
            packed = torch.cat(
                [
                    gradient.reshape(-1)
                    if rows is None
                    else gradient.index_select(0, rows).reshape(-1)
                    for gradient, rows in parts
                ]
            )
        packed.div_(dist.get_world_size())

        def unpack(future: torch.futures.Future) -> torch.Tensor:
            if packed is buffer:
                return buffer
            for (gradient, rows), values in zip(
                parts, packed.split(sizes), strict=True
            ):
                if rows is None:
                    gradient.copy_(values.view_as(gradient))
                else:
                    gradient.index_copy_(
                        0, rows, values.view(rows.numel(), *gradient.shape[1:])
                    )
            return buffer

        return dist.all_reduce(packed, async_op=True).get_future().then(unpack)

    def _selected_rows(self, parameter: torch.Tensor) -> torch.Tensor | None:
        """The rows of a bank that the step trains; None for another parameter."""
        component = self._bank_component.get(id(parameter))
        return None if component is None else self._rows[component]
