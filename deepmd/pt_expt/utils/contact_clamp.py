# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch form of the contact clamp of a bridged model.

The pt and pt_expt backends share this module. It keeps the contact record and
the data edges as persistent buffers, records them from the training batches
in place, so that a compiled evaluation graph reading the edges sees every
update, and merges the records of the ranks of a distributed training.
"""

from typing import (
    Any,
)

import torch
import torch.distributed as dist

from deepmd.dpmodel.utils.contact_clamp import ContactClamp as ContactClampDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class ContactClamp(ContactClampDP):
    """Contact clamp whose record and edges are buffers updated in place.

    The clamp acts in training as in evaluation, so a training batch is read
    as the deployed model reads it: the contacts of the fewer frames below an
    edge are read through the clamp. Every batch is observed before it is
    evaluated, and a graph that reads the edges sees each update.
    """

    @torch.no_grad()
    def observe(
        self,
        distance: torch.Tensor,
        center_type: torch.Tensor,
        neighbor_type: torch.Tensor,
        mask: torch.Tensor,
        frame: torch.Tensor,
    ) -> None:
        contacts = self.observed(
            distance.detach(), center_type, neighbor_type, mask, frame
        )
        self.adopt(contacts)

    @torch.no_grad()
    def adopt(self, contacts: torch.Tensor) -> None:
        """Take over a contact record and the edges it sets.

        Parameters
        ----------
        contacts : torch.Tensor
            Record with shape ``(T + 1, T + 1, K)`` in Å.
        """
        self.contacts.copy_(contacts.to(dtype=self.contacts.dtype))
        self.edges.copy_(self.edges_of(self.contacts).to(dtype=self.edges.dtype))

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        # A state written without a clamp, such as an unbridged model that a
        # bridged one is fine-tuned from, records no contacts; the clamp then
        # keeps the empty record and the covalent edges it starts from.
        for name in ("edges", "contacts"):
            state_dict.setdefault(prefix + name, getattr(self, name).detach().clone())
        # A record of another support, such as that of a model fine-tuned with
        # a different ``bridging_support``, keeps its smallest contacts: it is
        # cut to the K smallest or padded with unfilled slots, and the edges
        # follow from it.
        contacts = state_dict[prefix + "contacts"]
        if contacts.shape[-1] != self.support:
            kept = contacts[..., : self.support]
            missing = self.support - kept.shape[-1]
            if missing > 0:
                kept = torch.cat(
                    [kept, kept.new_full((*kept.shape[:-1], missing), float("inf"))],
                    dim=-1,
                )
            state_dict[prefix + "contacts"] = kept
            state_dict[prefix + "edges"] = self.edges_of(kept.to(torch.float64)).to(
                dtype=self.edges.dtype
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


register_dpmodel_mapping(
    ContactClampDP,
    lambda v: ContactClamp.deserialize(v.serialize()),
)


@torch.no_grad()
def synchronize_contact_edges(module: torch.nn.Module) -> None:
    """Merge the contact records of every contact clamp of a module over all ranks.

    Each rank observes its own batches, so after the merge every rank holds
    the record, and hence the edges, of all of them. Every rank calls this at
    the same point of every training step, whichever task it trained.

    Parameters
    ----------
    module : torch.nn.Module
        The trained module, possibly wrapped for distributed training.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    clamps = [m for m in module.modules() if isinstance(m, ContactClamp)]
    if not clamps:
        return
    flat = torch.cat([clamp.contacts.reshape(-1) for clamp in clamps])
    # Infinite entries mark unfilled slots; the all-gather carries them as is.
    world_size = dist.get_world_size()
    # The collective concatenates the inputs of the ranks into one flat output.
    gathered = flat.new_empty(world_size * flat.numel())
    dist.all_gather_into_tensor(gathered, flat)
    gathered = gathered.reshape(world_size, flat.numel())
    offset = 0
    for clamp in clamps:
        size = clamp.contacts.numel()
        records = gathered[:, offset : offset + size].reshape(-1, *clamp.contacts.shape)
        clamp.adopt(clamp.merged(records))
        offset += size
