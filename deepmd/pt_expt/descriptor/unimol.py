# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.unimol import DescrptUniMol as DescrptUniMolDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("unimol")
@torch_module
class DescrptUniMol(DescrptUniMolDP):
    """Uni-Mol v1 backbone on the PyTorch-Exportable backend."""

    def share_params(
        self,
        base_class: "DescrptUniMol",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share parameters with ``base_class`` for multi-task training.

        Level 0 shares the whole backbone, level 1 only the token embedding.
        There are no environment statistics to merge, so ``model_prob`` and
        ``resume`` play no part.
        """
        del model_prob, resume
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            for key in ("gbf", "gbf_proj", "encoder"):
                self._modules[key] = base_class._modules[key]
            self.embed_tokens = base_class.embed_tokens
        elif shared_level == 1:
            self.embed_tokens = base_class.embed_tokens
        else:
            raise NotImplementedError
