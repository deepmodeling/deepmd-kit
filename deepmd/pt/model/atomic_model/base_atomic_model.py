# SPDX-License-Identifier: LGPL-3.0-or-later


import torch

from deepmd.dpmodel.atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel_ = make_base_atomic_model(torch.Tensor)


class BaseAtomicModel(BaseAtomicModel_):
    # export public methods that are not abstract
    get_nsel = torch.jit.export(BaseAtomicModel_.get_nsel)
    get_nnei = torch.jit.export(BaseAtomicModel_.get_nnei)
