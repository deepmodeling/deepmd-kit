# SPDX-License-Identifier: LGPL-3.0-or-later


import torch

from deepmd.dpmodel.model import (
    make_base_atomic_model,
)

BaseAtomicModel_ = make_base_atomic_model(torch.Tensor)


class BaseAtomicModel(BaseAtomicModel_):
    # export public methods that are not abstract
    get_nsel = torch.jit.export(BaseAtomicModel_.get_nsel)
    get_nnei = torch.jit.export(BaseAtomicModel_.get_nnei)
    get_dim_fparam = torch.jit.export(BaseAtomicModel_.get_dim_fparam)
    get_dim_aparam = torch.jit.export(BaseAtomicModel_.get_dim_aparam)
    get_sel_type = torch.jit.export(BaseAtomicModel_.get_sel_type)
    get_numb_dos = torch.jit.export(BaseAtomicModel_.get_numb_dos)
    get_has_efield = torch.jit.export(BaseAtomicModel_.get_has_efield)
    get_ntypes_spin = torch.jit.export(BaseAtomicModel_.get_ntypes_spin)
