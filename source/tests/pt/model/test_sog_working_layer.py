# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

try:
    import pytorch_finufft  # noqa: F401

    HAS_FINUFFT = True
except Exception:
    HAS_FINUFFT = False

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.sog_model import (
    SOGEnergyModel,
)
from deepmd.pt.model.task.sog_energy_fitting import (
    SOGEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


dtype = env.GLOBAL_PT_FLOAT_PRECISION


def _reduce_extended_tensor(extended_tensor: torch.Tensor, mapping: torch.Tensor, nloc: int):
    nframes = extended_tensor.shape[0]
    ext_dims = extended_tensor.shape[2:]
    reduced_tensor = torch.zeros(
        [nframes, nloc, *ext_dims],
        dtype=extended_tensor.dtype,
        device=extended_tensor.device,
    )
    mldims = list(mapping.shape)
    mapping_exp = mapping.view(mldims + [1] * len(ext_dims)).expand(
        [-1] * len(mldims) + list(ext_dims)
    )
    reduced_tensor = torch.scatter_reduce(
        reduced_tensor,
        1,
        index=mapping_exp,
        src=extended_tensor,
        reduce="sum",
    )
    return reduced_tensor


@unittest.skipIf(not HAS_FINUFFT, "pytorch_finufft is required for SOG tests")
class TestSOGWorkingLayer(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.nf = 2
        self.nloc = 4
        self.nt = 2
        self.rcut = 4.0
        self.rcut_smth = 3.5
        self.sel = [8, 8]

        self.descriptor = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        self.fitting = SOGEnergyFittingNet(
            var_name="energy",
            ntypes=self.nt,
            dim_descrpt=self.descriptor.get_dim_out(),
            dim_out_sr=1,
            dim_out_lr=1,
            mixed_types=self.descriptor.mixed_types(),
            n_dl=2,
        ).to(env.DEVICE)
        self.model = SOGEnergyModel(
            descriptor=self.descriptor,
            fitting=self.fitting,
            type_map=["A", "B"],
        ).to(env.DEVICE)

        coord = torch.rand(
            (self.nf, self.nloc, 3),
            dtype=dtype,
            device=env.DEVICE,
        )
        cell = torch.eye(3, dtype=dtype, device=env.DEVICE).unsqueeze(0).repeat(self.nf, 1, 1)
        cell = cell * 5.0
        self.coord = coord.reshape(self.nf, self.nloc * 3)
        self.cell = cell.reshape(self.nf, 9)
        self.atype = torch.tensor(
            [[0, 0, 1, 1], [1, 0, 1, 0]],
            dtype=torch.int64,
            device=env.DEVICE,
        )

    def test_frame_correction_applies_once_per_frame(self) -> None:
        coord3 = self.coord.view(self.nf, self.nloc, 3)
        cell33 = self.cell.view(self.nf, 3, 3)
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord3,
            self.atype,
            self.model.get_rcut(),
            self.model.get_sel(),
            mixed_types=True,
            box=cell33,
        )

        lower_ret = self.model.forward_common_lower(
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            do_atomic_virial=False,
            comm_dict={"box": cell33},
        )

        frame_corr = self.model._compute_sog_frame_correction(
            extended_coord[:, : self.nloc, :],
            lower_ret["latent_charge"],
            cell33,
        ).to(lower_ret["energy_redu"].dtype)
        expected_energy_redu = lower_ret["energy"].sum(dim=1) + frame_corr

        torch.testing.assert_close(
            lower_ret["energy_redu"],
            expected_energy_redu,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_forward_and_forward_lower_consistency(self) -> None:
        fw = self.model.forward(
            self.coord,
            self.atype,
            box=self.cell,
            do_atomic_virial=True,
        )

        coord3 = self.coord.view(self.nf, self.nloc, 3)
        cell33 = self.cell.view(self.nf, 3, 3)
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord3,
            self.atype,
            self.model.get_rcut(),
            self.model.get_sel(),
            mixed_types=True,
            box=cell33,
        )

        fw_lower = self.model.forward_lower(
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            do_atomic_virial=True,
            comm_dict={"box": cell33},
        )

        torch.testing.assert_close(fw_lower["energy"], fw["energy"], rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(fw_lower["virial"], fw["virial"], rtol=1e-7, atol=1e-7)

        reduced_force = _reduce_extended_tensor(fw_lower["extended_force"], mapping, self.nloc)
        torch.testing.assert_close(reduced_force, fw["force"], rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
