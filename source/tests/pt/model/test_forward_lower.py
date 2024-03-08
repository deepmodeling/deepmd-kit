# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.infer.deep_eval import (
    eval_model,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from .test_permutation import (  # model_dpau,
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
    model_spin,
    model_zbl,
)

dtype = torch.float64


def reduce_tensor(extended_tensor, mapping, nloc: int):
    nframes, nall = extended_tensor.shape[:2]
    ext_dims = extended_tensor.shape[2:]
    reduced_tensor = torch.zeros(
        [nframes, nloc, *ext_dims],
        dtype=extended_tensor.dtype,
        device=extended_tensor.device,
    )
    mldims = list(mapping.shape)
    mapping = mapping.view(mldims + [1] * len(ext_dims)).expand(
        [-1] * len(mldims) + list(ext_dims)
    )
    # nf x nloc x (*ext_dims)
    reduced_tensor = torch.scatter_reduce(
        reduced_tensor,
        1,
        index=mapping,
        src=extended_tensor,
        reduce="sum",
    )
    return reduced_tensor


class ForwardLowerTest:
    def test(
        self,
    ):
        prec = self.prec
        natoms = 5
        cell = 4.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        coord = 3.0 * torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        spin = 0.5 * torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=env.DEVICE)
        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag"]

        result_forward = eval_model(
            self.model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spins=spin.unsqueeze(0),
        )
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            self.model.get_rcut(),
            self.model.get_sel(),
            mixed_types=self.model.mixed_types(),
            box=cell.unsqueeze(0),
        )
        extended_spin = torch.gather(
            spin.unsqueeze(0), index=mapping.unsqueeze(-1).tile((1, 1, 3)), dim=1
        )
        input_dict = {
            "extended_coord": extended_coord,
            "extended_atype": extended_atype,
            "nlist": nlist,
            "mapping": mapping,
            "do_atomic_virial": False,
        }
        if test_spin:
            input_dict["extended_spin"] = extended_spin
        result_forward_lower = self.model.forward_lower(**input_dict)
        for key in test_keys:
            if key in ["energy"]:
                torch.testing.assert_close(
                    result_forward_lower[key], result_forward[key], rtol=prec, atol=prec
                )
            elif key in ["force", "force_mag"]:
                reduced_vv = reduce_tensor(
                    result_forward_lower[f"extended_{key}"], mapping, natoms
                )
                torch.testing.assert_close(
                    reduced_vv, result_forward[key], rtol=prec, atol=prec
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    torch.testing.assert_close(
                        result_forward_lower[key],
                        result_forward[key],
                        rtol=prec,
                        atol=prec,
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")


class TestEnergyModelSeA(unittest.TestCase, ForwardLowerTest):
    def setUp(self):
        self.prec = 1e-10
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, ForwardLowerTest):
    def setUp(self):
        self.prec = 1e-10
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, ForwardLowerTest):
    def setUp(self):
        self.prec = 1e-10
        model_params_sample = copy.deepcopy(model_dpa2)
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBL(unittest.TestCase, ForwardLowerTest):
    def setUp(self):
        self.prec = 1e-10
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeA(unittest.TestCase, ForwardLowerTest):
    def setUp(self):
        # still need to figure out why only 1e-5 rtol and atol
        self.prec = 1e-5
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
