# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils import (
    decomp,
    env,
)
from deepmd.pd.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
)
from .test_permutation import (  # model_dpau,
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
    model_spin,
    model_zbl,
)

dtype = paddle.float64


def reduce_tensor(extended_tensor, mapping, nloc: int):
    nframes, nall = extended_tensor.shape[:2]
    ext_dims = extended_tensor.shape[2:]
    reduced_tensor = paddle.zeros(
        [nframes, nloc, *ext_dims],
        dtype=extended_tensor.dtype,
    ).to(device=extended_tensor.place)
    mldims = list(mapping.shape)
    mapping = mapping.reshape(mldims + [1] * len(ext_dims)).expand(
        [-1] * len(mldims) + list(ext_dims)
    )
    # nf x nloc x (*ext_dims)
    reduced_tensor = decomp.scatter_reduce(
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
        cell = 4.0 * paddle.eye(3, dtype=dtype).to(device=env.DEVICE)
        generator = paddle.seed(GLOBAL_SEED)
        coord = 3.0 * paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        spin = 0.5 * paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        atype = paddle.to_tensor([0, 0, 0, 1, 1], dtype=paddle.int64).to(
            device=env.DEVICE
        )
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
            self.model.get_rcut() + 1.0
            if test_spin
            else self.model.get_rcut(),  # buffer region for spin nlist
            self.model.get_sel(),
            mixed_types=self.model.mixed_types(),
            box=cell.unsqueeze(0),
        )
        extended_spin = paddle.take_along_axis(
            spin.unsqueeze(0), indices=mapping.unsqueeze(-1).tile((1, 1, 3)), axis=1
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
                np.testing.assert_allclose(
                    result_forward_lower[key].numpy(),
                    result_forward[key].numpy(),
                    rtol=prec,
                    atol=prec,
                )
            elif key in ["force", "force_mag"]:
                reduced_vv = reduce_tensor(
                    result_forward_lower[f"extended_{key}"], mapping, natoms
                )
                np.testing.assert_allclose(
                    reduced_vv.numpy(),
                    result_forward[key].numpy(),
                    rtol=prec,
                    atol=prec,
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    np.testing.assert_allclose(
                        result_forward_lower[key].numpy(),
                        result_forward[key].numpy(),
                        rtol=prec,
                        atol=prec,
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")


class TestEnergyModelSeA(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_se_e2_a)
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_dpa1)
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_dpa2)
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelZBL(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_zbl)
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelSpinSeA(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_spin)
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelSpinDPA1(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_dpa1)["descriptor"]
        # double sel for virtual atoms to avoid large error
        model_params["descriptor"]["sel"] *= 2
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelSpinDPA2(unittest.TestCase, ForwardLowerTest):
    def setUp(self) -> None:
        self.prec = 1e-10
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_dpa2)["descriptor"]
        # double sel for virtual atoms to avoid large error
        model_params["descriptor"]["repinit"]["nsel"] *= 2
        model_params["descriptor"]["repformer"]["nsel"] *= 2
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
