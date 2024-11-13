# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
)
from .test_permutation import (
    model_dpa2,
)

dtype = torch.float64


class TestUnusedParamsDPA2(unittest.TestCase):
    def test_unused(self) -> None:
        import itertools

        for conv, drrd, grrg, attn1, g1g1, attn2, h2 in itertools.product(
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
        ):
            if (not drrd) and (not grrg) and h2:
                # skip the case h2 is not involved
                continue
            if (not grrg) and (not conv):
                # skip the case g2 is not involved
                continue
            model = copy.deepcopy(model_dpa2)
            model["descriptor"]["repformer"]["nlayers"] = 2
            # model["descriptor"]["combine_grrg"] = cmbg2
            model["descriptor"]["repformer"]["update_g1_has_conv"] = conv
            model["descriptor"]["repformer"]["update_g1_has_drrd"] = drrd
            model["descriptor"]["repformer"]["update_g1_has_grrg"] = grrg
            model["descriptor"]["repformer"]["update_g1_has_attn"] = attn1
            model["descriptor"]["repformer"]["update_g2_has_g1g1"] = g1g1
            model["descriptor"]["repformer"]["update_g2_has_attn"] = attn2
            model["descriptor"]["repformer"]["update_h2"] = h2
            model["fitting_net"]["neuron"] = [12, 12, 12]
            self._test_unused(model)

    def _test_unused(self, model_params) -> None:
        self.model = get_model(model_params).to(env.DEVICE)
        natoms = 5
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE, generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        coord = torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        idx_perm = [1, 0, 4, 3, 2]
        result_0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        test_keys = ["energy", "force", "virial"]
        ret0 = {key: result_0[key].squeeze(0) for key in test_keys}

        # use computation graph to find all contributing tensors
        def get_contributing_params(y, top_level=True):
            nf = y.grad_fn.next_functions if top_level else y.next_functions
            for f, _ in nf:
                try:
                    yield f.variable
                except AttributeError:
                    pass  # node has no tensor
                if f is not None:
                    yield from get_contributing_params(f, top_level=False)

        contributing_parameters = set(get_contributing_params(ret0["energy"]))
        all_parameters = set(self.model.parameters())
        non_contributing = all_parameters - contributing_parameters
        # 2 for compression
        self.assertEqual(len(non_contributing), 2)


if __name__ == "__main__":
    unittest.main()
