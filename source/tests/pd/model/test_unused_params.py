# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils import (
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

dtype = paddle.float64


@unittest.skip("paddle do not support unpacking grad_fn.next_functions")
class TestUnusedParamsDPA2(unittest.TestCase):
    def test_unused(self):
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
                # skip the case h2 is not envolved
                continue
            if (not grrg) and (not conv):
                # skip the case g2 is not envolved
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

    def _test_unused(self, model_params):
        self.model = get_model(model_params).to(env.DEVICE)
        natoms = 5
        generator = paddle.seed(GLOBAL_SEED)
        cell = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        cell = (cell + cell.T) + 5.0 * paddle.eye(3).to(device=env.DEVICE)
        coord = paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        coord = paddle.matmul(coord, cell)
        atype = paddle.to_tensor([0, 0, 0, 1, 1]).to(env.DEVICE)
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
        self.assertEqual(len(non_contributing), 0)


if __name__ == "__main__":
    unittest.main()
