# SPDX-License-Identifier: LGPL-3.0-or-later
"""This module ensures input in the examples directory
could pass the argument checking.
"""

import unittest
from pathlib import (
    Path,
)

from deepmd.common import (
    j_loader,
)
from deepmd.utils.argcheck import (
    normalize,
)

from ..pt.test_multitask import (
    preprocess_shared_params,
)

p_examples = Path(__file__).parent.parent.parent.parent / "examples"

input_files = (
    p_examples / "water" / "se_e2_a" / "input.json",
    p_examples / "water" / "se_e2_r" / "input.json",
    p_examples / "water" / "se_e3" / "input.json",
    p_examples / "water" / "se_e2_a_tebd" / "input.json",
    p_examples / "water" / "se_e2_a_mixed_prec" / "input.json",
    p_examples / "water" / "se_atten" / "input.json",
    p_examples / "water" / "se_atten_compressible" / "input.json",
    p_examples / "water" / "se_atten_dpa1_compat" / "input.json",
    p_examples / "water" / "zbl" / "input.json",
    p_examples / "water" / "hybrid" / "input.json",
    p_examples / "water" / "dplr" / "train" / "dw.json",
    p_examples / "water" / "dplr" / "train" / "ener.json",
    p_examples / "water" / "d3" / "input_pt.json",
    p_examples / "water" / "linear" / "input.json",
    p_examples / "water" / "linear" / "input_pt.json",
    p_examples / "nopbc" / "train" / "input.json",
    p_examples / "water_tensor" / "dipole" / "dipole_input.json",
    p_examples / "water_tensor" / "polar" / "polar_input.json",
    p_examples / "water_tensor" / "dipole" / "dipole_input_torch.json",
    p_examples / "water_tensor" / "polar" / "polar_input_torch.json",
    p_examples / "fparam" / "train" / "input.json",
    p_examples / "fparam" / "train" / "input_aparam.json",
    p_examples / "zinc_protein" / "zinc_se_a_mask.json",
    p_examples / "dos" / "train" / "input.json",
    p_examples / "dos" / "train" / "input_torch.json",
    p_examples / "spin" / "se_e2_a" / "input_tf.json",
    p_examples / "spin" / "se_e2_a" / "input_torch.json",
    p_examples / "dprc" / "normal" / "input.json",
    p_examples / "dprc" / "pairwise" / "input.json",
    p_examples / "dprc" / "generalized_force" / "input.json",
    p_examples / "water" / "se_e2_a" / "input_torch.json",
    p_examples / "water" / "se_atten" / "input_torch.json",
    p_examples / "water" / "dpa2" / "input_torch_small.json",
    p_examples / "water" / "dpa2" / "input_torch_medium.json",
    p_examples / "water" / "dpa2" / "input_torch_large.json",
    p_examples / "water" / "dpa2" / "input_torch_compressible.json",
    p_examples / "water" / "dpa3" / "input_torch.json",
    p_examples / "water" / "dpa3" / "input_torch_dynamic.json",
    p_examples / "property" / "train" / "input_torch.json",
    p_examples / "water" / "se_e3_tebd" / "input_torch.json",
    p_examples / "hessian" / "single_task" / "input.json",
)

input_files_multi = (
    p_examples / "water_multi_task" / "pytorch_example" / "input_torch.json",
    p_examples / "water_multi_task" / "pytorch_example" / "input_torch_sharefit.json",
    p_examples / "hessian" / "multi_task" / "input.json",
)


class TestExamples(unittest.TestCase):
    def test_arguments(self) -> None:
        for fn in input_files + input_files_multi:
            multi_task = fn in input_files_multi
            fn = str(fn)
            with self.subTest(fn=fn):
                jdata = j_loader(fn)
                if multi_task:
                    jdata["model"], _ = preprocess_shared_params(jdata["model"])
                normalize(jdata, multi_task=multi_task)
