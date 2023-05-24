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

p_examples = Path(__file__).parent.parent.parent / "examples"

input_files = (
    p_examples / "water" / "se_e2_a" / "input.json",
    p_examples / "water" / "se_e2_r" / "input.json",
    p_examples / "water" / "se_e3" / "input.json",
    p_examples / "water" / "se_e2_a_tebd" / "input.json",
    p_examples / "water" / "se_e2_a_mixed_prec" / "input.json",
    p_examples / "water" / "se_atten" / "input.json",
    p_examples / "water" / "hybrid" / "input.json",
    p_examples / "water" / "dplr" / "train" / "dw.json",
    p_examples / "water" / "dplr" / "train" / "ener.json",
    p_examples / "nopbc" / "train" / "input.json",
    p_examples / "water_tensor" / "dipole" / "dipole_input.json",
    p_examples / "water_tensor" / "polar" / "polar_input.json",
    p_examples / "water_multi_task" / "ener_dipole" / "input.json",
    p_examples / "fparam" / "train" / "input.json",
    p_examples / "fparam" / "train" / "input_aparam.json",
    p_examples / "zinc_protein" / "zinc_se_a_mask.json",
    p_examples / "dos" / "train" / "input.json",
    p_examples / "spin" / "se_e2_a" / "input.json",
)


class TestExamples(unittest.TestCase):
    def test_arguments(self):
        for fn in input_files:
            fn = str(fn)
            with self.subTest(fn=fn):
                jdata = j_loader(fn)
                normalize(jdata)
