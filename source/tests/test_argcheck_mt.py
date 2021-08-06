import os,sys,shutil,copy
import numpy as np
import unittest

from deepmd.utils.argcheck_mt import normalize_mt
from deepmd.utils.compat import updata_deepmd_input
from common import j_loader

class TestArgcheckMt (unittest.TestCase) :
    def test_argcheck_mt (self) :
        jdata = j_loader('input_origin.json')
        jdata = updata_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
        jdata = normalize_mt(jdata)
        jdata1 = j_loader('input_correct.json')
        self.assertEqual(jdata,jdata1)
