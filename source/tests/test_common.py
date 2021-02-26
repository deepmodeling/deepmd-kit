import os,sys,shutil,fnmatch
import numpy as np
import unittest
from pathlib import Path

from deepmd.common import expand_sys_str

# compute relative path
# https://stackoverflow.com/questions/38083555/using-pathlibs-relative-to-for-directories-on-the-same-level
def relpath(path_to, path_from):
    path_to = Path(path_to).resolve()
    path_from = Path(path_from).resolve()
    try:
        for p in (*reversed(path_from.parents), path_from):
            head, tail = p, path_to.relative_to(p)
    except ValueError:  # Stop when the paths diverge.
        pass
    return Path('../' * (len(path_from.parents) - len(head.parents))).joinpath(tail)

class TestCommonExpandSysDir(unittest.TestCase) :
    def setUp(self):
        self.match_file = Path('type.raw')
        Path('test_sys').mkdir()
        self.dir = Path('test_sys')
        self.dira = Path('test_sys/a')
        self.dirb = Path('test_sys/a/b')
        self.dirc = Path('test_sys/c')
        self.dird = Path('test_sys/c/d')
        self.dire = Path('test_sys/c/type.raw')
        self.dira.mkdir()
        self.dirb.mkdir()
        self.dirc.mkdir()
        for ii in [self.dir, self.dira, self.dirb]:
            (ii/self.match_file).touch()
        relb = relpath(self.dirb, self.dirc)
        absb = self.dirb.resolve()
        self.dird.symlink_to(relb)
        self.dire.symlink_to(absb)
        self.expected_out = ['test_sys', 'test_sys/a', 'test_sys/a/b', 'test_sys/c/d', 'test_sys/c/type.raw']
        self.expected_out.sort()

    def tearDown(self):
        shutil.rmtree('test_sys')

    def test_expand(self):
        ret = expand_sys_str('test_sys')
        ret.sort()
        self.assertEqual(ret, self.expected_out)
