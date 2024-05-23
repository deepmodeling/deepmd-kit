# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.common import (
    NativeOP,
)

from ..common.backend import (
    BackendTestCase,
)


class DPTestCase(BackendTestCase):
    """Common test case."""

    module: NativeOP
    """DP module to test."""

    def forward_wrapper(self, x):
        return x

    @property
    def deserialized_module(self):
        return self.module.deserialize(self.module.serialize())

    @property
    def modules_to_test(self):
        modules = [
            self.module,
            self.deserialized_module,
        ]
        return modules
