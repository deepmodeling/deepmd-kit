# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common test case."""

from abc import (
    ABC,
    abstractmethod,
)


class BackendTestCase(ABC):
    """Backend test case."""

    module: object
    """Module to test."""

    @property
    @abstractmethod
    def modules_to_test(self) -> list:
        pass

    @abstractmethod
    def forward_wrapper(self, x):
        pass

    @abstractmethod
    def forward_wrapper_cpu_ref(self, module):
        pass

    @classmethod
    @abstractmethod
    def convert_to_numpy(cls, xx):
        pass

    @classmethod
    @abstractmethod
    def convert_from_numpy(cls, xx):
        pass
