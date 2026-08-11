# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.fitting.make_base_fitting import (
    make_base_fitting,
)


class MinimalFitting(make_base_fitting(np.ndarray)):
    """Smallest concrete fitting: exercises the base-declared defaults."""

    def output_def(self):
        raise NotImplementedError

    def fwd(self, *args, **kwargs):
        raise NotImplementedError

    def get_type_map(self):
        return []

    def change_type_map(self, type_map, model_with_new_type_stat=None):
        raise NotImplementedError

    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data):
        return cls()


def test_reinit_exclude_default_noop_on_empty() -> None:
    MinimalFitting().reinit_exclude([])


def test_reinit_exclude_default_raises_on_nonempty() -> None:
    with pytest.raises(NotImplementedError):
        MinimalFitting().reinit_exclude([0])
