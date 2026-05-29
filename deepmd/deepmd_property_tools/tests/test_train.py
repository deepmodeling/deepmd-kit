# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import pytest
from deepmd_property_tools import (
    PropertyTrain,
)


def test_property_train_rejects_unknown_arguments() -> None:
    with pytest.raises(TypeError, match="Unexpected PropertyTrain argument"):
        PropertyTrain(unknown_option=True)


def test_epochs_to_steps() -> None:
    assert PropertyTrain._epochs_to_steps(None) == 1000000
    assert PropertyTrain._epochs_to_steps(2) == 2000
    assert PropertyTrain._epochs_to_steps(0) == 1000
