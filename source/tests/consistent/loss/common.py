# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)


def _to_tf2_loss_data(value: Any) -> Any:
    from deepmd.tf2.common import (
        to_tensorflow_array,
    )

    if isinstance(value, dict):
        return {kk: _to_tf2_loss_data(vv) for kk, vv in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_tf2_loss_data(vv) for vv in value)
    if isinstance(value, list):
        return [_to_tf2_loss_data(vv) for vv in value]
    if isinstance(value, np.ndarray):
        return to_tensorflow_array(value)
    return value


class LossTest:
    """Useful utilities for loss tests."""

    def eval_tf2_loss(
        self,
        tf2_obj: Any,
        predict: dict[str, Any],
        label: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        loss, more_loss = tf2_obj(
            self.learning_rate,
            self.natoms,
            _to_tf2_loss_data(predict),
            _to_tf2_loss_data(label),
            **kwargs,
        )
        loss = to_numpy_array(loss)
        more_loss = {kk: to_numpy_array(vv) for kk, vv in more_loss.items()}
        return loss, more_loss
