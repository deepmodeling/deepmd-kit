# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from .make_base_fitting import (
    make_base_fitting,
)

BaseFitting = make_base_fitting(np.ndarray, fwd_method_name="call")
