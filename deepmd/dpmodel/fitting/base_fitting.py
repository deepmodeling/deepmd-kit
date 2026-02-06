# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from .make_base_fitting import (
    make_base_fitting,
)

if TYPE_CHECKING:
    # For type checking, define BaseFitting as Any to avoid type errors
    BaseFitting: Any = object
else:
    BaseFitting = make_base_fitting(np.ndarray, fwd_method_name="call")
