# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from .make_base_descriptor import (
    make_base_descriptor,
)

if TYPE_CHECKING:
    # For type checking, define BaseDescriptor as Any to avoid type errors
    BaseDescriptor: Any = object
else:
    BaseDescriptor = make_base_descriptor(np.ndarray, "call")
