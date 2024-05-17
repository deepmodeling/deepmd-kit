# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import numpy as np

from .make_base_descriptor import (
    make_base_descriptor,
)

BaseDescriptor = make_base_descriptor(np.ndarray, "call")
