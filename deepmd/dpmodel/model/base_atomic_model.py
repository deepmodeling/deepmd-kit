# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from .make_base_atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel = make_base_atomic_model(np.ndarray)
