# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.make_base_descriptor import (
    make_base_descriptor,
)

# no type annotations standard in array api
BaseDescriptor = make_base_descriptor(Any)
