# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.modifier.base_modifier import (
    make_base_modifier,
)
from deepmd.tf.infer import (
    DeepPot,
)


class BaseModifier(DeepPot, make_base_modifier()):
    def __init__(self, *args, **kwargs) -> None:
        """Construct a basic model for different tasks."""
        DeepPot.__init__(self, *args, **kwargs)
