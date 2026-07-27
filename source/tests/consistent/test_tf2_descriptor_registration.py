# SPDX-License-Identifier: LGPL-3.0-or-later
"""Every TF2 descriptor wrapper must register its config type names.

TF2 standard-model construction resolves descriptors through
``BaseDescriptor.get_class_by_type(<type>)``.  The wrapper classes exist and are
exported by ``deepmd.tf2.descriptor``, but several used to be defined without the
``@BaseDescriptor.register(...)`` decorators the JAX wrappers carry, so their
config type names could not be resolved and model construction failed with an
unknown-descriptor error.
"""

import unittest

from .common import (
    INSTALLED_TF2,
)

if INSTALLED_TF2:
    import deepmd.tf2.descriptor  # noqa: F401
    from deepmd.tf2.descriptor.base_descriptor import (
        BaseDescriptor,
    )

# type names that must resolve on the TF2 descriptor registry, mirroring the
# JAX wrapper registrations for the same descriptors.
TF2_DESCRIPTOR_TYPES = [
    "se_e2_a",
    "se_a",
    "se_e2_r",
    "se_r",
    "se_e3",  # se_t
    "se_at",  # se_t
    "se_a_3be",  # se_t
    "se_e3_tebd",  # se_t_tebd
    "se_atten_v2",
    "se_atten",  # dpa1
    "dpa1",
    "dpa2",
    "dpa3",
    "hybrid",
]


@unittest.skipUnless(INSTALLED_TF2, "TF2 backend is not installed")
class TestTF2DescriptorRegistration(unittest.TestCase):
    def test_all_types_resolve(self) -> None:
        for descriptor_type in TF2_DESCRIPTOR_TYPES:
            with self.subTest(descriptor_type=descriptor_type):
                cls = BaseDescriptor.get_class_by_type(descriptor_type)
                self.assertTrue(callable(cls))


if __name__ == "__main__":
    unittest.main()
