# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class-level defaults for the ``set_davg_zero`` / ``set_stddev_constant``
stat-behavior flags on ``DescriptorBlock`` (issue #5897): stat machinery
must be able to read these flags on any block without a ``getattr`` probe.
"""

from deepmd.dpmodel.descriptor.descriptor import (
    DescriptorBlock,
)


def test_block_stat_flags_have_class_defaults() -> None:
    """Every DescriptorBlock answers the stat-behavior flags without a
    getattr probe: class-level defaults False, blocks override in __init__.
    """
    assert DescriptorBlock.set_davg_zero is False
    assert DescriptorBlock.set_stddev_constant is False


def test_block_stat_flags_override_branch() -> None:
    """A block constructed with ``set_davg_zero=True`` shadows the class
    default with an instance attribute; a block constructed with the
    default arguments keeps reading the class default (False).
    """
    from deepmd.dpmodel.descriptor.dpa1 import (
        DescrptBlockSeAtten,
    )

    blk_default = DescrptBlockSeAtten(
        rcut=4.0,
        rcut_smth=0.5,
        sel=[6, 6],
        ntypes=2,
    )
    assert blk_default.set_davg_zero is False

    blk_override = DescrptBlockSeAtten(
        rcut=4.0,
        rcut_smth=0.5,
        sel=[6, 6],
        ntypes=2,
        set_davg_zero=True,
    )
    assert blk_override.set_davg_zero is True
