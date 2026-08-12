# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class-level defaults for the ``set_davg_zero`` / ``set_stddev_constant``
stat-behavior flags on ``DescriptorBlock`` (issue #5897): stat machinery
must be able to read these flags on any block without a ``getattr`` probe.
"""

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.dpmodel.descriptor.make_base_descriptor import (
    make_base_descriptor,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    merge_env_stat,
)


def test_block_stat_flags_have_class_defaults() -> None:
    """Every DescriptorBlock answers the stat-behavior flags without a
    getattr probe: class-level defaults False, blocks override in __init__.
    """
    assert DescriptorBlock.set_davg_zero is False
    assert DescriptorBlock.set_stddev_constant is False


def test_base_descriptor_stat_flags_have_class_defaults() -> None:
    """The ``BD`` base in ``make_base_descriptor`` (the ``Descriptor``-side
    twin of ``DescriptorBlock`` above) carries the same concrete class
    defaults, so ``merge_env_stat`` -- which accepts either a ``Descriptor``
    or a ``DescriptorBlock`` as ``base_obj`` -- can read the flags on a bare
    ``Descriptor`` without a ``getattr`` probe.
    """
    bd = make_base_descriptor(np.ndarray, "call")
    assert bd.set_davg_zero is False
    assert bd.set_stddev_constant is False


def _sample() -> dict:
    rng = np.random.default_rng(0)
    nf, nloc = 2, 6
    coord = rng.normal(size=(nf, nloc, 3)) * 2.0
    atype = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.int64)
    box = np.tile((np.eye(3) * 12.0).reshape(1, 9), (nf, 1))
    return {"coord": coord, "atype": atype, "box": box}


def test_merge_env_stat_on_bare_descriptor_no_attribute_error() -> None:
    """``merge_env_stat`` reads ``base_obj.set_davg_zero`` /
    ``set_stddev_constant`` unconditionally (no ``getattr`` probe). Pin that
    this does not raise ``AttributeError`` when ``base_obj`` is a bare
    se-family ``Descriptor`` (not a ``DescriptorBlock``) which never sets
    those flags itself and instead relies on the ``BD`` base's class
    defaults.
    """
    base = DescrptSeA(6.0, 0.5, [10, 10])
    link = DescrptSeA(6.0, 0.5, [10, 10])
    sample = _sample()
    base.compute_input_stats([sample])
    link.compute_input_stats([sample])
    # Would raise AttributeError before the BD-base class defaults existed
    # if a descriptor never assigned instance attributes for these flags.
    merge_env_stat(base, link)


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
