# SPDX-License-Identifier: LGPL-3.0-or-later

from collections.abc import (
    Iterator,
)
from contextlib import (
    contextmanager,
)
from typing import (
    Any,
)

from deepmd.jax.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.update_sel import (
    BaseUpdateSel,
)


class UpdateSel(BaseUpdateSel):
    @property
    def neighbor_stat(self) -> type[NeighborStat]:
        return NeighborStat


_MISSING = object()


def _get_update_sel_descriptors() -> tuple[type[Any], ...]:
    import deepmd.dpmodel.descriptor as _dpmodel_descriptor  # noqa: F401
    from deepmd.dpmodel.descriptor.base_descriptor import (
        BaseDescriptor,
    )

    return tuple(
        {
            descriptor_cls
            for descriptor_cls in BaseDescriptor.get_plugins().values()
            if hasattr(descriptor_cls, "_update_sel_cls")
        }
    )


@contextmanager
def use_jax_update_sel() -> Iterator[None]:
    """Use JAX neighbor statistics in dpmodel descriptor update_sel methods."""
    descriptor_classes = _get_update_sel_descriptors()
    saved_update_sel = {
        descriptor_cls: descriptor_cls.__dict__.get("_update_sel_cls", _MISSING)
        for descriptor_cls in descriptor_classes
    }
    try:
        for descriptor_cls in descriptor_classes:
            descriptor_cls._update_sel_cls = UpdateSel
        yield
    finally:
        for descriptor_cls, update_sel_cls in saved_update_sel.items():
            if update_sel_cls is _MISSING:
                del descriptor_cls._update_sel_cls
            else:
                descriptor_cls._update_sel_cls = update_sel_cls
