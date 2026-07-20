# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4/SeZM native-spin model (NeighborGraph route, autograd force_mag)."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.model.dpa4_native_spin_model import (
    DPA4NativeSpinModel as DPA4NativeSpinModelDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class DPA4NativeSpinModel(DPA4NativeSpinModelDP):
    """pt_expt native-spin DPA4/SeZM model.

    Mirrors :class:`deepmd.dpmodel.model.dpa4_native_spin_model.DPA4NativeSpinModel`
    (construction, delegation, output defs, (de)serialization are all
    inherited unchanged), but overrides :meth:`forward`: the pt_expt
    backbone's ``call_common`` (Task 3 of the "DPA4 native spin on the
    NeighborGraph route" plan) produces REAL autograd
    ``energy_derv_r``/``energy_derv_r_mag``/``energy_derv_c_redu`` tensors,
    unlike the dpmodel parent's energy-only ``call`` (which is restricted to
    ``force``/``force_mag``/``virial`` as ``None`` placeholders because
    dpmodel has no autograd).
    """

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the wrapped model.

        In torch.nn.Module, submodules (e.g. ``backbone_model``) are stored
        in ``_modules``, not ``__dict__``. The dpmodel parent's
        ``__getattr__`` guards its ``backbone_model`` delegation with
        ``"backbone_model" not in self.__dict__``, which is always true here
        and would incorrectly raise ``AttributeError`` for every submodule
        access (including ``self.backbone_model`` itself). Mirrors
        :class:`deepmd.pt_expt.model.spin_model.SpinModel`'s override: try
        ``torch.nn.Module``'s own ``__getattr__`` (checks ``_parameters``,
        ``_buffers``, ``_modules``) first, then fall back to
        ``backbone_model`` delegation for arbitrary attributes.
        """
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # backbone_model is in _modules, access via _modules directly to
        # avoid re-entering __getattr__.
        modules = self.__dict__.get("_modules", {})
        backbone = modules.get("backbone_model")
        if backbone is not None:
            return getattr(backbone, name)
        raise AttributeError(name)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return native-spin model predictions with public output keys.

        Parameters
        ----------
        coord
            The coordinates of the atoms. shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
        spin
            The per-local-atom spin. shape: nf x (nloc x 3)
        box
            The simulation box. shape: nf x 9
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            If calculate the atomic virial.

        Returns
        -------
        ret_dict
            The result dict with keys ``atom_energy``, ``energy``,
            ``force``, ``force_mag``, ``virial``, ``mask_mag``, and
            (when ``do_atomic_virial``) ``atom_virial``. ``force`` and
            ``force_mag`` are real autograd tensors (``-dE/dcoord`` and
            ``-dE/dspin``), NOT placeholders.
        """
        # ``spin=`` rides the NeighborGraph lower only; ``neighbor_graph_method``
        # is left at its default (None) so pt_expt's own default-flip resolves
        # it to the carry-all graph builder for this (DPA4) descriptor.
        model_ret = self.backbone_model.call_common(
            coord,
            atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            spin=spin,
        )
        out: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "force": model_ret["energy_derv_r"].squeeze(-2),
            "force_mag": model_ret["energy_derv_r_mag"].squeeze(-2),
            "virial": model_ret["energy_derv_c_redu"].squeeze(-2),
            # Non-magnetic atoms already carry an exactly-zero magnetic force:
            # the descriptor gates the spin embedding by type, so the
            # autograd gradient w.r.t. their (inert) spin input is zero by
            # construction -- mirrors pt's ``SeZMNativeSpinModel.forward``
            # docstring, which asserts the same and does NOT re-mask the
            # force. Re-masking here would be a defensive backstop the
            # project's design principles forbid.
            "mask_mag": (self.spin_mask[atype] > 0).unsqueeze(-1),
        }
        if do_atomic_virial:
            out["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return out
