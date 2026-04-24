# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections import (
    defaultdict,
)
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)
from deepmd.pt.utils import (
    env,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, PropertyFittingNet):
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_compute_stats_distinguish_types(self) -> bool:
        """Get whether the fitting net computes stats which are not distinguished between different types of atoms."""
        return False

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.fitting_net.get_intensive()

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the stat to each atomic output.
        In property fitting, each output will be multiplied by label std and then plus the label average value.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc. It is useless in property fitting.

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            ret[kk] = ret[kk] * out_std[kk][0] + out_bias[kk][0]
        return ret


class DPXASAtomicModel(DPPropertyAtomicModel):
    """Atomic model for XAS spectrum fitting.

    Extends :class:`DPPropertyAtomicModel` with per-(absorbing_type, edge)
    statistics buffers: ``xas_e_ref`` [ntypes, nfparam, 2],
    ``xas_intensity_ref`` and ``xas_intensity_std`` [ntypes, nfparam, n_pts].

    These buffers are computed by :meth:`compute_or_load_out_stat` (called via
    the standard :meth:`compute_or_load_stat` pipeline before training starts)
    and saved in the checkpoint so that absolute edge energies and intensity
    scales are available at inference time.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        super().__init__(descriptor, fitting, type_map, **kwargs)
        nfparam: int = getattr(fitting, "numb_fparam", 0)
        if nfparam > 0:
            ntypes: int = len(type_map)
            n_pts: int = max(getattr(fitting, "dim_out", 2) - 2, 0)
            self.register_buffer(
                "xas_e_ref",
                torch.zeros(ntypes, nfparam, 2, dtype=torch.float64),
            )
            # maps edge_idx (argmax of fparam one-hot) → absorbing atom type index
            self.register_buffer(
                "xas_edge_to_seltype",
                torch.zeros(nfparam, dtype=torch.long),
            )
            # per-(type, edge, point) intensity statistics for inference denormalisation
            self.register_buffer(
                "xas_intensity_ref",
                torch.zeros(ntypes, nfparam, n_pts, dtype=torch.float64),
            )
            self.register_buffer(
                "xas_intensity_std",
                torch.ones(ntypes, nfparam, n_pts, dtype=torch.float64),
            )
        else:
            self.xas_e_ref: torch.Tensor | None = None
            self.xas_edge_to_seltype: torch.Tensor | None = None
            self.xas_intensity_ref: torch.Tensor | None = None
            self.xas_intensity_std: torch.Tensor | None = None

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: Any = None,
    ) -> None:
        """Compute per-(absorbing_type, edge) statistics from training data.

        Populates ``xas_e_ref``, ``xas_intensity_ref``, ``xas_intensity_std``,
        and sets ``out_bias``/``out_std`` so the NN trains in a normalised space.
        Falls back to the parent implementation when ``nfparam == 0``.
        """
        if self.xas_e_ref is None:
            super().compute_or_load_out_stat(merged, stat_file_path)
            return

        sampled = merged() if callable(merged) else merged

        nfparam: int = self.xas_e_ref.shape[1]
        ntypes: int = self.xas_e_ref.shape[0]
        n_pts: int = self.xas_intensity_ref.shape[2]
        task_dim: int = 2 + n_pts
        var_name: str = self.bias_keys[0]

        accum: dict[tuple[int, int], list] = defaultdict(list)
        for frame in sampled:
            if (
                var_name not in frame
                or "sel_type" not in frame
                or "fparam" not in frame
            ):
                continue
            xas = frame[var_name].reshape(-1, task_dim)
            sel_type = frame["sel_type"].reshape(-1).long()
            edge_idx = frame["fparam"].reshape(-1, nfparam).argmax(dim=-1)
            for i in range(xas.shape[0]):
                t = int(sel_type[i].item())
                e = int(edge_idx[i].item())
                if 0 <= t < ntypes and 0 <= e < nfparam:
                    accum[(t, e)].append(xas[i].detach().cpu().numpy())

        if not accum:
            log.warning(
                "DPXASAtomicModel.compute_or_load_out_stat: no XAS frames found; "
                "stats remain at defaults. Training may be unstable."
            )
            return

        e_ref = torch.zeros(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        e_std = torch.ones(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        intensity_ref = torch.zeros(
            ntypes, nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        intensity_std = torch.ones(
            ntypes, nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )

        for (t, e), vals in accum.items():
            arr = np.array(vals)  # [n, task_dim]
            e_ref[t, e] = torch.tensor(
                np.mean(arr[:, :2], axis=0), dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
            e_std[t, e] = torch.tensor(
                np.std(arr[:, :2], axis=0).clip(min=1.0),
                dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            )
            if n_pts > 0:
                intensity_ref[t, e] = torch.tensor(
                    np.mean(arr[:, 2:], axis=0), dtype=env.GLOBAL_PT_FLOAT_PRECISION
                )
                intensity_std[t, e] = torch.tensor(
                    np.std(arr[:, 2:], axis=0).clip(min=1e-6),
                    dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                )
            log.info(
                f"DPXASAtomicModel stats: type={t}, edge={e} | "
                f"E_ref=[{float(e_ref[t,e,0]):.2f}, {float(e_ref[t,e,1]):.2f}] eV | "
                f"n={len(vals)}"
            )

        self.xas_e_ref.copy_(e_ref.to(self.xas_e_ref.dtype))
        self.xas_intensity_ref.copy_(intensity_ref.to(self.xas_intensity_ref.dtype))
        self.xas_intensity_std.copy_(intensity_std.to(self.xas_intensity_std.dtype))

        # Legacy fallback mapping used by XASModel.forward when sel_type is not provided.
        if self.xas_edge_to_seltype is not None:
            mapping = torch.zeros(
                nfparam, dtype=torch.long, device=self.xas_edge_to_seltype.device
            )
            for t, e in accum.keys():
                mapping[e] = t
            self.xas_edge_to_seltype.copy_(mapping)

        key_idx = self.bias_keys.index(var_name)
        populated = e_std.abs().gt(1.0)
        e_std_global = (
            e_std[populated].mean(dim=0)
            if populated.any()
            else torch.ones(2, dtype=e_std.dtype)
        )
        with torch.no_grad():
            self.out_bias[key_idx, :, :2] = 0.0
            self.out_std[key_idx, :, :2] = e_std_global.to(self.out_std.dtype)
            if n_pts > 0:
                self.out_bias[key_idx, :, 2:] = 0.0
                self.out_std[key_idx, :, 2:] = 1.0

        log.info(
            f"DPXASAtomicModel: stats computed for {len(accum)} (type, edge) groups. "
            f"out_std[:2]={e_std_global.tolist()} eV."
        )
