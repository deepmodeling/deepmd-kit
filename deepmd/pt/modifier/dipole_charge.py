# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import torch
from torch_admp.pme import (
    CoulombForceModule,
)
from torch_admp.utils import (
    calc_grads,
)

from deepmd.pt.model.model import (
    DipoleModel,
)
from deepmd.pt.modifier.base_modifier import (
    BaseModifier,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.serialization import (
    serialize_from_file,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)


@BaseModifier.register("dipole_charge")
class DipoleChargeModifier(BaseModifier):
    """Parameters
    ----------
    model_name
            The model file for the DeepDipole model
    model_charge_map
            Gives the amount of charge for the wfcc
    sys_charge_map
            Gives the amount of charge for the real atoms
    ewald_h
            Grid spacing of the reciprocal part of Ewald sum. Unit: A
    ewald_beta
            Splitting parameter of the Ewald sum. Unit: A^{-1}
    """

    def __new__(
        cls, *args: tuple, model_name: str | None = None, **kwargs: dict
    ) -> "DipoleChargeModifier":
        return super().__new__(cls, model_name)

    def __init__(
        self,
        model_name: str | None,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 1.0,
        ewald_batch_size: int = 5,
        dp_batch_size: int | None = None,
        model: DipoleModel | None = None,
        use_cache: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__(use_cache=use_cache)
        self.modifier_type = "dipole_charge"

        if model_name is None and model is None:
            raise AttributeError("`model_name` or `model` should be specified.")
        if model_name is not None and model is not None:
            raise AttributeError(
                "`model_name` and `model` cannot be used simultaneously."
            )

        if model is not None:
            self._model = model.to(env.DEVICE)
        if model_name is not None:
            data = serialize_from_file(model_name)
            self._model = DipoleModel.deserialize(data["model"]).to(env.DEVICE)
        self._model.eval()

        # use jit model for inference
        self.model = torch.jit.script(self._model)
        self.rcut = self.model.get_rcut()
        self.type_map = self.model.get_type_map()
        sel_type = self.model.get_sel_type()
        self.sel_type = to_torch_tensor(np.array(sel_type))
        self.model_charge_map = to_torch_tensor(np.array(model_charge_map))
        self.sys_charge_map = to_torch_tensor(np.array(sys_charge_map))
        self._model_charge_map = model_charge_map
        self._sys_charge_map = sys_charge_map

        # Validate that model_charge_map and sel_type have matching lengths
        if len(model_charge_map) != len(sel_type):
            raise ValueError(
                f"model_charge_map length ({len(model_charge_map)}) must match "
                f"sel_type length ({len(sel_type)})"
            )

        # init ewald recp
        self.ewald_h = ewald_h
        self.ewald_beta = ewald_beta
        self.er = CoulombForceModule(
            rcut=self.rcut,
            rspace=False,
            kappa=ewald_beta,
            spacing=ewald_h,
        )
        self.placeholder_pairs = torch.ones((1, 2), device=env.DEVICE, dtype=torch.long)
        self.placeholder_ds = torch.ones((1), device=env.DEVICE, dtype=torch.float64)
        self.placeholder_buffer_scales = torch.zeros(
            (1), device=env.DEVICE, dtype=torch.float64
        )

        self.ewald_batch_size = ewald_batch_size
        if dp_batch_size is None:
            dp_batch_size = int(os.environ.get("DP_INFER_BATCH_SIZE", 1))
        self.dp_batch_size = dp_batch_size

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        dd = BaseModifier.serialize(self)
        dd.update(
            {
                "model": self._model.serialize(),
                "model_charge_map": self._model_charge_map,
                "sys_charge_map": self._sys_charge_map,
                "ewald_h": self.ewald_h,
                "ewald_beta": self.ewald_beta,
                "ewald_batch_size": self.ewald_batch_size,
                "dp_batch_size": self.dp_batch_size,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "DipoleChargeModifier":
        data = data.copy()
        data.pop("@class", None)
        data.pop("type", None)
        data.pop("@version", None)
        model_obj = DipoleModel.deserialize(data.pop("model"))
        data["model"] = model_obj
        data["model_name"] = None
        return cls(**data)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute energy, force, and virial corrections for dipole-charge systems.

        This method extends the system with Wannier Function Charge Centers (WFCC)
        by adding dipole vectors to atomic coordinates for selected atom types.
        It then calculates the electrostatic interactions using Ewald reciprocal
        summation to obtain energy, force, and virial corrections.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms, 3)
        atype : torch.Tensor
            The atom types with shape (nframes, natoms)
        box : torch.Tensor | None, optional
            The simulation box with shape (nframes, 3, 3), by default None
            Note: This modifier can only be applied for periodic systems
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None
        do_atomic_virial : bool, optional
            Whether to compute atomic virial, by default False
            Note: This parameter is currently not implemented and is ignored

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the correction terms:
            - energy: Energy correction tensor with shape (nframes, 1)
            - force: Force correction tensor with shape (nframes, natoms, 3)
            - virial: Virial correction tensor with shape (nframes, 3, 3)
        """
        if box is None:
            raise RuntimeError(
                "dipole_charge data modifier can only be applied for periodic systems."
            )
        else:
            modifier_pred = {}
            nframes = coord.shape[0]
            natoms = coord.shape[1]

            input_box = box.reshape(nframes, 9)
            input_box.requires_grad_(True)

            detached_box = input_box.detach()
            sfactor = torch.matmul(
                torch.inverse(detached_box.reshape(nframes, 3, 3)),
                input_box.reshape(nframes, 3, 3),
            )
            input_coord = torch.matmul(coord, sfactor).reshape(nframes, -1)

            extended_coord, extended_charge, _atomic_dipole = self.extend_system(
                input_coord,
                atype,
                input_box,
                fparam,
                aparam,
            )

            # add Ewald reciprocal correction
            tot_e: list[torch.Tensor] = []
            chunk_coord = torch.split(
                extended_coord.reshape(nframes, -1, 3), self.dp_batch_size, dim=0
            )
            chunk_box = torch.split(
                input_box.reshape(nframes, 3, 3), self.dp_batch_size, dim=0
            )
            chunk_charge = torch.split(
                extended_charge.reshape(nframes, -1), self.dp_batch_size, dim=0
            )
            for _coord, _box, _charge in zip(chunk_coord, chunk_box, chunk_charge):
                self.er(
                    _coord,
                    _box,
                    self.placeholder_pairs,
                    self.placeholder_ds,
                    self.placeholder_buffer_scales,
                    {"charge": _charge},
                )
                tot_e.append(self.er.reciprocal_energy.unsqueeze(0))
            # nframe,
            tot_e = torch.concat(tot_e, dim=0)
            # nframe, nat * 3
            tot_f = -calc_grads(tot_e, input_coord)
            # nframe, nat, 3
            tot_f = torch.reshape(tot_f, (nframes, natoms, 3))
            # nframe, 9
            tot_v = calc_grads(tot_e, input_box)
            tot_v = torch.reshape(tot_v, (nframes, 3, 3))
            # nframe, 3, 3
            tot_v = -torch.matmul(
                tot_v.transpose(2, 1), input_box.reshape(nframes, 3, 3)
            )

            modifier_pred["energy"] = tot_e
            modifier_pred["force"] = tot_f
            modifier_pred["virial"] = tot_v
            return modifier_pred

    def extend_system(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extend the system with WFCC (Wannier Function Charge Centers).

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms * 3)
        atype : torch.Tensor
            The atom types with shape (nframes, natoms)
        box : torch.Tensor
            The simulation box with shape (nframes, 9)
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing three tensors:
            - extended_coord : torch.Tensor
                Extended coordinates with shape (nframes, 2 * natoms * 3)
            - extended_charge : torch.Tensor
                Extended charges with shape (nframes, 2 * natoms)
            - atomic_dipole : torch.Tensor
                Atomic dipoles with shape (nframes, natoms, 3)
        """
        # nframes, natoms, 3
        extended_coord, atomic_dipole = self.extend_system_coord(
            coord,
            atype,
            box,
            fparam,
            aparam,
        )
        # Get ion charges based on atom types
        # nframe x nat
        ion_charge = self.sys_charge_map[atype]
        # Initialize wfcc charges
        wc_charge = torch.zeros_like(ion_charge)
        # Assign charges to selected atom types
        for ii, charge in enumerate(self.model_charge_map):
            wc_charge[atype == self.sel_type[ii]] = charge
        # Concatenate ion charges and wfcc charges
        extended_charge = torch.cat([ion_charge, wc_charge], dim=1)
        return extended_coord, extended_charge, atomic_dipole

    def extend_system_coord(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the system with WFCC (Wannier Function Charge Centers).

        This function calculates Wannier Function Charge Centers (WFCC) by adding dipole
        vectors to atomic coordinates for selected atom types, then concatenates these
        WFCC coordinates with the original atomic coordinates.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms * 3)
        atype : torch.Tensor
            The atom types with shape (nframes, natoms)
        box : torch.Tensor
            The simulation box with shape (nframes, 9)
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
            - all_coord : torch.Tensor
                Extended coordinates with shape (nframes, 2 * natoms * 3)
                where nsel is the number of selected atoms
            - dipole_reshaped : torch.Tensor
                Atomic dipoles with shape (nframes, natoms, 3)
        """
        nframes = coord.shape[0]
        natoms = coord.shape[1] // 3

        all_dipole: list[torch.Tensor] = []
        chunk_coord = torch.split(coord, self.dp_batch_size, dim=0)
        chunk_atype = torch.split(atype, self.dp_batch_size, dim=0)
        chunk_box = torch.split(box, self.dp_batch_size, dim=0)
        # use placeholder to make the jit happy for fparam/aparam is None
        chunk_fparam = (
            torch.split(fparam, self.dp_batch_size, dim=0)
            if fparam is not None
            else chunk_atype
        )
        chunk_aparam = (
            torch.split(aparam, self.dp_batch_size, dim=0)
            if aparam is not None
            else chunk_atype
        )
        for _coord, _atype, _box, _fparam, _aparam in zip(
            chunk_coord, chunk_atype, chunk_box, chunk_fparam, chunk_aparam
        ):
            dipole_batch = self.model(
                coord=_coord,
                atype=_atype,
                box=_box,
                do_atomic_virial=False,
                fparam=_fparam if fparam is not None else None,
                aparam=_aparam if aparam is not None else None,
            )
            # Extract dipole from the output dictionary
            all_dipole.append(dipole_batch["dipole"])

        # nframe x natoms x 3
        dipole = torch.cat(all_dipole, dim=0)
        if dipole.shape[0] != nframes:
            raise RuntimeError(
                f"Dipole shape mismatch: expected {nframes} frames, got {dipole.shape[0]}"
            )

        dipole_reshaped = dipole.reshape(nframes, natoms, 3)
        coord_reshaped = coord.reshape(nframes, natoms, 3)
        wfcc_coord = coord_reshaped + dipole_reshaped
        all_coord = torch.cat((coord_reshaped, wfcc_coord), dim=1)
        return all_coord, dipole_reshaped
