# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import torch
from torch_admp.pme import (
    CoulombForceModule,
)
from torch_admp.utils import (
    calc_grads,
)

from deepmd.pt.modifier.base_modifier import (
    BaseModifier,
)
from deepmd.pt.utils import (
    env,
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
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 1.0,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.modifier_type = "dipole_charge"
        self.model_name = model_name

        self.model = torch.jit.load(model_name, map_location=env.DEVICE)
        self.rcut = self.model.get_rcut()
        self.type_map = self.model.get_type_map()
        sel_type = self.model.get_sel_type()
        self.sel_type = to_torch_tensor(np.array(sel_type))
        self.model_charge_map = to_torch_tensor(np.array(model_charge_map))
        self.sys_charge_map = to_torch_tensor(np.array(sys_charge_map))
        self._model_charge_map = model_charge_map
        self._sys_charge_map = sys_charge_map

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

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Modifier",
            "type": self.modifier_type,
            "@version": 3,
            "model_name": self.model_name,
            "model_charge_map": self._model_charge_map,
            "sys_charge_map": self._sys_charge_map,
            "ewald_h": self.ewald_h,
            "ewald_beta": self.ewald_beta,
        }
        return data

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

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the correction terms:
            - energy: Energy correction tensor with shape (nframes, 1)
            - force: Force correction tensor with shape (nframes, natoms+nsel, 3)
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
                torch.linalg.inv(detached_box.reshape(nframes, 3, 3)),
                input_box.reshape(nframes, 3, 3),
            )
            input_coord = torch.matmul(coord, sfactor).reshape(nframes, -1)

            extended_coord, extended_charge = self.extend_system(
                input_coord,
                atype,
                input_box,
                fparam,
                aparam,
            )

            tot_e = []
            # add Ewald reciprocal correction
            for ii in range(nframes):
                self.er(
                    extended_coord[ii].reshape((-1, 3)),
                    input_box[ii].reshape((3, 3)),
                    self.placeholder_pairs,
                    self.placeholder_ds,
                    self.placeholder_buffer_scales,
                    {"charge": extended_charge[ii].reshape((-1,))},
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        tuple
            (extended_coord, extended_charge)
            extended_coord : torch.Tensor
                Extended coordinates with shape (nframes, (natoms + nsel) * 3)
            extended_charge : torch.Tensor
                Extended charges with shape (nframes, natoms + nsel)
        """
        nframes = coord.shape[0]
        mask = make_mask(self.sel_type, atype)

        extended_coord = self.extend_system_coord(
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
        # Get the charges for selected atoms only
        wc_charge_selected = wc_charge[mask].reshape(nframes, -1)
        # Concatenate ion charges and wfcc charges
        extended_charge = torch.cat([ion_charge, wc_charge_selected], dim=1)
        return extended_coord, extended_charge

    def extend_system_coord(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        all_coord : torch.Tensor
            Extended coordinates with shape (nframes, (natoms + nsel) * 3)
            where nsel is the number of selected atoms
        """
        mask = make_mask(self.sel_type, atype)

        nframes = coord.shape[0]
        natoms = coord.shape[1] // 3

        all_dipole = []
        for ii in range(nframes):
            dipole_batch = self.model(
                coord=coord[ii].reshape(1, -1),
                atype=atype[ii].reshape(1, -1),
                box=box[ii].reshape(1, -1),
                do_atomic_virial=False,
                fparam=fparam[ii].reshape(1, -1) if fparam is not None else None,
                aparam=aparam[ii].reshape(1, -1) if aparam is not None else None,
            )
            # Extract dipole from the output dictionary
            all_dipole.append(dipole_batch["dipole"])

        # nframe x natoms x 3
        dipole = torch.cat(all_dipole, dim=0)
        assert dipole.shape[0] == nframes

        dipole_reshaped = dipole.reshape(nframes, natoms, 3)
        coord_reshaped = coord.reshape(nframes, natoms, 3)
        _wfcc_coord = coord_reshaped + dipole_reshaped
        # Apply mask and reshape
        wfcc_coord = _wfcc_coord[mask.unsqueeze(-1).expand_as(_wfcc_coord)]
        wfcc_coord = wfcc_coord.reshape(nframes, -1)
        all_coord = torch.cat((coord, wfcc_coord), dim=1)
        return all_coord


@torch.jit.export
def make_mask(
    sel_type: torch.Tensor,
    atype: torch.Tensor,
) -> torch.Tensor:
    """Create a boolean mask for selected atom types.

    Parameters
    ----------
    sel_type : torch.Tensor
        The selected atom types to create a mask for
    atype : torch.Tensor
        The atom types in the system

    Returns
    -------
    mask : torch.Tensor
        Boolean mask where True indicates atoms of selected types
    """
    # Ensure tensors are of the right type
    sel_type = sel_type.to(torch.long)
    atype = atype.to(torch.long)

    # Create mask using broadcasting
    mask = torch.zeros_like(atype, dtype=torch.bool)
    for t in sel_type:
        mask = mask | (atype == t)
    return mask
