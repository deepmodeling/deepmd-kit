# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch
from deepmd.model_format import (
    ModelOutputDef,
)

from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)


def make_model(T_AtomicModel):
    class CM(T_AtomicModel):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

        def get_model_output_def(self):
            return ModelOutputDef(self.get_fitting_output_def())

        # cannot use the name forward. torch script does not work
        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Return total energy of the system.
            Args:
            - coord: Atom coordinates with shape [nframes, natoms[1]*3].
            - atype: Atom types with shape [nframes, natoms[1]].
            - natoms: Atom statisics with shape [self.ntypes+2].
            - box: Simulation box with shape [nframes, 9].
            - atomic_virial: Whether or not compoute the atomic virial.

            Returns
            -------
            - energy: Energy per atom.
            - force: XYZ force per atom.
            """
            nframes, nloc = atype.shape[:2]
            if box is not None:
                coord_normalized = normalize_coord(coord, box.reshape(-1, 3, 3))
            else:
                coord_normalized = coord.clone()
            extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
                coord_normalized, atype, box, self.get_rcut()
            )
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                nloc,
                self.get_rcut(),
                self.get_sel(),
                distinguish_types=self.distinguish_types(),
            )
            extended_coord = extended_coord.reshape(nframes, -1, 3)
            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.get_model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        def forward_common_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ):
            """Return model prediction.

            Parameters
            ----------
            extended_coord
                coodinates in extended region
            extended_atype
                atomic type in extended region
            nlist
                neighbor list. nf x nloc x nsel
            mapping
                mapps the extended indices to local indices
            do_atomic_virial
                whether do atomic virial

            Returns
            -------
            result_dict
                the result dict, defined by the fitting net output def.

            """
            atomic_ret = self.forward_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.get_fitting_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

    return CM
