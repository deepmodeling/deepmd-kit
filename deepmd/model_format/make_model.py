# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import numpy as np

from .nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from .output_def import (
    ModelOutputDef,
)
from .region import (
    normalize_coord,
)
from .transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
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

        def call(
            self,
            coord,
            atype,
            box: Optional[np.ndarray] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, np.ndarray]:
            """Return predictions of a model.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,np.ndarray].
                The keys are defined by the `ModelOutputDef`.

            """
            nframes, nloc = atype.shape[:2]
            if box is not None:
                coord_normalized = normalize_coord(
                    coord.reshape(nframes, nloc, 3),
                    box.reshape(nframes, 3, 3),
                )
            else:
                coord_normalized = coord.copy()
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
            model_predict_lower = self.call_lower(
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

        def call_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[np.ndarray] = None,
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
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.reshape(nframes, -1, 3)
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
