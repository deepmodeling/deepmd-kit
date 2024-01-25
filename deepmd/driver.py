# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpdata driver."""
# Derived from https://github.com/deepmodeling/dpdata/blob/18a0ed5ebced8b1f6887038883d46f31ae9990a4/dpdata/plugins/deepmd.py#L361-L443
# under LGPL-3.0-or-later license.
# The original deepmd driver maintained in the dpdata package will be overriden.
# The class in the dpdata package needs to handle different situations for v1 and v2 interface,
# which is too complex with the development of deepmd-kit.
# So, it will be a good idea to ship it with DeePMD-kit itself.
import dpdata
from dpdata.utils import (
    sort_atom_names,
)


@dpdata.driver.Driver.register("dp")
@dpdata.driver.Driver.register("deepmd")
@dpdata.driver.Driver.register("deepmd-kit")
class DPDriver(dpdata.driver.Driver):
    """DeePMD-kit driver.

    Parameters
    ----------
    dp : deepmd.DeepPot or str
        The deepmd-kit potential class or the filename of the model.

    Examples
    --------
    >>> DPDriver("frozen_model.pb")
    """

    def __init__(self, dp: str) -> None:
        from deepmd.infer.deep_pot import (
            DeepPot,
        )

        if not isinstance(dp, DeepPot):
            self.dp = DeepPot(dp, auto_batch_size=True)
        else:
            self.dp = dp

    def label(self, data: dict) -> dict:
        """Label a system data by deepmd-kit. Returns new data with energy, forces, and virials.

        Parameters
        ----------
        data : dict
            data with coordinates and atom types

        Returns
        -------
        dict
            labeled data with energies and forces
        """
        nframes = data["coords"].shape[0]
        natoms = data["coords"].shape[1]
        type_map = self.dp.get_type_map()
        # important: dpdata type_map may not be the same as the model type_map
        # note: while we want to change the type_map when feeding to DeepPot,
        # we don't want to change the type_map in the returned data
        sorted_data = sort_atom_names(data.copy(), type_map=type_map)
        atype = sorted_data["atom_types"]

        coord = data["coords"].reshape((nframes, natoms * 3))
        if "nopbc" not in data:
            cell = data["cells"].reshape((nframes, 9))
        else:
            cell = None
        e, f, v = self.dp.eval(coord, cell, atype)
        data = data.copy()
        data["energies"] = e.reshape((nframes,))
        data["forces"] = f.reshape((nframes, natoms, 3))
        data["virials"] = v.reshape((nframes, 3, 3))
        return data
