# SPDX-License-Identifier: LGPL-3.0-or-later
from enum import (
    Enum,
)
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    global_cvt_2_ener_float,
    op_module,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.utils.pair_tab import (
    PairTab,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)


@Model.register("pairtab")
class PairTabModel(Model):
    """Pairwise tabulation energy model.

    This model can be used to tabulate the pairwise energy between atoms for either
    short-range or long-range interactions, such as D3, LJ, ZBL, etc. It should not
    be used alone, but rather as one submodel of a linear (sum) model, such as
    DP+D3.

    Do not put the model on the first model of a linear model, since the linear
    model fetches the type map from the first model.

    At this moment, the model does not smooth the energy at the cutoff radius, so
    one needs to make sure the energy has been smoothed to zero.

    Parameters
    ----------
    tab_file : str
        The path to the tabulation file.
    rcut : float
        The cutoff radius
    sel : int or list[int]
        The maxmum number of atoms in the cut-off radius
    """

    model_type = "ener"

    def __init__(
        self, tab_file: str, rcut: float, sel: Union[int, List[int]], **kwargs
    ):
        super().__init__()
        self.tab_file = tab_file
        self.tab = PairTab(self.tab_file)
        self.ntypes = self.tab.ntypes
        self.rcut = rcut
        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        frz_model: Optional[str] = None,
        ckpt_meta: Optional[str] = None,
        suffix: str = "",
        reuse: Optional[Union[bool, Enum]] = None,
    ):
        """Build the model.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinates of atoms
        atype_ : tf.Tensor
            The atom types of atoms
        natoms : tf.Tensor
            The number of atoms
        box : tf.Tensor
            The box vectors
        mesh : tf.Tensor
            The mesh vectors
        input_dict : dict
            The input dict
        frz_model : str, optional
            The path to the frozen model
        ckpt_meta : str, optional
            The path prefix of the checkpoint and meta files
        suffix : str, optional
            The suffix of the scope
        reuse : bool or tf.AUTO_REUSE, optional
            Whether to reuse the variables

        Returns
        -------
        dict
            The output dict
        """
        tab_info, tab_data = self.tab.get()
        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            self.tab_info = tf.get_variable(
                "t_tab_info",
                tab_info.shape,
                dtype=tf.float64,
                trainable=False,
                initializer=tf.constant_initializer(tab_info, dtype=tf.float64),
            )
            self.tab_data = tf.get_variable(
                "t_tab_data",
                tab_data.shape,
                dtype=tf.float64,
                trainable=False,
                initializer=tf.constant_initializer(tab_data, dtype=tf.float64),
            )
            t_tmap = tf.constant(" ".join(self.type_map), name="tmap", dtype=tf.string)
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)

        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(0, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(0, name="daparam", dtype=tf.int32)
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_rcut = tf.constant(
                self.rcut, name="rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
            )
        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        box = tf.reshape(box, [-1, 9])
        # perhaps we need a OP that only outputs rij and nlist
        (
            _,
            _,
            rij,
            nlist,
            _,
            _,
        ) = op_module.prod_env_mat_a_mix(
            coord,
            atype,
            natoms,
            box,
            mesh,
            np.zeros([self.ntypes, self.sel * 4]),
            np.ones([self.ntypes, self.sel * 4]),
            rcut_a=-1,
            rcut_r=self.rcut,
            rcut_r_smth=self.rcut,
            sel_a=[self.sel],
            sel_r=[0],
        )
        scale = tf.ones([tf.shape(coord)[0], natoms[0]], dtype=tf.float64)
        tab_atom_ener, tab_force, tab_atom_virial = op_module.pair_tab(
            self.tab_info,
            self.tab_data,
            atype,
            rij,
            nlist,
            natoms,
            scale,
            sel_a=[self.sel],
            sel_r=[0],
        )
        energy_raw = tf.reshape(
            tab_atom_ener, [-1, natoms[0]], name="o_atom_energy" + suffix
        )
        energy = tf.reduce_sum(
            global_cvt_2_ener_float(energy_raw), axis=1, name="o_energy" + suffix
        )
        force = tf.reshape(tab_force, [-1, 3 * natoms[1]], name="o_force" + suffix)
        virial = tf.reshape(
            tf.reduce_sum(tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis=1),
            [-1, 9],
            name="o_virial" + suffix,
        )
        atom_virial = tf.reshape(
            tab_atom_virial, [-1, 9 * natoms[1]], name="o_atom_virial" + suffix
        )
        model_dict = {}
        model_dict["energy"] = energy
        model_dict["force"] = force
        model_dict["virial"] = virial
        model_dict["atom_ener"] = energy_raw
        model_dict["atom_virial"] = atom_virial
        model_dict["coord"] = coord
        model_dict["atype"] = atype

        return model_dict

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        model_type: str = "original_model",
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given frozen model.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        # skip. table can be initialized from the file

    def get_fitting(self) -> Union[Fitting, dict]:
        """Get the fitting(s)."""
        # nothing needs to do
        return {}

    def get_loss(self, loss: dict, lr) -> Optional[Union[Loss, dict]]:
        """Get the loss function(s)."""
        # nothing nees to do
        return

    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of types."""
        return self.ntypes

    def data_stat(self, data: dict):
        """Data staticis."""
        # nothing needs to do

    def enable_compression(self, suffix: str = "") -> None:
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        # nothing needs to do

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict) -> dict:
        """Update the selection and perform neighbor statistics.

        Notes
        -----
        Do not modify the input data without copying it.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        """
        local_jdata_cpy = local_jdata.copy()
        return UpdateSel().update_one_sel(global_jdata, local_jdata_cpy, True)
