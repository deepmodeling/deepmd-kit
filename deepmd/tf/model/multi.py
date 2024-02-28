# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.tf.descriptor.descriptor import (
    Descriptor,
)
from deepmd.tf.env import (
    MODEL_VERSION,
    global_cvt_2_ener_float,
    op_module,
    tf,
)
from deepmd.tf.fit import (
    DipoleFittingSeA,
    DOSFitting,
    EnerFitting,
    GlobalPolarFittingSeA,
    PolarFittingSeA,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.utils.argcheck import (
    type_embedding_args,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.pair_tab import (
    PairTab,
)
from deepmd.tf.utils.spin import (
    Spin,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)

from .model import (
    Model,
)
from .model_stat import (
    make_stat_input,
    merge_sys_stat,
)


@Model.register("multi")
class MultiModel(Model):
    """Multi-task model.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net_dict
            Dictionary of fitting nets
    fitting_type_dict
        deprecated argument
    type_embedding
            Type embedding net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    data_stat_nbatch
            Number of frames used for data statistic
    data_stat_protect
            Protect parameter for atomic energy regression
    use_srtab
            The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
    smin_alpha
            The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.
    sw_rmin
            The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    sw_rmin
            The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    """

    model_type = "multi_task"

    def __init__(
        self,
        descriptor: dict,
        fitting_net_dict: dict,
        fitting_type_dict: Optional[dict] = None,  # deprecated
        type_embedding=None,
        type_map: Optional[List[str]] = None,
        data_stat_nbatch: int = 10,
        data_stat_protect: float = 1e-2,
        use_srtab: Optional[str] = None,  # all the ener fitting will do this
        smin_alpha: Optional[float] = None,
        sw_rmin: Optional[float] = None,
        sw_rmax: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Constructor."""
        super().__init__(
            descriptor=descriptor,
            fitting_net_dict=fitting_net_dict,
            type_embedding=type_embedding,
            type_map=type_map,
            data_stat_nbatch=data_stat_nbatch,
            data_stat_protect=data_stat_protect,
            use_srtab=use_srtab,
            smin_alpha=smin_alpha,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
        )
        if self.spin is not None and not isinstance(self.spin, Spin):
            self.spin = Spin(**self.spin)
        if isinstance(descriptor, Descriptor):
            self.descrpt = descriptor
        else:
            self.descrpt = Descriptor(
                **descriptor,
                ntypes=len(self.get_type_map()),
                multi_task=True,
                spin=self.spin,
            )

        fitting_dict = {}
        for item in fitting_net_dict:
            item_fitting_param = fitting_net_dict[item]
            if isinstance(item_fitting_param, Fitting):
                fitting_dict[item] = item_fitting_param
            else:
                if item_fitting_param["type"] in ["dipole", "polar"]:
                    item_fitting_param[
                        "embedding_width"
                    ] = self.descrpt.get_dim_rot_mat_1()
                fitting_dict[item] = Fitting(
                    **item_fitting_param,
                    descrpt=self.descrpt,
                    spin=self.spin,
                    ntypes=self.descrpt.get_ntypes(),
                    dim_descrpt=self.descrpt.get_dim_out(),
                )

        # type embedding
        if type_embedding is not None and isinstance(type_embedding, TypeEmbedNet):
            self.typeebd = type_embedding
        elif type_embedding is not None:
            self.typeebd = TypeEmbedNet(
                **type_embedding,
                padding=self.descrpt.explicit_ntypes,
            )
        elif self.descrpt.explicit_ntypes:
            default_args = type_embedding_args()
            default_args_dict = {i.name: i.default for i in default_args}
            default_args_dict["activation_function"] = None
            self.typeebd = TypeEmbedNet(
                **default_args_dict,
                padding=True,
            )
        else:
            self.typeebd = None

        # descriptor
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting_dict = fitting_dict
        self.numb_fparam_dict = {
            item: self.fitting_dict[item].get_numb_fparam()
            for item in self.fitting_dict
            if isinstance(self.fitting_dict[item], EnerFitting)
        }
        # other inputs
        if type_map is None:
            self.type_map = []
        else:
            self.type_map = type_map
        self.data_stat_nbatch = data_stat_nbatch
        self.data_stat_protect = data_stat_protect
        self.srtab_name = use_srtab
        if self.srtab_name is not None:
            self.srtab = PairTab(self.srtab_name)
            self.smin_alpha = smin_alpha
            self.sw_rmin = sw_rmin
            self.sw_rmax = sw_rmax
        else:
            self.srtab = None

    def get_rcut(self):
        return self.rcut

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return self.type_map

    def data_stat(self, data):
        for fitting_key in data:
            all_stat = make_stat_input(
                data[fitting_key], self.data_stat_nbatch, merge_sys=False
            )
            m_all_stat = merge_sys_stat(all_stat)
            self._compute_input_stat(
                m_all_stat,
                protection=self.data_stat_protect,
                mixed_type=data[fitting_key].mixed_type,
                fitting_key=fitting_key,
            )
            self._compute_output_stat(
                all_stat,
                mixed_type=data[fitting_key].mixed_type,
                fitting_key=fitting_key,
            )
        self.descrpt.merge_input_stats(self.descrpt.stat_dict)

    def _compute_input_stat(
        self, all_stat, protection=1e-2, mixed_type=False, fitting_key=""
    ):
        if mixed_type:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
                mixed_type,
                all_stat["real_natoms_vec"],
            )
        else:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
            )
        if hasattr(self.fitting_dict[fitting_key], "compute_input_stats"):
            self.fitting_dict[fitting_key].compute_input_stats(
                all_stat, protection=protection
            )

    def _compute_output_stat(self, all_stat, mixed_type=False, fitting_key=""):
        if hasattr(self.fitting_dict[fitting_key], "compute_output_stats"):
            if mixed_type:
                self.fitting_dict[fitting_key].compute_output_stats(
                    all_stat, mixed_type=mixed_type
                )
            else:
                self.fitting_dict[fitting_key].compute_output_stats(all_stat)

    def build(
        self,
        coord_,
        atype_,
        natoms,
        box,
        mesh,
        input_dict,
        frz_model=None,
        ckpt_meta: Optional[str] = None,
        suffix="",
        reuse=None,
    ):
        if input_dict is None:
            input_dict = {}
        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            t_tmap = tf.constant(" ".join(self.type_map), name="tmap", dtype=tf.string)
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)
            t_st = {}
            t_od = {}
            sel_type = {}
            natomsel = {}
            nout = {}
            for fitting_key in self.fitting_dict:
                if isinstance(
                    self.fitting_dict[fitting_key],
                    (DipoleFittingSeA, PolarFittingSeA, GlobalPolarFittingSeA),
                ):
                    sel_type[fitting_key] = self.fitting_dict[
                        fitting_key
                    ].get_sel_type()
                    natomsel[fitting_key] = sum(
                        natoms[2 + type_i] for type_i in sel_type[fitting_key]
                    )
                    nout[fitting_key] = self.fitting_dict[fitting_key].get_out_size()
                    t_st[fitting_key] = tf.constant(
                        sel_type[fitting_key],
                        name=f"sel_type_{fitting_key}",
                        dtype=tf.int32,
                    )
                    t_od[fitting_key] = tf.constant(
                        nout[fitting_key],
                        name=f"output_dim_{fitting_key}",
                        dtype=tf.int32,
                    )

            if self.srtab is not None:
                tab_info, tab_data = self.srtab.get()
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

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        input_dict["nframes"] = tf.shape(coord)[0]

        # type embedding if any
        if self.typeebd is not None:
            type_embedding = self.build_type_embedding(
                self.ntypes,
                reuse=reuse,
                suffix=suffix,
                frz_model=frz_model,
                ckpt_meta=ckpt_meta,
            )
            input_dict["type_embedding"] = type_embedding
        input_dict["atype"] = atype_

        dout = self.build_descrpt(
            coord,
            atype,
            natoms,
            box,
            mesh,
            input_dict,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix=suffix,
            reuse=reuse,
        )
        dout = tf.identity(dout, name="o_descriptor")

        if self.srtab is not None:
            nlist, rij, sel_a, sel_r = self.descrpt.get_nlist()
            nnei_a = np.cumsum(sel_a)[-1]
            nnei_r = np.cumsum(sel_r)[-1]
            sw_lambda, sw_deriv = op_module.soft_min_switch(
                atype,
                rij,
                nlist,
                natoms,
                sel_a=sel_a,
                sel_r=sel_r,
                alpha=self.smin_alpha,
                rmin=self.sw_rmin,
                rmax=self.sw_rmax,
            )
            inv_sw_lambda = 1.0 - sw_lambda
            # NOTICE:
            # atom energy is not scaled,
            # force and virial are scaled
            tab_atom_ener, tab_force, tab_atom_virial = op_module.pair_tab(
                self.tab_info,
                self.tab_data,
                atype,
                rij,
                nlist,
                natoms,
                sw_lambda,
                sel_a=sel_a,
                sel_r=sel_r,
            )

        rot_mat = self.descrpt.get_rot_mat()
        rot_mat = tf.identity(rot_mat, name="o_rot_mat" + suffix)
        self.atom_ener = {}
        model_dict = {}
        for fitting_key in self.fitting_dict:
            if isinstance(self.fitting_dict[fitting_key], EnerFitting):
                atom_ener = self.fitting_dict[fitting_key].build(
                    dout,
                    natoms,
                    input_dict,
                    reuse=reuse,
                    suffix=f"_{fitting_key}" + suffix,
                )
                self.atom_ener[fitting_key] = atom_ener
                if self.srtab is not None:
                    energy_diff = tab_atom_ener - tf.reshape(atom_ener, [-1, natoms[0]])
                    tab_atom_ener = tf.reshape(sw_lambda, [-1]) * tf.reshape(
                        tab_atom_ener, [-1]
                    )
                    atom_ener = tf.reshape(inv_sw_lambda, [-1]) * atom_ener
                    energy_raw = tab_atom_ener + atom_ener
                else:
                    energy_raw = atom_ener
                energy_raw = tf.reshape(
                    energy_raw,
                    [-1, natoms[0]],
                    name=f"o_atom_energy_{fitting_key}" + suffix,
                )
                energy = tf.reduce_sum(
                    global_cvt_2_ener_float(energy_raw),
                    axis=1,
                    name=f"o_energy_{fitting_key}" + suffix,
                )
                force, virial, atom_virial = self.descrpt.prod_force_virial(
                    atom_ener, natoms
                )

                if self.srtab is not None:
                    sw_force = op_module.soft_min_force(
                        energy_diff,
                        sw_deriv,
                        nlist,
                        natoms,
                        n_a_sel=nnei_a,
                        n_r_sel=nnei_r,
                    )
                    force = force + sw_force + tab_force

                force = tf.reshape(
                    force,
                    [-1, 3 * natoms[1]],
                    name=f"o_force_{fitting_key}" + suffix,
                )

                if self.srtab is not None:
                    sw_virial, sw_atom_virial = op_module.soft_min_virial(
                        energy_diff,
                        sw_deriv,
                        rij,
                        nlist,
                        natoms,
                        n_a_sel=nnei_a,
                        n_r_sel=nnei_r,
                    )
                    atom_virial = atom_virial + sw_atom_virial + tab_atom_virial
                    virial = (
                        virial
                        + sw_virial
                        + tf.reduce_sum(
                            tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis=1
                        )
                    )

                virial = tf.reshape(
                    virial, [-1, 9], name=f"o_virial_{fitting_key}" + suffix
                )
                atom_virial = tf.reshape(
                    atom_virial,
                    [-1, 9 * natoms[1]],
                    name=f"o_atom_virial_{fitting_key}" + suffix,
                )

                model_dict[fitting_key] = {}
                model_dict[fitting_key]["energy"] = energy
                model_dict[fitting_key]["force"] = force
                model_dict[fitting_key]["virial"] = virial
                model_dict[fitting_key]["atom_ener"] = energy_raw
                model_dict[fitting_key]["atom_virial"] = atom_virial
                model_dict[fitting_key]["coord"] = coord
                model_dict[fitting_key]["atype"] = atype
            elif isinstance(
                self.fitting_dict[fitting_key],
                (DipoleFittingSeA, PolarFittingSeA, GlobalPolarFittingSeA),
            ):
                tensor_name = {
                    DipoleFittingSeA: "dipole",
                    PolarFittingSeA: "polar",
                    GlobalPolarFittingSeA: "global_polar",
                }[type(self.fitting_dict[fitting_key])]
                output = self.fitting_dict[fitting_key].build(
                    dout,
                    rot_mat,
                    natoms,
                    input_dict,
                    reuse=reuse,
                    suffix=f"_{fitting_key}" + suffix,
                )
                framesize = (
                    nout
                    if "global" in tensor_name
                    else natomsel[fitting_key] * nout[fitting_key]
                )
                output = tf.reshape(
                    output,
                    [-1, framesize],
                    name=f"o_{tensor_name}_{fitting_key}" + suffix,
                )

                model_dict[fitting_key] = {}
                model_dict[fitting_key][tensor_name] = output

                if "global" not in tensor_name:
                    gname = "global_" + tensor_name
                    atom_out = tf.reshape(
                        output, [-1, natomsel[fitting_key], nout[fitting_key]]
                    )
                    global_out = tf.reduce_sum(atom_out, axis=1)
                    global_out = tf.reshape(
                        global_out,
                        [-1, nout[fitting_key]],
                        name=f"o_{gname}_{fitting_key}" + suffix,
                    )

                    out_cpnts = tf.split(atom_out, nout[fitting_key], axis=-1)
                    force_cpnts = []
                    virial_cpnts = []
                    atom_virial_cpnts = []

                    for out_i in out_cpnts:
                        (
                            force_i,
                            virial_i,
                            atom_virial_i,
                        ) = self.descrpt.prod_force_virial(out_i, natoms)
                        force_cpnts.append(tf.reshape(force_i, [-1, 3 * natoms[1]]))
                        virial_cpnts.append(tf.reshape(virial_i, [-1, 9]))
                        atom_virial_cpnts.append(
                            tf.reshape(atom_virial_i, [-1, 9 * natoms[1]])
                        )

                    # [nframe x nout x (natom x 3)]
                    force = tf.concat(
                        force_cpnts,
                        axis=1,
                        name=f"o_force_{fitting_key}" + suffix,
                    )
                    # [nframe x nout x 9]
                    virial = tf.concat(
                        virial_cpnts,
                        axis=1,
                        name=f"o_virial_{fitting_key}" + suffix,
                    )
                    # [nframe x nout x (natom x 9)]
                    atom_virial = tf.concat(
                        atom_virial_cpnts,
                        axis=1,
                        name=f"o_atom_virial_{fitting_key}" + suffix,
                    )

                    model_dict[fitting_key][gname] = global_out
                    model_dict[fitting_key]["force"] = force
                    model_dict[fitting_key]["virial"] = virial
                    model_dict[fitting_key]["atom_virial"] = atom_virial
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
        # self.frz_model will control the self.model to import the descriptor from the given frozen model instead of building from scratch...
        # initialize fitting net with the given compressed frozen model
        assert (
            model_type == "original_model"
        ), "Initialization in multi-task mode does not support compressed model!"
        self.descrpt.init_variables(graph, graph_def, suffix=suffix)
        old_jdata = json.loads(
            get_tensor_by_name_from_graph(graph, "train_attr/training_script")
        )
        old_fitting_keys = list(old_jdata["model"]["fitting_net_dict"].keys())
        newly_added_fittings = set(self.fitting_dict.keys()) - set(old_fitting_keys)
        reused_fittings = set(self.fitting_dict.keys()) - newly_added_fittings
        for fitting_key in reused_fittings:
            self.fitting_dict[fitting_key].init_variables(
                graph, graph_def, suffix=f"_{fitting_key}" + suffix
            )
        tf.constant("original_model", name="model_type", dtype=tf.string)
        if self.typeebd is not None:
            self.typeebd.init_variables(graph, graph_def, suffix=suffix)

    def enable_mixed_precision(self, mixed_prec: dict):
        """Enable mixed precision for the model.

        Parameters
        ----------
        mixed_prec : dict
            The mixed precision config
        """
        self.descrpt.enable_mixed_precision(mixed_prec)
        for fitting_key in self.fitting_dict:
            self.fitting_dict[fitting_key].enable_mixed_precision(self.mixed_prec)

    def get_numb_fparam(self) -> dict:
        """Get the number of frame parameters."""
        numb_fparam_dict = {}
        for fitting_key in self.fitting_dict:
            if isinstance(self.fitting_dict[fitting_key], (EnerFitting, DOSFitting)):
                numb_fparam_dict[fitting_key] = self.fitting_dict[
                    fitting_key
                ].get_numb_fparam()
            else:
                numb_fparam_dict[fitting_key] = 0
        return numb_fparam_dict

    def get_numb_aparam(self) -> dict:
        """Get the number of atomic parameters."""
        numb_aparam_dict = {}
        for fitting_key in self.fitting_dict:
            if isinstance(self.fitting_dict[fitting_key], (EnerFitting, DOSFitting)):
                numb_aparam_dict[fitting_key] = self.fitting_dict[
                    fitting_key
                ].get_numb_aparam()
            else:
                numb_aparam_dict[fitting_key] = 0
        return numb_aparam_dict

    def get_numb_dos(self) -> dict:
        """Get the number of gridpoints in energy space."""
        numb_dos_dict = {}
        for fitting_key in self.fitting_dict:
            if isinstance(self.fitting_dict[fitting_key], DOSFitting):
                numb_dos_dict[fitting_key] = self.fitting_dict[
                    fitting_key
                ].get_numb_dos()
            else:
                numb_dos_dict[fitting_key] = 0
        return numb_dos_dict

    def get_fitting(self) -> dict:
        """Get the fitting(s)."""
        return self.fitting_dict.copy()

    def get_loss(self, loss: dict, lr: dict) -> Dict[str, Loss]:
        loss_dict = {}
        for fitting_key in self.fitting_dict:
            loss_param = loss.get(fitting_key, {})
            loss_dict[fitting_key] = self.fitting_dict[fitting_key].get_loss(
                loss_param, lr[fitting_key]
            )
        return loss_dict

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"] = Descriptor.update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy
