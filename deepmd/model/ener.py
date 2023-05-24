from typing import (
    List,
    Optional,
)

import numpy as np

from deepmd.env import (
    MODEL_VERSION,
    global_cvt_2_ener_float,
    op_module,
    tf,
)
from deepmd.utils.pair_tab import (
    PairTab,
)
from deepmd.utils.spin import (
    Spin,
)

from .model import (
    Model,
)
from .model_stat import (
    make_stat_input,
    merge_sys_stat,
)


class EnerModel(Model):
    """Energy model.

    Parameters
    ----------
    descrpt
            Descriptor
    fitting
            Fitting net
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

    model_type = "ener"

    def __init__(
        self,
        descrpt,
        fitting,
        typeebd=None,
        type_map: Optional[List[str]] = None,
        data_stat_nbatch: int = 10,
        data_stat_protect: float = 1e-2,
        use_srtab: Optional[str] = None,
        smin_alpha: Optional[float] = None,
        sw_rmin: Optional[float] = None,
        sw_rmax: Optional[float] = None,
        spin: Optional[Spin] = None,
    ) -> None:
        """Constructor."""
        # descriptor
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting = fitting
        self.numb_fparam = self.fitting.get_numb_fparam()
        # type embedding
        self.typeebd = typeebd
        # spin
        self.spin = spin
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
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys=False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(
            m_all_stat, protection=self.data_stat_protect, mixed_type=data.mixed_type
        )
        self._compute_output_stat(all_stat, mixed_type=data.mixed_type)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat(self, all_stat, protection=1e-2, mixed_type=False):
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
        self.fitting.compute_input_stats(all_stat, protection=protection)

    def _compute_output_stat(self, all_stat, mixed_type=False):
        if mixed_type:
            self.fitting.compute_output_stats(all_stat, mixed_type=mixed_type)
        else:
            self.fitting.compute_output_stats(all_stat)

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
            type_embedding = self.typeebd.build(
                self.ntypes,
                reuse=reuse,
                suffix=suffix,
            )
            input_dict["type_embedding"] = type_embedding
        # spin if any
        if self.spin is not None:
            type_spin = self.spin.build(
                reuse=reuse,
                suffix=suffix,
            )
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

        if self.srtab is not None:
            nlist, rij, sel_a, sel_r = self.descrpt.get_nlist()
            nnei_a = np.cumsum(sel_a)[-1]
            nnei_r = np.cumsum(sel_r)[-1]

        atom_ener = self.fitting.build(
            dout, natoms, input_dict, reuse=reuse, suffix=suffix
        )
        self.atom_ener = atom_ener

        if self.srtab is not None:
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
            energy_diff = tab_atom_ener - tf.reshape(atom_ener, [-1, natoms[0]])
            tab_atom_ener = tf.reshape(sw_lambda, [-1]) * tf.reshape(
                tab_atom_ener, [-1]
            )
            atom_ener = tf.reshape(inv_sw_lambda, [-1]) * atom_ener
            energy_raw = tab_atom_ener + atom_ener
        else:
            energy_raw = atom_ener

        nloc_atom = (
            natoms[0]
            if self.spin is None
            else tf.reduce_sum(natoms[2 : 2 + len(self.spin.use_spin)])
        )
        energy_raw = tf.reshape(
            energy_raw, [-1, nloc_atom], name="o_atom_energy" + suffix
        )
        energy = tf.reduce_sum(
            global_cvt_2_ener_float(energy_raw), axis=1, name="o_energy" + suffix
        )

        force, virial, atom_virial = self.descrpt.prod_force_virial(atom_ener, natoms)

        if self.srtab is not None:
            sw_force = op_module.soft_min_force(
                energy_diff, sw_deriv, nlist, natoms, n_a_sel=nnei_a, n_r_sel=nnei_r
            )
            force = force + sw_force + tab_force

        force = tf.reshape(force, [-1, 3 * natoms[1]])
        if self.spin is not None:
            # split and concatenate force to compute local atom force and magnetic force
            judge = tf.equal(natoms[0], natoms[1])
            force = tf.cond(
                judge,
                lambda: self.natoms_match(force, natoms),
                lambda: self.natoms_not_match(force, natoms, atype),
            )

        force = tf.reshape(force, [-1, 3 * natoms[1]], name="o_force" + suffix)

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
                + tf.reduce_sum(tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis=1)
            )

        virial = tf.reshape(virial, [-1, 9], name="o_virial" + suffix)
        atom_virial = tf.reshape(
            atom_virial, [-1, 9 * natoms[1]], name="o_atom_virial" + suffix
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
        # self.frz_model will control the self.model to import the descriptor from the given frozen model instead of building from scratch...
        # initialize fitting net with the given compressed frozen model
        if model_type == "original_model":
            self.descrpt.init_variables(graph, graph_def, suffix=suffix)
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("original_model", name="model_type", dtype=tf.string)
        elif model_type == "compressed_model":
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            raise RuntimeError("Unknown model type %s" % model_type)
        if self.typeebd is not None:
            self.typeebd.init_variables(graph, graph_def, suffix=suffix)

    def natoms_match(self, force, natoms):
        use_spin = self.spin.use_spin
        virtual_len = self.spin.virtual_len
        spin_norm = self.spin.spin_norm
        natoms_index = tf.concat([[0], tf.cumsum(natoms[2:])], axis=0)
        force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_real_list.append(
                    tf.slice(
                        force, [0, natoms_index[idx] * 3], [-1, natoms[idx + 2] * 3]
                    )
                    + tf.slice(
                        force,
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [-1, natoms[idx + 2 + len(use_spin)] * 3],
                    )
                )
            else:
                force_real_list.append(
                    tf.slice(
                        force, [0, natoms_index[idx] * 3], [-1, natoms[idx + 2] * 3]
                    )
                )
        force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_mag_list.append(
                    tf.slice(
                        force,
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [-1, natoms[idx + 2 + len(use_spin)] * 3],
                    )
                )
                force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        force_real = tf.concat(force_real_list, axis=1)
        force_mag = tf.concat(force_mag_list, axis=1)
        loc_force = tf.concat([force_real, force_mag], axis=1)
        force = loc_force
        return force

    def natoms_not_match(self, force, natoms, atype):
        # if ghost atoms exist, compute ghost atom force and magnetic force
        # compute ghost atom force and magnetic force
        use_spin = self.spin.use_spin
        virtual_len = self.spin.virtual_len
        spin_norm = self.spin.spin_norm
        loc_force = self.natoms_match(force, natoms)
        aatype = atype[0, :]
        ghost_atype = aatype[natoms[0] :]
        _, _, ghost_natoms = tf.unique_with_counts(ghost_atype)
        ghost_natoms_index = tf.concat([[0], tf.cumsum(ghost_natoms)], axis=0)
        ghost_natoms_index += natoms[0]

        ghost_force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_real_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx] * 3],
                        [-1, ghost_natoms[idx] * 3],
                    )
                    + tf.slice(
                        force,
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [-1, ghost_natoms[idx + len(use_spin)] * 3],
                    )
                )
            else:
                ghost_force_real_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx] * 3],
                        [-1, ghost_natoms[idx] * 3],
                    )
                )
        ghost_force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_mag_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [-1, ghost_natoms[idx + len(use_spin)] * 3],
                    )
                )
                ghost_force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        ghost_force_real = tf.concat(ghost_force_real_list, axis=1)
        ghost_force_mag = tf.concat(ghost_force_mag_list, axis=1)
        ghost_force = tf.concat([ghost_force_real, ghost_force_mag], axis=1)
        force = tf.concat([loc_force, ghost_force], axis=1)
        return force
