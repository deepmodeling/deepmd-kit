from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

import numpy as np

from deepmd.env import (
    MODEL_VERSION,
    paddle,
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

if TYPE_CHECKING:
    import paddle  # noqa: F811

    from deepmd.fit import ener  # noqa: F811


class EnerModel(Model, paddle.nn.Layer):
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
        fitting: "ener.EnerFitting",
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
        super().__init__()
        # super(EnerModel, self).__init__(name_scope="EnerModel")
        """Constructor."""
        super().__init__()
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

        # content of build below
        self.t_tmap = " ".join(self.type_map)
        self.t_mt = self.model_type
        self.t_ver = str(MODEL_VERSION)
        # NOTE: workaround for string type is not supported in Paddle
        self.register_buffer(
            "buffer_tmap",
            paddle.to_tensor([ord(c) for c in self.t_tmap], dtype="int32"),
        )
        self.register_buffer(
            "buffer_model_type",
            paddle.to_tensor([ord(c) for c in self.t_mt], dtype="int32"),
        )
        self.register_buffer(
            "buffer_model_version",
            paddle.to_tensor([ord(c) for c in self.t_ver], dtype="int32"),
        )
        if self.srtab is not None:
            tab_info, tab_data = self.srtab.get()
            self.tab_info = paddle.register_buffer(
                "buffer_t_tab_info",
                paddle.to_tensor(tab_info, dtype=paddle.float64),
            )
            self.tab_data = paddle.register_buffer(
                "buffer_t_tab_data",
                paddle.to_tensor(tab_data, dtype=paddle.float64),
            )

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

    def forward(
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

        coord = paddle.reshape(coord_, [-1, natoms[1] * 3])
        atype = paddle.reshape(atype_, [-1, natoms[1]])
        # input_dict["nframes"] = paddle.shape(coord)[0] # 推理模型导出的时候注释掉这里，否则会报错

        # type embedding if any
        # if self.typeebd is not None:
        #     type_embedding = self.typeebd.build(
        #         self.ntypes,
        #         reuse=reuse,
        #         suffix=suffix,
        #     )
        #     input_dict["type_embedding"] = type_embedding
        # spin if any
        # if self.spin is not None:
        #     type_spin = self.spin.build(
        #         reuse=reuse,
        #         suffix=suffix,
        #     )
        input_dict["atype"] = atype_

        dout = self.descrpt(
            coord,
            atype,
            natoms,
            box,
            mesh,
            input_dict,
            suffix=suffix,
            reuse=reuse,
        )

        if self.srtab is not None:
            nlist, rij, sel_a, sel_r = self.descrpt.get_nlist()
            nnei_a = np.cumsum(sel_a)[-1]
            nnei_r = np.cumsum(sel_r)[-1]

        atom_ener = self.fitting(dout, natoms, input_dict, reuse=reuse, suffix=suffix)
        self.atom_ener = atom_ener

        if self.srtab is not None:
            raise NotImplementedError(
                f"srtab not implemented in {self.__class__.__name__}"
            )
            # sw_lambda, sw_deriv = op_module.soft_min_switch(
            #     atype,
            #     rij,
            #     nlist,
            #     natoms,
            #     sel_a=sel_a,
            #     sel_r=sel_r,
            #     alpha=self.smin_alpha,
            #     rmin=self.sw_rmin,
            #     rmax=self.sw_rmax,
            # )
            # inv_sw_lambda = 1.0 - sw_lambda
            # # NOTICE:
            # # atom energy is not scaled,
            # # force and virial are scaled
            # tab_atom_ener, tab_force, tab_atom_virial = op_module.pair_tab(
            #     self.tab_info,
            #     self.tab_data,
            #     atype,
            #     rij,
            #     nlist,
            #     natoms,
            #     sw_lambda,
            #     sel_a=sel_a,
            #     sel_r=sel_r,
            # )
            # energy_diff = tab_atom_ener - tf.reshape(atom_ener, [-1, natoms[0]])
            # tab_atom_ener = tf.reshape(sw_lambda, [-1]) * tf.reshape(
            #     tab_atom_ener, [-1]
            # )
            # atom_ener = tf.reshape(inv_sw_lambda, [-1]) * atom_ener
            # energy_raw = tab_atom_ener + atom_ener
        else:
            energy_raw = atom_ener

        nloc_atom = (
            natoms[0]
            if self.spin is None
            else paddle.sum(natoms[2 : 2 + len(self.spin.use_spin)]).item()
        )
        energy_raw = paddle.reshape(
            energy_raw, [-1, nloc_atom], name="o_atom_energy" + suffix
        )
        energy = paddle.sum(energy_raw, axis=1, name="o_energy" + suffix)

        force, virial, atom_virial = self.descrpt.prod_force_virial(atom_ener, natoms)
        # force: [1, all_atoms*3]
        # virial: [1, 9]
        # atom_virial: [1, all_atoms*9]

        if self.srtab is not None:
            raise NotImplementedError()
            # sw_force = op_module.soft_min_force(
            #     energy_diff, sw_deriv, nlist, natoms, n_a_sel=nnei_a, n_r_sel=nnei_r
            # )
            # force = force + sw_force + tab_force

        force = paddle.reshape(force, [-1, 3 * natoms[1]])  # [1, all_atoms*3]
        if self.spin is not None:
            # split and concatenate force to compute local atom force and magnetic force
            judge = paddle.equal(natoms[0], natoms[1])
            if judge.item():
                force = self.natoms_match(force, natoms)
            else:
                force = self.natoms_not_match(force, natoms, atype)

        force = paddle.reshape(force, [-1, 3 * natoms[1]], name="o_force" + suffix)

        if self.srtab is not None:
            raise NotImplementedError()
            # sw_virial, sw_atom_virial = op_module.soft_min_virial(
            #     energy_diff,
            #     sw_deriv,
            #     rij,
            #     nlist,
            #     natoms,
            #     n_a_sel=nnei_a,
            #     n_r_sel=nnei_r,
            # )
            # atom_virial = atom_virial + sw_atom_virial + tab_atom_virial
            # virial = (
            #     virial
            #     + sw_virial
            #     + tf.sum(tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis=1)
            # )

        virial = paddle.reshape(virial, [-1, 9], name="o_virial" + suffix)
        atom_virial = paddle.reshape(
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
        # natoms_index = paddle.concat(
        #     [
        #         paddle.to_tensor([0]),
        #         paddle.sum(natoms[2:])
        #     ],
        #     axis=0,
        # )
        natoms_index = paddle.concat(
            [
                paddle.to_tensor([0], dtype=natoms.dtype),
                paddle.cumsum(natoms[2:]),
            ],
            axis=0,
        )
        force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_real_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, natoms_index[idx] * 3],
                        [
                            force.shape[0],
                            natoms_index[idx] * 3 + natoms[idx + 2].item() * 3,
                        ],
                    )
                    + paddle.slice(
                        force,
                        [0, 1],
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [
                            force.shape[0],
                            natoms_index[idx + len(use_spin)] * 3
                            + natoms[idx + 2 + len(use_spin)].item() * 3,
                        ],
                    )
                )
            else:
                force_real_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, natoms_index[idx] * 3],
                        [
                            force.shape[0],
                            natoms_index[idx] * 3 + natoms[idx + 2].item() * 3,
                        ],
                    )
                )
        force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_mag_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [
                            force.shape[0],
                            natoms_index[idx + len(use_spin)] * 3
                            + natoms[idx + 2 + len(use_spin)].item() * 3,
                        ],
                    )
                )
                force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        force_real = paddle.concat(force_real_list, axis=1)
        force_mag = paddle.concat(force_mag_list, axis=1)
        loc_force = paddle.concat([force_real, force_mag], axis=1)
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

        # TODO: paddle.unique和tf.unique的返回值顺序是不一致, 下面的代码实现不正确，需要修改
        ghost_atype = aatype[natoms[0] :]
        _, idx, ghost_natoms = paddle.unique(
            ghost_atype, return_index=True, return_counts=True
        )
        idx_inv = paddle.empty_like(idx)
        # NOTE: Use inverse permutation to get equaivalent result as tf
        idx_inv[idx] = paddle.arange(0, len(idx))
        ghost_natoms = ghost_natoms[idx_inv]

        # ghost_natoms_index = [0] + paddle.cumsum(ghost_natoms).tolist()
        # for i in range(len(ghost_natoms_index)):
        #     ghost_natoms_index[i] +=  natoms[0].item()
        ghost_natoms_index = paddle.concat(
            [
                paddle.to_tensor([0], dtype=ghost_natoms.dtype),
                paddle.cumsum(ghost_natoms),
            ],
            axis=0,
        )
        ghost_natoms_index += natoms[0]

        ghost_force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_real_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, ghost_natoms_index[idx] * 3],
                        [
                            force.shape[0],
                            ghost_natoms_index[idx] * 3 + ghost_natoms[idx].item() * 3,
                        ],
                    )
                    + paddle.slice(
                        force,
                        [0, 1],
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [
                            force.shape[0],
                            ghost_natoms_index[idx + len(use_spin)] * 3
                            + ghost_natoms[idx + len(use_spin)].item() * 3,
                        ],
                    )
                )
            else:
                ghost_force_real_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, ghost_natoms_index[idx] * 3],
                        [
                            force.shape[0],
                            ghost_natoms_index[idx] * 3 + ghost_natoms[idx].item() * 3,
                        ],
                    )
                )
        ghost_force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_mag_list.append(
                    paddle.slice(
                        force,
                        [0, 1],
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [
                            force.shape[0],
                            ghost_natoms_index[idx + len(use_spin)] * 3
                            + ghost_natoms[idx + len(use_spin)].item() * 3,
                        ],
                    )
                )
                ghost_force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        ghost_force_real = paddle.concat(ghost_force_real_list, axis=1)
        ghost_force_mag = paddle.concat(ghost_force_mag_list, axis=1)
        ghost_force = paddle.concat([ghost_force_real, ghost_force_mag], axis=1)
        force = paddle.concat([loc_force, ghost_force], axis=1)
        return force
