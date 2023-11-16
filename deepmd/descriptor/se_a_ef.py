# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.common import (
    add_data_requirement,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.utils.sess import (
    run_sess,
)

from .descriptor import (
    Descriptor,
)
from .se import (
    DescrptSe,
)
from .se_a import (
    DescrptSeA,
)


@Descriptor.register("se_a_ef")
class DescrptSeAEf(DescrptSe):
    r"""Smooth edition descriptor with Ef.

    Parameters
    ----------
    rcut
            The cut-off radius
    rcut_smth
            From where the environment matrix should be smoothed
    sel : list[str]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    axis_neuron
            Number of the axis neuron (number of columns of the sub-matrix of the embedding matrix)
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed=False,
        **kwargs,
    ) -> None:
        """Constructor."""
        self.descrpt_para = DescrptSeAEfLower(
            op_module.descrpt_se_a_ef_para,
            rcut,
            rcut_smth,
            sel,
            neuron,
            axis_neuron,
            resnet_dt,
            trainable,
            seed,
            type_one_side,
            exclude_types,
            set_davg_zero,
            activation_function,
            precision,
            uniform_seed,
        )
        self.descrpt_vert = DescrptSeAEfLower(
            op_module.descrpt_se_a_ef_vert,
            rcut,
            rcut_smth,
            sel,
            neuron,
            axis_neuron,
            resnet_dt,
            trainable,
            seed,
            type_one_side,
            exclude_types,
            set_davg_zero,
            activation_function,
            precision,
            uniform_seed,
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.descrpt_vert.rcut_r

    def get_ntypes(self) -> int:
        """Returns the number of atom types."""
        return self.descrpt_vert.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.descrpt_vert.get_dim_out() + self.descrpt_para.get_dim_out()

    def get_dim_rot_mat_1(self) -> int:
        """Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3."""
        return self.descrpt_vert.filter_neuron[-1]

    def get_rot_mat(self) -> tf.Tensor:
        """Get rotational matrix."""
        return self.qmat

    def get_nlist(self) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """Returns neighbor information.

        Returns
        -------
        nlist
            Neighbor list
        rij
            The relative distance between the neighbor and the center atom.
        sel_a
            The number of neighbors with full information
        sel_r
            The number of neighbors with only radial information
        """
        return (
            self.descrpt_vert.nlist,
            self.descrpt_vert.rij,
            self.descrpt_vert.sel_a,
            self.descrpt_vert.sel_r,
        )

    def compute_input_stats(
        self,
        data_coord: list,
        data_box: list,
        data_atype: list,
        natoms_vec: list,
        mesh: list,
        input_dict: dict,
        **kwargs,
    ) -> None:
        """Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.

        Parameters
        ----------
        data_coord
            The coordinates. Can be generated by deepmd.model.make_stat_input
        data_box
            The box. Can be generated by deepmd.model.make_stat_input
        data_atype
            The atom types. Can be generated by deepmd.model.make_stat_input
        natoms_vec
            The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.model.make_stat_input
        mesh
            The mesh for neighbor searching. Can be generated by deepmd.model.make_stat_input
        input_dict
            Dictionary for additional input
        **kwargs
            Additional keyword arguments.
        """
        self.descrpt_vert.compute_input_stats(
            data_coord, data_box, data_atype, natoms_vec, mesh, input_dict
        )
        self.descrpt_para.compute_input_stats(
            data_coord, data_box, data_atype, natoms_vec, mesh, input_dict
        )

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        """Build the computational graph for the descriptor.

        Parameters
        ----------
        coord_
            The coordinate of atoms
        atype_
            The type of atoms
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box_ : tf.Tensor
            The box of the system
        mesh
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict
            Dictionary for additional inputs. Should have 'efield'.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor
            The output descriptor
        """
        self.dout_vert = self.descrpt_vert.build(
            coord_, atype_, natoms, box_, mesh, input_dict
        )
        self.dout_para = self.descrpt_para.build(
            coord_, atype_, natoms, box_, mesh, input_dict, reuse=True
        )
        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        nframes = tf.shape(coord)[0]
        self.dout_vert = tf.reshape(
            self.dout_vert, [nframes * natoms[0], self.descrpt_vert.get_dim_out()]
        )
        self.dout_para = tf.reshape(
            self.dout_para, [nframes * natoms[0], self.descrpt_para.get_dim_out()]
        )
        self.dout = tf.concat([self.dout_vert, self.dout_para], axis=1)
        self.dout = tf.reshape(self.dout, [nframes, natoms[0], self.get_dim_out()])
        self.qmat = self.descrpt_vert.qmat + self.descrpt_para.qmat

        tf.summary.histogram("embedding_net_output", self.dout)

        return self.dout

    def prod_force_virial(
        self, atom_ener: tf.Tensor, natoms: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute force and virial.

        Parameters
        ----------
        atom_ener
            The atomic energy
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        Returns
        -------
        force
            The force on atoms
        virial
            The total virial
        atom_virial
            The atomic virial
        """
        f_vert, v_vert, av_vert = self.descrpt_vert.prod_force_virial(atom_ener, natoms)
        f_para, v_para, av_para = self.descrpt_para.prod_force_virial(atom_ener, natoms)
        force = f_vert + f_para
        virial = v_vert + v_para
        atom_vir = av_vert + av_para
        return force, virial, atom_vir


class DescrptSeAEfLower(DescrptSeA):
    """Helper class for implementing DescrptSeAEf."""

    def __init__(
        self,
        op,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
    ) -> None:
        DescrptSeA.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            neuron,
            axis_neuron,
            resnet_dt,
            trainable,
            seed,
            type_one_side,
            exclude_types,
            set_davg_zero,
            activation_function,
            precision,
            uniform_seed,
        )
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.filter_neuron = neuron
        self.n_axis_neuron = axis_neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.trainable = trainable
        self.op = op

        # descrpt config
        self.sel_r = [0 for ii in range(len(self.sel_a))]
        self.ntypes = len(self.sel_a)
        assert self.ntypes == len(self.sel_r)
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.useBN = False
        self.dstd = None
        self.davg = None

        add_data_requirement("efield", 3, atomic=True, must=True, high_prec=False)

        self.place_holders = {}
        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = "d_sea_ef_"
            for ii in ["coord", "box"]:
                self.place_holders[ii] = tf.placeholder(
                    GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_" + ii
                )
            self.place_holders["type"] = tf.placeholder(
                tf.int32, [None, None], name=name_pfx + "t_type"
            )
            self.place_holders["natoms_vec"] = tf.placeholder(
                tf.int32, [self.ntypes + 2], name=name_pfx + "t_natoms"
            )
            self.place_holders["default_mesh"] = tf.placeholder(
                tf.int32, [None], name=name_pfx + "t_mesh"
            )
            self.place_holders["efield"] = tf.placeholder(
                GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_efield"
            )
            self.stat_descrpt, descrpt_deriv, rij, nlist = self.op(
                self.place_holders["coord"],
                self.place_holders["type"],
                self.place_holders["natoms_vec"],
                self.place_holders["box"],
                self.place_holders["default_mesh"],
                self.place_holders["efield"],
                tf.constant(avg_zero),
                tf.constant(std_ones),
                rcut_a=self.rcut_a,
                rcut_r=self.rcut_r,
                rcut_r_smth=self.rcut_r_smth,
                sel_a=self.sel_a,
                sel_r=self.sel_r,
            )
        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)

    def compute_input_stats(
        self,
        data_coord,
        data_box,
        data_atype,
        natoms_vec,
        mesh,
        input_dict,
        **kwargs,
    ):
        data_efield = input_dict["efield"]
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc, bb, tt, nn, mm, ee in zip(
                data_coord, data_box, data_atype, natoms_vec, mesh, data_efield
            ):
                sysr, sysr2, sysa, sysa2, sysn = self._compute_dstats_sys_smth(
                    cc, bb, tt, nn, mm, ee
                )
                sumr.append(sysr)
                suma.append(sysa)
                sumn.append(sysn)
                sumr2.append(sysr2)
                suma2.append(sysa2)
            sumr = np.sum(sumr, axis=0)
            suma = np.sum(suma, axis=0)
            sumn = np.sum(sumn, axis=0)
            sumr2 = np.sum(sumr2, axis=0)
            suma2 = np.sum(suma2, axis=0)
            for type_i in range(self.ntypes):
                davgunit = [sumr[type_i] / sumn[type_i], 0, 0, 0]
                dstdunit = [
                    self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]),
                    self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]),
                    self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]),
                    self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]),
                ]
                davg = np.tile(davgunit, self.ndescrpt // 4)
                dstd = np.tile(dstdunit, self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)

        self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)

    def _normalize_3d(self, a):
        na = tf.norm(a, axis=1)
        na = tf.tile(tf.reshape(na, [-1, 1]), tf.constant([1, 3]))
        return tf.divide(a, na)

    def build(
        self, coord_, atype_, natoms, box_, mesh, input_dict, suffix="", reuse=None
    ):
        efield = input_dict["efield"]
        davg = self.davg
        dstd = self.dstd
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])
            if dstd is None:
                dstd = np.ones([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(
                np.max([self.rcut_r, self.rcut_a]),
                name="rcut",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, name="ndescrpt", dtype=tf.int32)
            t_sel = tf.constant(self.sel_a, name="sel", dtype=tf.int32)
            self.t_avg = tf.get_variable(
                "t_avg",
                davg.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(davg),
            )
            self.t_std = tf.get_variable(
                "t_std",
                dstd.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(dstd),
            )

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        efield = tf.reshape(efield, [-1, 3])
        efield = self._normalize_3d(efield)
        efield = tf.reshape(efield, [-1, natoms[0] * 3])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist = self.op(
            coord,
            atype,
            natoms,
            box,
            mesh,
            efield,
            self.t_avg,
            self.t_std,
            rcut_a=self.rcut_a,
            rcut_r=self.rcut_r,
            rcut_r_smth=self.rcut_r_smth,
            sel_a=self.sel_a,
            sel_r=self.sel_r,
        )

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name="o_rmat")
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name="o_rmat_deriv")
        self.rij = tf.identity(self.rij, name="o_rij")
        self.nlist = tf.identity(self.nlist, name="o_nlist")

        # only used when tensorboard was set as true
        tf.summary.histogram("descrpt", self.descrpt)
        tf.summary.histogram("rij", self.rij)
        tf.summary.histogram("nlist", self.nlist)

        self.dout, self.qmat = self._pass_filter(
            self.descrpt_reshape,
            atype,
            natoms,
            input_dict,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )
        tf.summary.histogram("embedding_net_output", self.dout)

        return self.dout

    def _compute_dstats_sys_smth(
        self, data_coord, data_box, data_atype, natoms_vec, mesh, data_efield
    ):
        dd_all = run_sess(
            self.sub_sess,
            self.stat_descrpt,
            feed_dict={
                self.place_holders["coord"]: data_coord,
                self.place_holders["type"]: data_atype,
                self.place_holders["natoms_vec"]: natoms_vec,
                self.place_holders["box"]: data_box,
                self.place_holders["default_mesh"]: mesh,
                self.place_holders["efield"]: data_efield,
            },
        )
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2 + type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index
            # compute
            dd = np.reshape(dd, [-1, 4])
            ddr = dd[:, :1]
            dda = dd[:, 1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.0
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.0
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn
