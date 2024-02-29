# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)
from deepmd.tf.common import (
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.network import (
    embedding_net,
    embedding_net_rand_seed_shift,
)
from deepmd.tf.utils.sess import (
    run_sess,
)
from deepmd.tf.utils.spin import (
    Spin,
)
from deepmd.tf.utils.tabulate import (
    DPTabulate,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .descriptor import (
    Descriptor,
)
from .se import (
    DescrptSe,
)


@Descriptor.register("se_e2_r")
@Descriptor.register("se_r")
class DescrptSeR(DescrptSe):
    r"""DeepPot-SE constructed from radial information of atomic configurations.

    The embedding takes the distance between atoms as input.

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
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        multi_task: bool = False,
        spin: Optional[Spin] = None,
        **kwargs,
    ) -> None:
        """Constructor."""
        if rcut < rcut_smth:
            raise RuntimeError(
                f"rcut_smth ({rcut_smth:f}) should be no more than rcut ({rcut:f})!"
            )
        self.sel_r = sel
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.filter_neuron = neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = embedding_net_rand_seed_shift(self.filter_neuron)
        self.trainable = trainable
        self.filter_activation_fn = get_activation_func(activation_function)
        self.activation_function_name = activation_function
        self.filter_precision = get_precision(precision)
        self.orig_exclude_types = exclude_types
        self.exclude_types = set()
        for tt in exclude_types:
            assert len(tt) == 2
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero
        self.type_one_side = type_one_side
        self.spin = spin

        # extend sel_r for spin system
        if self.spin is not None:
            self.ntypes_spin = self.spin.get_ntypes_spin()
            self.sel_r_spin = self.sel_r[: self.ntypes_spin]
            self.sel_r.extend(self.sel_r_spin)
        else:
            self.ntypes_spin = 0

        # descrpt config
        self.sel_a = [0 for ii in range(len(self.sel_r))]
        self.ntypes = len(self.sel_r)
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.nnei_r
        self.useBN = False
        self.davg = None
        self.dstd = None
        self.compress = False
        self.embedding_net_variables = None

        self.place_holders = {}
        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = "d_ser_"
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
            self.stat_descrpt, descrpt_deriv, rij, nlist = op_module.prod_env_mat_r(
                self.place_holders["coord"],
                self.place_holders["type"],
                self.place_holders["natoms_vec"],
                self.place_holders["box"],
                self.place_holders["default_mesh"],
                tf.constant(avg_zero),
                tf.constant(std_ones),
                rcut=self.rcut,
                rcut_smth=self.rcut_smth,
                sel=self.sel_r,
            )
            self.sub_sess = tf.Session(
                graph=sub_graph, config=default_tf_session_config
            )
        self.multi_task = multi_task
        if multi_task:
            self.stat_dict = {"sumr": [], "sumn": [], "sumr2": []}

    def get_rcut(self):
        """Returns the cut-off radius."""
        return self.rcut

    def get_ntypes(self):
        """Returns the number of atom types."""
        return self.ntypes

    def get_dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1]

    def get_nlist(self):
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
        return self.nlist, self.rij, self.sel_a, self.sel_r

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
        """Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.

        Parameters
        ----------
        data_coord
            The coordinates. Can be generated by deepmd.tf.model.make_stat_input
        data_box
            The box. Can be generated by deepmd.tf.model.make_stat_input
        data_atype
            The atom types. Can be generated by deepmd.tf.model.make_stat_input
        natoms_vec
            The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.tf.model.make_stat_input
        mesh
            The mesh for neighbor searching. Can be generated by deepmd.tf.model.make_stat_input
        input_dict
            Dictionary for additional input
        **kwargs
            Additional keyword arguments.
        """
        sumr = []
        sumn = []
        sumr2 = []
        for cc, bb, tt, nn, mm in zip(
            data_coord, data_box, data_atype, natoms_vec, mesh
        ):
            sysr, sysr2, sysn = self._compute_dstats_sys_se_r(cc, bb, tt, nn, mm)
            sumr.append(sysr)
            sumn.append(sysn)
            sumr2.append(sysr2)
        if not self.multi_task:
            stat_dict = {"sumr": sumr, "sumn": sumn, "sumr2": sumr2}
            self.merge_input_stats(stat_dict)
        else:
            self.stat_dict["sumr"] += sumr
            self.stat_dict["sumn"] += sumn
            self.stat_dict["sumr2"] += sumr2

    def merge_input_stats(self, stat_dict):
        """Merge the statisitcs computed from compute_input_stats to obtain the self.davg and self.dstd.

        Parameters
        ----------
        stat_dict
                The dict of statisitcs computed from compute_input_stats, including:
            sumr
                    The sum of radial statisitcs.
            sumn
                    The sum of neighbor numbers.
            sumr2
                    The sum of square of radial statisitcs.
        """
        all_davg = []
        all_dstd = []
        sumr = np.sum(stat_dict["sumr"], axis=0)
        sumn = np.sum(stat_dict["sumn"], axis=0)
        sumr2 = np.sum(stat_dict["sumr2"], axis=0)
        for type_i in range(self.ntypes):
            davgunit = [sumr[type_i] / sumn[type_i]]
            dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i])]
            davg = np.tile(davgunit, self.ndescrpt // 1)
            dstd = np.tile(dstdunit, self.ndescrpt // 1)
            all_davg.append(davg)
            all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)

    def enable_compression(
        self,
        min_nbor_dist: float,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
        suffix: str = "",
    ) -> None:
        """Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        graph : tf.Graph
            The graph of the model
        graph_def : tf.GraphDef
            The graph_def of the model
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        suffix : str, optional
            The suffix of the scope
        """
        assert (
            not self.filter_resnet_dt
        ), "Model compression error: descriptor resnet_dt must be false!"

        self.compress = True
        self.table = DPTabulate(
            self,
            self.filter_neuron,
            graph,
            graph_def,
            activation_fn=self.filter_activation_fn,
            suffix=suffix,
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )

        self.davg = get_tensor_by_name_from_graph(
            graph, "descrpt_attr%s/t_avg" % suffix
        )
        self.dstd = get_tensor_by_name_from_graph(
            graph, "descrpt_attr%s/t_std" % suffix
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
            Dictionary for additional inputs
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor
            The output descriptor
        """
        davg = self.davg
        dstd = self.dstd
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])
            if dstd is None:
                dstd = np.ones([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(
                self.rcut, name="rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
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

        (
            self.descrpt,
            self.descrpt_deriv,
            self.rij,
            self.nlist,
        ) = op_module.prod_env_mat_r(
            coord,
            atype,
            natoms,
            box,
            mesh,
            self.t_avg,
            self.t_std,
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel_r,
        )

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self._identity_tensors(suffix=suffix)

        # only used when tensorboard was set as true
        tf.summary.histogram("descrpt", self.descrpt)
        tf.summary.histogram("rij", self.rij)
        tf.summary.histogram("nlist", self.nlist)

        self.dout = self._pass_filter(
            self.descrpt_reshape,
            atype,
            natoms,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )
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
        [net_deriv] = tf.gradients(atom_ener, self.descrpt_reshape)
        tf.summary.histogram("net_derivative", net_deriv)
        net_deriv_reshape = tf.reshape(
            net_deriv,
            [
                np.asarray(-1, dtype=np.int64),
                natoms[0] * np.asarray(self.ndescrpt, dtype=np.int64),
            ],
        )
        force = op_module.prod_force_se_r(
            net_deriv_reshape, self.descrpt_deriv, self.nlist, natoms
        )
        virial, atom_virial = op_module.prod_virial_se_r(
            net_deriv_reshape, self.descrpt_deriv, self.rij, self.nlist, natoms
        )
        tf.summary.histogram("force", force)
        tf.summary.histogram("virial", virial)
        tf.summary.histogram("atom_virial", atom_virial)

        return force, virial, atom_virial

    def _pass_filter(
        self, inputs, atype, natoms, reuse=None, suffix="", trainable=True
    ):
        start_index = 0
        inputs = tf.reshape(inputs, [-1, natoms[0], self.ndescrpt])
        output = []
        if not self.type_one_side:
            for type_i in range(self.ntypes):
                inputs_i = tf.slice(
                    inputs, [0, start_index, 0], [-1, natoms[2 + type_i], -1]
                )
                inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
                filter_name = "filter_type_" + str(type_i) + suffix
                layer = self._filter_r(
                    inputs_i,
                    type_i,
                    name=filter_name,
                    natoms=natoms,
                    reuse=reuse,
                    trainable=trainable,
                    activation_fn=self.filter_activation_fn,
                )
                layer = tf.reshape(
                    layer, [tf.shape(inputs)[0], natoms[2 + type_i], self.get_dim_out()]
                )
                output.append(layer)
                start_index += natoms[2 + type_i]
        else:
            inputs_i = inputs
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            type_i = -1
            if len(self.exclude_types):
                atype_nloc = tf.reshape(
                    tf.slice(atype, [0, 0], [-1, natoms[0]]), [-1]
                )  # when nloc != nall, pass nloc to mask
                mask = self.build_type_exclude_mask(
                    self.exclude_types,
                    self.ntypes,
                    self.sel_r,
                    self.ndescrpt,
                    atype_nloc,
                    tf.shape(inputs_i)[0],
                )
                inputs_i *= mask
            layer = self._filter_r(
                inputs_i,
                type_i,
                name="filter_type_all" + suffix,
                natoms=natoms,
                reuse=reuse,
                trainable=trainable,
                activation_fn=self.filter_activation_fn,
            )
            layer = tf.reshape(
                layer, [tf.shape(inputs)[0], natoms[0], self.get_dim_out()]
            )
            output.append(layer)
        output = tf.concat(output, axis=1)
        return output

    def _compute_dstats_sys_se_r(
        self, data_coord, data_box, data_atype, natoms_vec, mesh
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
            },
        )
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysn = []
        sysr2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2 + type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index
            # compute
            dd = np.reshape(dd, [-1, 1])
            ddr = dd[:, :1]
            sumr = np.sum(ddr)
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            sysr.append(sumr)
            sysn.append(sumn)
            sysr2.append(sumr2)
        return sysr, sysr2, sysn

    def _compute_std(self, sumv2, sumv, sumn):
        val = np.sqrt(sumv2 / sumn - np.multiply(sumv / sumn, sumv / sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val

    @cast_precision
    def _filter_r(
        self,
        inputs,
        type_input,
        natoms,
        activation_fn=tf.nn.tanh,
        stddev=1.0,
        bavg=0.0,
        name="linear",
        reuse=None,
        trainable=True,
    ):
        # natom x nei
        outputs_size = [1, *self.filter_neuron]
        with tf.variable_scope(name, reuse=reuse):
            start_index = 0
            xyz_scatter_total = []
            for type_i in range(self.ntypes):
                # cut-out inputs
                # with natom x nei_type_i
                inputs_i = tf.slice(inputs, [0, start_index], [-1, self.sel_r[type_i]])
                start_index += self.sel_r[type_i]
                shape_i = inputs_i.get_shape().as_list()
                # with (natom x nei_type_i) x 1
                xyz_scatter = tf.reshape(inputs_i, [-1, 1])
                if self.compress and ((type_input, type_i) not in self.exclude_types):
                    net = "filter_" + str(type_input) + "_net_" + str(type_i)
                    info = [
                        self.lower[net],
                        self.upper[net],
                        self.upper[net] * self.table_config[0],
                        self.table_config[1],
                        self.table_config[2],
                        self.table_config[3],
                    ]
                    xyz_scatter = op_module.tabulate_fusion_se_r(
                        tf.cast(self.table.data[net], self.filter_precision),
                        info,
                        inputs_i,
                        last_layer_size=outputs_size[-1],
                    )
                elif (type_input, type_i) not in self.exclude_types:
                    xyz_scatter = embedding_net(
                        xyz_scatter,
                        self.filter_neuron,
                        self.filter_precision,
                        activation_fn=activation_fn,
                        resnet_dt=self.filter_resnet_dt,
                        name_suffix="_" + str(type_i),
                        stddev=stddev,
                        bavg=bavg,
                        seed=self.seed,
                        trainable=trainable,
                        uniform_seed=self.uniform_seed,
                        initial_variables=self.embedding_net_variables,
                    )
                    if (not self.uniform_seed) and (self.seed is not None):
                        self.seed += self.seed_shift
                    # natom x nei_type_i x out_size
                    xyz_scatter = tf.reshape(
                        xyz_scatter, (-1, shape_i[1], outputs_size[-1])
                    )
                else:
                    natom = tf.shape(inputs)[0]
                    xyz_scatter = tf.cast(
                        tf.fill((natom, shape_i[1], outputs_size[-1]), 0.0),
                        self.filter_precision,
                    )
                xyz_scatter_total.append(xyz_scatter)

            # natom x nei x outputs_size
            xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
            # natom x outputs_size
            #
            res_rescale = 1.0 / 5.0
            result = tf.reduce_mean(xyz_scatter, axis=1) * res_rescale

        return result

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        if cls is not DescrptSeR:
            raise NotImplementedError("Not implemented in class %s" % cls.__name__)
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        embedding_net_variables = cls.deserialize_network(
            data.pop("embeddings"), suffix=suffix
        )
        data.pop("env_mat")
        variables = data.pop("@variables")
        descriptor = cls(**data)
        descriptor.embedding_net_variables = embedding_net_variables
        descriptor.davg = variables["davg"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.dstd = variables["dstd"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        return descriptor

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Parameters
        ----------
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The serialized data
        """
        if type(self) is not DescrptSeR:
            raise NotImplementedError(
                "Not implemented in class %s" % self.__class__.__name__
            )
        if self.embedding_net_variables is None:
            raise RuntimeError("init_variables must be called before serialize")
        if self.spin is not None:
            raise NotImplementedError("spin is unsupported")
        assert self.davg is not None
        assert self.dstd is not None
        # TODO: not sure how to handle type embedding - type embedding is not a model parameter,
        # but instead a part of the input data. Maybe the interface should be refactored...
        return {
            "@class": "Descriptor",
            "type": "se_r",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel_r,
            "neuron": self.filter_neuron,
            "resnet_dt": self.filter_resnet_dt,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "exclude_types": list(self.orig_exclude_types),
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function_name,
            "precision": self.filter_precision.name,
            "embeddings": self.serialize_network(
                ntypes=self.ntypes,
                ndim=(1 if self.type_one_side else 2),
                in_dim=1,
                neuron=self.filter_neuron,
                activation_function=self.activation_function_name,
                resnet_dt=self.filter_resnet_dt,
                variables=self.embedding_net_variables,
                excluded_types=self.exclude_types,
                suffix=suffix,
            ),
            "env_mat": EnvMat(self.rcut, self.rcut_smth).serialize(),
            "@variables": {
                "davg": self.davg.reshape(self.ntypes, self.nnei_r, 1),
                "dstd": self.dstd.reshape(self.ntypes, self.nnei_r, 1),
            },
            "spin": self.spin,
        }
