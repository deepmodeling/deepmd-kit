import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Callable
import os
import json

import numpy as np
from deepmd.common import make_default_mesh
from deepmd.env import default_tf_session_config, tf
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.infer.deep_eval import DeepEval
from deepmd.utils.sess import run_sess
from deepmd.utils.batch_size import AutoBatchSize
from deepmd.env import op_module, GLOBAL_CONFIG

# import Ascend npu ops
dp_variant = GLOBAL_CONFIG.get("dp_variant", "cpu")
if dp_variant == "ascend":
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class DeepPot(DeepEval):
    """Constructor.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    auto_batch_size : bool or int or AutomaticBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.

    Examples
    --------
    >>> from deepmd.infer import DeepPot
    >>> import numpy as np
    >>> dp = DeepPot('graph.pb')
    >>> coord = np.array([[1,0,0], [0,0,1.5], [1,0,3]]).reshape([1, -1])
    >>> cell = np.diag(10 * np.ones(3)).reshape([1, -1])
    >>> atype = [1,0,1]
    >>> e, f, v = dp.eval(coord, cell, atype)
    
    where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.

    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
    ) -> None:

        # add these tensors on top of what is defined by DeepTensor Class
        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # descrpt attrs
                "t_ntypes": "descrpt_attr/ntypes:0",
                "t_rcut": "descrpt_attr/rcut:0",
                # fitting attrs
                "t_dfparam": "fitting_attr/dfparam:0",
                "t_daparam": "fitting_attr/daparam:0",
                # model attrs
                "t_tmap": "model_attr/tmap:0",
                # inputs
                "t_coord": "t_coord:0",
                "t_type": "t_type:0",
                "t_natoms": "t_natoms:0",
                "t_box": "t_box:0",
                "t_mesh": "t_mesh:0",
                # add output tensors
                "t_energy": "o_energy:0",
                "t_force": "o_force:0",
                "t_virial": "o_virial:0",
                "t_ae": "o_atom_energy:0",
                "t_av": "o_atom_virial:0",
                "t_descriptor": "o_descriptor:0",
            },
        )
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            auto_batch_size=auto_batch_size,
        )

        if dp_variant == "ascend":
            config = tf.ConfigProto()
            custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["mix_compile_mode"].b = False
            custom_op.parameter_map["op_debug_level"].i = 0
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            self.npu_sess = tf.Session(graph=self.graph, config=config)
            self.ASCEND_NEI_LEN = 1024

        # load optional tensors
        operations = [op.name for op in self.graph.get_operations()]
        # check if the graph has these operations:
        # if yes add them
        if 't_efield' in operations:
            self._get_tensor("t_efield:0", "t_efield")
            self.has_efield = True
        else:
            log.debug(f"Could not get tensor 't_efield:0'")
            self.t_efield = None
            self.has_efield = False

        if 'load/t_fparam' in operations:
            self.tensors.update({"t_fparam": "t_fparam:0"})
            self.has_fparam = True
        else:
            log.debug(f"Could not get tensor 't_fparam:0'")
            self.t_fparam = None
            self.has_fparam = False

        if 'load/t_aparam' in operations:
            self.tensors.update({"t_aparam": "t_aparam:0"})
            self.has_aparam = True
        else:
            log.debug(f"Could not get tensor 't_aparam:0'")
            self.t_aparam = None
            self.has_aparam = False

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_tensor(tensor_name, attr_name)

        self._run_default_sess()
        self.tmap = self.tmap.decode('UTF-8').split()        

        # setup modifier
        try:
            t_modifier_type = self._get_tensor("modifier_attr/type:0")
            self.modifier_type = run_sess(self.sess, t_modifier_type).decode("UTF-8")
        except (ValueError, KeyError):
            self.modifier_type = None

        if self.modifier_type == "dipole_charge":
            t_mdl_name = self._get_tensor("modifier_attr/mdl_name:0")
            t_mdl_charge_map = self._get_tensor("modifier_attr/mdl_charge_map:0")
            t_sys_charge_map = self._get_tensor("modifier_attr/sys_charge_map:0")
            t_ewald_h = self._get_tensor("modifier_attr/ewald_h:0")
            t_ewald_beta = self._get_tensor("modifier_attr/ewald_beta:0")
            [mdl_name, mdl_charge_map, sys_charge_map, ewald_h, ewald_beta] = run_sess(self.sess, [t_mdl_name, t_mdl_charge_map, t_sys_charge_map, t_ewald_h, t_ewald_beta])
            mdl_name = mdl_name.decode("UTF-8")
            mdl_charge_map = [int(ii) for ii in mdl_charge_map.decode("UTF-8").split()]
            sys_charge_map = [int(ii) for ii in sys_charge_map.decode("UTF-8").split()]
            self.dm = DipoleChargeModifier(mdl_name, mdl_charge_map, sys_charge_map, ewald_h = ewald_h, ewald_beta = ewald_beta)

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.dfparam, self.daparam, self.tmap] = run_sess(self.sess, 
            [self.t_ntypes, self.t_rcut, self.t_dfparam, self.t_daparam, self.t_tmap]
        )

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_type_map(self) -> List[int]:
        """Get the type map (element name of the atom types) of this model."""
        return self.tmap

    def get_sel_type(self) -> List[int]:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dfparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.daparam
    
    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
        """Wrapper method with auto batch size.
        
        Parameters
        ----------
        inner_func : Callable
            the method to be wrapped
        numb_test: int
            number of tests
        natoms : int
            number of atoms
        
        Returns
        -------
        Callable
            the wrapper
        """
        if self.auto_batch_size is not None:
            def eval_func(*args, **kwargs):
                return self.auto_batch_size.execute_all(inner_func, numb_test, natoms, *args, **kwargs)
        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(self, coords: np.ndarray, atom_types: List[int]) -> Tuple[int, int]:
        natoms = len(atom_types)
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _get_neigh(
        self,
        natoms_vec: np.ndarray,
        feed_dict: dict
        ) -> Tuple[list, list]: 
        """get neighbor list for Ascend prod_env_mat op, which does not support generate neighbor list in current version.

        Parameters
        ----------
        natoms_vec : np.ndarray of int
            Data natoms.
            The array should be of size type size + 2
        feed_dict : dict
            Session original feed_dict includes coord, type, box, mesh, natoms.
        
        Returns
        -------
        neighbor : list of np.array
            Neighbor list data.
        graph_para : list of np.array
            Graph_para is the parameters of graph, used by the following operations.
        """
        t_natoms = natoms_vec.astype(np.int32)
        avg_tensor = self.sess.graph.get_tensor_by_name('load/descrpt_attr/t_avg/read:0')
        std_tensor = self.sess.graph.get_tensor_by_name('load/descrpt_attr/t_std/read:0')
        script_tensor = self.sess.graph.get_tensor_by_name('load/train_attr/training_script:0')
        natoms_tensor = self.sess.graph.get_tensor_by_name('load/t_natoms:0')
        coord_tensor = feed_dict[self.sess.graph.get_tensor_by_name('load/t_coord:0')]
        type_tensor = feed_dict[self.sess.graph.get_tensor_by_name('load/t_type:0')]
        box_tensor = feed_dict[self.sess.graph.get_tensor_by_name('load/t_box:0')]
        mesh_tensor = feed_dict[self.sess.graph.get_tensor_by_name('load/t_mesh:0')]

        coord_tensor = np.reshape(coord_tensor, [-1, natoms_vec[1] * 3])
        type_tensor = np.reshape(np.array(type_tensor, dtype=np.int32), [-1, natoms_vec[1]])
        box_tensor = np.reshape(box_tensor, [-1, 9])
        graph_para = run_sess(self.sess, [avg_tensor, std_tensor, script_tensor, natoms_tensor], feed_dict={})
        script_val = json.loads(graph_para[2].decode("utf-8"))["model"]['descriptor']
        rcut_a_default = -1.0
        sel_r_default = [0 for ii in range(len(script_val['sel']))]
        with tf.Graph().as_default() as sub_graph:
            coord_sub = np.reshape(feed_dict[self.sess.graph.get_tensor_by_name('load/t_coord:0')],
                                   [-1, natoms_vec[1] * 3])
            temp_type = feed_dict[self.sess.graph.get_tensor_by_name('load/t_type:0')]
            type_sub = np.reshape(np.array(temp_type, dtype=np.int32), [-1, natoms_vec[1]])
            box_sub = np.reshape(feed_dict[self.sess.graph.get_tensor_by_name('load/t_box:0')], [-1, 9])
            coord_new, type_new, idx_mapping, nlist_new \
                = op_module.ProdEnvMatAMesh(coord=tf.constant(coord_tensor),
                                            type=tf.constant(type_tensor),
                                            natoms=tf.constant(t_natoms),
                                            box=tf.constant(box_tensor),
                                            mesh=tf.constant(mesh_tensor),
                                            davg=tf.constant(graph_para[0]),
                                            dstd=tf.constant(graph_para[1]),
                                            rcut_a=rcut_a_default,
                                            rcut_r=script_val['rcut'],
                                            rcut_r_smth=script_val['rcut_smth'],
                                            sel_a=script_val['sel'],
                                            sel_r=sel_r_default)

        with tf.Session(graph=sub_graph, config=default_tf_session_config) as sub_sess:
            neighbor = run_sess(sub_sess, [sub_graph.get_tensor_by_name('ProdEnvMatAMesh:0'),
                                           sub_graph.get_tensor_by_name('ProdEnvMatAMesh:1'),
                                           sub_graph.get_tensor_by_name('ProdEnvMatAMesh:2'),
                                           sub_graph.get_tensor_by_name('ProdEnvMatAMesh:3')])

        return neighbor, graph_para

    def _init_padding_input(
        self,
        nloc_padding: int,
        nall_padding: int
        ) -> Tuple:
        """init padding input for Ascend inference.

        Parameters
        ----------
        nloc_padding : int
            Number of N local atoms after padding. The value is provided by the Ascend transfered model.
        nall_padding : int
            Number of N all atoms after padding. The value is provided by the Ascend transfered model.
        
        Returns
        -------
        Tuple
            init inputs consists of coord, type, map, and neighbor list. The first value in the neighbor
            list is the number of N local atoms after padding, followed by the index of the N local atoms.
        """
        coords_init = np.array([-1.0] * nall_padding * 3, dtype=np.float32)
        type_init = np.array([-1] * nall_padding)
        map_init = np.array([-1] * nall_padding)
        nlist_init = np.array([-1] * (nloc_padding * (self.ASCEND_NEI_LEN + 2) + 1))
        nlist_init[0] = nloc_padding
        nlist_init[1 : nloc_padding + 1] = [ii for ii in range(nloc_padding)]
        return (coords_init, type_init, map_init, nlist_init)

    def _prepare_padding_feed_dict(
        self,
        init_input: tuple,
        graph_para: list, 
        natoms_vec: np.ndarray,
        coords_i: np.ndarray,
        ntypes_i: np.ndarray,
        idx_mapping_i: np.ndarray,
        nlist_i: np.ndarray,
        box_i: np.ndarray
        ) -> Tuple[dict, np.ndarray]:
        """init padding input for Ascend inference.

        Parameters
        ----------
        init_input : tuple of np.array
            Initialized input generated by _init_padding_input function.
        graph_para : list of np.array
            Parameters of graph.
        natoms_vec : np.ndarray
            The number of atoms in datasets. This tensor has the length of Ntypes + 2.
        coords_i : np.ndarray
            Coordinates of the i-th data.
        ntypes_i : np.ndarray
            Types of the i-th data.
        idx_mapping_i : np.ndarray
            Idx_mapping of the i-th data, which include the mapping of nall atom indexes.
        nlist_i : np.ndarray
            Neighbor list
        box_i : np.ndarray
            Box of the i-th data, it is one of inputs of ProdEnvMatA op.
        
        Returns
        -------
        feed_dict_padding : dict
            feed_dict with padding.
        offset_mapping : np.ndarray
            The map is used to transform the atom indexes after padding to the original indexes.
        """
        natoms = natoms_vec.astype(np.int32)[0]
        nloc_padding = graph_para[3][0]
        type_count = natoms_vec.astype(np.int32)[2:]
        type_count_padding = graph_para[3][2:]
        nall_new = int(idx_mapping_i[0])
        coords_padding = init_input[0]
        type_padding = init_input[1]
        map_padding = init_input[2]
        nlist_padding = init_input[3]
        neigh_len = self.ASCEND_NEI_LEN
        feed_dict_padding = {}

        # generate offset mapping
        offset_mapping = np.array([0 for ii in range(nall_new)])
        offset_padding = 0
        offset = 0
        assert nloc_padding >= natoms
        for type_i in range(len(type_count)):
            offset_mapping[offset : offset + type_count[type_i]] = np.arange(type_count[type_i]) + offset_padding
            offset += type_count[type_i]
            offset_padding += type_count_padding[type_i]
        assert offset == natoms
        offset_mapping[offset : offset + nall_new - natoms] = np.arange(nall_new - natoms) + offset_padding
        offset_mapping = np.append(offset_mapping, -1)

        # one frame padding begin
        coords_padding[3 * offset_mapping[:-1]] = coords_i[np.arange(nall_new) * 3]
        coords_padding[3 * offset_mapping[:-1] + 1] = coords_i[np.arange(nall_new) * 3 + 1]
        coords_padding[3 * offset_mapping[:-1] + 2] = coords_i[np.arange(nall_new) * 3 + 2]
        type_padding[offset_mapping[:-1]] = ntypes_i[np.arange(nall_new)]
        map_padding[offset_mapping[:nall_new]] = offset_mapping[idx_mapping_i[1:nall_new+1].astype(int)]
        nlist_padding[nloc_padding + offset_mapping[:natoms] + 1] = nlist_i[natoms + np.arange(natoms) + 1]
        for ii in range(natoms):
            nlist_padding[2 * nloc_padding + offset_mapping[ii] * neigh_len + 1: 2 * nloc_padding + (
                            offset_mapping[ii] + 1) * neigh_len + 1] = offset_mapping[nlist_i[
                            2 * natoms + ii * neigh_len + 1 : 2 * natoms + (ii + 1) * neigh_len + 1]]
        mesh = np.append(nlist_padding, map_padding)
        feed_dict_padding[self.t_coord] = np.reshape(coords_padding, [-1])
        feed_dict_padding[self.t_type] = np.reshape(type_padding, [-1])
        feed_dict_padding[self.t_mesh] = np.reshape(mesh, [-1])
        feed_dict_padding[self.t_box] = np.reshape(box_i, [-1])
        return feed_dict_padding, offset_mapping


    def _run_with_padding(
        self,
        t_out: tf.Tensor,
        feed_dict: dict,
        nframes: int,
        atomic: bool
        ) -> List:
        """Evaluate the energy, force and virial, Ascend platform needs padding inputs.

        Parameters
        ----------
        t_out : list of tensor
            Output tensor includes energy, force, virial, atom_force, atom_virial.
        feed_dict : dict of tensor
            Session original feed_dict includes coord, type, box, mesh.
        nframes : int
            Number of samples in one dataset.
        atomic : bool
            Calculate the atomic energy and virial.

        Returns
        -------
        energy : np.ndarray
            The system energy.
        force : np.ndarray
            The force on each atom. 
        virial : np.ndarray
            The virial on each atom. 
        atom_energy : np.ndarray
            The atomic energy. Only returned when atomic == True.
        atom_virial : np.ndarray
            The atomic virial. Only returned when atomic == True.
        """
        if not atomic:
            t_out += [self.t_ae, 
                        self.t_av]
        # initialize input and output
        natoms_vec = feed_dict[self.t_natoms]
        natoms = natoms_vec[0]
        del feed_dict[self.t_natoms]
        energy, force, virial, ae, av = [], [], [], [], []
        coord_nlist, graph_para = self._get_neigh(natoms_vec, feed_dict)
        nloc_padding = graph_para[3][0]
        nall_padding = graph_para[3][1]
        init_input = self._init_padding_input(nloc_padding, nall_padding)
        box_all = np.reshape(feed_dict[self.sess.graph.get_tensor_by_name('load/t_box:0')], [-1, 9])

        for i in range(0, nframes):
            coords_i = coord_nlist[0][i]
            type_i = coord_nlist[1][i]
            idx_mapping_i = coord_nlist[2][i]
            nlist_i = coord_nlist[3][i]
            box_i = box_all[i]
            feed_dict_padding, offset_mapping = self._prepare_padding_feed_dict(init_input=init_input,
                                                                                graph_para=graph_para, 
                                                                                natoms_vec=natoms_vec,
                                                                                coords_i=coords_i,
                                                                                ntypes_i=type_i,
                                                                                idx_mapping_i=idx_mapping_i,
                                                                                nlist_i=nlist_i,
                                                                                box_i=box_i)

            sess_out = self.npu_sess.run(t_out, feed_dict=feed_dict_padding)
            force_i = sess_out[1]
            virial_i = sess_out[2]
            ae_padding = sess_out[3]
            av_padding = sess_out[4]
            force_i = np.reshape(force_i, [nall_padding, 3])
            av_padding = np.reshape(av_padding, [nall_padding, 9])
            ae_i = ae_padding[0, offset_mapping[:natoms]]
            av_i = av_padding[offset_mapping[:natoms], :]
            energy.append(np.sum(ae_i, dtype=np.float64))
            force.append(force_i[offset_mapping[:natoms], :])
            virial.append(virial_i)
            av.append(av_i)
            ae.append(ae_i)

        if atomic:
            return energy, force, virial, ae, av
        return energy, force, virial

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate the energy, force and virial by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        efield
            The external field on atoms.
            The array should be of size nframes x natoms x 3

        Returns
        -------
        energy
            The system energy.
        force
            The force on each atom
        virial
            The virial
        atom_energy
            The atomic energy. Only returned when atomic == True
        atom_virial
            The atomic virial. Only returned when atomic == True
        """
        # reshape coords before getting shape
        natoms, numb_test = self._get_natoms_and_nframes(coords, atom_types)
        output = self._eval_func(self._eval_inner, numb_test, natoms)(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic, efield = efield)

        if self.modifier_type is not None:
            if atomic:
                raise RuntimeError('modifier does not support atomic modification')
            me, mf, mv = self.dm.eval(coords, cells, atom_types)
            output = list(output) # tuple to list
            e, f, v = output[:3]
            output[0] += me.reshape(e.shape)
            output[1] += mf.reshape(f.shape)
            output[2] += mv.reshape(v.shape)
            output = tuple(output)
        return output

    def _prepare_feed_dict(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        atomic=False,
        efield=None
    ):
        # standarize the shape of inputs
        natoms, nframes = self._get_natoms_and_nframes(coords, atom_types)
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        if cells is None:
            pbc = False
            # make cells to work around the requirement of pbc
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
        
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)
        if self.has_aparam :
            assert(aparam is not None)
            aparam = np.array(aparam)
        if self.has_efield :
            assert(efield is not None), "you are using a model with external field, parameter efield should be provided"
            efield = np.array(efield)

        # reshape the inputs 
        if self.has_fparam :
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim :
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim :
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d or %d' % (nframes, fdim, fdim))
        if self.has_aparam :
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d' % (nframes, natoms, fdim, natoms, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self.sort_input(coords, atom_types)
        if self.has_efield:
            efield = np.reshape(efield, [nframes, natoms, 3])
            efield = efield[:,imap,:]
            efield = np.reshape(efield, [nframes, natoms*3])            

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes, 1]).reshape([-1])
        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
        feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
        if self.has_efield:
            feed_dict_test[self.t_efield]= np.reshape(efield, [-1])
        if pbc:
            feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
        else:
            feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)
        if self.has_fparam:
            feed_dict_test[self.t_fparam] = np.reshape(fparam, [-1])
        if self.has_aparam:
            feed_dict_test[self.t_aparam] = np.reshape(aparam, [-1])
        return feed_dict_test, imap

    def _eval_inner(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        atomic=False,
        efield=None
    ):
        natoms, nframes = self._get_natoms_and_nframes(coords, atom_types)
        feed_dict_test, imap = self._prepare_feed_dict(coords, cells, atom_types, fparam, aparam, efield)
        t_out = [self.t_energy, 
                 self.t_force, 
                 self.t_virial]
        if atomic :
            t_out += [self.t_ae, 
                      self.t_av]

        if dp_variant == "ascend":
            v_out = self._run_with_padding(t_out, feed_dict=feed_dict_test, nframes=nframes, atomic=atomic)
        else:
            v_out = run_sess(self.sess, t_out, feed_dict = feed_dict_test)
        energy = v_out[0]
        force = v_out[1]
        virial = v_out[2]
        if atomic:
            ae = v_out[3]
            av = v_out[4]

        # reverse map of the outputs
        force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)
        if atomic :
            ae  = self.reverse_map(np.reshape(ae, [nframes,-1,1]), imap)
            av  = self.reverse_map(np.reshape(av, [nframes,-1,9]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, natoms, 3])
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms, 1])
            av = np.reshape(av, [nframes, natoms, 9])
            return energy, force, virial, ae, av
        else :
            return energy, force, virial

    def eval_descriptor(self,
            coords: np.ndarray,
            cells: np.ndarray,
            atom_types: List[int],
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            efield: Optional[np.ndarray] = None,
            ) -> np.array:
        """Evaluate descriptors by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        efield
            The external field on atoms.
            The array should be of size nframes x natoms x 3

        Returns
        -------
        descriptor
            Descriptors.
        """
        natoms, numb_test = self._get_natoms_and_nframes(coords, atom_types)
        descriptor = self._eval_func(self._eval_descriptor_inner, numb_test, natoms)(coords, cells, atom_types, fparam = fparam, aparam = aparam, efield = efield)
        return descriptor
    
    def _eval_descriptor_inner(self,
            coords: np.ndarray,
            cells: np.ndarray,
            atom_types: List[int],
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            efield: Optional[np.ndarray] = None,
            ) -> np.array:
        natoms, nframes = self._get_natoms_and_nframes(coords, atom_types)
        feed_dict_test, imap = self._prepare_feed_dict(coords, cells, atom_types, fparam, aparam, efield)
        descriptor, = run_sess(self.sess, [self.t_descriptor], feed_dict = feed_dict_test)
        return self.reverse_map(np.reshape(descriptor, [nframes, natoms, -1]), imap)
