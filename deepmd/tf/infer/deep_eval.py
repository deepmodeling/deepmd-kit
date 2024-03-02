# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    lru_cache,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.common import (
    make_default_mesh,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.tf.env import (
    MODEL_VERSION,
    default_tf_session_config,
    tf,
)
from deepmd.tf.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.tf.utils.sess import (
    run_sess,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

    from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper


class DeepEval(DeepEvalBackend):
    """TensorFlow backend implementation for DeepEval.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    auto_batch_size : bool or int or AutomaticBatchSize, default: False
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    input_map : dict, optional
        The input map for tf.import_graph_def. Only work with default tf graph
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        model_file: "Path",
        output_def: ModelOutputDef,
        *args: list,
        load_prefix: str = "load",
        default_tf_graph: bool = False,
        auto_batch_size: Union[bool, int, AutoBatchSize] = False,
        input_map: Optional[dict] = None,
        neighbor_list=None,
        **kwargs: dict,
    ):
        self.graph = self._load_graph(
            model_file,
            prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
        )
        self.load_prefix = load_prefix

        # graph_compatable should be called after graph and prefix are set
        if not self._graph_compatable():
            raise RuntimeError(
                f"model in graph (version {self.model_version}) is incompatible"
                f"with the model (version {MODEL_VERSION}) supported by the current code."
                "See https://deepmd.rtfd.io/compatability/ for details."
            )

        # set default to False, as subclasses may not support
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

        self.neighbor_list = neighbor_list

        self.output_def = output_def
        self._init_tensors()
        self._init_attr()
        self.has_efield = self.tensors["efield"] is not None
        self.has_fparam = self.tensors["fparam"] is not None
        self.has_aparam = self.tensors["aparam"] is not None
        self.has_spin = self.ntypes_spin > 0
        self.modifier_type = None

        # looks ugly...
        if self.modifier_type == "dipole_charge":
            from deepmd.tf.infer.data_modifier import (
                DipoleChargeModifier,
            )

            t_mdl_name = self._get_tensor("modifier_attr/mdl_name:0")
            t_mdl_charge_map = self._get_tensor("modifier_attr/mdl_charge_map:0")
            t_sys_charge_map = self._get_tensor("modifier_attr/sys_charge_map:0")
            t_ewald_h = self._get_tensor("modifier_attr/ewald_h:0")
            t_ewald_beta = self._get_tensor("modifier_attr/ewald_beta:0")
            [mdl_name, mdl_charge_map, sys_charge_map, ewald_h, ewald_beta] = run_sess(
                self.sess,
                [
                    t_mdl_name,
                    t_mdl_charge_map,
                    t_sys_charge_map,
                    t_ewald_h,
                    t_ewald_beta,
                ],
            )
            mdl_name = mdl_name.decode("UTF-8")
            mdl_charge_map = [int(ii) for ii in mdl_charge_map.decode("UTF-8").split()]
            sys_charge_map = [int(ii) for ii in sys_charge_map.decode("UTF-8").split()]
            self.dm = DipoleChargeModifier(
                mdl_name,
                mdl_charge_map,
                sys_charge_map,
                ewald_h=ewald_h,
                ewald_beta=ewald_beta,
            )

    def _init_tensors(self):
        tensor_names = {
            # descrpt attrs
            "ntypes": "descrpt_attr/ntypes:0",
            "rcut": "descrpt_attr/rcut:0",
            # model attrs
            "tmap": "model_attr/tmap:0",
            # inputs
            "coord": "t_coord:0",
            "type": "t_type:0",
            "natoms": "t_natoms:0",
            "box": "t_box:0",
            "mesh": "t_mesh:0",
        }
        optional_tensor_names = {
            # fitting attrs
            "dfparam": "fitting_attr/dfparam:0",
            "daparam": "fitting_attr/daparam:0",
            "numb_dos": "fitting_attr/numb_dos:0",
            # model attrs
            "sel_type": "model_attr/sel_type:0",
            # additonal inputs
            "efield": "t_efield:0",
            "fparam": "t_fparam:0",
            "aparam": "t_aparam:0",
            "ntypes_spin": "spin_attr/ntypes_spin:0",
            # descriptor
            "descriptor": "o_descriptor:0",
        }
        # output tensors
        output_tensor_names = {}
        for vv in self.output_def.var_defs:
            output_tensor_names[vv] = f"o_{self._OUTDEF_DP2BACKEND[vv]}:0"

        self.tensors = {}
        for tensor_key, tensor_name in tensor_names.items():
            self.tensors[tensor_key] = self._get_tensor(tensor_name)
        for tensor_key, tensor_name in optional_tensor_names.items():
            try:
                self.tensors[tensor_key] = self._get_tensor(tensor_name)
            except KeyError:
                self.tensors[tensor_key] = None
        self.output_tensors = {}
        removed_defs = []
        for ii, (tensor_key, tensor_name) in enumerate(output_tensor_names.items()):
            try:
                self.output_tensors[tensor_key] = self._get_tensor(tensor_name)
            except KeyError:
                # do not output
                removed_defs.append(ii)
        for ii in sorted(removed_defs, reverse=True):
            del self.output_def.var_defs[list(self.output_def.var_defs.keys())[ii]]

    def _init_attr(self):
        [
            self.ntypes,
            self.rcut,
            tmap,
        ] = run_sess(
            self.sess,
            [
                self.tensors["ntypes"],
                self.tensors["rcut"],
                self.tensors["tmap"],
            ],
        )
        if self.tensors["ntypes_spin"] is not None:
            self.ntypes_spin = run_sess(self.sess, [self.tensors["ntypes_spin"]])[0]
        else:
            self.ntypes_spin = 0
        if self.tensors["dfparam"] is not None:
            self.dfparam = run_sess(self.sess, [self.tensors["dfparam"]])[0]
        else:
            self.dfparam = 0
        if self.tensors["daparam"] is not None:
            self.daparam = run_sess(self.sess, [self.tensors["daparam"]])[0]
        else:
            self.daparam = 0
        if self.tensors["sel_type"] is not None:
            self.sel_type = run_sess(self.sess, [self.tensors["sel_type"]])[0]
        else:
            self.sel_type = None
        if self.tensors["numb_dos"] is not None:
            self.numb_dos = run_sess(self.sess, [self.tensors["numb_dos"]])[0]
        else:
            self.numb_dos = 0
        self.tmap = tmap.decode("utf-8").split()

    @property
    @lru_cache(maxsize=None)
    def model_type(self) -> "DeepEvalWrapper":
        """Get type of model.

        :type:str
        """
        t_mt = self._get_tensor("model_attr/model_type:0")
        [mt] = run_sess(self.sess, [t_mt], feed_dict={})
        model_type = mt.decode("utf-8")
        if model_type == "ener":
            return DeepPot
        elif model_type == "dos":
            return DeepDOS
        elif model_type == "dipole":
            return DeepDipole
        elif model_type == "polar":
            return DeepPolar
        elif model_type == "global_polar":
            return DeepGlobalPolar
        elif model_type == "wfc":
            return DeepWFC
        else:
            raise RuntimeError(f"unknown model type {model_type}")

    @property
    @lru_cache(maxsize=None)
    def model_version(self) -> str:
        """Get version of model.

        Returns
        -------
        str
            version of model
        """
        try:
            t_mt = self._get_tensor("model_attr/model_version:0")
        except KeyError:
            # For deepmd-kit version 0.x - 1.x, set model version to 0.0
            return "0.0"
        else:
            [mt] = run_sess(self.sess, [t_mt], feed_dict={})
            return mt.decode("utf-8")

    @property
    @lru_cache(maxsize=None)
    def sess(self) -> tf.Session:
        """Get TF session."""
        # start a tf session associated to the graph
        return tf.Session(graph=self.graph, config=default_tf_session_config)

    def _graph_compatable(self) -> bool:
        """Check the model compatability.

        Returns
        -------
        bool
            If the model stored in the graph file is compatable with the current code
        """
        model_version_major = int(self.model_version.split(".")[0])
        model_version_minor = int(self.model_version.split(".")[1])
        MODEL_VERSION_MAJOR = int(MODEL_VERSION.split(".")[0])
        MODEL_VERSION_MINOR = int(MODEL_VERSION.split(".")[1])
        if (model_version_major != MODEL_VERSION_MAJOR) or (
            model_version_minor > MODEL_VERSION_MINOR
        ):
            return False
        else:
            return True

    def _get_tensor(
        self,
        tensor_name: str,
    ) -> tf.Tensor:
        """Get TF graph tensor.

        Parameters
        ----------
        tensor_name : str
            name of tensor to get

        Returns
        -------
        tf.Tensor
            loaded tensor
        """
        # do not use os.path.join as it doesn't work on Windows
        tensor_path = "/".join((self.load_prefix, tensor_name))
        tensor = self.graph.get_tensor_by_name(tensor_path)
        return tensor

    @staticmethod
    def _load_graph(
        frozen_graph_filename: "Path",
        prefix: str = "load",
        default_tf_graph: bool = False,
        input_map: Optional[dict] = None,
    ):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            if default_tf_graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=input_map,
                    return_elements=None,
                    name=prefix,
                    producer_op_list=None,
                )
                graph = tf.get_default_graph()
            else:
                # Then, we can use again a convenient built-in function to import
                # a graph_def into the  current default Graph
                with tf.Graph().as_default() as graph:
                    tf.import_graph_def(
                        graph_def,
                        input_map=None,
                        return_elements=None,
                        name=prefix,
                        producer_op_list=None,
                    )

            return graph

    @staticmethod
    def sort_input(
        coord: np.ndarray,
        atom_type: np.ndarray,
        sel_atoms: Optional[List[int]] = None,
    ):
        """Sort atoms in the system according their types.

        Parameters
        ----------
        coord
            The coordinates of atoms.
            Should be of shape [nframes, natoms, 3]
        atom_type
            The type of atoms
            Should be of shape [natoms]
        sel_atoms
            The selected atoms by type

        Returns
        -------
        coord_out
            The coordinates after sorting
        atom_type_out
            The atom types after sorting
        idx_map
            The index mapping from the input to the output.
            For example coord_out = coord[:,idx_map,:]
        sel_atom_type
            Only output if sel_atoms is not None
            The sorted selected atom types
        sel_idx_map
            Only output if sel_atoms is not None
            The index mapping from the selected atoms to sorted selected atoms.
        """
        natoms = atom_type.shape[1]
        if sel_atoms is not None:
            selection = np.array([False] * natoms, dtype=bool)
            for ii in sel_atoms:
                selection += atom_type[0] == ii
            sel_atom_type = atom_type[:, selection]
        idx = np.arange(natoms)
        idx_map = np.lexsort((idx, atom_type[0]))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:, idx_map, :], [nframes, -1])
        atom_type = atom_type[:, idx_map]
        if sel_atoms is not None:
            sel_natoms = sel_atom_type.shape[1]
            sel_idx = np.arange(sel_natoms)
            sel_idx_map = np.lexsort((sel_idx, sel_atom_type[0]))
            sel_atom_type = sel_atom_type[:, sel_idx_map]
            return coord, atom_type, idx_map, sel_atom_type, sel_idx_map
        else:
            return coord, atom_type, idx_map, atom_type, idx_map

    @staticmethod
    def reverse_map(vec: np.ndarray, imap: List[int]) -> np.ndarray:
        """Reverse mapping of a vector according to the index map.

        Parameters
        ----------
        vec
            Input vector. Be of shape [nframes, natoms, -1]
        imap
            Index map. Be of shape [natoms]

        Returns
        -------
        vec_out
            Reverse mapped vector.
        """
        ret = np.zeros(vec.shape)
        ret[:, imap, :] = vec
        return ret

    def make_natoms_vec(
        self,
        atom_types: np.ndarray,
    ) -> np.ndarray:
        """Make the natom vector used by deepmd-kit.

        Parameters
        ----------
        atom_types
            The type of atoms

        Returns
        -------
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        """
        natoms_vec = np.zeros(self.ntypes + 2).astype(int)
        natoms = atom_types[0].size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        for ii in range(self.ntypes):
            natoms_vec[ii + 2] = np.count_nonzero(atom_types[0] == ii)
        return natoms_vec

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.tf.utils.type_embed.TypeEmbedNet : The type embedding network.

        Examples
        --------
        Get the output of type embedding network of `graph.pb`:

        >>> from deepmd.tf.infer import DeepPotential
        >>> dp = DeepPotential("graph.pb")
        >>> dp.eval_typeebd()
        """
        t_typeebd = self._get_tensor("t_typeebd:0")
        [typeebd] = run_sess(self.sess, [t_typeebd], feed_dict={})
        return typeebd

    def build_neighbor_list(
        self,
        coords: np.ndarray,
        cell: Optional[np.ndarray],
        atype: np.ndarray,
        imap: np.ndarray,
        neighbor_list,
    ):
        """Make the mesh with neighbor list for a single frame.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of atoms. Should be of shape [natoms, 3]
        cell : Optional[np.ndarray]
            The cell of the system. Should be of shape [3, 3]
        atype : np.ndarray
            The type of atoms. Should be of shape [natoms]
        imap : np.ndarray
            The index map of atoms. Should be of shape [natoms]
        neighbor_list : ase.neighborlist.NewPrimitiveNeighborList
            ASE neighbor list. The following method or attribute will be
            used/set: bothways, self_interaction, update, build, first_neigh,
            pair_second, offset_vec.

        Returns
        -------
        natoms_vec : np.ndarray
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: nloc
            natoms[1]: nall
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms for nloc
        coords : np.ndarray
            The coordinates of atoms, including ghost atoms. Should be of
            shape [nframes, nall, 3]
        atype : np.ndarray
            The type of atoms, including ghost atoms. Should be of shape [nall]
        mesh : np.ndarray
            The mesh in nei_mode=4.
        imap : np.ndarray
            The index map of atoms. Should be of shape [nall]
        ghost_map : np.ndarray
            The index map of ghost atoms. Should be of shape [nghost]
        """
        pbc = np.repeat(cell is not None, 3)
        cell = cell.reshape(3, 3)
        positions = coords.reshape(-1, 3)
        neighbor_list.bothways = True
        neighbor_list.self_interaction = False
        if neighbor_list.update(pbc, cell, positions):
            neighbor_list.build(pbc, cell, positions)
        first_neigh = neighbor_list.first_neigh.copy()
        pair_second = neighbor_list.pair_second.copy()
        offset_vec = neighbor_list.offset_vec.copy()
        # get out-of-box neighbors
        out_mask = np.any(offset_vec != 0, axis=1)
        out_idx = pair_second[out_mask]
        out_offset = offset_vec[out_mask]
        out_coords = positions[out_idx] + out_offset.dot(cell)
        atype = np.array(atype, dtype=int).reshape(-1)
        out_atype = atype[out_idx]

        nloc = positions.shape[0]
        nghost = out_idx.size
        all_coords = np.concatenate((positions, out_coords), axis=0)
        all_atype = np.concatenate((atype, out_atype), axis=0)
        # convert neighbor indexes
        ghost_map = pair_second[out_mask]
        pair_second[out_mask] = np.arange(nloc, nloc + nghost)
        # get the mesh
        mesh = np.zeros(16 + nloc * 2 + pair_second.size, dtype=int)
        mesh[0] = nloc
        # ilist
        mesh[16 : 16 + nloc] = np.arange(nloc)
        # numnei
        mesh[16 + nloc : 16 + nloc * 2] = first_neigh[1:] - first_neigh[:-1]
        # jlist
        mesh[16 + nloc * 2 :] = pair_second

        # natoms_vec
        natoms_vec = np.zeros(self.ntypes + 2).astype(int)
        natoms_vec[0] = nloc
        natoms_vec[1] = nloc + nghost
        for ii in range(self.ntypes):
            natoms_vec[ii + 2] = np.count_nonzero(atype == ii)
        # imap append ghost atoms
        imap = np.concatenate((imap, np.arange(nloc, nloc + nghost)))
        return natoms_vec, all_coords, all_atype, mesh, imap, ghost_map

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model."""
        return self.ntypes_spin

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_type_map(self) -> List[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.tmap

    def get_sel_type(self) -> Optional[np.ndarray]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return np.array(self.sel_type).ravel()

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
        numb_test : int
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
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: Union[List[int], np.ndarray],
    ) -> Tuple[int, int]:
        natoms = len(atom_types[0])
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
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
        output_dict : dict
            The output of the evaluation. The keys are the names of the output
            variables, and the values are the corresponding output arrays.
        """
        # reshape coords before getting shape
        natoms, numb_test = self._get_natoms_and_nframes(
            coords,
            atom_types,
        )
        output = self._eval_func(self._eval_inner, numb_test, natoms)(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            atomic=atomic,
            efield=efield,
        )
        if not isinstance(output, tuple):
            output = (output,)

        output_dict = {
            odef.name: oo for oo, odef in zip(output, self.output_def.var_defs.values())
        }
        # ugly!!
        if self.modifier_type is not None and isinstance(self.model_type, DeepPot):
            if atomic:
                raise RuntimeError("modifier does not support atomic modification")
            me, mf, mv = self.dm.eval(coords, cells, atom_types)
            output = list(output)  # tuple to list
            e, f, v = output[:3]
            output_dict["energy_redu"] += me.reshape(e.shape)
            output_dict["energy_deri_r"] += mf.reshape(f.shape)
            output_dict["energy_deri_c_redu"] += mv.reshape(v.shape)
        return output_dict

    def _prepare_feed_dict(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        efield=None,
    ):
        # standarize the shape of inputs
        natoms, nframes = self._get_natoms_and_nframes(
            coords,
            atom_types,
        )
        atom_types = np.array(atom_types, dtype=int).reshape([nframes, natoms])
        coords = np.reshape(np.array(coords), [nframes, natoms * 3])
        if cells is None:
            pbc = False
            # make cells to work around the requirement of pbc
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])

        if self.has_fparam:
            assert fparam is not None
            fparam = np.array(fparam)
        if self.has_aparam:
            assert aparam is not None
            aparam = np.array(aparam)
        if self.has_efield:
            assert (
                efield is not None
            ), "you are using a model with external field, parameter efield should be provided"
            efield = np.array(efield)

        # reshape the inputs
        if self.has_fparam:
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim:
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim:
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else:
                raise RuntimeError(
                    "got wrong size of frame param, should be either %d x %d or %d"
                    % (nframes, fdim, fdim)
                )
        if self.has_aparam:
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim:
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim:
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else:
                raise RuntimeError(
                    "got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d"
                    % (nframes, natoms, fdim, natoms, fdim, fdim)
                )

        # sort inputs
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(
            coords, atom_types, sel_atoms=self.get_sel_type()
        )
        if self.has_efield:
            efield = np.reshape(efield, [nframes, natoms, 3])
            efield = efield[:, imap, :]
            efield = np.reshape(efield, [nframes, natoms * 3])
        if self.has_aparam:
            aparam = np.reshape(aparam, [nframes, natoms, fdim])
            aparam = aparam[:, imap, :]
            aparam = np.reshape(aparam, [nframes, natoms * fdim])

        # make natoms_vec and default_mesh
        if self.neighbor_list is None:
            natoms_vec = self.make_natoms_vec(atom_types)
            assert natoms_vec[0] == natoms
            mesh = make_default_mesh(pbc, not self._check_mixed_types(atom_types))
            ghost_map = None
        else:
            if nframes > 1:
                raise NotImplementedError(
                    "neighbor_list does not support multiple frames"
                )
            (
                natoms_vec,
                coords,
                atom_types,
                mesh,
                imap,
                ghost_map,
            ) = self.build_neighbor_list(
                coords,
                cells if cells is not None else None,
                atom_types,
                imap,
                self.neighbor_list,
            )

        # evaluate
        feed_dict_test = {}
        feed_dict_test[self.tensors["natoms"]] = natoms_vec
        feed_dict_test[self.tensors["type"]] = atom_types.reshape([-1])
        feed_dict_test[self.tensors["coord"]] = np.reshape(coords, [-1])

        if len(self.tensors["box"].shape) == 1:
            feed_dict_test[self.tensors["box"]] = np.reshape(cells, [-1])
        elif len(self.tensors["box"].shape) == 2:
            feed_dict_test[self.tensors["box"]] = cells
        else:
            raise RuntimeError
        if self.has_efield:
            feed_dict_test[self.tensors["efield"]] = np.reshape(efield, [-1])
        feed_dict_test[self.tensors["mesh"]] = mesh
        if self.has_fparam:
            feed_dict_test[self.tensors["fparam"]] = np.reshape(fparam, [-1])
        if self.has_aparam:
            feed_dict_test[self.tensors["aparam"]] = np.reshape(aparam, [-1])
        return feed_dict_test, imap, natoms_vec, ghost_map, sel_at, sel_imap

    def _eval_inner(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        efield=None,
        **kwargs,
    ):
        natoms, nframes = self._get_natoms_and_nframes(
            coords,
            atom_types,
        )
        (
            feed_dict_test,
            imap,
            natoms_vec,
            ghost_map,
            sel_at,
            sel_imap,
        ) = self._prepare_feed_dict(
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            efield,
        )

        nloc = natoms_vec[0]
        nloc_sel = sel_at.shape[1]
        nall = natoms_vec[1]

        t_out = list(self.output_tensors.values())

        v_out = run_sess(self.sess, t_out, feed_dict=feed_dict_test)

        if nloc_sel == 0:
            nloc_sel = nloc
            sel_imap = imap
        if self.has_spin:
            ntypes_real = self.ntypes - self.ntypes_spin
            natoms_real = sum(
                [
                    np.count_nonzero(np.array(atom_types[0]) == ii)
                    for ii in range(ntypes_real)
                ]
            )
        else:
            natoms_real = nloc_sel
        if ghost_map is not None:
            # add the value of ghost atoms to real atoms
            for ii, odef in enumerate(self.output_def.var_defs.values()):
                # when the shape is nall
                if odef.category in (
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C,
                ):
                    odef_shape = self._get_output_shape(odef, nframes, nall)
                    tmp_shape = [np.prod(odef_shape[:-2]), *odef_shape[-2:]]
                    v_out[ii] = np.reshape(v_out[ii], tmp_shape)
                    for jj in range(v_out[ii].shape[0]):
                        np.add.at(v_out[ii][jj], ghost_map, v_out[ii][jj, nloc:])

        for ii, odef in enumerate(self.output_def.var_defs.values()):
            if odef.category in (
                OutputVariableCategory.DERV_R,
                OutputVariableCategory.DERV_C,
            ):
                odef_shape = self._get_output_shape(odef, nframes, nall)
                tmp_shape = [np.prod(odef_shape[:-2]), *odef_shape[-2:]]
                # reverse map of the outputs
                v_out[ii] = self.reverse_map(np.reshape(v_out[ii], tmp_shape), imap)
                v_out[ii] = np.reshape(v_out[ii], odef_shape)
                if nloc < nall:
                    v_out[ii] = v_out[ii][:, :, :nloc]
            elif odef.category == OutputVariableCategory.OUT:
                odef_shape = self._get_output_shape(odef, nframes, natoms_real)
                v_out[ii] = self.reverse_map(
                    np.reshape(v_out[ii], odef_shape), sel_imap[:natoms_real]
                )
                if nloc_sel < nloc:
                    # convert shape from nsel to nloc
                    # sel_atoms was applied before sort; see sort_input
                    # do not consider mixed_types here (as it is never supported)
                    sel_mask = np.isin(atom_types[0], self.sel_type)
                    out_nsel = v_out[ii]
                    out_nloc = np.zeros(
                        (nframes, nloc, *out_nsel.shape[2:]), dtype=out_nsel.dtype
                    )
                    out_nloc[:, sel_mask] = out_nsel
                    v_out[ii] = out_nloc
                    odef_shape = self._get_output_shape(odef, nframes, nloc)
                v_out[ii] = np.reshape(v_out[ii], odef_shape)
            elif odef.category in (
                OutputVariableCategory.REDU,
                OutputVariableCategory.DERV_C_REDU,
            ):
                odef_shape = self._get_output_shape(odef, nframes, 0)
                v_out[ii] = np.reshape(v_out[ii], odef_shape)
            else:
                raise RuntimeError("unknown category")
        return tuple(v_out)

    def _get_output_shape(self, odef, nframes, natoms):
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            # virial
            return [nframes, *odef.shape[:-1], 9]
        elif odef.category == OutputVariableCategory.REDU:
            # energy
            return [nframes, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_C:
            # atom_virial
            return [nframes, *odef.shape[:-1], natoms, 9]
        elif odef.category == OutputVariableCategory.DERV_R:
            # force
            return [nframes, *odef.shape[:-1], natoms, 3]
        elif odef.category == OutputVariableCategory.OUT:
            # atom_energy, atom_tensor
            # Something wrong here?
            # return [nframes, *shape, natoms, 1]
            return [nframes, natoms, *odef.shape, 1]
        else:
            raise RuntimeError("unknown category")

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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
        natoms, numb_test = self._get_natoms_and_nframes(
            coords,
            atom_types,
        )
        descriptor = self._eval_func(self._eval_descriptor_inner, numb_test, natoms)(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            efield=efield,
        )
        return descriptor

    def _eval_descriptor_inner(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        natoms, nframes = self._get_natoms_and_nframes(
            coords,
            atom_types,
        )
        (
            feed_dict_test,
            imap,
            natoms_vec,
            ghost_map,
            sel_at,
            sel_imap,
        ) = self._prepare_feed_dict(
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            efield,
        )
        (descriptor,) = run_sess(
            self.sess, [self.tensors["descriptor"]], feed_dict=feed_dict_test
        )
        imap = imap[:natoms]
        return self.reverse_map(np.reshape(descriptor, [nframes, natoms, -1]), imap)

    def get_numb_dos(self) -> int:
        return self.numb_dos

    def get_has_efield(self) -> bool:
        return self.has_efield


class DeepEvalOld:
    # old class for DipoleChargeModifier only
    """Common methods for DeepPot, DeepWFC, DeepPolar, ...

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    auto_batch_size : bool or int or AutomaticBatchSize, default: False
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    input_map : dict, optional
        The input map for tf.import_graph_def. Only work with default tf graph
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    """

    load_prefix: str  # set by subclass

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False,
        auto_batch_size: Union[bool, int, AutoBatchSize] = False,
        input_map: Optional[dict] = None,
        neighbor_list=None,
    ):
        self.graph = self._load_graph(
            model_file,
            prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
        )
        self.load_prefix = load_prefix

        # graph_compatable should be called after graph and prefix are set
        if not self._graph_compatable():
            raise RuntimeError(
                f"model in graph (version {self.model_version}) is incompatible"
                f"with the model (version {MODEL_VERSION}) supported by the current code."
                "See https://deepmd.rtfd.io/compatability/ for details."
            )

        # set default to False, as subclasses may not support
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

        self.neighbor_list = neighbor_list

    @property
    @lru_cache(maxsize=None)
    def model_type(self) -> str:
        """Get type of model.

        :type:str
        """
        t_mt = self._get_tensor("model_attr/model_type:0")
        [mt] = run_sess(self.sess, [t_mt], feed_dict={})
        return mt.decode("utf-8")

    @property
    @lru_cache(maxsize=None)
    def model_version(self) -> str:
        """Get version of model.

        Returns
        -------
        str
            version of model
        """
        try:
            t_mt = self._get_tensor("model_attr/model_version:0")
        except KeyError:
            # For deepmd-kit version 0.x - 1.x, set model version to 0.0
            return "0.0"
        else:
            [mt] = run_sess(self.sess, [t_mt], feed_dict={})
            return mt.decode("utf-8")

    @property
    @lru_cache(maxsize=None)
    def sess(self) -> tf.Session:
        """Get TF session."""
        # start a tf session associated to the graph
        return tf.Session(graph=self.graph, config=default_tf_session_config)

    def _graph_compatable(self) -> bool:
        """Check the model compatability.

        Returns
        -------
        bool
            If the model stored in the graph file is compatable with the current code
        """
        model_version_major = int(self.model_version.split(".")[0])
        model_version_minor = int(self.model_version.split(".")[1])
        MODEL_VERSION_MAJOR = int(MODEL_VERSION.split(".")[0])
        MODEL_VERSION_MINOR = int(MODEL_VERSION.split(".")[1])
        if (model_version_major != MODEL_VERSION_MAJOR) or (
            model_version_minor > MODEL_VERSION_MINOR
        ):
            return False
        else:
            return True

    def _get_tensor(
        self, tensor_name: str, attr_name: Optional[str] = None
    ) -> tf.Tensor:
        """Get TF graph tensor and assign it to class namespace.

        Parameters
        ----------
        tensor_name : str
            name of tensor to get
        attr_name : Optional[str], optional
            if specified, class attribute with this name will be created and tensor will
            be assigned to it, by default None

        Returns
        -------
        tf.Tensor
            loaded tensor
        """
        # do not use os.path.join as it doesn't work on Windows
        tensor_path = "/".join((self.load_prefix, tensor_name))
        tensor = self.graph.get_tensor_by_name(tensor_path)
        if attr_name:
            setattr(self, attr_name, tensor)
            return tensor
        else:
            return tensor

    @staticmethod
    def _load_graph(
        frozen_graph_filename: "Path",
        prefix: str = "load",
        default_tf_graph: bool = False,
        input_map: Optional[dict] = None,
    ):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            if default_tf_graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=input_map,
                    return_elements=None,
                    name=prefix,
                    producer_op_list=None,
                )
                graph = tf.get_default_graph()
            else:
                # Then, we can use again a convenient built-in function to import
                # a graph_def into the  current default Graph
                with tf.Graph().as_default() as graph:
                    tf.import_graph_def(
                        graph_def,
                        input_map=None,
                        return_elements=None,
                        name=prefix,
                        producer_op_list=None,
                    )

            return graph

    @staticmethod
    def sort_input(
        coord: np.ndarray,
        atom_type: np.ndarray,
        sel_atoms: Optional[List[int]] = None,
        mixed_type: bool = False,
    ):
        """Sort atoms in the system according their types.

        Parameters
        ----------
        coord
            The coordinates of atoms.
            Should be of shape [nframes, natoms, 3]
        atom_type
            The type of atoms
            Should be of shape [natoms]
        sel_atoms
            The selected atoms by type
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        coord_out
            The coordinates after sorting
        atom_type_out
            The atom types after sorting
        idx_map
            The index mapping from the input to the output.
            For example coord_out = coord[:,idx_map,:]
        sel_atom_type
            Only output if sel_atoms is not None
            The sorted selected atom types
        sel_idx_map
            Only output if sel_atoms is not None
            The index mapping from the selected atoms to sorted selected atoms.
        """
        if mixed_type:
            # mixed_type need not to resort
            natoms = atom_type[0].size
            idx_map = np.arange(natoms)
            return coord, atom_type, idx_map
        if sel_atoms is not None:
            selection = [False] * np.size(atom_type)
            for ii in sel_atoms:
                selection += atom_type == ii
            sel_atom_type = atom_type[selection]
        natoms = atom_type.size
        idx = np.arange(natoms)
        idx_map = np.lexsort((idx, atom_type))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:, idx_map, :], [nframes, -1])
        atom_type = atom_type[idx_map]
        if sel_atoms is not None:
            sel_natoms = np.size(sel_atom_type)
            sel_idx = np.arange(sel_natoms)
            sel_idx_map = np.lexsort((sel_idx, sel_atom_type))
            sel_atom_type = sel_atom_type[sel_idx_map]
            return coord, atom_type, idx_map, sel_atom_type, sel_idx_map
        else:
            return coord, atom_type, idx_map

    @staticmethod
    def reverse_map(vec: np.ndarray, imap: List[int]) -> np.ndarray:
        """Reverse mapping of a vector according to the index map.

        Parameters
        ----------
        vec
            Input vector. Be of shape [nframes, natoms, -1]
        imap
            Index map. Be of shape [natoms]

        Returns
        -------
        vec_out
            Reverse mapped vector.
        """
        ret = np.zeros(vec.shape)
        # for idx,ii in enumerate(imap) :
        #     ret[:,ii,:] = vec[:,idx,:]
        ret[:, imap, :] = vec
        return ret

    def make_natoms_vec(
        self, atom_types: np.ndarray, mixed_type: bool = False
    ) -> np.ndarray:
        """Make the natom vector used by deepmd-kit.

        Parameters
        ----------
        atom_types
            The type of atoms
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        """
        natoms_vec = np.zeros(self.ntypes + 2).astype(int)
        if mixed_type:
            natoms = atom_types[0].size
        else:
            natoms = atom_types.size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        if mixed_type:
            natoms_vec[2] = natoms
            return natoms_vec
        for ii in range(self.ntypes):
            natoms_vec[ii + 2] = np.count_nonzero(atom_types == ii)
        return natoms_vec

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.tf.utils.type_embed.TypeEmbedNet : The type embedding network.

        Examples
        --------
        Get the output of type embedding network of `graph.pb`:

        >>> from deepmd.tf.infer import DeepPotential
        >>> dp = DeepPotential("graph.pb")
        >>> dp.eval_typeebd()
        """
        t_typeebd = self._get_tensor("t_typeebd:0")
        [typeebd] = run_sess(self.sess, [t_typeebd], feed_dict={})
        return typeebd

    def build_neighbor_list(
        self,
        coords: np.ndarray,
        cell: Optional[np.ndarray],
        atype: np.ndarray,
        imap: np.ndarray,
        neighbor_list,
    ):
        """Make the mesh with neighbor list for a single frame.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of atoms. Should be of shape [natoms, 3]
        cell : Optional[np.ndarray]
            The cell of the system. Should be of shape [3, 3]
        atype : np.ndarray
            The type of atoms. Should be of shape [natoms]
        imap : np.ndarray
            The index map of atoms. Should be of shape [natoms]
        neighbor_list : ase.neighborlist.NewPrimitiveNeighborList
            ASE neighbor list. The following method or attribute will be
            used/set: bothways, self_interaction, update, build, first_neigh,
            pair_second, offset_vec.

        Returns
        -------
        natoms_vec : np.ndarray
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: nloc
            natoms[1]: nall
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms for nloc
        coords : np.ndarray
            The coordinates of atoms, including ghost atoms. Should be of
            shape [nframes, nall, 3]
        atype : np.ndarray
            The type of atoms, including ghost atoms. Should be of shape [nall]
        mesh : np.ndarray
            The mesh in nei_mode=4.
        imap : np.ndarray
            The index map of atoms. Should be of shape [nall]
        ghost_map : np.ndarray
            The index map of ghost atoms. Should be of shape [nghost]
        """
        pbc = np.repeat(cell is not None, 3)
        cell = cell.reshape(3, 3)
        positions = coords.reshape(-1, 3)
        neighbor_list.bothways = True
        neighbor_list.self_interaction = False
        if neighbor_list.update(pbc, cell, positions):
            neighbor_list.build(pbc, cell, positions)
        first_neigh = neighbor_list.first_neigh.copy()
        pair_second = neighbor_list.pair_second.copy()
        offset_vec = neighbor_list.offset_vec.copy()
        # get out-of-box neighbors
        out_mask = np.any(offset_vec != 0, axis=1)
        out_idx = pair_second[out_mask]
        out_offset = offset_vec[out_mask]
        out_coords = positions[out_idx] + out_offset.dot(cell)
        atype = np.array(atype, dtype=int)
        out_atype = atype[out_idx]

        nloc = positions.shape[0]
        nghost = out_idx.size
        all_coords = np.concatenate((positions, out_coords), axis=0)
        all_atype = np.concatenate((atype, out_atype), axis=0)
        # convert neighbor indexes
        ghost_map = pair_second[out_mask]
        pair_second[out_mask] = np.arange(nloc, nloc + nghost)
        # get the mesh
        mesh = np.zeros(16 + nloc * 2 + pair_second.size, dtype=int)
        mesh[0] = nloc
        # ilist
        mesh[16 : 16 + nloc] = np.arange(nloc)
        # numnei
        mesh[16 + nloc : 16 + nloc * 2] = first_neigh[1:] - first_neigh[:-1]
        # jlist
        mesh[16 + nloc * 2 :] = pair_second

        # natoms_vec
        natoms_vec = np.zeros(self.ntypes + 2).astype(int)
        natoms_vec[0] = nloc
        natoms_vec[1] = nloc + nghost
        for ii in range(self.ntypes):
            natoms_vec[ii + 2] = np.count_nonzero(atype == ii)
        # imap append ghost atoms
        imap = np.concatenate((imap, np.arange(nloc, nloc + nghost)))
        return natoms_vec, all_coords, all_atype, mesh, imap, ghost_map
