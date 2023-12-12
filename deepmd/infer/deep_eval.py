# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    lru_cache,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.env import (
    MODEL_VERSION,
    default_tf_session_config,
    tf,
)
from deepmd.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.utils.sess import (
    run_sess,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )


class DeepEval:
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
        deepmd.utils.type_embed.TypeEmbedNet : The type embedding network.

        Examples
        --------
        Get the output of type embedding network of `graph.pb`:

        >>> from deepmd.infer import DeepPotential
        >>> dp = DeepPotential('graph.pb')
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
