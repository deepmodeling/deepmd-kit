import os
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from deepmd.common import make_default_mesh
from deepmd.env import default_tf_session_config, tf

if TYPE_CHECKING:
    from pathlib import Path


class DeepEval:
    """Common methods for DeepPot, DeepWFC, DeepPolar, ..."""

    _model_type: Optional[str] = None
    load_prefix: str  # set by subclass

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False
    ):
        self.graph = self._load_graph(
            model_file, prefix=load_prefix, default_tf_graph=default_tf_graph
        )
        self.load_prefix = load_prefix

    @property
    def model_type(self) -> str:
        """Get type of model.

        :type:str
        """
        if not self._model_type:
            t_mt = self._get_tensor("model_attr/model_type:0")
            sess = tf.Session(graph=self.graph, config=default_tf_session_config)
            [mt] = sess.run([t_mt], feed_dict={})
            self._model_type = mt.decode("utf-8")

        return self._model_type

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
        tensor_path = os.path.join(self.load_prefix, tensor_name)
        tensor = self.graph.get_tensor_by_name(tensor_path)
        if attr_name:
            setattr(self, attr_name, tensor)
            return tensor
        else:
            return tensor

    @staticmethod
    def _load_graph(
        frozen_graph_filename: "Path", prefix: str = "load", default_tf_graph: bool = False
    ):


        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            if default_tf_graph:
                tf.import_graph_def(
                    graph_def, 
                    input_map=None, 
                    return_elements=None, 
                    name=prefix, 
                    producer_op_list=None
                )
                graph = tf.get_default_graph()
            else :
                # Then, we can use again a convenient built-in function to import
                # a graph_def into the  current default Graph
                with tf.Graph().as_default() as graph:
                    tf.import_graph_def(
                        graph_def,
                        input_map=None,
                        return_elements=None,
                        name=prefix,
                        producer_op_list=None
                    )

            return graph

class DeepTensor(DeepEval):
    """Evaluates a tensor model.

    Constructor

    Parameters
    ----------
    model_file: str
        The name of the frozen model file.
    variable_dof: int
        The DOF of the variable to evaluate.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    """

    tensors = {
        "t_ntypes": "descrpt_attr/ntypes:0",
        "t_rcut": "descrpt_attr/rcut:0",
        "t_tmap": "model_attr/tmap:0",
        # inputs
        "t_coord": "t_coord:0",
        "t_type": "t_type:0",
        "t_natoms": "t_natoms:0",
        "t_box": "t_box:0",
        "t_mesh": "t_mesh:0",
    }

    def __init__(
        self,
        model_file: "Path",
        variable_dof: Optional[int],
        load_prefix: str = 'load',
        default_tf_graph: bool = False
    ) -> None:
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph
        )
        self.variable_dof = variable_dof

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_tensor(tensor_name, attr_name)

        # start a tf session associated to the graph
        self.sess = tf.Session(graph=self.graph, config=default_tf_session_config)
        self._run_default_sess()
        self.tmap = self.tmap.decode('UTF-8').split()

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.tmap, self.tselt] = self.sess.run(
            [self.t_ntypes, self.t_rcut, self.t_tmap, self.t_sel_type]
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
        """Get the selected atom types of this model."""        
        return self.tselt

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dfparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.daparam

    def eval(
        self,
        coords: np.array,
        cells: np.array,
        atom_types: List[int],
        atomic: bool = True,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
        efield: Optional[np.array] = None
    ) -> np.array:
        """Evaluate the model.

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
            Not used in this model
        aparam
            Not used in this model
        efield
            Not used in this model

        Returns
        -------
        tensor
                The returned tensor
                If atomic == False then of size nframes x variable_dof
                else of size nframes x natoms x variable_dof
        """
        # standarize the shape of inputs
        coords = np.array(coords)
        cells = np.array(cells)
        atom_types = np.array(atom_types, dtype = int)

        # reshape the inputs 
        cells = np.reshape(cells, [-1, 9])
        nframes = cells.shape[0]
        coords = np.reshape(coords, [nframes, -1])
        natoms = coords.shape[1] // 3

        # sort inputs
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(coords, atom_types, sel_atoms = self.get_sel_type())

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        tensor = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_tensor]
        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            feed_dict_test[self.t_mesh ] = make_default_mesh(cells[ii:ii+1, :])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            tensor.append(v_out[0])

        # reverse map of the outputs
        if atomic:
            tensor = np.array(tensor)
            tensor = self.reverse_map(np.reshape(tensor, [nframes,-1,self.variable_dof]), sel_imap)
            tensor = np.reshape(tensor, [nframes, len(sel_at), self.variable_dof])
        else:
            tensor = np.reshape(tensor, [nframes, self.variable_dof])
        
        return tensor

    @staticmethod
    def sort_input(
        coord : np.array, atom_type : np.array, sel_atoms : List[int] = None
    ):
        """
        Sort atoms in the system according their types.
        
        Parameters
        ----------
        coord
                The coordinates of atoms.
                Should be of shape [nframes, natoms, 3]
        atom_type
                The type of atoms
                Should be of shape [natoms]
        sel_atom
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
        if sel_atoms is not None:
            selection = [False] * np.size(atom_type)
            for ii in sel_atoms:
                selection += (atom_type == ii)
            sel_atom_type = atom_type[selection]
        natoms = atom_type.size
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:,idx_map,:], [nframes, -1])
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
    def reverse_map(vec : np.ndarray, imap : List[int]) -> np.ndarray:
        """Reverse mapping of a vector according to the index map

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
        for idx,ii in enumerate(imap) :
            ret[:,ii,:] = vec[:,idx,:]
        return ret

    def make_natoms_vec(self, atom_types : np.ndarray) -> np.ndarray :
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
        natoms_vec = np.zeros (self.ntypes+2).astype(int)
        natoms = atom_types.size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        for ii in range (self.ntypes) :
            natoms_vec[ii+2] = np.count_nonzero(atom_types == ii)
        return natoms_vec

