import os
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import json
from deepmd.common import make_default_mesh, j_must_have
from deepmd.env import MODEL_VERSION, paddle, tf
from deepmd.fit import EnerFitting
from deepmd.descriptor import DescrptSeA
from deepmd.model import EnerModel

if TYPE_CHECKING:
    from pathlib import Path


class DeepEval:
    """Common methods for DeepPot, DeepWFC, DeepPolar, ..."""

    _model_type: Optional[str] = None
    _model_version: Optional[str] = None
    load_prefix: str  # set by subclass

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False
    ):
        ##### Hard code, will change to use dy2stat, avoid to build model #######
        ##### Now use paddle.load temporarily#######
        with open("out.json", 'r') as load_f:
            jdata = json.load(load_f)
        
        model_param = j_must_have(jdata, 'model')
        descrpt_param = j_must_have(model_param, 'descriptor')
        descrpt_param.pop('type', None)
        self.descrpt = descrpt = DescrptSeA(**descrpt_param)

        fitting_param = j_must_have(model_param, 'fitting_net')
        fitting_param.pop('type', None)
        fitting_param['descrpt'] = self.descrpt
        self.fitting = EnerFitting(**fitting_param)
        
        # self.model = EnerModel(
        #         self.descrpt, 
        #         self.fitting, 
        #         model_param.get('type_map'),
        #         model_param.get('data_stat_nbatch', 10),
        #         model_param.get('data_stat_protect', 1e-2),
        #         model_param.get('use_srtab'),
        #         model_param.get('smin_alpha'),
        #         model_param.get('sw_rmin'),
        #         model_param.get('sw_rmax')
        #     )
        # self.model.set_dict(paddle.load(model_file))
        self.model = paddle.jit.load(model_file)
        ################################################################

        self.load_prefix = load_prefix

        # graph_compatable should be called after graph and prefix are set
        if not self._graph_compatable():
            raise RuntimeError(
                f"model in graph (version {self.model_version}) is incompatible"
                f"with the model (version {MODEL_VERSION}) supported by the current code."
            )

    @property
    def model_type(self) -> str:
        """Get type of model.

        :type:str
        """
        self._model_type = self.model.t_mt
        return self._model_type

    @property
    def model_version(self) -> str:
        """Get type of model.

        :type:str
        """

        if not self._model_version:
            try:
                self._model_version = self.model.t_ver
            except KeyError:
                # For deepmd-kit version 0.x - 1.x, set model version to 0.0
                self._model_version = "0.0"
        return self._model_version

    def _graph_compatable(
        self
    ) -> bool :
        """ Check the model compatability
        
        Return
        bool
            If the model stored in the graph file is compatable with the current code
        """
        model_version_major = int(self.model_version.split('.')[0])
        model_version_minor = int(self.model_version.split('.')[1])
        MODEL_VERSION_MAJOR = int(MODEL_VERSION.split('.')[0])
        MODEL_VERSION_MINOR = int(MODEL_VERSION.split('.')[1])
        if (model_version_major != MODEL_VERSION_MAJOR) or \
           (model_version_minor >  MODEL_VERSION_MINOR) :
            return False
        else:
            return True

    def _get_value(
        self, tensor_name: str, attr_name: Optional[str] = None
    ):
        """
        """
        value = None
        for name, tensor in self.model.named_buffers():
            if tensor_name in name:
                value = tensor.numpy()[0] if tensor.shape == [1] else tensor.numpy()
        if attr_name:
            setattr(self, attr_name, value)
            return value
        else:
            return value

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
