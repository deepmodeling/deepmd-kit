import os
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from deepmd.common import make_default_mesh
from deepmd.env import default_tf_session_config, tf
from deepmd.infer.deep_eval import DeepEval

if TYPE_CHECKING:
    from pathlib import Path

class DeepTensor(DeepEval):
    """Evaluates a tensor model.

    Constructor

    Parameters
    ----------
    model_file: str
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    """

    tensors = {
        # descriptor attrs
        "t_ntypes": "descrpt_attr/ntypes:0",
        "t_rcut": "descrpt_attr/rcut:0",
        # model attrs
        "t_tmap": "model_attr/tmap:0",
        "t_sel_type": "model_attr/sel_type:0",
        "t_ouput_dim": "model_attr/output_dim:0",
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
        load_prefix: str = 'load',
        default_tf_graph: bool = False
    ) -> None:
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph
        )
        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_tensor(tensor_name, attr_name)

        # start a tf session associated to the graph
        self.sess = tf.Session(graph=self.graph, config=default_tf_session_config)
        self._run_default_sess()
        self.tmap = self.tmap.decode('UTF-8').split()

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.tmap, self.tselt, self.output_dim] \
            = self.sess.run(
                [self.t_ntypes, self.t_rcut, self.t_tmap, self.t_sel_type, self.t_ouput_dim]
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
                If atomic == False then of size nframes x output_dim
                else of size nframes x natoms x output_dim
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
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes,1]).reshape([-1])
        t_out = [self.t_tensor]
        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
        feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
        feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
        v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
        tensor = v_out[0]

        # reverse map of the outputs
        if atomic:
            tensor = np.array(tensor)
            tensor = self.reverse_map(np.reshape(tensor, [nframes,-1,self.output_dim]), sel_imap)
            tensor = np.reshape(tensor, [nframes, len(sel_at), self.output_dim])
        else:
            tensor = np.reshape(tensor, [nframes, self.output_dim])
        
        return tensor
