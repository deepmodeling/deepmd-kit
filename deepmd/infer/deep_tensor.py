import os
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np
from deepmd.common import make_default_mesh
from deepmd.env import default_tf_session_config, tf
from deepmd.infer.deep_eval import DeepEval
from deepmd.utils.sess import run_sess

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
        # check model type
        model_type = self.tensors["t_tensor"][2:-2]
        assert self.model_type == model_type, \
            f"expect {model_type} model but got {self.model_type}"

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_tensor(tensor_name, attr_name)
        
        # load optional tensors if possible
        optional_tensors = {
            "t_global_tensor": f"o_global_{model_type}:0",
            "t_force": "o_force:0",
            "t_virial": "o_virial:0",
            "t_atom_virial": "o_atom_virial:0"
        }
        try:
            # first make sure these tensor all exists (but do not modify self attr)
            for attr_name, tensor_name in optional_tensors.items():
                self._get_tensor(tensor_name)
            # then put those into self.attrs
            for attr_name, tensor_name in optional_tensors.items():
                self._get_tensor(tensor_name, attr_name)
            self.tensors.update(optional_tensors)
            self._support_gfv = True
        except KeyError:
            self._support_gfv = False
            
        # start a tf session associated to the graph
        self.sess = tf.Session(graph=self.graph, config=default_tf_session_config)
        self._run_default_sess()
        self.tmap = self.tmap.decode('UTF-8').split()

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.tmap, self.tselt, self.output_dim] \
            = run_sess(self.sess, 
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
            If True (default), return the atomic tensor
            Otherwise return the global tensor
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
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        if cells is None:
            pbc = False
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])

        # sort inputs
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(coords, atom_types, sel_atoms = self.get_sel_type())

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes,1]).reshape([-1])
        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
        feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
        if pbc:
            feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
        else:
            feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)

        if atomic:
            assert "global" not in self.model_type, \
                f"cannot do atomic evaluation with model type {self.model_type}"
            t_out = [self.t_tensor]
        else:
            assert self._support_gfv or "global" in self.model_type, \
                f"do not support global tensor evaluation with old {self.model_type} model"
            t_out = [self.t_global_tensor if self._support_gfv else self.t_tensor]
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

    def eval_full(
        self,
        coords: np.array,
        cells: np.array,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
        efield: Optional[np.array] = None
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate the model with interface similar to the energy model.
        Will return global tensor, component-wise force and virial
        and optionally atomic tensor and atomic virial.

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
            Whether to calculate atomic tensor and virial
        fparam
            Not used in this model
        aparam
            Not used in this model
        efield
            Not used in this model

        Returns
        -------
        tensor
            The global tensor. 
            shape: [nframes x nout]
        force
            The component-wise force (negative derivative) on each atom.
            shape: [nframes x nout x natoms x 3]
        virial
            The component-wise virial of the tensor.
            shape: [nframes x nout x 9]
        atom_tensor
            The atomic tensor. Only returned when atomic == True
            shape: [nframes x natoms x nout]
        atom_virial
            The atomic virial. Only returned when atomic == True
            shape: [nframes x nout x natoms x 9]
        """
        assert self._support_gfv, \
            f"do not support eval_full with old tensor model"

        # standarize the shape of inputs
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        if cells is None:
            pbc = False
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
        nout = self.output_dim

        # sort inputs
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(coords, atom_types, sel_atoms = self.get_sel_type())

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes,1]).reshape([-1])
        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
        feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
        if pbc:
            feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
        else:
            feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)
        
        t_out = [self.t_global_tensor, 
                 self.t_force, 
                 self.t_virial]
        if atomic :
            t_out += [self.t_tensor, 
                      self.t_atom_virial]
        
        v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
        gt     = v_out[0] # global tensor
        force  = v_out[1]
        virial = v_out[2]
        if atomic:
            at = v_out[3] # atom tensor
            av = v_out[4] # atom virial

        # please note here the shape are wrong!
        force  = self.reverse_map(np.reshape(force, [nframes*nout, natoms ,3]), imap)
        if atomic:
            at = self.reverse_map(np.reshape(at, [nframes, len(sel_at), nout]), sel_imap)
            av = self.reverse_map(np.reshape(av, [nframes*nout, natoms, 9]), imap)
        
        # make sure the shapes are correct here
        gt     = np.reshape(gt, [nframes, nout])
        force  = np.reshape(force, [nframes, nout, natoms, 3])
        virial = np.reshape(virial, [nframes, nout, 9])
        if atomic:
            at = np.reshape(at, [nframes, len(sel_at), self.output_dim])
            av = np.reshape(av, [nframes, nout, natoms, 9])
            return gt, force, virial, at, av
        else:
            return gt, force, virial