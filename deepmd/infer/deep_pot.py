import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from collections import defaultdict
from deepmd.common import make_default_mesh
from deepmd.env import paddle, GLOBAL_PD_FLOAT_PRECISION, GLOBAL_NP_FLOAT_PRECISION
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.infer.deep_eval import DeepEval

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
        default_tf_graph: bool = False
    ) -> None:

        # add these tensors on top of what is defined by DeepTensor Class
        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # descrpt attrs
                "ntypes": "descrpt.t_ntypes",
                "rcut": "descrpt.t_rcut",
                # fitting attrs
                "dfparam": "fitting.t_dfparam",
                "daparam": "fitting.t_daparam",
            },
        )
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
        )

        if self._get_value("t_efield") is not None:
            self._get_tensor("t_efield", "t_efield")
            self.has_efield = True
        else:
            log.debug(f"Could not get tensor 't_efield:0'")
            self.t_efield = None
            self.has_efield = False

        if self._get_value("t_fparam") is not None:
            self.tensors.update({"t_fparam": "t_fparam"})
            self.has_fparam = True
        else:
            log.debug(f"Could not get tensor 't_fparam'")
            self.t_fparam = None
            self.has_fparam = False

        if self._get_value("t_aparam") is not None:
            self.tensors.update({"t_aparam": "t_aparam"})
            self.has_aparam = True
        else:
            log.debug(f"Could not get tensor 't_aparam'")
            self.t_aparam = None
            self.has_aparam = False

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_value(tensor_name, attr_name)

        self.t_tmap = self.model.t_tmap.split()

        # setup modifier
        try:
            self.modifier_type = self._get_value("modifier_attr.type")
        except (ValueError, KeyError):
            self.modifier_type = None

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_type_map(self) -> List[int]:
        """Get the type map (element name of the atom types) of this model."""
        return self.t_tmap

    def get_sel_type(self) -> List[int]:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
    
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
        atomic: bool = False,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
        efield: Optional[np.array] = None
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
        if atomic:
            if self.modifier_type is not None:
                raise RuntimeError('modifier does not support atomic modification')
            return self._eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic, efield = efield)
        else :
            e, f, v = self._eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic, efield = efield)
            if self.modifier_type is not None:
                me, mf, mv = self.dm.eval(coords, cells, atom_types)
                e += me.reshape(e.shape)
                f += mf.reshape(f.shape)
                v += mv.reshape(v.shape)
            return e, f, v

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
        # standarize the shape of inputs
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
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
        eval_inputs = {}
        eval_inputs['coord'] = paddle.to_tensor(np.reshape(coords, [-1]), dtype=GLOBAL_PD_FLOAT_PRECISION)
        print(eval_inputs['coord'])
        eval_inputs['type'] = paddle.to_tensor(np.tile(atom_types, [nframes, 1]).reshape([-1]), dtype="int32")
        eval_inputs['natoms_vec'] = paddle.to_tensor(natoms_vec, dtype="int32")
        eval_inputs['box'] = paddle.to_tensor(np.reshape(cells , [-1]), dtype=GLOBAL_PD_FLOAT_PRECISION)
          
        if self.has_fparam:
            eval_inputs["fparam"] = paddle.to_tensor(np.reshape(fparam, [-1], dtype=GLOBAL_PD_FLOAT_PRECISION))
        if self.has_aparam:
            eval_inputs["aparam"] = paddle.to_tensor(np.reshape(aparam, [-1], dtype=GLOBAL_PD_FLOAT_PRECISION))
        if pbc:
            eval_inputs['default_mesh'] = paddle.to_tensor(make_default_mesh(cells), dtype="int32")
        else:
            eval_inputs['default_mesh'] = paddle.to_tensor(np.array([], dtype = np.int32))

        eval_outputs = self.model(eval_inputs['coord'], eval_inputs['type'], eval_inputs['natoms_vec'], eval_inputs['box'], eval_inputs['default_mesh'], eval_inputs, suffix = "", reuse = False)

        energy = eval_outputs['energy'].numpy()
        force = eval_outputs['force'].numpy()
        virial = eval_outputs['virial'].numpy()
        if atomic:
            ae = eval_outputs['atom_ener'].numpy()
            av = eval_outputs['atom_virial'].numpy()

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
