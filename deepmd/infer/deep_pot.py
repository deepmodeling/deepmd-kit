# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.common import (
    make_default_mesh,
)
from deepmd.infer.data_modifier import (
    DipoleChargeModifier,
)
from deepmd.infer.deep_eval import (
    DeepEval,
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
    input_map : dict, optional
        The input map for tf.import_graph_def. Only work with default tf graph
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.

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
        input_map: Optional[dict] = None,
        neighbor_list=None,
    ) -> None:
        # add these tensors on top of what is defined by DeepTensor Class
        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = {
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
        }
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            auto_batch_size=auto_batch_size,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )

        # load optional tensors
        operations = [op.name for op in self.graph.get_operations()]
        # check if the graph has these operations:
        # if yes add them

        if (f"{load_prefix}/t_efield") in operations:
            self.tensors.update({"t_efield": "t_efield:0"})
            self.has_efield = True
        else:
            log.debug("Could not get tensor 't_efield:0'")
            self.t_efield = None
            self.has_efield = False

        if (f"{load_prefix}/t_fparam") in operations:
            self.tensors.update({"t_fparam": "t_fparam:0"})
            self.has_fparam = True
        else:
            log.debug("Could not get tensor 't_fparam:0'")
            self.t_fparam = None
            self.has_fparam = False

        if (f"{load_prefix}/t_aparam") in operations:
            self.tensors.update({"t_aparam": "t_aparam:0"})
            self.has_aparam = True
        else:
            log.debug("Could not get tensor 't_aparam:0'")
            self.t_aparam = None
            self.has_aparam = False

        if (f"{load_prefix}/spin_attr/ntypes_spin") in operations:
            self.tensors.update({"t_ntypes_spin": "spin_attr/ntypes_spin:0"})
            self.has_spin = True
        else:
            self.ntypes_spin = 0
            self.has_spin = False

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            try:
                self._get_tensor(tensor_name, attr_name)
            except KeyError:
                if attr_name != "t_descriptor":
                    raise

        self._run_default_sess()
        self.tmap = self.tmap.decode("UTF-8").split()

        # setup modifier
        try:
            t_modifier_type = self._get_tensor("modifier_attr/type:0")
            self.modifier_type = run_sess(self.sess, t_modifier_type).decode("UTF-8")
        except (ValueError, KeyError):
            self.modifier_type = None

        try:
            t_jdata = self._get_tensor("train_attr/training_script:0")
            jdata = run_sess(self.sess, t_jdata).decode("UTF-8")
            import json

            jdata = json.loads(jdata)
            self.descriptor_type = jdata["model"]["descriptor"]["type"]
        except (ValueError, KeyError):
            self.descriptor_type = None

        if self.modifier_type == "dipole_charge":
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

    def _run_default_sess(self):
        if self.has_spin is True:
            [
                self.ntypes,
                self.ntypes_spin,
                self.rcut,
                self.dfparam,
                self.daparam,
                self.tmap,
            ] = run_sess(
                self.sess,
                [
                    self.t_ntypes,
                    self.t_ntypes_spin,
                    self.t_rcut,
                    self.t_dfparam,
                    self.t_daparam,
                    self.t_tmap,
                ],
            )
        else:
            [self.ntypes, self.rcut, self.dfparam, self.daparam, self.tmap] = run_sess(
                self.sess,
                [
                    self.t_ntypes,
                    self.t_rcut,
                    self.t_dfparam,
                    self.t_daparam,
                    self.t_tmap,
                ],
            )

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_ntypes_spin(self):
        """Get the number of spin atom types of this model."""
        return self.ntypes_spin

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_type_map(self) -> List[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.tmap

    def get_sel_type(self) -> List[int]:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_descriptor_type(self) -> List[int]:
        """Get the descriptor type of this model."""
        return self.descriptor_type

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
        mixed_type: bool = False,
    ) -> Tuple[int, int]:
        if mixed_type:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
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
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
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
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

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
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        output = self._eval_func(self._eval_inner, numb_test, natoms)(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            atomic=atomic,
            efield=efield,
            mixed_type=mixed_type,
        )

        if self.modifier_type is not None:
            if atomic:
                raise RuntimeError("modifier does not support atomic modification")
            me, mf, mv = self.dm.eval(coords, cells, atom_types)
            output = list(output)  # tuple to list
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
        efield=None,
        mixed_type=False,
    ):
        # standarize the shape of inputs
        natoms, nframes = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        if mixed_type:
            atom_types = np.array(atom_types, dtype=int).reshape([-1, natoms])
        else:
            atom_types = np.array(atom_types, dtype=int).reshape([-1])
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
        coords, atom_types, imap = self.sort_input(
            coords, atom_types, mixed_type=mixed_type
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
            natoms_vec = self.make_natoms_vec(atom_types, mixed_type=mixed_type)
            assert natoms_vec[0] == natoms
            mesh = make_default_mesh(pbc, mixed_type)
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
        feed_dict_test[self.t_natoms] = natoms_vec
        if mixed_type:
            feed_dict_test[self.t_type] = atom_types.reshape([-1])
        else:
            feed_dict_test[self.t_type] = np.tile(atom_types, [nframes, 1]).reshape(
                [-1]
            )
        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])

        if len(self.t_box.shape) == 1:
            feed_dict_test[self.t_box] = np.reshape(cells, [-1])
        elif len(self.t_box.shape) == 2:
            feed_dict_test[self.t_box] = cells
        else:
            raise RuntimeError
        if self.has_efield:
            feed_dict_test[self.t_efield] = np.reshape(efield, [-1])
        feed_dict_test[self.t_mesh] = mesh
        if self.has_fparam:
            feed_dict_test[self.t_fparam] = np.reshape(fparam, [-1])
        if self.has_aparam:
            feed_dict_test[self.t_aparam] = np.reshape(aparam, [-1])
        return feed_dict_test, imap, natoms_vec, ghost_map

    def _eval_inner(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        atomic=False,
        efield=None,
        mixed_type=False,
    ):
        natoms, nframes = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        feed_dict_test, imap, natoms_vec, ghost_map = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, efield, mixed_type=mixed_type
        )

        nloc = natoms_vec[0]
        nall = natoms_vec[1]

        t_out = [self.t_energy, self.t_force, self.t_virial]
        if atomic:
            t_out += [self.t_ae, self.t_av]

        v_out = run_sess(self.sess, t_out, feed_dict=feed_dict_test)
        energy = v_out[0]
        force = v_out[1]
        virial = v_out[2]
        if atomic:
            ae = v_out[3]
            av = v_out[4]

        if self.has_spin:
            ntypes_real = self.ntypes - self.ntypes_spin
            natoms_real = sum(
                [
                    np.count_nonzero(np.array(atom_types) == ii)
                    for ii in range(ntypes_real)
                ]
            )
        else:
            natoms_real = natoms
        if ghost_map is not None:
            # add the value of ghost atoms to real atoms
            force = np.reshape(force, [nframes, -1, 3])
            np.add.at(force[0], ghost_map, force[0, nloc:])
            if atomic:
                av = np.reshape(av, [nframes, -1, 9])
                np.add.at(av[0], ghost_map, av[0, nloc:])

        # reverse map of the outputs
        force = self.reverse_map(np.reshape(force, [nframes, -1, 3]), imap)
        if atomic:
            ae = self.reverse_map(np.reshape(ae, [nframes, -1, 1]), imap[:natoms_real])
            av = self.reverse_map(np.reshape(av, [nframes, -1, 9]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, nall, 3])
        if nloc < nall:
            force = force[:, :nloc, :]
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms_real, 1])
            av = np.reshape(av, [nframes, nall, 9])
            if nloc < nall:
                av = av[:, :nloc, :]
            return energy, force, virial, ae, av
        else:
            return energy, force, virial

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
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
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        descriptor
            Descriptors.
        """
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        descriptor = self._eval_func(self._eval_descriptor_inner, numb_test, natoms)(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            efield=efield,
            mixed_type=mixed_type,
        )
        return descriptor

    def _eval_descriptor_inner(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
    ) -> np.array:
        natoms, nframes = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        feed_dict_test, imap, natoms_vec, ghost_map = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, efield, mixed_type=mixed_type
        )
        (descriptor,) = run_sess(
            self.sess, [self.t_descriptor], feed_dict=feed_dict_test
        )
        imap = imap[:natoms]
        return self.reverse_map(np.reshape(descriptor, [nframes, natoms, -1]), imap)
