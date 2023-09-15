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


class DeepDOS(DeepEval):
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
            "t_numb_dos": "fitting_attr/numb_dos:0",
            # model attrs
            "t_tmap": "model_attr/tmap:0",
            # inputs
            "t_coord": "t_coord:0",
            "t_type": "t_type:0",
            "t_natoms": "t_natoms:0",
            "t_box": "t_box:0",
            "t_mesh": "t_mesh:0",
            # add output tensors
            "t_dos": "o_dos:0",
            "t_atom_dos": "o_atom_dos:0",
            "t_descriptor": "o_descriptor:0",
        }
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            auto_batch_size=auto_batch_size,
            input_map=input_map,
        )

        # load optional tensors
        operations = [op.name for op in self.graph.get_operations()]
        # check if the graph has these operations:
        # if yes add them
        if "load/t_fparam" in operations:
            self.tensors.update({"t_fparam": "t_fparam:0"})
            self.has_fparam = True
        else:
            log.debug("Could not get tensor 't_fparam:0'")
            self.t_fparam = None
            self.has_fparam = False

        if "load/t_aparam" in operations:
            self.tensors.update({"t_aparam": "t_aparam:0"})
            self.has_aparam = True
        else:
            log.debug("Could not get tensor 't_aparam:0'")
            self.t_aparam = None
            self.has_aparam = False

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

    def _run_default_sess(self):
        [
            self.ntypes,
            self.rcut,
            self.numb_dos,
            self.dfparam,
            self.daparam,
            self.tmap,
        ] = run_sess(
            self.sess,
            [
                self.t_ntypes,
                self.t_rcut,
                self.t_numb_dos,
                self.t_dfparam,
                self.t_daparam,
                self.t_tmap,
            ],
        )

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_numb_dos(self) -> int:
        """Get the length of DOS output of this DP model."""
        return self.numb_dos

    def get_type_map(self) -> List[str]:
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
        mixed_type: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate the dos, atom_dos by using this model.

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
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        dos
            The electron density of state.
        atom_dos
            The atom-sited density of state. Only returned when atomic == True
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
            mixed_type=mixed_type,
        )

        return output

    def _prepare_feed_dict(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        atomic=False,
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
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
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

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types, mixed_type=mixed_type)
        assert natoms_vec[0] == natoms

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
        feed_dict_test[self.t_mesh] = make_default_mesh(pbc, mixed_type)
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
        mixed_type=False,
    ):
        natoms, nframes = self._get_natoms_and_nframes(
            coords, atom_types, mixed_type=mixed_type
        )
        feed_dict_test, imap = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, mixed_type=mixed_type
        )
        t_out = [self.t_dos]
        if atomic:
            t_out += [self.t_atom_dos]

        v_out = run_sess(self.sess, t_out, feed_dict=feed_dict_test)
        dos = v_out[0]
        if atomic:
            atom_dos = v_out[1]

        # reverse map of the outputs
        if atomic:
            atom_dos = self.reverse_map(
                np.reshape(atom_dos, [nframes, -1, self.numb_dos]), imap
            )
            dos = np.sum(atom_dos, axis=1)

        dos = np.reshape(dos, [nframes, self.numb_dos])
        if atomic:
            atom_dos = np.reshape(atom_dos, [nframes, natoms, self.numb_dos])
            return dos, atom_dos
        else:
            return dos

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
        feed_dict_test, imap = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, efield, mixed_type=mixed_type
        )
        (descriptor,) = run_sess(
            self.sess, [self.t_descriptor], feed_dict=feed_dict_test
        )
        return self.reverse_map(np.reshape(descriptor, [nframes, natoms, -1]), imap)
