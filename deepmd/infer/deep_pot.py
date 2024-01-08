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
from deepmd.env import (
    paddle,
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
        self.tensors = {
                # descrpt attrs
                "ntypes": "descrpt.ntypes",
                "rcut": "descrpt.rcut",
                # fitting attrs
                "dfparam": "fitting.t_dfparam",
                "daparam": "fitting.t_daparam",
            }
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            # default_tf_graph=default_tf_graph,
            auto_batch_size=auto_batch_size,
        )

        # # load optional tensors
        self.has_efield = False

        self.has_fparam = False

        self.has_aparam = False
        self.ntypes_spin = 0
        self.has_spin = False

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            try:
                self._get_value(tensor_name, attr_name)
            except KeyError:
                if attr_name != "t_descriptor":
                    raise

        self.ntypes = int(self.model.descrpt.buffer_ntypes)
        self.rcut = float(self.model.descrpt.buffer_rcut)
        self.dfparam = 0
        self.daparam = 0
        self.t_tmap = [chr(idx) for idx in self.model.buffer_tmap.tolist()]
        self.t_tmap = [c for c in self.t_tmap if c != " "]

        # setup modifier
        try:
            self.modifier_type = self._get_value("modifier_attr.type")
        except (ValueError, KeyError):
            self.modifier_type = None
        self.modifier_type = None
        self.descriptor_type = "se_e2_a"

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
        return self.t_tmap

    def get_sel_type(self) -> List[int]:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_descriptor_type(self) -> List[int]:
        """Get the descriptor type of this model."""
        return self.descriptor_type

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.model.fitting.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.model.fitting.numb_aparam

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
        )  # 192, 30
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
        atomic=False,
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
        # coords, atom_types, imap = self.sort_input(
        #     coords, atom_types, mixed_type=mixed_type
        # )
        # if self.has_efield:
        #     efield = np.reshape(efield, [nframes, natoms, 3])
        #     efield = efield[:, imap, :]
        #     efield = np.reshape(efield, [nframes, natoms * 3])

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types, mixed_type=mixed_type)
        assert natoms_vec[0] == natoms

        # evaluate
        return None, None, natoms_vec

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
        feed_dict_test, imap, natoms_vec = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, efield, mixed_type=mixed_type
        )

        eval_inputs = {}
        eval_inputs["coord"] = paddle.to_tensor(
            np.reshape(coords, [-1]), dtype="float64"
        )
        eval_inputs["type"] = paddle.to_tensor(
            np.tile(atom_types, [nframes, 1]).reshape([-1]), dtype="int32"
        )
        eval_inputs["natoms_vec"] = paddle.to_tensor(
            natoms_vec, dtype="int32", place="cpu"
        )
        eval_inputs["box"] = paddle.to_tensor(np.reshape(cells, [-1]), dtype="float64")

        if self.has_fparam:
            eval_inputs["fparam"] = paddle.to_tensor(
                np.reshape(fparam, [-1], dtype="float64")
            )
        if self.has_aparam:
            eval_inputs["aparam"] = paddle.to_tensor(
                np.reshape(aparam, [-1], dtype="float64")
            )
        eval_inputs["default_mesh"] = paddle.to_tensor(
            make_default_mesh(cells), dtype="int32"
        )

        if hasattr(self, "st_model"):
            # NOTE: 使用静态图模型推理
            eval_outputs = self.st_model(
                eval_inputs["coord"],
                eval_inputs["type"],
                eval_inputs["natoms_vec"],
                eval_inputs["box"],
                eval_inputs["default_mesh"],
            )
            eval_outputs = {
                "atom_ener": eval_outputs[0],
                "atom_virial": eval_outputs[1],
                "atype": eval_outputs[2],
                "coord": eval_outputs[3],
                "energy": eval_outputs[4],
                "force": eval_outputs[5],
                "virial": eval_outputs[6],
            }
        else:
            # NOTE: 使用动态图模型推理
            eval_outputs = self.model(
                eval_inputs["coord"],
                eval_inputs["type"],
                eval_inputs["natoms_vec"],
                eval_inputs["box"],
                eval_inputs["default_mesh"],
                eval_inputs,
                suffix="",
                reuse=False,
            )
        energy = eval_outputs["energy"].numpy()
        force = eval_outputs["force"].numpy()
        virial = eval_outputs["virial"].numpy()
        if atomic:
            ae = eval_outputs["atom_ener"].numpy()
            av = eval_outputs["atom_virial"].numpy()
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
        feed_dict_test, imap, natoms_vec = self._prepare_feed_dict(
            coords, cells, atom_types, fparam, aparam, efield, mixed_type=mixed_type
        )
        (descriptor,) = run_sess(
            self.sess, [self.t_descriptor], feed_dict=feed_dict_test
        )
        return self.reverse_map(np.reshape(descriptor, [nframes, natoms, -1]), imap)
