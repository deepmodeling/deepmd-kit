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
from deepmd.utils.sess import (
    run_sess,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )


class DeepTensor(DeepEval):
    """Evaluates a tensor model.

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
        load_prefix: str = "load",
        default_tf_graph: bool = False,
    ) -> None:
        """Constructor."""
        DeepEval.__init__(
            self, model_file, load_prefix=load_prefix, default_tf_graph=default_tf_graph
        )
        # check model type
        model_type = self.tensors["t_tensor"][2:-2]
        assert (
            self.model_type == model_type
        ), f"expect {model_type} model but got {self.model_type}"

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_value(tensor_name, attr_name)

        # load optional tensors if possible
        optional_tensors = {
            "t_global_tensor": f"o_global_{model_type}:0",
            "t_force": "o_force:0",
            "t_virial": "o_virial:0",
            "t_atom_virial": "o_atom_virial:0",
        }
        try:
            # first make sure these tensor all exists (but do not modify self attr)
            for attr_name, tensor_name in optional_tensors.items():
                self._get_value(tensor_name)
            # then put those into self.attrs
            for attr_name, tensor_name in optional_tensors.items():
                self._get_value(tensor_name, attr_name)
        except KeyError:
            self._support_gfv = False
        else:
            self.tensors.update(optional_tensors)
            self._support_gfv = True

        # self._run_default_sess()
        # self.tmap = self.tmap.decode("UTF-8").split()
        self.ntypes = int(self.model.descrpt.buffer_ntypes)
        self.tselt = self.model.fitting.sel_type

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.tmap, self.tselt, self.output_dim] = run_sess(
            self.sess,
            [
                self.t_ntypes,
                self.t_rcut,
                self.t_tmap,
                self.t_sel_type,
                self.t_ouput_dim,
            ],
        )

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

        # if self.has_fparam:
        #     assert fparam is not None
        #     fparam = np.array(fparam)
        # if self.has_aparam:
        #     assert aparam is not None
        #     aparam = np.array(aparam)
        # if self.has_efield:
        #     assert (
        #         efield is not None
        #     ), "you are using a model with external field, parameter efield should be provided"
        #     efield = np.array(efield)

        # reshape the inputs
        # if self.has_fparam:
        #     fdim = self.get_dim_fparam()
        #     if fparam.size == nframes * fdim:
        #         fparam = np.reshape(fparam, [nframes, fdim])
        #     elif fparam.size == fdim:
        #         fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
        #     else:
        #         raise RuntimeError(
        #             "got wrong size of frame param, should be either %d x %d or %d"
        #             % (nframes, fdim, fdim)
        #         )
        # if self.has_aparam:
        #     fdim = self.get_dim_aparam()
        #     if aparam.size == nframes * natoms * fdim:
        #         aparam = np.reshape(aparam, [nframes, natoms * fdim])
        #     elif aparam.size == natoms * fdim:
        #         aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
        #     elif aparam.size == fdim:
        #         aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
        #     else:
        #         raise RuntimeError(
        #             "got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d"
        #             % (nframes, natoms, fdim, natoms, fdim, fdim)
        #         )

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
        # feed_dict_test = {}
        # feed_dict_test[self.t_natoms] = natoms_vec
        # if mixed_type:
        #     feed_dict_test[self.t_type] = atom_types.reshape([-1])
        # else:
        #     feed_dict_test[self.t_type] = np.tile(atom_types, [nframes, 1]).reshape(
        #         [-1]
        #     )
        # feed_dict_test[self.t_coord] = np.reshape(coords, [-1])

        # if len(self.t_box.shape) == 1:
        #     feed_dict_test[self.t_box] = np.reshape(cells, [-1])
        # elif len(self.t_box.shape) == 2:
        #     feed_dict_test[self.t_box] = cells
        # else:
        #     raise RuntimeError
        # if self.has_efield:
        #     feed_dict_test[self.t_efield] = np.reshape(efield, [-1])
        # if pbc:
        #     feed_dict_test[self.t_mesh] = make_default_mesh(cells)
        # else:
        #     feed_dict_test[self.t_mesh] = np.array([], dtype=np.int32)
        # if self.has_fparam:
        #     feed_dict_test[self.t_fparam] = np.reshape(fparam, [-1])
        # if self.has_aparam:
        #     feed_dict_test[self.t_aparam] = np.reshape(aparam, [-1])
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
        if cells is None:
            pbc = False
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
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

        # if self.has_fparam:
        #     eval_inputs["fparam"] = paddle.to_tensor(
        #         np.reshape(fparam, [-1], dtype="float64")
        #     )
        # if self.has_aparam:
        #     eval_inputs["aparam"] = paddle.to_tensor(
        #         np.reshape(aparam, [-1], dtype="float64")
        #     )
        # if se.pbc:
        #     eval_inputs["default_mesh"] = paddle.to_tensor(
        #     make_default_mesh(cells), dtype="int32"
        # )
        # else:
        eval_inputs["default_mesh"] = paddle.to_tensor(np.array([], dtype=np.int32))

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
        dipole = eval_outputs["dipole"].numpy()

        return dipole

        # if atomic:
        #     ae = eval_outputs["atom_ener"].numpy()
        #     av = eval_outputs["atom_virial"].numpy()
        #     return energy, force, virial, ae, av
        # else:
        #     return energy, force, virial

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut

    def get_type_map(self) -> List[str]:
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
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
    ) -> Tuple[np.ndarray, ...]:
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
        # standarize the shape of inputs
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
        # if self.modifier_type is not None:
        #     if atomic:
        #         raise RuntimeError("modifier does not support atomic modification")
        #     me, mf, mv = self.dm.eval(coords, cells, atom_types)
        #     output = list(output)  # tuple to list
        #     e, f, v = output[:3]
        #     output[0] += me.reshape(e.shape)
        #     output[1] += mf.reshape(f.shape)
        #     output[2] += mv.reshape(v.shape)
        #     output = tuple(output)

        return output

    def eval_full(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
        efield: Optional[np.array] = None,
        mixed_type: bool = False,
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
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

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
        assert self._support_gfv, "do not support eval_full with old tensor model"

        # standarize the shape of inputs
        if mixed_type:
            natoms = atom_types[0].size
            atom_types = np.array(atom_types, dtype=int).reshape([-1, natoms])
        else:
            atom_types = np.array(atom_types, dtype=int).reshape([-1])
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
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(
            coords, atom_types, sel_atoms=self.get_sel_type(), mixed_type=mixed_type
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
        feed_dict_test[self.t_box] = np.reshape(cells, [-1])
        if pbc:
            feed_dict_test[self.t_mesh] = make_default_mesh(cells)
        else:
            feed_dict_test[self.t_mesh] = np.array([], dtype=np.int32)

        t_out = [self.t_global_tensor, self.t_force, self.t_virial]
        if atomic:
            t_out += [self.t_tensor, self.t_atom_virial]

        v_out = self.sess.run(t_out, feed_dict=feed_dict_test)
        gt = v_out[0]  # global tensor
        force = v_out[1]
        virial = v_out[2]
        if atomic:
            at = v_out[3]  # atom tensor
            av = v_out[4]  # atom virial

        # please note here the shape are wrong!
        force = self.reverse_map(np.reshape(force, [nframes * nout, natoms, 3]), imap)
        if atomic:
            at = self.reverse_map(
                np.reshape(at, [nframes, len(sel_at), nout]), sel_imap
            )
            av = self.reverse_map(np.reshape(av, [nframes * nout, natoms, 9]), imap)

        # make sure the shapes are correct here
        gt = np.reshape(gt, [nframes, nout])
        force = np.reshape(force, [nframes, nout, natoms, 3])
        virial = np.reshape(virial, [nframes, nout, 9])
        if atomic:
            at = np.reshape(at, [nframes, len(sel_at), self.output_dim])
            av = np.reshape(av, [nframes, nout, natoms, 9])
            return gt, force, virial, at, av
        else:
            return gt, force, virial
