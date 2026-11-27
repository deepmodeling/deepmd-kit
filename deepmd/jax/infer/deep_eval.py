# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import warnings
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.default_neighbor_list import (
    DefaultNeighborList,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_polar import (
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.jax.common import (
    to_jax_array,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.hlo import (
    HLO,
)
from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.jax.utils.graph_batch import (
    prepare_graph_batch,
)
from deepmd.jax.utils.precision import (
    MatmulPrecisionPolicy,
)
from deepmd.utils.charge_state import (
    validate_charge_states,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEval(DeepEvalBackend):
    """JAX checkpoint and frozen-artifact implementation of DeepEval.

    Parameters
    ----------
    model_file : str
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: Any,
        auto_batch_size: bool | int | AutoBatchSize = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file
        self._live_model = False

        if model_file.endswith(".jax"):
            from deepmd.jax.utils.serialization import (
                _set_model_min_nbor_dist_from_data,
                serialize_from_file,
            )

            data = serialize_from_file(model_file, training=False)
            model_data = data["model"]
            model_script = data["model_def_script"]
            if "model_dict" in model_data:
                head = kwargs.get("head")
                if head is None:
                    raise ValueError(
                        "A multi-task checkpoint requires a head name; available heads: "
                        + ", ".join(model_data["model_dict"])
                    )
                model_data = model_data["model_dict"][head]
                model_script = model_script["model_dict"][head]
            self.dp = BaseModel.deserialize(model_data)
            _set_model_min_nbor_dist_from_data(
                self.dp,
                {**data, "model_def_script": model_script},
            )
            self.dp.model_def_script = json.dumps(model_script)
            if model_script.get("hessian_mode", False):
                self.dp.enable_hessian()
            self.dp.eval()
            self._live_model = True
            precision = MatmulPrecisionPolicy()
            graphdef, _ = nnx.split(self.dp)

            def call_graph(
                state: nnx.State,
                inputs: dict[str, jax.Array | None],
            ) -> dict[str, jax.Array]:
                model = nnx.merge(graphdef, state)
                with precision.applied(training=False):
                    return model.call_common_lower_graph(**inputs)

            def call_lower(
                state: nnx.State,
                inputs: dict[str, jax.Array | None],
                *,
                atomic_virial: bool,
            ) -> dict[str, jax.Array | None]:
                model = nnx.merge(graphdef, state)
                with precision.applied(training=False):
                    return model.call_common_lower(
                        **inputs, do_atomic_virial=atomic_virial
                    )

            # Inference returns predictions only. Mutable NNX transforms also
            # publish state outputs, which could replace healthy parameter
            # arrays with failed asynchronous executions during OOM retries.
            self._call_graph = jax.jit(call_graph)
            self._call_lower = jax.jit(
                call_lower,
                static_argnames=("atomic_virial",),
            )
        elif model_file.endswith(".hlo"):
            model_data = load_dp_model(model_file)
            self.dp = HLO(
                stablehlo=model_data["@variables"]["stablehlo"].tobytes(),
                stablehlo_atomic_virial=model_data["@variables"][
                    "stablehlo_atomic_virial"
                ].tobytes(),
                stablehlo_no_ghost=model_data["@variables"][
                    "stablehlo_no_ghost"
                ].tobytes(),
                stablehlo_atomic_virial_no_ghost=model_data["@variables"][
                    "stablehlo_atomic_virial_no_ghost"
                ].tobytes(),
                model_def_script=json.dumps(model_data["model_def_script"]),
                **model_data["constants"],
            )
        elif model_file.endswith(".savedmodel"):
            from deepmd.jax.jax2tf.tfmodel import (
                TFModelWrapper,
            )

            self.dp = TFModelWrapper(model_file)
        else:
            raise ValueError("Unsupported file extension")
        self.rcut = self.dp.get_rcut()
        self.type_map = self.dp.get_type_map()
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

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dp.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.dp.get_dim_aparam()

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The evaluator of the model type."""
        model = self.get_model()
        model_output_type = model.model_output_type()
        if "energy" in model_output_type:
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type or "polarizability" in model_output_type:
            return DeepPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        elif self._get_property_var_name(model) in model_output_type:
            return DeepProperty
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.dp.get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return self.dp.get_numb_dos()

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model."""
        return 0

    def get_has_spin(self) -> bool:
        """Whether this model accepts native-spin vectors."""
        return self.dp.has_spin() if hasattr(self.dp, "has_spin") else False

    def get_use_spin(self) -> list[bool]:
        """Get the magnetic-type mask from the model configuration."""
        if not self.get_has_spin():
            return []
        if self._live_model:
            return self.dp.spin.use_spin.tolist()
        from deepmd.utils.spin import (
            normalize_spin_use_spin,
        )

        return normalize_spin_use_spin(
            self.get_model_def_script()["spin"]["use_spin"], self.type_map
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the loaded model as a model tree.

        JAX-native ``.jax``/``.hlo`` inputs return the lossless, weight-bearing
        ``model`` subtree from the file payload. TensorFlow-wrapped
        ``.savedmodel`` inputs cannot be converted back losslessly; for that
        format this method reconstructs the model tree from the definition
        script, so trained weights are not preserved.
        """
        if str(self.model_path).endswith(".savedmodel"):
            from deepmd.jax.model.model import (
                get_model,
            )

            return get_model(self.get_model_def_script()).serialize()

        from deepmd.jax.utils.serialization import (
            serialize_from_file,
        )

        data = serialize_from_file(self.model_path, training=False)
        if "model" not in data:
            raise RuntimeError("Serialized model data does not contain key 'model'.")
        return data["model"]

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Evaluate the energy, force and virial by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 3 x 3
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
        **kwargs
            Other parameters.
            charge_spin : array-like, optional
                The per-frame charge/spin conditioning input. The array can be
                of size nframes x dim_chg_spin, or dim_chg_spin to reuse the
                same value for all frames.

        Returns
        -------
        output_dict : dict
            The output of the evaluation. The keys are the names of the output
            variables, and the values are the corresponding output arrays.
        """
        # convert all of the input to numpy array
        charge_spin = kwargs.pop("charge_spin", None)
        spin = kwargs.pop("spin", None)
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        if self.get_has_spin() and spin is None:
            raise ValueError("This model requires eval(..., spin=spin_array).")
        if not self.get_has_spin() and spin is not None:
            raise ValueError("The model does not accept per-atom spin vectors.")
        if spin is not None:
            spin = np.asarray(spin, dtype=GLOBAL_NP_FLOAT_PRECISION).reshape(
                -1, natoms, 3
            )
            spin = np.broadcast_to(spin, (numb_test, natoms, 3))
        if self.has_chg_spin_ebd():
            if charge_spin is None:
                charge_spin = self.dp.get_default_chg_spin()
            if charge_spin is None:
                raise ValueError("The model requires charge_spin conditioning.")
            charge_spin = np.asarray(
                charge_spin, dtype=GLOBAL_NP_FLOAT_PRECISION
            ).reshape(-1, self.get_dim_chg_spin())
            charge_spin = np.broadcast_to(
                charge_spin, (numb_test, self.get_dim_chg_spin())
            )
            validate_charge_states(charge_spin)
        request_defs = self._get_request_defs(atomic)
        out = self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, fparam, aparam, charge_spin, request_defs, spin
        )
        # ``AutoBatchSize.execute_all`` unwraps a single-output result out of
        # its tuple, which would make ``zip`` iterate over the array's frame
        # axis. Re-wrap so the request-def names line up (a single request def
        # arises for global-only DOS/property inference at atomic=False).
        if not isinstance(out, tuple):
            out = (out,)
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
                strict=True,
            )
        )

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
        """Get the requested output definitions.

        When atomic is True, all output_def are requested.
        When atomic is False, only energy (tensor), force, and virial
        are requested.

        Parameters
        ----------
        atomic : bool
            Whether to request the atomic output.

        Returns
        -------
        list[OutputVariableDef]
            The requested output definitions.
        """
        if atomic:
            output_defs = list(self.output_def.var_defs.values())
        else:
            output_defs = [
                x
                for x in self.output_def.var_defs.values()
                if x.category
                in (
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
                    OutputVariableCategory.DERV_R_DERV_R,
                    OutputVariableCategory.DERV_R_MAG,
                )
                or x.name == "mask_mag"
            ]
        # Avoid allocating the quadratic Hessian placeholder when the frozen
        # model does not provide that output.
        if not self.get_has_hessian():
            output_defs = [
                x
                for x in output_defs
                if x.category != OutputVariableCategory.DERV_R_DERV_R
            ]
        return output_defs

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

            def eval_func(*args: Any, **kwargs: Any) -> Any:
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
        mixed_type: bool = False,
    ) -> tuple[int, int]:
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

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        charge_spin: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        spin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        model = self.dp

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = coords.reshape([-1, natoms, 3])
        type_input = atom_types
        if cells is not None:
            box_input = cells.reshape([-1, 3, 3])
        else:
            box_input = None
        if fparam is not None:
            fparam_input = fparam.reshape(nframes, self.get_dim_fparam())
        elif self.dp.has_default_fparam():
            # JAX (XLA) requires static shapes, so default must be implemented here
            default_fparam = self.dp.get_default_fparam()
            assert default_fparam is not None
            fparam_input = np.tile(
                np.array(default_fparam, dtype=GLOBAL_NP_FLOAT_PRECISION),
                (nframes, 1),
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = aparam.reshape(nframes, natoms, self.get_dim_aparam())
        else:
            aparam_input = None
        if charge_spin is not None and not self.has_chg_spin_ebd():
            warnings.warn(
                "charge_spin was provided, but this model does not support "
                "charge/spin conditioning. The provided charge_spin will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        charge_spin_input = (
            np.asarray(charge_spin, dtype=GLOBAL_NP_FLOAT_PRECISION)
            if self.has_chg_spin_ebd() and charge_spin is not None
            else None
        )

        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )
        model_kwargs = {
            "box": to_jax_array(box_input),
            "fparam": to_jax_array(fparam_input),
            "aparam": to_jax_array(aparam_input),
            "do_atomic_virial": do_atomic_virial,
        }
        if self.has_chg_spin_ebd():
            model_kwargs["charge_spin"] = to_jax_array(charge_spin_input)
        if self.get_has_spin():
            model_kwargs["spin"] = to_jax_array(spin)
        if self._live_model:
            batch_output = self._eval_live_model(
                coord_input,
                type_input,
                box_input,
                fparam_input,
                aparam_input,
                spin,
                charge_spin_input,
                do_atomic_virial,
            )
        else:
            batch_output = model(
                to_jax_array(coord_input),
                to_jax_array(type_input),
                **model_kwargs,
            )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        for kk, vv in batch_output.items():
            batch_output[kk] = to_numpy_array(vv)

        results = []
        for odef in request_defs:
            # HLO and TFModelWrapper return raw internal keys (not translated),
            # so no key mapping is needed here.
            dp_name = odef.name
            if dp_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                if batch_output[dp_name] is not None:
                    out = batch_output[dp_name].reshape(shape)
                else:
                    out = np.full(shape, np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )  # this is kinda hacky
        return tuple(results)

    def _eval_live_model(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        box: np.ndarray | None,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        spin: np.ndarray | None,
        charge_spin: np.ndarray | None,
        atomic_virial: bool,
    ) -> dict[str, np.ndarray | jax.Array | None]:
        """Evaluate a checkpoint with a host-built sparse or dense neighbor set."""
        nf, nloc = atype.shape
        state = nnx.state(self.dp)
        if self.dp.uses_full_neighbor_list():
            batch = {
                "coord": coord,
                "type": atype,
                "box": box,
                "find_box": box is not None,
                "fparam": fparam,
                "aparam": aparam,
                "spin": spin,
                "charge_spin": charge_spin,
            }
            _, inputs = prepare_graph_batch(
                batch,
                rcut=self.rcut,
                ntypes=len(self.type_map),
                pair_exclude_types=self.dp.atomic_model.pair_exclude_types,
            )
            outputs = self._call_graph(
                state,
                jax.tree.map(jnp.asarray, inputs),
            )
            physical = atype.reshape(-1) >= 0
            nphysical = np.count_nonzero(physical)
            result = {}
            for key, value in outputs.items():
                array = np.asarray(value)
                if key.endswith("_redu"):
                    result[key] = array[:nf]
                elif nphysical == nf * nloc:
                    result[key] = array[:nphysical].reshape(nf, nloc, *array.shape[1:])
                else:
                    rectangular = np.zeros(
                        (nf * nloc, *array.shape[1:]), dtype=array.dtype
                    )
                    rectangular[physical] = array[:nphysical]
                    result[key] = rectangular.reshape(nf, nloc, *array.shape[1:])
            return result
        coord_ext, atype_ext, nlist, mapping = DefaultNeighborList().build(
            coord,
            atype,
            box,
            self.rcut,
            self.dp.get_sel(),
            pair_excl=self.dp.atomic_model.pair_excl,
        )
        inputs = {
            "extended_coord": coord_ext,
            "extended_atype": atype_ext,
            "nlist": nlist,
            "mapping": mapping,
            "fparam": fparam,
            "aparam": aparam,
            "charge_spin": charge_spin,
        }
        outputs = self._call_lower(
            state,
            jax.tree.map(to_jax_array, inputs),
            atomic_virial=atomic_virial,
        )
        return communicate_extended_output(
            outputs,
            self.dp.model_output_def(),
            jnp.asarray(mapping),
            do_atomic_virial=atomic_virial,
        )

    def _get_output_shape(
        self, odef: OutputVariableDef, nframes: int, natoms: int
    ) -> list[int]:
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            # virial
            return [nframes, *odef.shape[:-1], 9]
        elif odef.category == OutputVariableCategory.REDU:
            # energy
            return [nframes, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_C:
            # atom_virial
            return [nframes, *odef.shape[:-1], natoms, 9]
        elif odef.category in (
            OutputVariableCategory.DERV_R,
            OutputVariableCategory.DERV_R_MAG,
        ):
            # force
            return [nframes, *odef.shape[:-1], natoms, 3]
        elif odef.category == OutputVariableCategory.OUT:
            # atom_energy, atom_tensor
            return [nframes, natoms, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:
            # hessian
            return [nframes, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        return json.loads(self.dp.get_model_def_script())

    def get_has_hessian(self) -> bool:
        """Check if the model has Hessian output."""
        return self.get_model_def_script().get("hessian_mode", False)

    def get_model(self) -> Any:
        """Get the JAX model as BaseModel.

        Returns
        -------
        BaseModel
            The JAX model as BaseModel instance.
        """
        return self.dp

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return self.dp.has_default_fparam()

    def has_chg_spin_ebd(self) -> bool:
        """Check if the model has charge spin embedding."""
        return self.dp.has_chg_spin_ebd()

    def get_dim_chg_spin(self) -> int:
        """Get the dimension of charge_spin input."""
        return self.dp.get_dim_chg_spin()

    def has_default_chg_spin(self) -> bool:
        """Check if the model has default charge_spin values.

        ``has_default_chg_spin`` was merged into ``get_default_chg_spin`` on
        the live-model interfaces; this wrapper method is kept for API
        stability and computes the predicate directly.
        """
        return self.dp.get_default_chg_spin() is not None
