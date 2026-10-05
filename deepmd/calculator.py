# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE calculator interface module."""

import logging
from functools import (
    partial,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Union,
)

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from deepmd.infer import (
    DeepPot,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from ase import (
        Atoms,
    )
    from ase.neighborlist import (
        NeighborList,
    )

    from deepmd.infer.deep_eval import (
        DeepEvalBackend,
    )

__all__ = ["DP"]

log = logging.getLogger(__name__)


def _compress_dpa4c_for_ase(model: Any) -> bool:
    """Select supported DPA4C compression without changing other model families."""
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compression_ineligibility_reason,
    )

    parts = model.atomic_model.fused_decomposition()
    if (
        parts is None
        or parts[0] is None
        or not isinstance(parts[0].descriptor, DescrptDPA4C)
    ):
        return False
    reason = compression_ineligibility_reason(parts[0].descriptor)
    if reason is None:
        return True
    log.warning(
        "DPA4C compression is unavailable: %s Freezing without compression.", reason
    )
    return False


def _checkpoint_freezer(
    evaluator: "DeepEvalBackend", head: str | None, *, freeze_requested: bool
) -> "Callable[..., None] | None":
    """Select an exporter for the loaded checkpoint's model and head."""
    from deepmd.pt.infer.deep_eval import DeepEval as PTDeepEval
    from deepmd.pt.model.model.sezm_model import (
        SeZMModel,
    )
    from deepmd.pt_expt.infer.deep_eval import DeepEval as PTExptDeepEval

    if isinstance(evaluator, PTDeepEval) and isinstance(
        evaluator.get_model(), SeZMModel
    ):
        from deepmd.pt.entrypoints.freeze_pt2 import (
            freeze_sezm_to_pt2,
        )

        return partial(
            freeze_sezm_to_pt2,
            head=head if evaluator.multi_task else None,
            with_comm=False,
        )
    if (
        isinstance(evaluator, PTExptDeepEval)
        and evaluator.metadata["lower_input_kind"] == "graph"
    ):
        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        # The checkpoint loader selects Default when no head is supplied.
        return partial(
            freeze,
            head="Default" if head is None else head,
            lower_kind="graph",
            with_comm=False,
            compress=(
                _compress_dpa4c_for_ase(evaluator._dpmodel)
                if freeze_requested
                else False
            ),
        )
    return None


class DP(Calculator):
    """Implementation of ASE deepmd calculator.

    Implemented properties are `energy`, `forces` and `stress`

    Parameters
    ----------
    model : Union[str, Path]
        path to the model
    label : str, optional
        calculator label, by default "DP"
    type_dict : dict[str, int], optional
        mapping of element types and their numbers, best left None and the calculator
        will infer this information from model, by default None
    neighbor_list : ase.neighborlist.NeighborList, optional
        The neighbor list object. If None, then build the native neighbor list.
    head : Union[str, None], optional
        a specific model branch choosing from pretrained model, by default None
    freeze : bool, optional
        Export an eligible ``.pt`` checkpoint to ``.pt2`` for inference,
        by default False.
        Supports PT DPA4/SeZM and graph-eligible PT-expt models. Each call
        exports ``<checkpoint-stem>-ase_single.pt2`` beside the checkpoint,
        using the backend's freeze-time kernel policy and omitting the
        multi-rank communication artifact. DPA4C models are also compressed
        when their configuration supports it; otherwise a warning explains
        why the export remains uncompressed. Other models ignore this option
        and are loaded directly. Direct checkpoint inference for supported
        models can be substantially slower; False is intended for debugging
        without the compilation overhead.

    Examples
    --------
    Compute potential energy

    >>> from ase import Atoms
    >>> from deepmd.tf.calculator import DP
    >>> water = Atoms('H2O',
    >>>             positions=[(0.7601, 1.9270, 1),
    >>>                        (1.9575, 1, 1),
    >>>                        (1., 1., 1.)],
    >>>             cell=[100, 100, 100],
    >>>             calculator=DP(model="frozen_model.pb"))
    >>> print(water.get_potential_energy())
    >>> print(water.get_forces())

    Run BFGS structure optimization

    >>> from ase.optimize import BFGS
    >>> dyn = BFGS(water)
    >>> dyn.run(fmax=1e-6)
    >>> print(water.get_positions())
    """

    name = "DP"
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "virial",
        "stress",
    ]

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP",
        type_dict: dict[str, int] | None = None,
        neighbor_list: Optional["NeighborList"] = None,
        head: str | None = None,
        nlist_backend: str = "auto",
        freeze: bool = False,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)
        model_path = Path(model).resolve()
        self.dp = DeepPot(
            str(model_path),
            neighbor_list=neighbor_list,
            head=head,
            nlist_backend=nlist_backend,
        )
        freezer = (
            _checkpoint_freezer(self.dp.deep_eval, head, freeze_requested=freeze)
            if model_path.suffix == ".pt"
            else None
        )
        if freezer is not None and freeze:
            output = model_path.with_name(f"{model_path.stem}-ase_single.pt2")
            # Release the eager model before the exporter constructs its model.
            self.dp.close()
            del self.dp
            log.info("Freezing ASE checkpoint %s to %s", model_path, output)
            freezer(str(model_path), str(output))
            self.dp = DeepPot(
                str(output), neighbor_list=neighbor_list, nlist_backend=nlist_backend
            )
        elif freezer is not None:
            log.warning(
                "ASE inference directly from a .pt checkpoint can be SUBSTANTIALLY "
                "slower than a frozen .pt2 model. Use DP(..., freeze=True), or "
                "freeze the checkpoint manually and load the .pt2 file. "
                "freeze=False is intended for debugging and similar short runs "
                "that avoid the upfront compilation overhead."
            )
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()), strict=True)
            )

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: list[str] = ["energy", "forces", "virial"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Run calculation with deepmd model.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            atoms object to run the calculation on, by default None
        properties : list[str], optional
            unused, only for function signature compatibility,
            by default ["energy", "forces", "stress"]
        system_changes : list[str], optional
            unused, only for function signature compatibility, by default all_changes
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]

        fparam = self.atoms.info.get("fparam", None)
        aparam = self.atoms.info.get("aparam", None)
        charge_spin = self.atoms.info.get("charge_spin", None)
        e, f, v = self.dp.eval(
            coords=coord,
            cells=cell,
            atom_types=atype,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )[:3]
        self.results["energy"] = e[0][0]
        # see https://gitlab.com/ase/ase/-/merge_requests/2485
        self.results["free_energy"] = e[0][0]
        self.results["forces"] = f[0]
        virial = v[0].reshape(3, 3)
        self.results["virial"] = virial

        # convert virial into stress for lattice relaxation
        if cell is not None:
            # the usual convention (tensile stress is positive)
            # stress = -virial / volume
            # ASE represents Cauchy stress as a symmetric tensor.  Reshape the
            # flat model output before transposing; ``.T`` on the original
            # one-dimensional virial array would otherwise be a no-op.
            stress = -0.5 * (virial + virial.T) / atoms.get_volume()
            # Voigt notation
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif "stress" in properties:
            raise PropertyNotImplementedError
        else:
            pass
