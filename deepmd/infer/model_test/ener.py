# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of energy models, including those carrying spin."""

from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    ClassVar,
    NamedTuple,
)

import numpy as np

from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    save_txt_file,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.eval_metrics import (
    DP_TEST_HESSIAN_METRIC_KEYS,
    DP_TEST_SPIN_WEIGHTED_METRIC_KEYS,
    DP_TEST_WEIGHTED_FORCE_METRIC_KEYS,
    DP_TEST_WEIGHTED_METRIC_KEYS,
    compute_energy_type_metrics,
    compute_error_stat,
    compute_spin_force_metrics,
    compute_weighted_error_stat,
)

if TYPE_CHECKING:
    from deepmd.infer.deep_pot import (
        DeepPot,
    )

__all__ = ["EnerTester", "SpinEnerTester"]


def _reshape_force_by_atom(force_array: np.ndarray, natoms: int) -> np.ndarray:
    """Reshape flattened force arrays into `[nframes, natoms, 3]`."""
    return np.reshape(force_array, [-1, natoms, 3])


def _concat_force_rows(
    force_blocks: list[np.ndarray], dtype: np.dtype | type[np.generic]
) -> np.ndarray:
    """Concatenate per-frame force rows into one 2D array."""
    if not force_blocks:
        return np.empty((0, 3), dtype=dtype)
    return np.concatenate(force_blocks, axis=0)


def _align_spin_force_arrays(
    *,
    dp: "DeepPot",
    atype: np.ndarray,
    natoms: int,
    prediction_force: np.ndarray,
    reference_force: np.ndarray,
    prediction_force_mag: np.ndarray | None,
    reference_force_mag: np.ndarray | None,
    mask_mag: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Align spin force arrays into real-atom and magnetic subsets."""
    prediction_force_by_atom = _reshape_force_by_atom(prediction_force, natoms)
    reference_force_by_atom = _reshape_force_by_atom(reference_force, natoms)
    if dp.get_ntypes_spin() != 0:  # old tf support for spin
        ntypes_real = dp.get_ntypes() - dp.get_ntypes_spin()
        atype_by_frame = np.reshape(atype, [-1, natoms])
        if atype_by_frame.shape[0] == 1 and prediction_force_by_atom.shape[0] != 1:
            atype_by_frame = np.broadcast_to(
                atype_by_frame,
                (prediction_force_by_atom.shape[0], natoms),
            )
        if atype_by_frame.shape[0] != prediction_force_by_atom.shape[0]:
            raise ValueError(
                "Spin atom types and force arrays must have matching frames."
            )
        force_real_prediction_chunks = []
        force_real_reference_chunks = []
        force_magnetic_prediction_chunks = []
        force_magnetic_reference_chunks = []
        for frame_atype, frame_prediction, frame_reference in zip(
            atype_by_frame,
            prediction_force_by_atom,
            reference_force_by_atom,
            strict=False,
        ):
            real_mask = frame_atype < ntypes_real
            magnetic_mask = ~real_mask
            force_real_prediction_chunks.append(frame_prediction[real_mask])
            force_real_reference_chunks.append(frame_reference[real_mask])
            force_magnetic_prediction_chunks.append(frame_prediction[magnetic_mask])
            force_magnetic_reference_chunks.append(frame_reference[magnetic_mask])
        return (
            _concat_force_rows(
                force_real_prediction_chunks,
                prediction_force_by_atom.dtype,
            ),
            _concat_force_rows(
                force_real_reference_chunks,
                reference_force_by_atom.dtype,
            ),
            _concat_force_rows(
                force_magnetic_prediction_chunks,
                prediction_force_by_atom.dtype,
            ),
            _concat_force_rows(
                force_magnetic_reference_chunks,
                reference_force_by_atom.dtype,
            ),
        )

    force_real_prediction = prediction_force_by_atom.reshape(-1, 3)
    force_real_reference = reference_force_by_atom.reshape(-1, 3)
    if prediction_force_mag is None or reference_force_mag is None or mask_mag is None:
        return force_real_prediction, force_real_reference, None, None
    magnetic_mask = mask_mag.reshape(-1).astype(bool)
    return (
        force_real_prediction,
        force_real_reference,
        prediction_force_mag.reshape(-1, 3)[magnetic_mask],
        reference_force_mag.reshape(-1, 3)[magnetic_mask],
    )


def _write_energy_test_details(
    *,
    detail_path: Path,
    system: str,
    natoms: int,
    append_detail: bool,
    reference_energy: np.ndarray,
    prediction_energy: np.ndarray,
    reference_force: np.ndarray,
    prediction_force: np.ndarray,
    reference_virial: np.ndarray | None,
    prediction_virial: np.ndarray | None,
    out_put_spin: bool,
    reference_stress: np.ndarray | None = None,
    prediction_stress: np.ndarray | None = None,
    reference_force_real: np.ndarray | None = None,
    prediction_force_real: np.ndarray | None = None,
    reference_force_magnetic: np.ndarray | None = None,
    prediction_force_magnetic: np.ndarray | None = None,
    reference_hessian: np.ndarray | None = None,
    prediction_hessian: np.ndarray | None = None,
) -> None:
    """Write energy-type detail outputs after arrays have been aligned."""
    pe = np.concatenate(
        (
            np.reshape(reference_energy, [-1, 1]),
            np.reshape(prediction_energy, [-1, 1]),
        ),
        axis=1,
    )
    save_txt_file(
        detail_path.with_suffix(".e.out"),
        pe,
        header=f"{system}: data_e pred_e",
        append=append_detail,
    )
    pe_atom = pe / natoms
    save_txt_file(
        detail_path.with_suffix(".e_peratom.out"),
        pe_atom,
        header=f"{system}: data_e pred_e",
        append=append_detail,
    )
    if not out_put_spin:
        pf = np.concatenate(
            (
                np.reshape(reference_force, [-1, 3]),
                np.reshape(prediction_force, [-1, 3]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".f.out"),
            pf,
            header=f"{system}: data_fx data_fy data_fz pred_fx pred_fy pred_fz",
            append=append_detail,
        )
    else:
        if reference_force_real is None or prediction_force_real is None:
            raise ValueError("Spin detail output requires aligned real-atom forces.")
        pf_real = np.concatenate(
            (
                np.reshape(reference_force_real, [-1, 3]),
                np.reshape(prediction_force_real, [-1, 3]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".fr.out"),
            pf_real,
            header=f"{system}: data_fx data_fy data_fz pred_fx pred_fy pred_fz",
            append=append_detail,
        )
        if (reference_force_magnetic is None) != (prediction_force_magnetic is None):
            raise ValueError(
                "Spin magnetic detail output requires both reference and prediction forces."
            )
        if (
            reference_force_magnetic is not None
            and prediction_force_magnetic is not None
        ):
            pf_mag = np.concatenate(
                (
                    np.reshape(reference_force_magnetic, [-1, 3]),
                    np.reshape(prediction_force_magnetic, [-1, 3]),
                ),
                axis=1,
            )
            save_txt_file(
                detail_path.with_suffix(".fm.out"),
                pf_mag,
                header=f"{system}: data_fmx data_fmy data_fmz pred_fmx pred_fmy pred_fmz",
                append=append_detail,
            )
    if (reference_virial is None) != (prediction_virial is None):
        raise ValueError(
            "Virial detail output requires both reference and prediction virials."
        )
    if reference_virial is not None and prediction_virial is not None:
        pv = np.concatenate(
            (
                np.reshape(reference_virial, [-1, 9]),
                np.reshape(prediction_virial, [-1, 9]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".v.out"),
            pv,
            header=f"{system}: data_vxx data_vxy data_vxz data_vyx data_vyy "
            "data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx "
            "pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz",
            append=append_detail,
        )
        pv_atom = pv / natoms
        save_txt_file(
            detail_path.with_suffix(".v_peratom.out"),
            pv_atom,
            header=f"{system}: data_vxx data_vxy data_vxz data_vyx data_vyy "
            "data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx "
            "pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz",
            append=append_detail,
        )
    if (reference_stress is None) != (prediction_stress is None):
        raise ValueError(
            "Stress detail output requires both reference and prediction stresses."
        )
    if reference_stress is not None and prediction_stress is not None:
        ps = np.concatenate(
            (
                np.reshape(reference_stress, [-1, 9]),
                np.reshape(prediction_stress, [-1, 9]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".s.out"),
            ps,
            header=f"{system} (eV/Å^3): data_sxx data_sxy data_sxz data_syx "
            "data_syy data_syz data_szx data_szy data_szz pred_sxx pred_sxy pred_sxz "
            "pred_syx pred_syy pred_syz pred_szx pred_szy pred_szz",
            append=append_detail,
        )
    if reference_hessian is not None and prediction_hessian is not None:
        hessian_detail = np.concatenate(
            (
                reference_hessian.reshape(-1, 1),
                prediction_hessian.reshape(-1, 1),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".h.out"),
            hessian_detail,
            header=f"{system}: data_h pred_h (3Na*3Na matrix in row-major order)",
            append=append_detail,
        )


class _OptionalEnerOutputs(NamedTuple):
    """The optional trailing outputs of ``DeepPot.eval`` (``None`` when absent)."""

    atom_energy: "np.ndarray | None"
    atom_virial: "np.ndarray | None"
    force_mag: "np.ndarray | None"
    mask_mag: "np.ndarray | None"
    hessian: "np.ndarray | None"


def _split_optional_ener_outputs(
    ret: tuple,
    *,
    has_atom_ener: bool,
    has_spin: bool,
    has_hessian: bool,
    numb_test: int,
) -> _OptionalEnerOutputs:
    """Split the optional trailing outputs of ``DeepPot.eval``.

    ``DeepPot.eval`` appends its optional outputs after ``(energy, force,
    virial)`` in a fixed order: atomic ``(atom_energy, atom_virial)``, then spin
    ``(force_mag, mask_mag)``, then ``hessian``. Read them by advancing an index
    through the tuple in that same order, so the hessian slot is not confused
    with atomic energy/virial or spin outputs when those are also present.
    """
    atom_energy = atom_virial = force_mag = mask_mag = hessian = None
    idx = 3
    if has_atom_ener:
        atom_energy = ret[idx].reshape([numb_test, -1])
        atom_virial = ret[idx + 1].reshape([numb_test, -1])
        idx += 2
    if has_spin:
        force_mag = ret[idx].reshape([numb_test, -1])
        mask_mag = ret[idx + 1].reshape([numb_test, -1])
        idx += 2
    if has_hessian:
        hessian = ret[idx].reshape([numb_test, -1])
    return _OptionalEnerOutputs(atom_energy, atom_virial, force_mag, mask_mag, hessian)


@dataclass(frozen=True)
class _ForceDetails:
    """Force arrays a spin detail file records, alongside the shared ones."""

    reference_real: np.ndarray | None = None
    prediction_real: np.ndarray | None = None
    reference_magnetic: np.ndarray | None = None
    prediction_magnetic: np.ndarray | None = None


class EnerTester(ModelTester):
    """Test an energy model against energies, forces and the virial."""

    report = (
        ("mae_e", "Energy MAE         : {} eV"),
        ("rmse_e", "Energy RMSE        : {} eV"),
        ("mae_ea", "Energy MAE/Natoms  : {} eV"),
        ("rmse_ea", "Energy RMSE/Natoms : {} eV"),
        ("mae_f", "Force  MAE         : {} eV/Å"),
        ("rmse_f", "Force  RMSE        : {} eV/Å"),
        ("mae_fw", "Force weighted MAE : {} eV/Å"),
        ("rmse_fw", "Force weighted RMSE: {} eV/Å"),
        ("mae_fr", "Force atom MAE      : {} eV/Å"),
        ("rmse_fr", "Force atom RMSE     : {} eV/Å"),
        ("mae_fm", "Force spin MAE      : {} eV/uB"),
        ("rmse_fm", "Force spin RMSE     : {} eV/uB"),
        ("mae_v", "Virial MAE         : {} eV"),
        ("rmse_v", "Virial RMSE        : {} eV"),
        ("mae_va", "Virial MAE/Natoms  : {} eV"),
        ("rmse_va", "Virial RMSE/Natoms : {} eV"),
        ("mae_s", "Stress MAE         : {} eV/Å^3"),
        ("rmse_s", "Stress RMSE        : {} eV/Å^3"),
        ("mae_ae", "Atomic ener MAE    : {} eV"),
        ("rmse_ae", "Atomic ener RMSE   : {} eV"),
        ("mae_h", "Hessian MAE        : {} eV/Å^2"),
        ("rmse_h", "Hessian RMSE       : {} eV/Å^2"),
    )
    per_system_only = ("mae_ae", "rmse_ae")

    #: Whether the force this model class reports is the plain atomic force.
    #: A spin model reports a real and a magnetic force instead.
    reports_plain_force: ClassVar[bool] = True

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels an energy test consumes."""
        dp = self.dp
        data.add("energy", 1, atomic=False, must=False, high_prec=True)
        data.add("force", 3, atomic=True, must=False, high_prec=False)
        data.add("atom_pref", 1, atomic=True, must=False, high_prec=False, repeat=3)
        data.add("virial", 9, atomic=False, must=False, high_prec=False)
        if dp.has_efield:
            data.add("efield", 3, atomic=True, must=True, high_prec=False)
        if self.atomic:
            data.add("atom_ener", 1, atomic=True, must=True, high_prec=False)
        if dp.get_dim_fparam() > 0:
            data.add(
                "fparam",
                dp.get_dim_fparam(),
                atomic=False,
                must=not dp.has_default_fparam(),
                high_prec=False,
            )
        if dp.get_dim_aparam() > 0:
            data.add(
                "aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False
            )
        if dp.has_chg_spin_ebd():
            data.add(
                "charge_spin",
                2,
                atomic=False,
                must=not dp.has_default_chg_spin(),
                high_prec=False,
            )
        if dp.has_spin:
            data.add("spin", 3, atomic=True, must=True, high_prec=False)
            data.add("force_mag", 3, atomic=True, must=False, high_prec=False)
        if dp.has_hessian:
            data.add("hessian", 1, atomic=True, must=True, high_prec=False)

    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one chunk of an energy test."""
        dp = self.dp
        errors: dict[str, tuple[float, float]] = {}
        find_energy = test_data.get("find_energy")
        find_force = test_data.get("find_force")
        find_virial = test_data.get("find_virial")
        find_atom_pref = test_data.get("find_atom_pref")
        mixed_type = data.mixed_type
        natoms = len(test_data["type"][0])
        nframes = test_data["box"].shape[0]

        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"] if data.pbc else None
        efield = test_data["efield"].reshape([nframes, -1]) if dp.has_efield else None
        spin = test_data["spin"].reshape([nframes, -1]) if dp.has_spin else None
        if mixed_type:
            atype = test_data["type"].reshape([nframes, -1])
        else:
            atype = test_data["type"][0]
        fparam = (
            test_data["fparam"]
            if dp.get_dim_fparam() > 0 and test_data["find_fparam"] != 0.0
            else None
        )
        aparam = test_data["aparam"] if dp.get_dim_aparam() > 0 else None
        charge_spin = (
            test_data["charge_spin"]
            if dp.has_chg_spin_ebd() and test_data.get("find_charge_spin", 0.0) != 0.0
            else None
        )

        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
            atomic=self.atomic,
            efield=efield,
            mixed_type=mixed_type,
            spin=spin,
            charge_spin=charge_spin,
        )
        energy = ret[0].reshape([nframes, 1])
        force = ret[1].reshape([nframes, -1])
        virial = ret[2].reshape([nframes, 9])
        optional_outputs = _split_optional_ener_outputs(
            ret,
            has_atom_ener=self.atomic,
            has_spin=dp.has_spin,
            has_hessian=dp.has_hessian,
            numb_test=nframes,
        )

        force_details = self.force_errors(
            errors,
            data=data,
            test_data=test_data,
            atype=atype,
            natoms=natoms,
            prediction_force=force,
            optional_outputs=optional_outputs,
            find_force=find_force,
            find_atom_pref=find_atom_pref,
        )

        reports_virial = find_virial == 1 and data.pbc
        shared_metrics = compute_energy_type_metrics(
            prediction={
                "energy": energy,
                "force": force,
                **({"virial": virial} if reports_virial else {}),
            },
            test_data={
                "find_energy": find_energy,
                "find_force": find_force if self.reports_plain_force else 0.0,
                "find_virial": find_virial,
                "energy": test_data["energy"],
                "force": test_data["force"],
                **({"virial": test_data["virial"]} if reports_virial else {}),
            },
            natoms=natoms,
            has_pbc=data.pbc,
        )
        errors.update(
            shared_metrics.as_weighted_average_errors(DP_TEST_WEIGHTED_METRIC_KEYS)
        )
        if find_energy == 1 and (
            shared_metrics.energy is None or shared_metrics.energy_per_atom is None
        ):
            raise RuntimeError("Energy metrics are unavailable for dp test.")

        prediction_stress = None
        reference_stress = None
        if reports_virial:
            if shared_metrics.virial is None or shared_metrics.virial_per_atom is None:
                raise RuntimeError("Virial metrics are unavailable for dp test.")
            # Stress sigma = -virial / volume, in eV/Å^3 (tensile-positive
            # convention).
            volume = np.abs(np.linalg.det(box.reshape([nframes, 3, 3]))).reshape(
                [nframes, 1]
            )
            prediction_stress = -virial / volume
            reference_stress = -test_data["virial"] / volume
            errors.update(
                compute_error_stat(
                    prediction_stress, reference_stress
                ).as_weighted_average_errors("mae_s", "rmse_s")
            )

        if dp.has_hessian:
            errors.update(
                compute_error_stat(
                    optional_outputs.hessian, test_data["hessian"]
                ).as_weighted_average_errors(*DP_TEST_HESSIAN_METRIC_KEYS)
            )
        if self.atomic:
            errors.update(
                compute_error_stat(
                    optional_outputs.atom_energy.reshape([-1]),
                    test_data["atom_ener"].reshape([-1]),
                ).as_weighted_average_errors(*self.per_system_only)
            )

        if context.detail_path is not None:
            _write_energy_test_details(
                detail_path=context.detail_path,
                system=context.system,
                natoms=natoms,
                append_detail=context.append_detail,
                reference_energy=test_data["energy"],
                prediction_energy=energy,
                reference_force=test_data["force"],
                prediction_force=force,
                reference_virial=test_data["virial"],
                prediction_virial=virial,
                reference_stress=reference_stress,
                prediction_stress=prediction_stress,
                out_put_spin=not self.reports_plain_force,
                reference_force_real=force_details.reference_real,
                prediction_force_real=force_details.prediction_real,
                reference_force_magnetic=force_details.reference_magnetic,
                prediction_force_magnetic=force_details.prediction_magnetic,
                reference_hessian=test_data["hessian"] if dp.has_hessian else None,
                prediction_hessian=optional_outputs.hessian if dp.has_hessian else None,
            )

        return errors

    def force_errors(
        self,
        errors: dict[str, tuple[float, float]],
        *,
        data: DeepmdData,
        test_data: dict,
        atype: np.ndarray,
        natoms: int,
        prediction_force: np.ndarray,
        optional_outputs: "_OptionalEnerOutputs",
        find_force: float | None,
        find_atom_pref: float | None,
    ) -> "_ForceDetails":
        """Add the force errors of a chunk and return what the details need.

        The plain atomic force is covered by the shared energy metrics, so only
        the optional per-atom weighting is added here.

        Parameters
        ----------
        errors : dict[str, tuple[float, float]]
            Errors of the chunk, extended in place.
        data : DeepmdData
            The system the chunk was drawn from.
        test_data : dict
            The chunk.
        atype : np.ndarray
            Atom types of the chunk.
        natoms : int
            Number of atoms per frame.
        prediction_force : np.ndarray
            Predicted force with shape ``(nframes, natoms * 3)``.
        optional_outputs : _OptionalEnerOutputs
            The optional trailing outputs of the evaluation.
        find_force : float or None
            Whether the chunk carries a force label.
        find_atom_pref : float or None
            Whether the chunk carries per-atom force weights.

        Returns
        -------
        _ForceDetails
            The force arrays the detail file records, empty for a model whose
            forces the shared detail writer already covers.
        """
        if find_force == 1 and find_atom_pref == 1:
            errors.update(
                compute_weighted_error_stat(
                    prediction_force,
                    test_data["force"],
                    test_data["atom_pref"],
                ).as_weighted_average_errors(*DP_TEST_WEIGHTED_FORCE_METRIC_KEYS)
            )
        return _ForceDetails()


class SpinEnerTester(EnerTester):
    """Test a spin energy model, whose force splits into real and magnetic.

    Everything else an energy model reports carries over unchanged: the
    magnetic degrees of freedom enter the virial only through the virtual
    atoms, whose displacement the model removes again, so the virial is with
    respect to the real atomic positions as for any other energy model.
    """

    reports_plain_force = False

    def force_errors(
        self,
        errors: dict[str, tuple[float, float]],
        *,
        data: DeepmdData,
        test_data: dict,
        atype: np.ndarray,
        natoms: int,
        prediction_force: np.ndarray,
        optional_outputs: "_OptionalEnerOutputs",
        find_force: float | None,
        find_atom_pref: float | None,
    ) -> "_ForceDetails":
        """Add the real and magnetic force errors of a chunk."""
        find_force_mag = test_data.get("find_force_mag")
        force_real, reference_real, force_mag, reference_mag = _align_spin_force_arrays(
            dp=self.dp,
            atype=atype,
            natoms=natoms,
            prediction_force=prediction_force,
            reference_force=test_data["force"],
            prediction_force_mag=optional_outputs.force_mag,
            reference_force_mag=test_data.get("force_mag"),
            mask_mag=optional_outputs.mask_mag,
        )
        if find_force_mag == 1 and (force_mag is None or reference_mag is None):
            raise RuntimeError(
                "Spin magnetic force metrics require magnetic force arrays and mask."
            )
        spin_metrics = compute_spin_force_metrics(
            force_real_prediction=force_real,
            force_real_reference=reference_real,
            force_magnetic_prediction=force_mag if find_force_mag == 1 else None,
            force_magnetic_reference=reference_mag if find_force_mag == 1 else None,
        )
        if spin_metrics.force_real is None:
            raise RuntimeError("Spin force metrics are unavailable for dp test.")
        if find_force == 1:
            errors.update(
                spin_metrics.as_weighted_average_errors(
                    {"force_real": DP_TEST_SPIN_WEIGHTED_METRIC_KEYS["force_real"]}
                )
            )
        if find_force_mag == 1:
            if spin_metrics.force_magnetic is None:
                raise RuntimeError("Spin magnetic force metrics are unavailable.")
            errors.update(
                spin_metrics.as_weighted_average_errors(
                    {
                        "force_magnetic": DP_TEST_SPIN_WEIGHTED_METRIC_KEYS[
                            "force_magnetic"
                        ]
                    }
                )
            )
        return _ForceDetails(
            reference_real=reference_real,
            prediction_real=force_real,
            reference_magnetic=reference_mag if find_force_mag == 1 else None,
            prediction_magnetic=force_mag if find_force_mag == 1 else None,
        )
