# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test trained DeePMD model."""

import logging
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.common import (
    j_loader,
)
from deepmd.dpmodel.utils.lmdb_data import (
    LmdbTestData,
    LmdbTestDataNlocView,
    is_lmdb,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
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
from deepmd.utils import random as dp_random
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.data_system import (
    process_systems,
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
    mae,
    rmse,
)
from deepmd.utils.weight_avg import (
    weighted_average,
)

if TYPE_CHECKING:
    from deepmd.infer.deep_tensor import (
        DeepTensor,
    )

__all__ = ["test"]

log = logging.getLogger(__name__)


def test(
    *,
    model: str,
    system: str | None,
    datafile: str | None,
    train_json: str | None = None,
    valid_json: str | None = None,
    numb_test: int,
    rand_seed: int | None,
    shuffle_test: bool,
    detail_file: str,
    atomic: bool,
    head: str | None = None,
    **kwargs: Any,
) -> None:
    """Test model predictions.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str, optional
        system directory
    datafile : str, optional
        the path to the list of systems to test
    train_json : Optional[str]
        Path to the input.json file provided via ``--train-data``. Training systems will be used for testing.
    valid_json : Optional[str]
        Path to the input.json file provided via ``--valid-data``. Validation systems will be used for testing.
    numb_test : int
        number of tests to do. 0 means all data.
    rand_seed : Optional[int]
        seed for random generator
    shuffle_test : bool
        whether to shuffle tests
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether per atom quantities should be computed
    head : Optional[str], optional
        (Supported backend: PyTorch) Task head to test if in multi-task mode.
    **kwargs
        additional arguments

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    if numb_test == 0:
        # only float has inf, but should work for min
        numb_test = float("inf")
    if train_json is not None:
        jdata = j_loader(train_json)
        jdata = update_deepmd_input(jdata)
        data_params = jdata.get("training", {}).get("training_data", {})
        systems = data_params.get("systems")
        if not systems:
            raise RuntimeError("No training data found in input json")
        root = Path(train_json).parent
        if isinstance(systems, str):
            systems = str((root / Path(systems)).resolve())
        else:
            systems = [str((root / Path(ss)).resolve()) for ss in systems]
        patterns = data_params.get("rglob_patterns", None)
        all_sys = process_systems(systems, patterns=patterns)
    elif valid_json is not None:
        jdata = j_loader(valid_json)
        jdata = update_deepmd_input(jdata)
        data_params = jdata.get("training", {}).get("validation_data", {})
        systems = data_params.get("systems")
        if not systems:
            raise RuntimeError("No validation data found in input json")
        root = Path(valid_json).parent
        if isinstance(systems, str):
            systems = str((root / Path(systems)).resolve())
        else:
            systems = [str((root / Path(ss)).resolve()) for ss in systems]
        patterns = data_params.get("rglob_patterns", None)
        all_sys = process_systems(systems, patterns=patterns)
    elif datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    elif system is not None:
        all_sys = process_systems(system)
    else:
        raise RuntimeError("No data source specified for testing")

    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")
    err_coll = []
    siz_coll = []

    # init random seed
    if rand_seed is not None:
        dp_random.seed(rand_seed % (2**32))

    # init model
    dp = DeepEval(model, head=head)

    for cc, system in enumerate(all_sys):
        log.info("# ---------------output of dp test--------------- ")
        log.info(f"# testing system : {system}")

        # create data class
        tmap = dp.get_type_map()
        if is_lmdb(system):
            lmdb_data = LmdbTestData(
                system,
                type_map=tmap,
                shuffle_test=shuffle_test,
            )
            # For mixed-nloc LMDB, test each nloc group separately
            nloc_keys = sorted(lmdb_data.nloc_groups.keys())
            if len(nloc_keys) > 1:
                group_summary = {
                    k: len(v) for k, v in sorted(lmdb_data.nloc_groups.items())
                }
                log.info(
                    f"# mixed-nloc LMDB: testing {len(nloc_keys)} groups: "
                    f"{group_summary}"
                )
            data_items: list[tuple[Any, str]] = []
            for nloc_val in nloc_keys:
                label = f"{system} [nloc={nloc_val}]" if len(nloc_keys) > 1 else system
                # Create a thin wrapper that returns only this nloc group
                data_items.append((LmdbTestDataNlocView(lmdb_data, nloc_val), label))
        else:
            data = DeepmdData(
                system,
                set_prefix="set",
                shuffle_test=shuffle_test,
                type_map=tmap,
                sort_atoms=False,
            )
            data_items = [(data, system)]

        for data, sys_label in data_items:
            if sys_label != system:
                log.info(f"# testing sub-group : {sys_label}")

            if isinstance(dp, DeepPot):
                err = test_ener(
                    dp,
                    data,
                    sys_label,
                    numb_test,
                    detail_file,
                    atomic,
                    append_detail=(cc != 0),
                )
            elif isinstance(dp, DeepDOS):
                err = test_dos(
                    dp,
                    data,
                    sys_label,
                    numb_test,
                    detail_file,
                    atomic,
                    append_detail=(cc != 0),
                )
            elif isinstance(dp, DeepProperty):
                err = test_property(
                    dp,
                    data,
                    sys_label,
                    numb_test,
                    detail_file,
                    atomic,
                    append_detail=(cc != 0),
                )
            elif isinstance(dp, DeepDipole):
                err = test_dipole(dp, data, numb_test, detail_file, atomic)
            elif isinstance(dp, DeepPolar):
                err = test_polar(dp, data, numb_test, detail_file, atomic=atomic)
            elif isinstance(
                dp, DeepGlobalPolar
            ):  # should not appear in this new version
                log.warning(
                    "Global polar model is not currently supported. Please directly use the polar mode and change loss parameters."
                )
                err = test_polar(
                    dp, data, numb_test, detail_file, atomic=False
                )  # YWolfeee: downward compatibility
            log.info("# ----------------------------------------------- ")
            err_coll.append(err)

    avg_err = weighted_average(err_coll)

    # For mixed-nloc LMDB, err_coll may have more entries than all_sys
    # (one per nloc group per system). Only warn if fewer.
    if len(err_coll) < len(all_sys):
        log.warning("Not all systems are tested! Check if the systems are valid")

    log.info("# ----------weighted average of errors----------- ")
    log.info(f"# number of systems : {len(all_sys)}")
    if isinstance(dp, DeepPot):
        print_ener_sys_avg(avg_err)
    elif isinstance(dp, DeepDOS):
        print_dos_sys_avg(avg_err)
    elif isinstance(dp, DeepProperty):
        print_property_sys_avg(avg_err)
    elif isinstance(dp, DeepDipole):
        print_dipole_sys_avg(avg_err)
    elif isinstance(dp, DeepPolar):
        print_polar_sys_avg(avg_err)
    elif isinstance(dp, DeepGlobalPolar):
        print_polar_sys_avg(avg_err)
    elif isinstance(dp, DeepWFC):
        print_wfc_sys_avg(avg_err)
    log.info("# ----------------------------------------------- ")


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
) -> None:
    """Save numpy array to test file.

    Parameters
    ----------
    fname : str
        filename
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended instead of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)


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


def test_ener(
    dp: "DeepPot",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: str | None,
    has_atom_ener: bool,
    append_detail: bool = False,
) -> dict[str, tuple[float, float]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data : DeepmdData
        data container object
    system : str
        system directory
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    has_atom_ener : bool
        whether per atom quantities should be computed
    append_detail : bool, optional
        if true append output detail file, by default False

    Returns
    -------
    dict[str, tuple[float, float]]
        weighted-average-ready metric pairs
    """
    dict_to_return = {}

    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    data.add("atom_pref", 1, atomic=True, must=False, high_prec=False, repeat=3)
    data.add("virial", 9, atomic=False, must=False, high_prec=False)
    if dp.has_efield:
        data.add("efield", 3, atomic=True, must=True, high_prec=False)
    if has_atom_ener:
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
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)
    if dp.has_spin:
        data.add("spin", 3, atomic=True, must=True, high_prec=False)
        data.add("force_mag", 3, atomic=True, must=False, high_prec=False)
    if dp.has_hessian:
        data.add("hessian", 1, atomic=True, must=True, high_prec=False)

    test_data = data.get_test()
    find_energy = test_data.get("find_energy")
    find_force = test_data.get("find_force")
    find_virial = test_data.get("find_virial")
    find_force_mag = test_data.get("find_force_mag")
    find_atom_pref = test_data.get("find_atom_pref")
    mixed_type = data.mixed_type
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    if dp.has_efield:
        efield = test_data["efield"][:numb_test].reshape([numb_test, -1])
    else:
        efield = None
    if dp.has_spin:
        spin = test_data["spin"][:numb_test].reshape([numb_test, -1])
    else:
        spin = None
    if not data.pbc:
        box = None
    if mixed_type:
        atype = test_data["type"][:numb_test].reshape([numb_test, -1])
    else:
        atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0 and test_data["find_fparam"] != 0.0:
        fparam = test_data["fparam"][:numb_test]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test]
    else:
        aparam = None

    ret = dp.eval(
        coord,
        box,
        atype,
        fparam=fparam,
        aparam=aparam,
        atomic=has_atom_ener,
        efield=efield,
        mixed_type=mixed_type,
        spin=spin,
    )
    energy = ret[0]
    force = ret[1]
    virial = ret[2]
    energy = energy.reshape([numb_test, 1])
    force = force.reshape([numb_test, -1])
    virial = virial.reshape([numb_test, 9])
    hessian = None
    force_m = None
    mask_mag = None
    if dp.has_hessian:
        hessian = ret[3]
        hessian = hessian.reshape([numb_test, -1])
    if has_atom_ener:
        ae = ret[3]
        av = ret[4]
        ae = ae.reshape([numb_test, -1])
        av = av.reshape([numb_test, -1])
        if dp.has_spin:
            force_m = ret[5]
            force_m = force_m.reshape([numb_test, -1])
            mask_mag = ret[6]
            mask_mag = mask_mag.reshape([numb_test, -1])
    else:
        if dp.has_spin:
            force_m = ret[3]
            force_m = force_m.reshape([numb_test, -1])
            mask_mag = ret[4]
            mask_mag = mask_mag.reshape([numb_test, -1])
    out_put_spin = dp.get_ntypes_spin() != 0 or dp.has_spin
    spin_metrics = None
    force_r = None
    test_force_r = None
    test_force_m = None
    if out_put_spin:
        force_r, test_force_r, force_m, test_force_m = _align_spin_force_arrays(
            dp=dp,
            atype=atype,
            natoms=natoms,
            prediction_force=force,
            reference_force=test_data["force"][:numb_test],
            prediction_force_mag=force_m,
            reference_force_mag=(
                test_data["force_mag"][:numb_test] if "force_mag" in test_data else None
            ),
            mask_mag=mask_mag,
        )
        if find_force_mag == 1 and (force_m is None or test_force_m is None):
            raise RuntimeError(
                "Spin magnetic force metrics require magnetic force arrays and mask."
            )
        spin_metrics = compute_spin_force_metrics(
            force_real_prediction=force_r,
            force_real_reference=test_force_r,
            force_magnetic_prediction=force_m if find_force_mag == 1 else None,
            force_magnetic_reference=test_force_m if find_force_mag == 1 else None,
        )

    energy_metric_input = {
        "find_energy": find_energy,
        "find_force": find_force if not out_put_spin else 0.0,
        "find_virial": find_virial if not out_put_spin else 0.0,
        "energy": test_data["energy"][:numb_test],
        "force": test_data["force"][:numb_test],
    }
    energy_metric_prediction = {
        "energy": energy,
        "force": force,
    }
    if find_virial == 1 and data.pbc and not out_put_spin:
        energy_metric_input["virial"] = test_data["virial"][:numb_test]
        energy_metric_prediction["virial"] = virial
    shared_metrics = compute_energy_type_metrics(
        prediction=energy_metric_prediction,
        test_data=energy_metric_input,
        natoms=natoms,
        has_pbc=data.pbc,
    )
    dict_to_return.update(
        shared_metrics.as_weighted_average_errors(DP_TEST_WEIGHTED_METRIC_KEYS)
    )

    weighted_force_metrics = None
    if find_energy == 1:
        if shared_metrics.energy is None or shared_metrics.energy_per_atom is None:
            raise RuntimeError("Energy metrics are unavailable for dp test.")
        mae_e = shared_metrics.energy.mae
        rmse_e = shared_metrics.energy.rmse
        mae_ea = shared_metrics.energy_per_atom.mae
        rmse_ea = shared_metrics.energy_per_atom.rmse

    if not out_put_spin and find_force == 1:
        if shared_metrics.force is None:
            raise RuntimeError("Force metrics are unavailable for dp test.")
        mae_f = shared_metrics.force.mae
        rmse_f = shared_metrics.force.rmse
        if find_atom_pref == 1:
            weighted_force_metrics = compute_weighted_error_stat(
                force,
                test_data["force"][:numb_test],
                test_data["atom_pref"][:numb_test],
            )
            mae_fw = weighted_force_metrics.mae
            rmse_fw = weighted_force_metrics.rmse

    if data.pbc and not out_put_spin and find_virial == 1:
        if shared_metrics.virial is None or shared_metrics.virial_per_atom is None:
            raise RuntimeError("Virial metrics are unavailable for dp test.")
        mae_v = shared_metrics.virial.mae
        rmse_v = shared_metrics.virial.rmse
        mae_va = shared_metrics.virial_per_atom.mae
        rmse_va = shared_metrics.virial_per_atom.rmse

    hessian_metrics = None
    if dp.has_hessian:
        hessian_metrics = compute_error_stat(
            hessian,
            test_data["hessian"][:numb_test],
        )
        mae_h = hessian_metrics.mae
        rmse_h = hessian_metrics.rmse
    if has_atom_ener:
        atomic_energy_metrics = compute_error_stat(
            ae.reshape([-1]),
            test_data["atom_ener"][:numb_test].reshape([-1]),
        )
        mae_ae = atomic_energy_metrics.mae
        rmse_ae = atomic_energy_metrics.rmse
    if out_put_spin:
        if spin_metrics is None or spin_metrics.force_real is None:
            raise RuntimeError("Spin force metrics are unavailable for dp test.")
        mae_fr = spin_metrics.force_real.mae
        rmse_fr = spin_metrics.force_real.rmse
        if find_force_mag == 1:
            if spin_metrics.force_magnetic is None:
                raise RuntimeError("Spin magnetic force metrics are unavailable.")
            mae_fm = spin_metrics.force_magnetic.mae
            rmse_fm = spin_metrics.force_magnetic.rmse

    log.info(f"# number of test data : {numb_test:d} ")
    if find_energy == 1:
        log.info(f"Energy MAE         : {mae_e:e} eV")
        log.info(f"Energy RMSE        : {rmse_e:e} eV")
        log.info(f"Energy MAE/Natoms  : {mae_ea:e} eV")
        log.info(f"Energy RMSE/Natoms : {rmse_ea:e} eV")
    if not out_put_spin and find_force == 1:
        log.info(f"Force  MAE         : {mae_f:e} eV/Å")
        log.info(f"Force  RMSE        : {rmse_f:e} eV/Å")
        if weighted_force_metrics is not None:
            log.info(f"Force weighted MAE : {mae_fw:e} eV/Å")
            log.info(f"Force weighted RMSE: {rmse_fw:e} eV/Å")
            dict_to_return.update(
                weighted_force_metrics.as_weighted_average_errors(
                    *DP_TEST_WEIGHTED_FORCE_METRIC_KEYS
                )
            )
    if out_put_spin and find_force == 1:
        log.info(f"Force atom MAE      : {mae_fr:e} eV/Å")
        log.info(f"Force atom RMSE     : {rmse_fr:e} eV/Å")
        dict_to_return.update(
            spin_metrics.as_weighted_average_errors(
                {"force_real": DP_TEST_SPIN_WEIGHTED_METRIC_KEYS["force_real"]}
            )
        )
    if out_put_spin and find_force_mag == 1:
        log.info(f"Force spin MAE      : {mae_fm:e} eV/uB")
        log.info(f"Force spin RMSE     : {rmse_fm:e} eV/uB")
        dict_to_return.update(
            spin_metrics.as_weighted_average_errors(
                {"force_magnetic": DP_TEST_SPIN_WEIGHTED_METRIC_KEYS["force_magnetic"]}
            )
        )
    if data.pbc and not out_put_spin and find_virial == 1:
        log.info(f"Virial MAE         : {mae_v:e} eV")
        log.info(f"Virial RMSE        : {rmse_v:e} eV")
        log.info(f"Virial MAE/Natoms  : {mae_va:e} eV")
        log.info(f"Virial RMSE/Natoms : {rmse_va:e} eV")
    if has_atom_ener:
        log.info(f"Atomic ener MAE    : {mae_ae:e} eV")
        log.info(f"Atomic ener RMSE   : {rmse_ae:e} eV")
    if dp.has_hessian:
        log.info(f"Hessian MAE        : {mae_h:e} eV/Å^2")
        log.info(f"Hessian RMSE       : {rmse_h:e} eV/Å^2")
        if hessian_metrics is None:
            raise RuntimeError("Hessian metrics are unavailable for dp test.")
        dict_to_return.update(
            hessian_metrics.as_weighted_average_errors(*DP_TEST_HESSIAN_METRIC_KEYS)
        )

    if detail_file is not None:
        _write_energy_test_details(
            detail_path=Path(detail_file),
            system=system,
            natoms=natoms,
            append_detail=append_detail,
            reference_energy=test_data["energy"][:numb_test],
            prediction_energy=energy,
            reference_force=test_data["force"][:numb_test],
            prediction_force=force,
            reference_virial=test_data["virial"][:numb_test],
            prediction_virial=virial,
            out_put_spin=out_put_spin,
            reference_force_real=test_force_r,
            prediction_force_real=force_r,
            reference_force_magnetic=test_force_m if find_force_mag == 1 else None,
            prediction_force_magnetic=force_m
            if out_put_spin and find_force_mag == 1
            else None,
            reference_hessian=test_data["hessian"][:numb_test]
            if dp.has_hessian
            else None,
            prediction_hessian=hessian if dp.has_hessian else None,
        )

    return dict_to_return


def print_ener_sys_avg(avg: dict[str, float]) -> None:
    """Print errors summary for energy type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Energy MAE         : {avg['mae_e']:e} eV")
    log.info(f"Energy RMSE        : {avg['rmse_e']:e} eV")
    log.info(f"Energy MAE/Natoms  : {avg['mae_ea']:e} eV")
    log.info(f"Energy RMSE/Natoms : {avg['rmse_ea']:e} eV")
    if "rmse_f" in avg:
        log.info(f"Force  MAE         : {avg['mae_f']:e} eV/Å")
        log.info(f"Force  RMSE        : {avg['rmse_f']:e} eV/Å")
        if "rmse_fw" in avg:
            log.info(f"Force weighted MAE : {avg['mae_fw']:e} eV/Å")
            log.info(f"Force weighted RMSE: {avg['rmse_fw']:e} eV/Å")
    else:
        log.info(f"Force atom MAE      : {avg['mae_fr']:e} eV/Å")
        log.info(f"Force atom RMSE     : {avg['rmse_fr']:e} eV/Å")
        if "rmse_fm" in avg:
            log.info(f"Force spin MAE      : {avg['mae_fm']:e} eV/uB")
            log.info(f"Force spin RMSE     : {avg['rmse_fm']:e} eV/uB")
    if "rmse_v" in avg:
        log.info(f"Virial MAE         : {avg['mae_v']:e} eV")
        log.info(f"Virial RMSE        : {avg['rmse_v']:e} eV")
        log.info(f"Virial MAE/Natoms  : {avg['mae_va']:e} eV")
        log.info(f"Virial RMSE/Natoms : {avg['rmse_va']:e} eV")
    if "rmse_h" in avg:
        log.info(f"Hessian MAE         : {avg['mae_h']:e} eV/Å^2")
        log.info(f"Hessian RMSE        : {avg['rmse_h']:e} eV/Å^2")


def test_dos(
    dp: "DeepDOS",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: str | None,
    has_atom_dos: bool,
    append_detail: bool = False,
) -> tuple[list[np.ndarray], list[int]]:
    """Test DOS type model.

    Parameters
    ----------
    dp : DeepDOS
        instance of deep potential
    data : DeepmdData
        data container object
    system : str
        system directory
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    has_atom_dos : bool
        whether per atom quantities should be computed
    append_detail : bool, optional
        if true append output detail file, by default False

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        arrays with results and their shapes
    """
    data.add("dos", dp.numb_dos, atomic=False, must=True, high_prec=True)
    if has_atom_dos:
        data.add("atom_dos", dp.numb_dos, atomic=True, must=False, high_prec=True)

    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)

    test_data = data.get_test()
    mixed_type = data.mixed_type
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]

    if not data.pbc:
        box = None
    if mixed_type:
        atype = test_data["type"][:numb_test].reshape([numb_test, -1])
    else:
        atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test]
    else:
        aparam = None

    ret = dp.eval(
        coord,
        box,
        atype,
        fparam=fparam,
        aparam=aparam,
        atomic=has_atom_dos,
        mixed_type=mixed_type,
    )
    dos = ret[0]

    dos = dos.reshape([numb_test, dp.numb_dos])

    if has_atom_dos:
        ados = ret[1]
        ados = ados.reshape([numb_test, natoms * dp.numb_dos])

    diff_dos = dos - test_data["dos"][:numb_test]
    mae_dos = mae(diff_dos)
    rmse_dos = rmse(diff_dos)

    mae_dosa = mae_dos / natoms
    rmse_dosa = rmse_dos / natoms

    if has_atom_dos:
        diff_ados = ados - test_data["atom_dos"][:numb_test]
        mae_ados = mae(diff_ados)
        rmse_ados = rmse(diff_ados)

    log.info(f"# number of test data : {numb_test:d} ")

    log.info(f"DOS MAE            : {mae_dos:e} Occupation/eV")
    log.info(f"DOS RMSE           : {rmse_dos:e} Occupation/eV")
    log.info(f"DOS MAE/Natoms     : {mae_dosa:e} Occupation/eV")
    log.info(f"DOS RMSE/Natoms    : {rmse_dosa:e} Occupation/eV")

    if has_atom_dos:
        log.info(f"Atomic DOS MAE     : {mae_ados:e} Occupation/eV")
        log.info(f"Atomic DOS RMSE    : {rmse_ados:e} Occupation/eV")

    if detail_file is not None:
        detail_path = Path(detail_file)

        for ii in range(numb_test):
            test_out = test_data["dos"][ii].reshape(-1, 1)
            pred_out = dos[ii].reshape(-1, 1)

            frame_output = np.hstack((test_out, pred_out))

            save_txt_file(
                detail_path.with_suffix(f".dos.out.{ii}"),
                frame_output,
                header=f"{system} - {ii}: data_dos pred_dos",
                append=append_detail,
            )

        if has_atom_dos:
            for ii in range(numb_test):
                test_out = test_data["atom_dos"][ii].reshape(-1, 1)
                pred_out = ados[ii].reshape(-1, 1)

                frame_output = np.hstack((test_out, pred_out))

                save_txt_file(
                    detail_path.with_suffix(f".ados.out.{ii}"),
                    frame_output,
                    header=f"{system} - {ii}: data_ados pred_ados",
                    append=append_detail,
                )

    return {
        "mae_dos": (mae_dos, dos.size),
        "mae_dosa": (mae_dosa, dos.size),
        "rmse_dos": (rmse_dos, dos.size),
        "rmse_dosa": (rmse_dosa, dos.size),
    }


def print_dos_sys_avg(avg: dict[str, float]) -> None:
    """Print errors summary for DOS type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"DOS MAE            : {avg['mae_dos']:e} Occupation/eV")
    log.info(f"DOS RMSE           : {avg['rmse_dos']:e} Occupation/eV")
    log.info(f"DOS MAE/Natoms     : {avg['mae_dosa']:e} Occupation/eV")
    log.info(f"DOS RMSE/Natoms    : {avg['rmse_dosa']:e} Occupation/eV")


def test_property(
    dp: "DeepProperty",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: str | None,
    has_atom_property: bool,
    append_detail: bool = False,
) -> tuple[list[np.ndarray], list[int]]:
    """Test Property type model.

    Parameters
    ----------
    dp : DeepProperty
        instance of deep potential
    data : DeepmdData
        data container object
    system : str
        system directory
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    has_atom_property : bool
        whether per atom quantities should be computed
    append_detail : bool, optional
        if true append output detail file, by default False

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        arrays with results and their shapes
    """
    var_name = dp.get_var_name()
    assert isinstance(var_name, str)
    data.add(var_name, dp.task_dim, atomic=False, must=True, high_prec=True)
    if has_atom_property:
        data.add(
            f"atom_{var_name}",
            dp.task_dim,
            atomic=True,
            must=False,
            high_prec=True,
        )

    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)

    test_data = data.get_test()
    mixed_type = data.mixed_type
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]

    if not data.pbc:
        box = None
    if mixed_type:
        atype = test_data["type"][:numb_test].reshape([numb_test, -1])
    else:
        atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test]
    else:
        aparam = None

    ret = dp.eval(
        coord,
        box,
        atype,
        fparam=fparam,
        aparam=aparam,
        atomic=has_atom_property,
        mixed_type=mixed_type,
    )

    property = ret[0]

    property = property.reshape([numb_test, dp.task_dim])

    if has_atom_property:
        aproperty = ret[1]
        aproperty = aproperty.reshape([numb_test, natoms * dp.task_dim])

    diff_property = property - test_data[var_name][:numb_test]
    mae_property = mae(diff_property)
    rmse_property = rmse(diff_property)

    if has_atom_property:
        diff_aproperty = aproperty - test_data[f"atom_{var_name}"][:numb_test]
        mae_aproperty = mae(diff_aproperty)
        rmse_aproperty = rmse(diff_aproperty)

    log.info(f"# number of test data : {numb_test:d} ")

    log.info(f"PROPERTY MAE            : {mae_property:e} units")
    log.info(f"PROPERTY RMSE           : {rmse_property:e} units")

    if has_atom_property:
        log.info(f"Atomic PROPERTY MAE     : {mae_aproperty:e} units")
        log.info(f"Atomic PROPERTY RMSE    : {rmse_aproperty:e} units")

    if detail_file is not None:
        detail_path = Path(detail_file)

        for ii in range(numb_test):
            test_out = test_data[var_name][ii].reshape(-1, 1)
            pred_out = property[ii].reshape(-1, 1)

            frame_output = np.hstack((test_out, pred_out))

            save_txt_file(
                detail_path.with_suffix(f".property.out.{ii}"),
                frame_output,
                header=f"{system} - {ii}: data_property pred_property",
                append=append_detail,
            )

        if has_atom_property:
            for ii in range(numb_test):
                test_out = test_data[f"atom_{var_name}"][ii].reshape(-1, 1)
                pred_out = aproperty[ii].reshape(-1, 1)

                frame_output = np.hstack((test_out, pred_out))

                save_txt_file(
                    detail_path.with_suffix(f".aproperty.out.{ii}"),
                    frame_output,
                    header=f"{system} - {ii}: data_aproperty pred_aproperty",
                    append=append_detail,
                )

    return {
        "mae_property": (mae_property, property.size),
        "rmse_property": (rmse_property, property.size),
    }


def print_property_sys_avg(avg: dict[str, float]) -> None:
    """Print errors summary for Property type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"PROPERTY MAE            : {avg['mae_property']:e} units")
    log.info(f"PROPERTY RMSE           : {avg['rmse_property']:e} units")


def run_test(
    dp: "DeepTensor", test_data: dict, numb_test: int, test_sys: DeepmdData
) -> dict:
    """Run tests.

    Parameters
    ----------
    dp : DeepTensor
        instance of deep potential
    test_data : dict
        dictionary with test data
    numb_test : int
        munber of tests to do
    test_sys : DeepmdData
        test system

    Returns
    -------
    [type]
        [description]
    """
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    if test_sys.pbc:
        box = test_data["box"][:numb_test]
    else:
        box = None
    atype = test_data["type"][0]
    prediction = dp.eval(coord, box, atype)

    return prediction.reshape([numb_test, -1]), numb_test, atype


def test_wfc(
    dp: "DeepWFC",
    data: DeepmdData,
    numb_test: int,
    detail_file: str | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data : DeepmdData
        data container object
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        arrays with results and their shapes
    """
    data.add(
        "wfc", 12, atomic=True, must=True, high_prec=False, type_sel=dp.get_sel_type()
    )
    test_data = data.get_test()
    wfc, numb_test, _ = run_test(dp, test_data, numb_test, data)
    rmse_f = rmse(wfc - test_data["wfc"][:numb_test])

    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"WFC  RMSE : {rmse_f:e}")

    if detail_file is not None:
        detail_path = Path(detail_file)
        pe = np.concatenate(
            (
                np.reshape(test_data["wfc"][:numb_test], [-1, 12]),
                np.reshape(wfc, [-1, 12]),
            ),
            axis=1,
        )
        np.savetxt(
            detail_path.with_suffix(".out"),
            pe,
            header="ref_wfc(12 dofs)   predicted_wfc(12 dofs)",
        )
    return {"rmse": (rmse_f, wfc.size)}


def print_wfc_sys_avg(avg: dict) -> None:
    """Print errors summary for wfc type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"WFC  RMSE : {avg['rmse']:e}")


def test_polar(
    dp: "DeepPolar",
    data: DeepmdData,
    numb_test: int,
    detail_file: str | None,
    *,
    atomic: bool,
) -> tuple[list[np.ndarray], list[int]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data : DeepmdData
        data container object
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether to use glovbal version of polar potential

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        arrays with results and their shapes
    """
    data.add(
        "polarizability" if not atomic else "atomic_polarizability",
        9,
        atomic=atomic,
        must=True,
        high_prec=False,
        type_sel=dp.get_sel_type(),
        output_natoms_for_type_sel=True,
    )

    test_data = data.get_test()
    polar, numb_test, atype = run_test(dp, test_data, numb_test, data)

    sel_type = dp.get_sel_type()
    sel_natoms = 0
    for ii in sel_type:
        sel_natoms += sum(atype == ii)

    # YWolfeee: do summation in global polar mode
    if not atomic:
        polar = np.sum(polar.reshape((polar.shape[0], -1, 9)), axis=1)
        rmse_f = rmse(polar - test_data["polarizability"][:numb_test])
        rmse_fs = rmse_f / np.sqrt(sel_natoms)
        rmse_fa = rmse_f / sel_natoms
    else:
        sel_mask = np.isin(atype, sel_type)
        polar = polar.reshape((polar.shape[0], -1, 9))[:, sel_mask, :].reshape(
            (polar.shape[0], -1)
        )
        label_polar = (
            test_data["atom_polarizability"][:numb_test]
            .reshape((numb_test, -1, 9))[:, sel_mask, :]
            .reshape((numb_test, -1))
        )
        rmse_f = rmse(polar - label_polar)

    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Polarizability  RMSE       : {rmse_f:e}")
    if not atomic:
        log.info(f"Polarizability  RMSE/sqrtN : {rmse_fs:e}")
        log.info(f"Polarizability  RMSE/N     : {rmse_fa:e}")
    log.info("The unit of error is the same as the unit of provided label.")

    if detail_file is not None:
        detail_path = Path(detail_file)

        if not atomic:
            pe = np.concatenate(
                (
                    np.reshape(test_data["polarizability"][:numb_test], [-1, 9]),
                    np.reshape(polar, [-1, 9]),
                ),
                axis=1,
            )
            header_text = (
                "data_pxx data_pxy data_pxz data_pyx data_pyy data_pyz data_pzx "
                "data_pzy data_pzz pred_pxx pred_pxy pred_pxz pred_pyx pred_pyy "
                "pred_pyz pred_pzx pred_pzy pred_pzz"
            )
        else:
            pe = np.concatenate(
                (
                    np.reshape(label_polar, [-1, 9 * sel_natoms]),
                    np.reshape(polar, [-1, 9 * sel_natoms]),
                ),
                axis=1,
            )
            header_text = [
                f"{letter}{number}"
                for number in range(1, sel_natoms + 1)
                for letter in [
                    "data_pxx",
                    "data_pxy",
                    "data_pxz",
                    "data_pyx",
                    "data_pyy",
                    "data_pyz",
                    "data_pzx",
                    "data_pzy",
                    "data_pzz",
                ]
            ] + [
                f"{letter}{number}"
                for number in range(1, sel_natoms + 1)
                for letter in [
                    "pred_pxx",
                    "pred_pxy",
                    "pred_pxz",
                    "pred_pyx",
                    "pred_pyy",
                    "pred_pyz",
                    "pred_pzx",
                    "pred_pzy",
                    "pred_pzz",
                ]
            ]
            header_text = " ".join(header_text)

        np.savetxt(
            detail_path.with_suffix(".out"),
            pe,
            header=header_text,
        )
    return {"rmse": (rmse_f, polar.size)}


def print_polar_sys_avg(avg: dict) -> None:
    """Print errors summary for polar type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Polarizability  RMSE : {avg['rmse']:e}")


def test_dipole(
    dp: "DeepDipole",
    data: DeepmdData,
    numb_test: int,
    detail_file: str | None,
    atomic: bool,
) -> tuple[list[np.ndarray], list[int]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data : DeepmdData
        data container object
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether atomic dipole is provided

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        arrays with results and their shapes
    """
    data.add(
        "dipole" if not atomic else "atomic_dipole",
        3,
        atomic=atomic,
        must=True,
        high_prec=False,
        type_sel=dp.get_sel_type(),
        output_natoms_for_type_sel=True,
    )

    test_data = data.get_test()
    dipole, numb_test, atype = run_test(dp, test_data, numb_test, data)

    sel_type = dp.get_sel_type()
    sel_natoms = 0
    for ii in sel_type:
        sel_natoms += sum(atype == ii)

    # do summation in atom dimension
    if not atomic:
        dipole = np.sum(dipole.reshape((dipole.shape[0], -1, 3)), axis=1)
        rmse_f = rmse(dipole - test_data["dipole"][:numb_test])
        rmse_fs = rmse_f / np.sqrt(sel_natoms)
        rmse_fa = rmse_f / sel_natoms
    else:
        sel_mask = np.isin(atype, sel_type)
        dipole = dipole.reshape((dipole.shape[0], -1, 3))[:, sel_mask, :].reshape(
            (dipole.shape[0], -1)
        )
        label_dipole = (
            test_data["atom_dipole"][:numb_test]
            .reshape((numb_test, -1, 3))[:, sel_mask, :]
            .reshape((numb_test, -1))
        )
        rmse_f = rmse(dipole - label_dipole)

    log.info(f"# number of test data : {numb_test:d}")
    log.info(f"Dipole  RMSE       : {rmse_f:e}")
    if not atomic:
        log.info(f"Dipole  RMSE/sqrtN : {rmse_fs:e}")
        log.info(f"Dipole  RMSE/N     : {rmse_fa:e}")
    log.info("The unit of error is the same as the unit of provided label.")

    if detail_file is not None:
        detail_path = Path(detail_file)
        if not atomic:
            pe = np.concatenate(
                (
                    np.reshape(test_data["dipole"][:numb_test], [-1, 3]),
                    np.reshape(dipole, [-1, 3]),
                ),
                axis=1,
            )
            header_text = "data_x data_y data_z pred_x pred_y pred_z"
        else:
            pe = np.concatenate(
                (
                    np.reshape(label_dipole, [-1, 3 * sel_natoms]),
                    np.reshape(dipole, [-1, 3 * sel_natoms]),
                ),
                axis=1,
            )
            header_text = [
                f"{letter}{number}"
                for number in range(1, sel_natoms + 1)
                for letter in ["data_x", "data_y", "data_z"]
            ] + [
                f"{letter}{number}"
                for number in range(1, sel_natoms + 1)
                for letter in ["pred_x", "pred_y", "pred_z"]
            ]
            header_text = " ".join(header_text)

        np.savetxt(
            detail_path.with_suffix(".out"),
            pe,
            header=header_text,
        )
    return {"rmse": (rmse_f, dipole.size)}


def print_dipole_sys_avg(avg: dict) -> None:
    """Print errors summary for dipole type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Dipole  RMSE         : {avg['rmse']:e}")
