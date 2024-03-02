# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test trained DeePMD model."""
import logging
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.common import (
    expand_sys_str,
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
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.weight_avg import (
    weighted_average,
)

if TYPE_CHECKING:
    from deepmd.tf.infer import (
        DeepDipole,
        DeepDOS,
        DeepPolar,
        DeepPot,
        DeepWFC,
    )
    from deepmd.tf.infer.deep_tensor import (
        DeepTensor,
    )

__all__ = ["test"]

log = logging.getLogger(__name__)


def test(
    *,
    model: str,
    system: str,
    datafile: str,
    set_prefix: str,
    numb_test: int,
    rand_seed: Optional[int],
    shuffle_test: bool,
    detail_file: str,
    atomic: bool,
    head: Optional[str] = None,
    **kwargs,
):
    """Test model predictions.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str
        system directory
    datafile : str
        the path to the list of systems to test
    set_prefix : str
        string prefix of set
    numb_test : int
        munber of tests to do. 0 means all data.
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
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    else:
        all_sys = expand_sys_str(system)

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
        tmap = dp.get_type_map() if isinstance(dp, DeepPot) else None
        data = DeepmdData(
            system,
            set_prefix,
            shuffle_test=shuffle_test,
            type_map=tmap,
            sort_atoms=False,
        )

        if isinstance(dp, DeepPot):
            err = test_ener(
                dp,
                data,
                system,
                numb_test,
                detail_file,
                atomic,
                append_detail=(cc != 0),
            )
        elif isinstance(dp, DeepDOS):
            err = test_dos(
                dp,
                data,
                system,
                numb_test,
                detail_file,
                atomic,
                append_detail=(cc != 0),
            )
        elif isinstance(dp, DeepDipole):
            err = test_dipole(dp, data, numb_test, detail_file, atomic)
        elif isinstance(dp, DeepPolar):
            err = test_polar(dp, data, numb_test, detail_file, atomic=atomic)
        elif isinstance(dp, DeepGlobalPolar):  # should not appear in this new version
            log.warning(
                "Global polar model is not currently supported. Please directly use the polar mode and change loss parameters."
            )
            err = test_polar(
                dp, data, numb_test, detail_file, atomic=False
            )  # YWolfeee: downward compatibility
        log.info("# ----------------------------------------------- ")
        err_coll.append(err)

    avg_err = weighted_average(err_coll)

    if len(all_sys) != len(err_coll):
        log.warning("Not all systems are tested! Check if the systems are valid")

    if len(all_sys) > 1:
        log.info("# ----------weighted average of errors----------- ")
        log.info(f"# number of systems : {len(all_sys)}")
        if isinstance(dp, DeepPot):
            print_ener_sys_avg(avg_err)
        elif isinstance(dp, DeepDOS):
            print_dos_sys_avg(avg_err)
        elif isinstance(dp, DeepDipole):
            print_dipole_sys_avg(avg_err)
        elif isinstance(dp, DeepPolar):
            print_polar_sys_avg(avg_err)
        elif isinstance(dp, DeepGlobalPolar):
            print_polar_sys_avg(avg_err)
        elif isinstance(dp, DeepGlobalPolar):
            print_wfc_sys_avg(avg_err)
        log.info("# ----------------------------------------------- ")


def mae(diff: np.ndarray) -> float:
    """Calcalte mean absulote error.

    Parameters
    ----------
    diff : np.ndarray
        difference

    Returns
    -------
    float
        mean absulote error
    """
    return np.mean(np.abs(diff))


def rmse(diff: np.ndarray) -> float:
    """Calculate root mean square error.

    Parameters
    ----------
    diff : np.ndarray
        difference

    Returns
    -------
    float
        root mean square error
    """
    return np.sqrt(np.average(diff * diff))


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
):
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
        if true file will be appended insted of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)


def test_ener(
    dp: "DeepPot",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: Optional[str],
    has_atom_ener: bool,
    append_detail: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
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
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    data.add("virial", 9, atomic=False, must=False, high_prec=False)
    if dp.has_efield:
        data.add("efield", 3, atomic=True, must=True, high_prec=False)
    if has_atom_ener:
        data.add("atom_ener", 1, atomic=True, must=True, high_prec=False)
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
    if dp.has_efield:
        efield = test_data["efield"][:numb_test].reshape([numb_test, -1])
    else:
        efield = None
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
        atomic=has_atom_ener,
        efield=efield,
        mixed_type=mixed_type,
    )
    energy = ret[0]
    force = ret[1]
    virial = ret[2]
    energy = energy.reshape([numb_test, 1])
    force = force.reshape([numb_test, -1])
    virial = virial.reshape([numb_test, 9])
    if has_atom_ener:
        ae = ret[3]
        av = ret[4]
        ae = ae.reshape([numb_test, -1])
        av = av.reshape([numb_test, -1])
    if dp.get_ntypes_spin() != 0:
        ntypes_real = dp.get_ntypes() - dp.get_ntypes_spin()
        nloc = natoms
        nloc_real = sum([np.count_nonzero(atype == ii) for ii in range(ntypes_real)])
        force_r = np.split(
            force, indices_or_sections=[nloc_real * 3, nloc * 3], axis=1
        )[0]
        force_m = np.split(
            force, indices_or_sections=[nloc_real * 3, nloc * 3], axis=1
        )[1]
        test_force_r = np.split(
            test_data["force"][:numb_test],
            indices_or_sections=[nloc_real * 3, nloc * 3],
            axis=1,
        )[0]
        test_force_m = np.split(
            test_data["force"][:numb_test],
            indices_or_sections=[nloc_real * 3, nloc * 3],
            axis=1,
        )[1]

    diff_e = energy - test_data["energy"][:numb_test].reshape([-1, 1])
    mae_e = mae(diff_e)
    rmse_e = rmse(diff_e)
    diff_f = force - test_data["force"][:numb_test]
    mae_f = mae(diff_f)
    rmse_f = rmse(diff_f)
    diff_v = virial - test_data["virial"][:numb_test]
    mae_v = mae(diff_v)
    rmse_v = rmse(diff_v)
    mae_ea = mae_e / natoms
    rmse_ea = rmse_e / natoms
    mae_va = mae_v / natoms
    rmse_va = rmse_v / natoms
    if has_atom_ener:
        diff_ae = test_data["atom_ener"][:numb_test].reshape([-1]) - ae.reshape([-1])
        mae_ae = mae(diff_ae)
        rmse_ae = rmse(diff_ae)
    if dp.get_ntypes_spin() != 0:
        mae_fr = mae(force_r - test_force_r)
        mae_fm = mae(force_m - test_force_m)
        rmse_fr = rmse(force_r - test_force_r)
        rmse_fm = rmse(force_m - test_force_m)

    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Energy MAE         : {mae_e:e} eV")
    log.info(f"Energy RMSE        : {rmse_e:e} eV")
    log.info(f"Energy MAE/Natoms  : {mae_ea:e} eV")
    log.info(f"Energy RMSE/Natoms : {rmse_ea:e} eV")
    if dp.get_ntypes_spin() == 0:
        log.info(f"Force  MAE         : {mae_f:e} eV/A")
        log.info(f"Force  RMSE        : {rmse_f:e} eV/A")
    else:
        log.info(f"Force atom MAE      : {mae_fr:e} eV/A")
        log.info(f"Force spin MAE      : {mae_fm:e} eV/uB")
        log.info(f"Force atom RMSE     : {rmse_fr:e} eV/A")
        log.info(f"Force spin RMSE     : {rmse_fm:e} eV/uB")

    if data.pbc:
        log.info(f"Virial MAE         : {mae_v:e} eV")
        log.info(f"Virial RMSE        : {rmse_v:e} eV")
        log.info(f"Virial MAE/Natoms  : {mae_va:e} eV")
        log.info(f"Virial RMSE/Natoms : {rmse_va:e} eV")
    if has_atom_ener:
        log.info(f"Atomic ener MAE    : {mae_ae:e} eV")
        log.info(f"Atomic ener RMSE   : {rmse_ae:e} eV")

    if detail_file is not None:
        detail_path = Path(detail_file)

        pe = np.concatenate(
            (
                np.reshape(test_data["energy"][:numb_test], [-1, 1]),
                np.reshape(energy, [-1, 1]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".e.out"),
            pe,
            header="%s: data_e pred_e" % system,
            append=append_detail,
        )
        pe_atom = pe / natoms
        save_txt_file(
            detail_path.with_suffix(".e_peratom.out"),
            pe_atom,
            header="%s: data_e pred_e" % system,
            append=append_detail,
        )
        if dp.get_ntypes_spin() == 0:
            pf = np.concatenate(
                (
                    np.reshape(test_data["force"][:numb_test], [-1, 3]),
                    np.reshape(force, [-1, 3]),
                ),
                axis=1,
            )
            save_txt_file(
                detail_path.with_suffix(".f.out"),
                pf,
                header="%s: data_fx data_fy data_fz pred_fx pred_fy pred_fz" % system,
                append=append_detail,
            )
        else:
            pf_real = np.concatenate(
                (np.reshape(test_force_r, [-1, 3]), np.reshape(force_r, [-1, 3])),
                axis=1,
            )
            pf_mag = np.concatenate(
                (np.reshape(test_force_m, [-1, 3]), np.reshape(force_m, [-1, 3])),
                axis=1,
            )
            save_txt_file(
                detail_path.with_suffix(".fr.out"),
                pf_real,
                header="%s: data_fx data_fy data_fz pred_fx pred_fy pred_fz" % system,
                append=append_detail,
            )
            save_txt_file(
                detail_path.with_suffix(".fm.out"),
                pf_mag,
                header="%s: data_fmx data_fmy data_fmz pred_fmx pred_fmy pred_fmz"
                % system,
                append=append_detail,
            )
        pv = np.concatenate(
            (
                np.reshape(test_data["virial"][:numb_test], [-1, 9]),
                np.reshape(virial, [-1, 9]),
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
    if dp.get_ntypes_spin() == 0:
        return {
            "mae_e": (mae_e, energy.size),
            "mae_ea": (mae_ea, energy.size),
            "mae_f": (mae_f, force.size),
            "mae_v": (mae_v, virial.size),
            "mae_va": (mae_va, virial.size),
            "rmse_e": (rmse_e, energy.size),
            "rmse_ea": (rmse_ea, energy.size),
            "rmse_f": (rmse_f, force.size),
            "rmse_v": (rmse_v, virial.size),
            "rmse_va": (rmse_va, virial.size),
        }
    else:
        return {
            "mae_e": (mae_e, energy.size),
            "mae_ea": (mae_ea, energy.size),
            "mae_fr": (mae_fr, force_r.size),
            "mae_fm": (mae_fm, force_m.size),
            "mae_v": (mae_v, virial.size),
            "mae_va": (mae_va, virial.size),
            "rmse_e": (rmse_e, energy.size),
            "rmse_ea": (rmse_ea, energy.size),
            "rmse_fr": (rmse_fr, force_r.size),
            "rmse_fm": (rmse_fm, force_m.size),
            "rmse_v": (rmse_v, virial.size),
            "rmse_va": (rmse_va, virial.size),
        }


def print_ener_sys_avg(avg: Dict[str, float]):
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
    if "rmse_f" in avg.keys():
        log.info(f"Force  MAE         : {avg['mae_f']:e} eV/A")
        log.info(f"Force  RMSE        : {avg['rmse_f']:e} eV/A")
    else:
        log.info(f"Force atom MAE      : {avg['mae_fr']:e} eV/A")
        log.info(f"Force spin MAE      : {avg['mae_fm']:e} eV/uB")
        log.info(f"Force atom RMSE     : {avg['rmse_fr']:e} eV/A")
        log.info(f"Force spin RMSE     : {avg['rmse_fm']:e} eV/uB")
    log.info(f"Virial MAE         : {avg['mae_v']:e} eV")
    log.info(f"Virial RMSE        : {avg['rmse_v']:e} eV")
    log.info(f"Virial MAE/Natoms  : {avg['mae_va']:e} eV")
    log.info(f"Virial RMSE/Natoms : {avg['rmse_va']:e} eV")


def test_dos(
    dp: "DeepDOS",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: Optional[str],
    has_atom_dos: bool,
    append_detail: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
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
    Tuple[List[np.ndarray], List[int]]
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
                detail_path.with_suffix(".dos.out.%.d" % ii),
                frame_output,
                header="%s - %.d: data_dos pred_dos" % (system, ii),
                append=append_detail,
            )

        if has_atom_dos:
            for ii in range(numb_test):
                test_out = test_data["atom_dos"][ii].reshape(-1, 1)
                pred_out = ados[ii].reshape(-1, 1)

                frame_output = np.hstack((test_out, pred_out))

                save_txt_file(
                    detail_path.with_suffix(".ados.out.%.d" % ii),
                    frame_output,
                    header="%s - %.d: data_ados pred_ados" % (system, ii),
                    append=append_detail,
                )

    return {
        "mae_dos": (mae_dos, dos.size),
        "mae_dosa": (mae_dosa, dos.size),
        "rmse_dos": (rmse_dos, dos.size),
        "rmse_dosa": (rmse_dosa, dos.size),
    }


def print_dos_sys_avg(avg: Dict[str, float]):
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


def run_test(dp: "DeepTensor", test_data: dict, numb_test: int, test_sys: DeepmdData):
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
    detail_file: Optional[str],
) -> Tuple[List[np.ndarray], List[int]]:
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
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add(
        "wfc", 12, atomic=True, must=True, high_prec=False, type_sel=dp.get_sel_type()
    )
    test_data = data.get_test()
    wfc, numb_test, _ = run_test(dp, test_data, numb_test, data)
    rmse_f = rmse(wfc - test_data["wfc"][:numb_test])

    log.info("# number of test data : {numb_test:d} ")
    log.info("WFC  RMSE : {rmse_f:e} eV/A")

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


def print_wfc_sys_avg(avg):
    """Print errors summary for wfc type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"WFC  RMSE : {avg['rmse']:e} eV/A")


def test_polar(
    dp: "DeepPolar",
    data: DeepmdData,
    numb_test: int,
    detail_file: Optional[str],
    *,
    atomic: bool,
) -> Tuple[List[np.ndarray], List[int]]:
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
        wheter to use glovbal version of polar potential

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add(
        "polarizability" if not atomic else "atomic_polarizability",
        9,
        atomic=atomic,
        must=True,
        high_prec=False,
        type_sel=dp.get_sel_type(),
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
        rmse_f = rmse(polar - test_data["atomic_polarizability"][:numb_test])

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
                    np.reshape(
                        test_data["atomic_polarizability"][:numb_test],
                        [-1, 9 * sel_natoms],
                    ),
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


def print_polar_sys_avg(avg):
    """Print errors summary for polar type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Polarizability  RMSE : {avg['rmse']:e} eV/A")


def test_dipole(
    dp: "DeepDipole",
    data: DeepmdData,
    numb_test: int,
    detail_file: Optional[str],
    atomic: bool,
) -> Tuple[List[np.ndarray], List[int]]:
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
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add(
        "dipole" if not atomic else "atomic_dipole",
        3,
        atomic=atomic,
        must=True,
        high_prec=False,
        type_sel=dp.get_sel_type(),
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
        rmse_f = rmse(dipole - test_data["atomic_dipole"][:numb_test])

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
                    np.reshape(
                        test_data["atomic_dipole"][:numb_test], [-1, 3 * sel_natoms]
                    ),
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


def print_dipole_sys_avg(avg):
    """Print errors summary for dipole type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Dipole  RMSE         : {avg['rmse']:e} eV/A")
