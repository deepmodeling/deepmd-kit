"""Test trained DeePMD model."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple

import numpy as np
from deepmd import DeepPotential
from deepmd.common import expand_sys_str
from deepmd.utils.data import DeepmdData
from deepmd.utils.weight_avg import weighted_average

if TYPE_CHECKING:
    from deepmd.infer import DeepDipole, DeepPolar, DeepPot, DeepWFC
    from deepmd.infer.deep_eval import DeepTensor

__all__ = ["test"]

log = logging.getLogger(__name__)


def test(
    *,
    model: str,
    system: str,
    set_prefix: str,
    numb_test: int,
    rand_seed: Optional[int],
    shuffle_test: bool,
    detail_file: str,
    atomic: bool,
    **kwargs,
):
    """Test model predictions.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str
        system directory
    set_prefix : str
        string prefix of set
    numb_test : int
        munber of tests to do
    rand_seed : Optional[int]
        seed for random generator
    shuffle_test : bool
        whether to shuffle tests
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether per atom quantities should be computed

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    all_sys = expand_sys_str(system)
    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")
    err_coll = []
    siz_coll = []

    # init random seed
    if rand_seed is not None:
        np.random.seed(rand_seed % (2 ** 32))

    # init model
    dp = DeepPotential(model)

    for cc, system in enumerate(all_sys):
        log.info("# ---------------output of dp test--------------- ")
        log.info(f"# testing system : {system}")

        # create data class
        tmap = dp.get_type_map() if dp.model_type == "ener" else None
        data = DeepmdData(system, set_prefix, shuffle_test=shuffle_test, type_map=tmap)

        if dp.model_type == "ener":
            err = test_ener(
                dp,
                data,
                system,
                numb_test,
                detail_file,
                atomic,
                append_detail=(cc != 0),
            )
        elif dp.model_type == "dipole":
            err = test_dipole(dp, data, numb_test, detail_file, atomic)
        elif dp.model_type == "polar":
            err = test_polar(dp, data, numb_test, detail_file, atomic=atomic)
        elif dp.model_type == "global_polar":   # should not appear in this new version
            log.warning("Global polar model is not currently supported. Please directly use the polar mode and change loss parameters.")
            err = test_polar(dp, data, numb_test, detail_file, atomic=False)    # YWolfeee: downward compatibility
        log.info("# ----------------------------------------------- ")
        err_coll.append(err)

    avg_err = weighted_average(err_coll)

    if len(all_sys) != len(err_coll):
        log.warning("Not all systems are tested! Check if the systems are valid")

    if len(all_sys) > 1:
        log.info("# ----------weighted average of errors----------- ")
        log.info(f"# number of systems : {len(all_sys)}")
        if dp.model_type == "ener":
            print_ener_sys_avg(avg_err)
        elif dp.model_type == "dipole":
            print_dipole_sys_avg(avg_err)
        elif dp.model_type == "polar":
            print_polar_sys_avg(avg_err)
        elif dp.model_type == "global_polar":
            print_polar_sys_avg(avg_err)
        elif dp.model_type == "wfc":
            print_wfc_sys_avg(avg_err)
        log.info("# ----------------------------------------------- ")


def rmse(diff: np.ndarray) -> np.ndarray:
    """Calculate average root mean square error.

    Parameters
    ----------
    diff: np.ndarray
        difference

    Returns
    -------
    np.ndarray
        array with normalized difference
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
    data: DeepmdData
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

    rmse_e = rmse(energy - test_data["energy"][:numb_test].reshape([-1, 1]))
    rmse_f = rmse(force - test_data["force"][:numb_test])
    rmse_v = rmse(virial - test_data["virial"][:numb_test])
    rmse_ea = rmse_e / natoms
    rmse_va = rmse_v / natoms
    if has_atom_ener:
        rmse_ae = rmse(
            test_data["atom_ener"][:numb_test].reshape([-1]) - ae.reshape([-1])
        )

    # print ("# energies: %s" % energy)
    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Energy RMSE        : {rmse_e:e} eV")
    log.info(f"Energy RMSE/Natoms : {rmse_ea:e} eV")
    log.info(f"Force  RMSE        : {rmse_f:e} eV/A")
    log.info(f"Virial RMSE        : {rmse_v:e} eV")
    log.info(f"Virial RMSE/Natoms : {rmse_va:e} eV")
    if has_atom_ener:
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
    return {
        "rmse_ea" : (rmse_ea, energy.size),
        "rmse_f" : (rmse_f, force.size),
        "rmse_va" : (rmse_va, virial.size),
    }


def print_ener_sys_avg(avg: Dict[str,float]):
    """Print errors summary for energy type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Energy RMSE/Natoms : {avg['rmse_ea']:e} eV")
    log.info(f"Force  RMSE        : {avg['rmse_f']:e} eV/A")
    log.info(f"Virial RMSE/Natoms : {avg['rmse_va']:e} eV")


def run_test(dp: "DeepTensor", test_data: dict, numb_test: int):
    """Run tests.

    Parameters
    ----------
    dp : DeepTensor
        instance of deep potential
    test_data : dict
        dictionary with test data
    numb_test : int
        munber of tests to do

    Returns
    -------
    [type]
        [description]
    """
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
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
    data: DeepmdData
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
    wfc, numb_test, _ = run_test(dp, test_data, numb_test)
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
    return {
        'rmse' : (rmse_f, wfc.size)
    }


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
    data: DeepmdData
        data container object
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    global_polar : bool
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
    polar, numb_test, atype = run_test(dp, test_data, numb_test)

    sel_type = dp.get_sel_type()
    sel_natoms = 0
    for ii in sel_type:
        sel_natoms += sum(atype == ii)

    # YWolfeee: do summation in global polar mode
    if not atomic:
        polar = np.sum(polar.reshape((polar.shape[0],-1,9)),axis=1)    
        rmse_f = rmse(polar - test_data["polarizability"][:numb_test])
        rmse_fs = rmse_f / np.sqrt(sel_natoms)
        rmse_fa = rmse_f / sel_natoms
    else:
        rmse_f = rmse(polar - test_data["atomic_polarizability"][:numb_test])
    
    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Polarizability  RMSE       : {rmse_f:e}")
    if not atomic:
        log.info(f"Polarizability  RMSE/sqrtN : {rmse_fs:e}")
        log.info(f"Polarizability  RMSE/N     : {rmse_fa:e}")
    log.info(f"The unit of error is the same as the unit of provided label.")

    if detail_file is not None:
        detail_path = Path(detail_file)

        pe = np.concatenate(
            (
                np.reshape(test_data["polarizability"][:numb_test], [-1, 9]),
                np.reshape(polar, [-1, 9]),
            ),
            axis=1,
        )
        np.savetxt(
            detail_path.with_suffix(".out"),
            pe,
            header="data_pxx data_pxy data_pxz data_pyx data_pyy data_pyz data_pzx "
            "data_pzy data_pzz pred_pxx pred_pxy pred_pxz pred_pyx pred_pyy pred_pyz "
            "pred_pzx pred_pzy pred_pzz",
        )
    return {
        "rmse" : (rmse_f, polar.size)
    }


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
    data: DeepmdData
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
        type_sel=dp.get_sel_type()
    )
    test_data = data.get_test()
    dipole, numb_test, atype = run_test(dp, test_data, numb_test)

    sel_type = dp.get_sel_type()
    sel_natoms = 0
    for ii in sel_type:
        sel_natoms += sum(atype == ii)
    
    # do summation in atom dimension
    if not atomic:
        dipole = np.sum(dipole.reshape((dipole.shape[0], -1, 3)),axis=1)
        rmse_f = rmse(dipole - test_data["dipole"][:numb_test])
        rmse_fs = rmse_f / np.sqrt(sel_natoms)
        rmse_fa = rmse_f / sel_natoms
    else:
        rmse_f = rmse(dipole - test_data["atomic_dipole"][:numb_test])
    
    log.info(f"# number of test data : {numb_test:d}")
    log.info(f"Dipole  RMSE       : {rmse_f:e}")
    if not atomic:
        log.info(f"Dipole  RMSE/sqrtN : {rmse_fs:e}")
        log.info(f"Dipole  RMSE/N     : {rmse_fa:e}")
    log.info(f"The unit of error is the same as the unit of provided label.")

    if detail_file is not None:
        detail_path = Path(detail_file)

        pe = np.concatenate(
            (
                np.reshape(test_data["dipole"][:numb_test], [-1, 3]),
                np.reshape(dipole, [-1, 3]),
            ),
            axis=1,
        )
        np.savetxt(
            detail_path.with_suffix(".out"),
            pe,
            header="data_x data_y data_z pred_x pred_y pred_z",
        )
    return {
        'rmse' : (rmse_f, dipole.size)
    }


def print_dipole_sys_avg(avg):
    """Print errors summary for dipole type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Dipole  RMSE         : {avg['rmse']:e} eV/A")
