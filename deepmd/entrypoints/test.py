"""Test trained DeePMD model."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from deepmd import DeepPotential
from deepmd.common import expand_sys_str
from deepmd.utils.data import DeepmdData

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
            err, siz = test_ener(
                dp,
                data,
                system,
                numb_test,
                detail_file,
                atomic,
                append_detail=(cc != 0),
            )
        elif dp.model_type == "dipole":
            err, siz = test_dipole(dp, data, numb_test, detail_file, atomic)
        elif dp.model_type == "polar":
            err, siz = test_polar(dp, data, numb_test, detail_file, global_polar=False)
        elif dp.model_type == "global_polar":
            err, siz = test_polar(dp, data, numb_test, detail_file, global_polar=True)
        elif dp.model_type == "wfc":
            err, siz = test_wfc(dp, data, numb_test, detail_file)
        log.info("# ----------------------------------------------- ")
        err_coll.append(err)
        siz_coll.append(siz)

    avg_err = weighted_average(err_coll, siz_coll)

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


def l2err(diff: np.ndarray) -> np.ndarray:
    """Calculate average l2 norm error.

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


def weighted_average(
    err_coll: List[List[np.ndarray]], siz_coll: List[List[int]]
) -> np.ndarray:
    """Compute wighted average of prediction errors for model.

    Parameters
    ----------
    err_coll : List[List[np.ndarray]]
        each item in list represents erros for one model
    siz_coll : List[List[int]]
        weight for each model errors

    Returns
    -------
    np.ndarray
        weighted averages
    """
    assert len(err_coll) == len(siz_coll)

    nitems = len(err_coll[0])
    sum_err = np.zeros(nitems)
    sum_siz = np.zeros(nitems)
    for sys_error, sys_size in zip(err_coll, siz_coll):
        for ii in range(nitems):
            ee = sys_error[ii]
            ss = sys_size[ii]
            sum_err[ii] += ee * ee * ss
            sum_siz[ii] += ss
    for ii in range(nitems):
        sum_err[ii] = np.sqrt(sum_err[ii] / sum_siz[ii])
    return sum_err


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

    l2e = l2err(energy - test_data["energy"][:numb_test].reshape([-1, 1]))
    l2f = l2err(force - test_data["force"][:numb_test])
    l2v = l2err(virial - test_data["virial"][:numb_test])
    l2ea = l2e / natoms
    l2va = l2v / natoms
    if has_atom_ener:
        l2ae = l2err(
            test_data["atom_ener"][:numb_test].reshape([-1]) - ae.reshape([-1])
        )

    # print ("# energies: %s" % energy)
    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Energy RMSE        : {l2e:e} eV")
    log.info(f"Energy RMSE/Natoms : {l2ea:e} eV")
    log.info(f"Force  RMSE        : {l2f:e} eV/A")
    log.info(f"Virial RMSE        : {l2v:e} eV")
    log.info(f"Virial RMSE/Natoms : {l2va:e} eV")
    if has_atom_ener:
        log.info(f"Atomic ener RMSE   : {l2ae:e} eV")

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
    return [l2ea, l2f, l2va], [energy.size, force.size, virial.size]


def print_ener_sys_avg(avg: np.ndarray):
    """Print errors summary for energy type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Energy RMSE/Natoms : {avg[0]:e} eV")
    log.info(f"Force  RMSE        : {avg[1]:e} eV/A")
    log.info(f"Virial RMSE/Natoms : {avg[2]:e} eV")


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
    l2f = l2err(wfc - test_data["wfc"][:numb_test])

    log.info("# number of test data : {numb_test:d} ")
    log.info("WFC  RMSE : {l2f:e} eV/A")

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
    return [l2f], [wfc.size]


def print_wfc_sys_avg(avg):
    """Print errors summary for wfc type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"WFC  RMSE : {avg[0]:e} eV/A")


def test_polar(
    dp: "DeepPolar",
    data: DeepmdData,
    numb_test: int,
    detail_file: Optional[str],
    *,
    global_polar: bool,
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
    if not global_polar:
        data.add(
            "polarizability",
            9,
            atomic=True,
            must=True,
            high_prec=False,
            type_sel=dp.get_sel_type(),
        )
    else:
        data.add(
            "polarizability",
            9,
            atomic=False,
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

    l2f = l2err(polar - test_data["polarizability"][:numb_test])
    l2fs = l2f / np.sqrt(sel_natoms)
    l2fa = l2f / sel_natoms

    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Polarizability  RMSE       : {l2f:e} eV/A")
    if global_polar:
        log.info(f"Polarizability  RMSE/sqrtN : {l2fs:e} eV/A")
        log.info(f"Polarizability  RMSE/N     : {l2fa:e} eV/A")

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
    return [l2f], [polar.size]


def print_polar_sys_avg(avg):
    """Print errors summary for polar type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Polarizability  RMSE : {avg[0]:e} eV/A")


def test_dipole(
    dp: "DeepDipole",
    data: DeepmdData,
    numb_test: int,
    detail_file: Optional[str],
    has_atom_dipole: bool,
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
    has_atom_dipole : bool
        whether atomic dipole is provided

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add(
        "dipole", 3, atomic=has_atom_dipole, must=True, high_prec=False, type_sel=dp.get_sel_type()
    )
    test_data = data.get_test()
    dipole, numb_test, _ = run_test(dp, test_data, numb_test)

    # do summation in atom dimension
    if has_atom_dipole == False:
        dipole = np.reshape(dipole,(dipole.shape[0], -1, 3))
        atoms = dipole.shape[1]
        dipole = np.sum(dipole,axis=1)

    l2f = l2err(dipole - test_data["dipole"][:numb_test])

    if has_atom_dipole == False:
        l2f = l2f / atoms

    log.info(f"# number of test data : {numb_test:d}")
    log.info(f"Dipole  RMSE         : {l2f:e} eV/A")

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
    return [l2f], [dipole.size]


def print_dipole_sys_avg(avg):
    """Print errors summary for dipole type potential.

    Parameters
    ----------
    avg : np.ndarray
        array with summaries
    """
    log.info(f"Dipole  RMSE         : {avg[0]:e} eV/A")
