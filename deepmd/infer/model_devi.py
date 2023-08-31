# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Tuple,
)

import numpy as np

from deepmd.common import (
    expand_sys_str,
)

from ..utils.batch_size import (
    AutoBatchSize,
)
from ..utils.data import (
    DeepmdData,
)
from .deep_pot import (
    DeepPot,
)


def calc_model_devi_f(
    fs: np.ndarray, real_f: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    """Calculate model deviation of force.

    Parameters
    ----------
    fs : numpy.ndarray
        size of `n_models x n_frames x n_atoms x 3`
    real_f : numpy.ndarray or None
        real force, size of `n_frames x n_atoms x 3`. If given,
        the RMS real error is calculated instead.

    Returns
    -------
    max_devi_f : numpy.ndarray
        maximum deviation of force in all atoms
    min_devi_f : numpy.ndarray
        minimum deviation of force in all atoms
    avg_devi_f : numpy.ndarray
        average deviation of force in all atoms
    """
    if real_f is None:
        fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    else:
        fs_devi = np.linalg.norm(
            np.sqrt(np.mean(np.square(fs - real_f), axis=0)), axis=-1
        )
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f


def calc_model_devi_e(
    es: np.ndarray, real_e: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate model deviation of total energy per atom.

    Here we don't use the atomic energy, as the decomposition
    of energy is arbitrary and not unique. There is no fitting
    target for atomic energy.

    Parameters
    ----------
    es : numpy.ndarray
        size of `n_models x n_frames x 1
    real_e : numpy.ndarray
        real energy, size of `n_frames x 1`. If given,
        the RMS real error is calculated instead.

    Returns
    -------
    max_devi_e : numpy.ndarray
        maximum deviation of energy
    """
    if real_e is None:
        es_devi = np.std(es, axis=0)
    else:
        es_devi = np.sqrt(np.mean(np.square(es - real_e), axis=0))
    es_devi = np.squeeze(es_devi, axis=-1)
    return es_devi


def calc_model_devi_v(
    vs: np.ndarray, real_v: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    """Calculate model deviation of virial.

    Parameters
    ----------
    vs : numpy.ndarray
        size of `n_models x n_frames x 9`
    real_v : numpy.ndarray
        real virial, size of `n_frames x 9`. If given,
        the RMS real error is calculated instead.

    Returns
    -------
    max_devi_v : numpy.ndarray
        maximum deviation of virial in 9 elements
    min_devi_v : numpy.ndarray
        minimum deviation of virial in 9 elements
    avg_devi_v : numpy.ndarray
        average deviation of virial in 9 elements
    """
    if real_v is None:
        vs_devi = np.std(vs, axis=0)
    else:
        vs_devi = np.sqrt(np.mean(np.square(vs - real_v), axis=0))
    max_devi_v = np.max(vs_devi, axis=-1)
    min_devi_v = np.min(vs_devi, axis=-1)
    avg_devi_v = np.linalg.norm(vs_devi, axis=-1) / 3
    return max_devi_v, min_devi_v, avg_devi_v


def write_model_devi_out(devi: np.ndarray, fname: str, header: str = ""):
    """Write output of model deviation.

    Parameters
    ----------
    devi : numpy.ndarray
        the first column is the steps index
    fname : str
        the file name to dump
    header : str, default=""
        the header to dump
    """
    assert devi.shape[1] == 8
    header = "%s\n%10s" % (header, "step")
    for item in "vf":
        header += "%19s%19s%19s" % (
            f"max_devi_{item}",
            f"min_devi_{item}",
            f"avg_devi_{item}",
        )
    header += "%19s" % "devi_e"
    with open(fname, "ab") as fp:
        np.savetxt(
            fp,
            devi,
            fmt=["%12d"] + ["%19.6e" for _ in range(7)],
            delimiter="",
            header=header,
        )
    return devi


def _check_tmaps(tmaps, ref_tmap=None):
    """Check whether type maps are identical."""
    assert isinstance(tmaps, list)
    if ref_tmap is None:
        ref_tmap = tmaps[0]
    assert isinstance(ref_tmap, list)

    flag = True
    for tmap in tmaps:
        if tmap != ref_tmap:
            flag = False
            break
    return flag


def calc_model_devi(
    coord,
    box,
    atype,
    models,
    fname=None,
    frequency=1,
    mixed_type=False,
    fparam: Optional[np.ndarray] = None,
    aparam: Optional[np.ndarray] = None,
    real_data: Optional[dict] = None,
):
    """Python interface to calculate model deviation.

    Parameters
    ----------
    coord : numpy.ndarray, `n_frames x n_atoms x 3`
        Coordinates of system to calculate
    box : numpy.ndarray or None, `n_frames x 3 x 3`
        Box to specify periodic boundary condition. If None, no pbc will be used
    atype : numpy.ndarray, `n_atoms x 1`
        Atom types
    models : list of DeepPot models
        Models used to evaluate deviation
    fname : str or None
        File to dump results, default None
    frequency : int
        Steps between frames (if the system is given by molecular dynamics engine), default 1
    mixed_type : bool
        Whether the input atype is in mixed_type format or not
    fparam : numpy.ndarray
        frame specific parameters
    aparam : numpy.ndarray
        atomic specific parameters
    real_data : dict, optional
        real data to calculate RMS real error

    Returns
    -------
    model_devi : numpy.ndarray, `n_frames x 8`
        Model deviation results. The first column is index of steps, the other 7 columns are
        max_devi_v, min_devi_v, avg_devi_v, max_devi_f, min_devi_f, avg_devi_f, devi_e.

    Examples
    --------
    >>> from deepmd.infer import calc_model_devi
    >>> from deepmd.infer import DeepPot as DP
    >>> import numpy as np
    >>> coord = np.array([[1,0,0], [0,0,1.5], [1,0,3]]).reshape([1, -1])
    >>> cell = np.diag(10 * np.ones(3)).reshape([1, -1])
    >>> atype = [1,0,1]
    >>> graphs = [DP("graph.000.pb"), DP("graph.001.pb")]
    >>> model_devi = calc_model_devi(coord, cell, atype, graphs)
    """
    energies = []
    forces = []
    virials = []
    natom = atype.shape[-1]
    for dp in models:
        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
        )
        energies.append(ret[0] / natom)
        forces.append(ret[1])
        virials.append(ret[2] / natom)

    energies = np.array(energies)
    forces = np.array(forces)
    virials = np.array(virials)

    devi = [np.arange(coord.shape[0]) * frequency]
    if real_data is None:
        devi += list(calc_model_devi_v(virials))
        devi += list(calc_model_devi_f(forces))
        devi.append(calc_model_devi_e(energies))
    else:
        devi += list(calc_model_devi_v(virials, real_data["virial"]))
        devi += list(calc_model_devi_f(forces, real_data["force"]))
        devi.append(calc_model_devi_e(energies, real_data["energy"]))
    devi = np.vstack(devi).T
    if fname:
        write_model_devi_out(devi, fname)
    return devi


def make_model_devi(
    *,
    models: list,
    system: str,
    set_prefix: str,
    output: str,
    frequency: int,
    real_error: bool = False,
    **kwargs,
):
    """Make model deviation calculation.

    Parameters
    ----------
    models : list
        A list of paths of models to use for making model deviation
    system : str
        The path of system to make model deviation calculation
    set_prefix : str
        The set prefix of the system
    output : str
        The output file for model deviation results
    frequency : int
        The number of steps that elapse between writing coordinates
        in a trajectory by a MD engine (such as Gromacs / Lammps).
        This paramter is used to determine the index in the output file.
    real_error : bool, default: False
        If True, calculate the RMS real error instead of model deviation.
    **kwargs
        Arbitrary keyword arguments.
    """
    auto_batch_size = AutoBatchSize()
    # init models
    dp_models = [DeepPot(model, auto_batch_size=auto_batch_size) for model in models]

    # check type maps
    tmaps = [dp.get_type_map() for dp in dp_models]
    if _check_tmaps(tmaps):
        tmap = tmaps[0]
    else:
        raise RuntimeError("The models does not have the same type map.")

    all_sys = expand_sys_str(system)
    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")
    devis_coll = []

    first_dp = dp_models[0]

    for system in all_sys:
        # create data-system
        dp_data = DeepmdData(system, set_prefix, shuffle_test=False, type_map=tmap)
        if first_dp.get_dim_fparam() > 0:
            dp_data.add(
                "fparam",
                first_dp.get_dim_fparam(),
                atomic=False,
                must=True,
                high_prec=False,
            )
        if first_dp.get_dim_aparam() > 0:
            dp_data.add(
                "aparam",
                first_dp.get_dim_aparam(),
                atomic=True,
                must=True,
                high_prec=False,
            )
        if real_error:
            dp_data.add(
                "energy",
                1,
                atomic=False,
                must=False,
                high_prec=True,
            )
            dp_data.add(
                "force",
                3,
                atomic=True,
                must=False,
                high_prec=False,
            )
            dp_data.add(
                "virial",
                9,
                atomic=False,
                must=False,
                high_prec=False,
            )

        mixed_type = dp_data.mixed_type

        data_sets = [dp_data._load_set(set_name) for set_name in dp_data.dirs]
        nframes_tot = 0
        devis = []
        for data in data_sets:
            coord = data["coord"]
            box = data["box"]
            if mixed_type:
                atype = data["type"]
            else:
                atype = data["type"][0]
            if not dp_data.pbc:
                box = None
            if first_dp.get_dim_fparam() > 0:
                fparam = data["fparam"]
            else:
                fparam = None
            if first_dp.get_dim_aparam() > 0:
                aparam = data["aparam"]
            else:
                aparam = None
            if real_error:
                natoms = atype.shape[-1]
                real_data = {
                    "energy": data["energy"] / natoms,
                    "force": data["force"].reshape([-1, natoms, 3]),
                    "virial": data["virial"] / natoms,
                }
            else:
                real_data = None
            devi = calc_model_devi(
                coord,
                box,
                atype,
                dp_models,
                mixed_type=mixed_type,
                fparam=fparam,
                aparam=aparam,
                real_data=real_data,
            )
            nframes_tot += coord.shape[0]
            devis.append(devi)
        devis = np.vstack(devis)
        devis[:, 0] = np.arange(nframes_tot) * frequency
        write_model_devi_out(devis, output, header=system)
        devis_coll.append(devis)
    return devis_coll
