from typing import (
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


def calc_model_devi_f(fs: np.ndarray) -> Tuple[np.ndarray]:
    """Calculate model deviation of force.

    Parameters
    ----------
    fs : numpy.ndarray
        size of `n_models x n_frames x n_atoms x 3`

    Returns
    -------
    max_devi_f : numpy.ndarray
        maximum deviation of force in all atoms
    min_devi_f : numpy.ndarray
        minimum deviation of force in all atoms
    avg_devi_f : numpy.ndarray
        average deviation of force in all atoms
    """
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f


def calc_model_devi_e(es: np.ndarray) -> np.ndarray:
    """Calculate model deviation of total energy per atom.

    Here we don't use the atomic energy, as the decomposition
    of energy is arbitrary and not unique. There is no fitting
    target for atomic energy.

    Parameters
    ----------
    es : numpy.ndarray
        size of `n_models x n_frames x 1

    Returns
    -------
    max_devi_e : numpy.ndarray
        maximum deviation of energy
    """
    es_devi = np.std(es, axis=0)
    es_devi = np.squeeze(es_devi, axis=-1)
    return es_devi


def calc_model_devi_v(vs: np.ndarray) -> Tuple[np.ndarray]:
    """Calculate model deviation of virial.

    Parameters
    ----------
    vs : numpy.ndarray
        size of `n_models x n_frames x 9`

    Returns
    -------
    max_devi_v : numpy.ndarray
        maximum deviation of virial in 9 elements
    min_devi_v : numpy.ndarray
        minimum deviation of virial in 9 elements
    avg_devi_v : numpy.ndarray
        average deviation of virial in 9 elements
    """
    vs_devi = np.std(vs, axis=0)
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
            mixed_type=mixed_type,
        )
        energies.append(ret[0] / natom)
        forces.append(ret[1])
        virials.append(ret[2] / natom)

    energies = np.array(energies)
    forces = np.array(forces)
    virials = np.array(virials)

    devi = [np.arange(coord.shape[0]) * frequency]
    devi += list(calc_model_devi_v(virials))
    devi += list(calc_model_devi_f(forces))
    devi.append(calc_model_devi_e(energies))
    devi = np.vstack(devi).T
    if fname:
        write_model_devi_out(devi, fname)
    return devi


def make_model_devi(
    *, models: list, system: str, set_prefix: str, output: str, frequency: int, **kwargs
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
    for system in all_sys:
        # create data-system
        dp_data = DeepmdData(system, set_prefix, shuffle_test=False, type_map=tmap)
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
            devi = calc_model_devi(coord, box, atype, dp_models, mixed_type=mixed_type)
            nframes_tot += coord.shape[0]
            devis.append(devi)
        devis = np.vstack(devis)
        devis[:, 0] = np.arange(nframes_tot) * frequency
        write_model_devi_out(devis, output, header=system)
        devis_coll.append(devis)
    return devis_coll
