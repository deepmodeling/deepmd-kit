import numpy as np
from .deep_pot import DeepPot
from ..utils.data import DeepmdData
        

def calc_model_devi_f(fs):
    '''
    fs : numpy.ndarray, size of `n_models x n_frames x n_atoms x 3`
    '''
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f

def calc_model_devi_e(es):
    '''
    es : numpy.ndarray, size of `n_models x n_frames x n_atoms
    '''
    es_devi = np.std(es, axis=0)
    max_devi_e = np.max(es_devi, axis=1)
    min_devi_e = np.min(es_devi, axis=1)
    avg_devi_e = np.mean(es_devi, axis=1)
    return max_devi_e, min_devi_e, avg_devi_e

def calc_model_devi_v(vs):
    '''
    vs : numpy.ndarray, size of `n_models x n_frames x 9`
    '''
    vs_devi = np.std(vs, axis=0)
    max_devi_v = np.max(vs_devi, axis=-1)
    min_devi_v = np.min(vs_devi, axis=-1)
    avg_devi_v = np.linalg.norm(vs_devi, axis=-1) / 3
    return max_devi_v, min_devi_v, avg_devi_v

def write_model_devi_out(devi, fname):
    '''
    devi : numpy.ndarray, the first column is the steps index
    fname : str, the file name to dump
    '''
    assert devi.shape[1] == 7
    header = "%10s" % "step"
    for item in 'vf':
        header += "%19s%19s%19s" % (f"max_devi_{item}", f"min_devi_{item}", f"avg_devi_{item}")
    np.savetxt(fname,
               devi,
               fmt=['%12d'] + ['%19.6e' for _ in range(6)],
               delimiter='',
               header=header)
    return devi

def _check_tmaps(tmaps, ref_tmap=None):
    '''
    Check whether type maps are identical
    '''
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

def calc_model_devi(coord,
                    box,
                    atype,
                    models,
                    fname=None,
                    frequency=1, 
                    nopbc=True):
    '''
    Python interface to calculate model deviation

    Parameters:
    -----------
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
    nopbc : bool
        Whether to use pbc conditions
    
    Return:
    -------
    model_devi : numpy.ndarray, `n_frames x 7`
        Model deviation results. The first column is index of steps, the other 6 columns are
        max_devi_v, min_devi_v, avg_devi_v, max_devi_f, min_devi_f, avg_devi_f.
    '''
    if nopbc:
        box = None

    forces = []
    virials = []
    for dp in models:
        ret = dp.eval(
            coord,
            box,
            atype,
        )
        forces.append(ret[1])
        virials.append(ret[2] / len(atype))
    
    forces = np.array(forces)
    virials = np.array(virials)
    
    devi = [np.arange(coord.shape[0]) * frequency]
    devi += list(calc_model_devi_v(virials))
    devi += list(calc_model_devi_f(forces))
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
    **kwargs
):
    '''
    Make model deviation calculation

    Parameters
    ----------

    models: list
        A list of paths of models to use for making model deviation
    system: str
        The path of system to make model deviation calculation
    set_prefix: str
        The set prefix of the system
    output: str
        The output file for model deviation results
    frequency: int
        The number of steps that elapse between writing coordinates 
        in a trajectory by a MD engine (such as Gromacs / Lammps).
        This paramter is used to determine the index in the output file.
    '''
    # init models
    dp_models = [DeepPot(model) for model in models]

    # check type maps
    tmaps = [dp.get_type_map() for dp in dp_models]
    if _check_tmaps(tmaps):
        tmap = tmaps[0]
    else:
        raise RuntimeError("The models does not have the same type map.")
    
    # create data-system
    dp_data = DeepmdData(system, set_prefix, shuffle_test=False, type_map=tmap)
    if dp_data.pbc:
        nopbc = False
    else:
        nopbc = True

    data_sets = [dp_data._load_set(set_name) for set_name in dp_data.dirs]
    nframes_tot = 0
    devis = []
    for data in data_sets:
        coord = data["coord"]
        box = data["box"]
        atype = data["type"][0] 
        devi = calc_model_devi(coord, box, atype, dp_models, nopbc=nopbc)
        nframes_tot += coord.shape[0]
        devis.append(devi)
    devis = np.vstack(devis)
    devis[:, 0] = np.arange(nframes_tot) * frequency
    write_model_devi_out(devis, output)
    return devis
