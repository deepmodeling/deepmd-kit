import numpy as np
from deepmd import DeepPotential
from deepmd.utils.data import DeepmdData


def calc_model_devi_f(fs):
    '''
        fs : numpy.ndarray, size of `n_models x n_frames x n_atoms x 3`
    '''
    fs_mean = np.mean(fs, axis=0)
    fs_err = np.sum((fs - fs_mean) ** 2, axis=-1)
    fs_devi = np.mean(fs_err, axis=0) ** 0.5
    max_devi_f = np.max(fs_devi, axis=1)
    min_devi_f = np.min(fs_devi, axis=1)
    avg_devi_f = np.mean(fs_devi, axis=1)
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
        vs : numpy.ndarray, size of `n_models x n_frames x 3 x 3`
    '''
    vs = np.reshape(vs, (vs.shape[1], vs.shape[2], 9))
    vs_devi = np.std(vs, axis=0)
    max_devi_v = np.max(vs_devi, axis=1)
    min_devi_v = np.min(vs_devi, axis=1)
    avg_devi_v = np.mean(vs_devi, axis=1)
    return max_devi_v, min_devi_v, avg_devi_v

def write_model_devi_out(devi, fname, items='vf'):
    '''
        devi : numpy.ndarray, the first column is the steps index
        fname : str, the file name to dump
        items : str, specify physical quantities of which model_devi is contained in `devi`
                    f - forces, v - virial
    '''
    assert devi.shape[1] == len(items) * 3 + 1
    header = "#%11s" % "step"
    for item in items:
        header += "%19s%19s%19s" % (f"max_devi_{item}", f"min_devi_{item}", f"avg_devi_{item}")
    np.savetxt(fname,
               devi,
               fmt=['%12d'] + ['%19.6e' for _ in range(len(items) * 3)],
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
    
def make_model_devi(
    *,
    models: list,
    system: str,
    set_prefix: str,
    output: str,
    frequency: int,
    items: str,
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
    items: str
        String to specify physical quantities of which model devi is calculated
        only f, v (force, virial) is supported.
    '''
    # init models
    dp_models = [DeepPotential(model) for model in models]

    # check type maps
    tmaps = [dp.get_type_map() for dp in dp_models]
    if _check_tmaps(tmaps):
        tmap = tmaps[0]
    else:
        raise RuntimeError("The models does not have the same type map.")
    
    # create data-system
    data = DeepmdData(system, set_prefix, shuffle_test=False, type_map=tmap)
    coord = data["coord"]
    box = data["box"]
    nframes = coord.shape[0]
    if dp_models[0].has_efield:
        efield = data["efield"]
    else:
        efield = None
    if not data.pbc:
        box = None
    atype = data["type"][0]
    if dp_models[0].get_dim_fparam() > 0:
        fparam = data["fparam"]
    else:
        fparam = None
    if dp_models[0].get_dim_aparam() > 0:
        aparam = data["aparam"]
    else:
        aparam = None
        
    forces = []
    virials = []
    for dp in dp_models:
        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
            atomic=False,
            efield=efield,
        )
        forces.append(ret[1])
        virials.append(ret[2])
    
    forces = np.array(forces)
    virials = np.array(virials)
    
    devi = [np.arange(nframes) * frequency]
    devi_f = calc_model_devi_f(forces)
    devi_v = calc_model_devi_v(virials)
    for item in items:
        if item == "v":
            devi += [devi_v[0], devi_v[1], devi_v[2]]
        elif item == "f":
            devi += [devi_f[0], devi_f[1], devi_f[2]]
        else:
            raise ValueError(f"Unvaild item {item}")
    devi = np.vstack(devi).T
    write_model_devi_out(devi, output, items)
    return devi