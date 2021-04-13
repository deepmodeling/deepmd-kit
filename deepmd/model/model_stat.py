import numpy as np
from collections import defaultdict

def _make_all_stat_ref(data, nbatches):
    all_stat = defaultdict(list)
    for ii in range(data.get_nsystems()) :
        for jj in range(nbatches) :
            stat_data = data.get_batch (sys_idx = ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32) 
                all_stat[dd].append(stat_data[dd])        
    return all_stat


def make_stat_input(data, nbatches, merge_sys = True):
    """
    pack data for statistics
    Parameters
    ----------
    data:
        The data
    merge_sys: bool (True)
        Merge system data
    Returns
    -------
    all_stat:
        A dictionary of list of list storing data for stat. 
        if merge_sys == False data can be accessed by 
            all_stat[key][sys_idx][batch_idx][frame_idx]
        else merge_sys == True can be accessed by 
            all_stat[key][batch_idx][frame_idx]
    """
    all_stat = defaultdict(list)
    for ii in range(data.get_nsystems()) :
        sys_stat =  defaultdict(list)
        for jj in range(nbatches) :
            stat_data = data.get_batch (sys_idx = ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32) 
                sys_stat[dd].append(stat_data[dd])
        for dd in sys_stat:
            if merge_sys:
                for bb in sys_stat[dd]:
                    all_stat[dd].append(bb)
            else:                    
                all_stat[dd].append(sys_stat[dd])
    return all_stat

def merge_sys_stat(all_stat):
    first_key = list(all_stat.keys())[0]
    nsys = len(all_stat[first_key])
    ret = defaultdict(list)
    for ii in range(nsys):
        for dd in all_stat:
            for bb in all_stat[dd][ii]:
                ret[dd].append(bb)
    return ret

