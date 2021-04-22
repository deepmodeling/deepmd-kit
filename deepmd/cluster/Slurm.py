#### https://github.com/deepsense-ai/tensorflow_on_slurm ####

# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division

import re
import os

def get_resource ():
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    nodelist = _expand_nodelist(nodelist)
    nodename = os.environ["SLURMD_NODENAME"]
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
    if len(nodelist) != num_nodes:
        raise ValueError("Number of slurm nodes {} not equal to {}".format(len(nodelist), num_nodes))
    if nodename not in nodelist:
        raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(nodename,nodelist))
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpus is not None :
        gpus = gpus.split(",")
        gpus = [int(ii) for ii in gpus]
    return nodename, nodelist, gpus

def _pad_zeros(iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)

def _expand_ids(ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            str_end = id.split('-')[1]
            begin, end = [int(token) for token in id.split('-')]
            result.extend(_pad_zeros(range(begin, end+1), len(str_end)))
        else:
            result.append(id)
    return result

def _expand_nodelist(nodelist):
    result = []
    interval_list = nodelist.split(',')
    for interval in interval_list:
        match = re.search("(.*)\[(.*)\]", interval)
        if match:
            prefix = match.group(1)
            ids = match.group(2)
            ids = _expand_ids(ids)
            result.extend([prefix + str(id) for id in ids])
        else:
            result.append(interval)
    return result

def _worker_task_id(nodelist, nodename):
    return nodelist.index(nodename)
