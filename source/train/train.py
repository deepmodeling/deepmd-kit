#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import argparse
import json
import tensorflow as tf

lib_path = os.path.dirname(os.path.realpath(__file__)) + "/../lib/"
sys.path.append (lib_path)

from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DataSystem
from deepmd.Model import NNPModel
from deepmd.Model import LearingRate

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json data base must provide key " + key )
    else :
        return jdata[key]

def _main () :
    default_num_inter_threads = 0
    parser = argparse.ArgumentParser(
        description="*** Train a model. ***")
    parser.add_argument('INPUT', 
                        help='the input json database ')
    parser.add_argument('-t','--inter-threads', type = int, default = default_num_inter_threads,
                        help=
                        'With default value %d. ' % default_num_inter_threads + 
                        'Setting the "inter_op_parallelism_threads" key for the tensorflow, '  +
                        'the "intra_op_parallelism_threads" will be set by the env variable OMP_NUM_THREADS')
    parser.add_argument('--init-model', type = str, 
                        help=
                        'Initialize the model by the provided checkpoint.')
    parser.add_argument('--restart', type = str, 
                        help=
                        'Restart the training from the provided checkpoint.')
    args = parser.parse_args()

    # load json database
    fp = open (args.INPUT, 'r')
    jdata = json.load (fp)

    # init params and run options
    systems = j_must_have(jdata, 'systems')
    set_pfx = j_must_have(jdata, 'set_prefix')
    numb_sys = len(systems)
    seed = None
    if 'seed' in jdata.keys() : seed = jdata['seed']
    seed = seed % (2**32)
    np.random.seed (seed)
    batch_size = j_must_have(jdata, 'batch_size')
    test_size = j_must_have(jdata, 'numb_test')
    stop_batch = j_must_have(jdata, 'stop_batch')
    rcut = j_must_have (jdata, 'rcut')
    print ("#")
    print ("# find %d system(s): " % numb_sys)    
    data = DataSystem(systems, set_pfx, batch_size, test_size, rcut)
    print ("#")
    tot_numb_batches = sum(data.get_nbatches())
    lr = LearingRate (jdata, tot_numb_batches)
    final_lr = lr.value (stop_batch)
    run_opt = RunOptions(args)
    print("# run with intra_op_parallelism_threads = %d, inter_op_parallelism_threads = %d " % 
          (run_opt.num_intra_threads, run_opt.num_inter_threads))

    # start tf
    tf.reset_default_graph()
    with tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=run_opt.num_intra_threads, 
                                  inter_op_parallelism_threads=run_opt.num_inter_threads
            )) as sess:
        # init the model
        model = NNPModel (sess, jdata, run_opt = run_opt)
        # build the model with stats from the first system
        model.build (data, lr)
        # train the model with the provided systems in a cyclic way
        start_time = time.time()
        cur_batch = model.get_global_step()
        print ("# start training, start lr is %e, final lr will be %e" % (lr.value(cur_batch), final_lr) )
        model.print_head()
        model.train (data, stop_batch)
        print ("# finished training")
        end_time = time.time()
        print ("# running time: %.3f s" % (end_time-start_time))

if __name__ == '__main__':
    _main()
    
