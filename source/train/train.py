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

from deepmd.Data import DataSets
from deepmd.Data import DataScan
from deepmd.Model import NNPModel
from deepmd.Model import LearingRate

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json data base must provide key " + key )
    else :
        return jdata[key]

def _main () :
    parser = argparse.ArgumentParser(
        description="*** Train a model. ***")
    parser.add_argument('INPUT', help='the input json database ')
    args = parser.parse_args()

    # load json database
    fp = open (args.INPUT, 'r')
    jdata = json.load (fp)

    # init params
    systems = j_must_have(jdata, 'systems')
    set_pfx = j_must_have(jdata, 'set_prefix')
    numb_sys = len(systems)
    seed = None
    if 'seed' in jdata.keys() : seed = jdata['seed']
    num_threads = j_must_have(jdata, 'num_threads')
    batch_size = j_must_have(jdata, 'batch_size')
    stop_batch = j_must_have(jdata, 'stop_batch')
    tot_numb_batches = 0
    print ("#")
    print ("# using %d system(s): " % numb_sys)
    for _sys in systems :
        s_data = DataScan (_sys, set_pfx)
        numb_batches = s_data.get_sys_numb_batch(batch_size)
        tot_numb_batches += numb_batches
        print ("# %s has %d batches, and was copied by %s " % (_sys, numb_batches, str(s_data.get_ncopies())))
    print ("#")
    lr = LearingRate (jdata, tot_numb_batches)
    final_lr = lr.value (stop_batch)

    # start tf
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
        # init the model
        model = NNPModel (jdata, sess)
        # build the model with stats from the first system
        data = DataSets (systems[0], set_pfx, seed = seed, do_norm = False)        
        model.build (data, lr)
        # train the model with the provided systems in a cyclic way
        start_time = time.time()
        count = 0
        cur_batch = model.get_global_step()
        cur_stop_batch = cur_batch
        print ("# start training, start lr is %e, final lr will be %e" % (lr.value(cur_stop_batch), final_lr) )
        model.print_head()
        while True :
            cur_sys = systems[count % numb_sys]
            data = DataSets (cur_sys, set_pfx, seed = seed, do_norm = False)
            cur_batch = cur_stop_batch
            cur_stop_batch += data.get_sys_numb_batch (batch_size)
            if cur_stop_batch > stop_batch : cur_stop_batch = stop_batch
            print ("# train with %s that has %d batches" % (cur_sys, cur_stop_batch - cur_batch))
            model.train (data, cur_stop_batch)
            if cur_stop_batch == stop_batch : break
            count += 1
        print ("# finished training")
        end_time = time.time()
        print ("# running time: %.3f s" % (end_time-start_time))

if __name__ == '__main__':
    _main()
    
