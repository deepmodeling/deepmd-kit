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

def create_done_queue(cluster_spec, task_index):
   with tf.device("/job:ps/task:%d" % (task_index)):
       queue = tf.FIFOQueue(cluster_spec.num_tasks("worker"), tf.int32,
                            shared_name = "done_queue" + str(task_index))
       return queue

def wait_done_queue(cluster_spec, server, queue, task_index):
    with tf.Session(server.target) as sess:
         for i in range(cluster_spec.num_tasks("worker")):
             sess.run(queue.dequeue())
         #     print("ps:%d received done from worker:%d" % (task_index, i))
         # print("ps:%d quitting" % task_index)

def connect_done_queue(cluster_spec, task_index):
     done_ops = []
     for i in range(cluster_spec.num_tasks("ps")):
         with tf.device("/job:ps/task:%d" % i):
             queue = tf.FIFOQueue(cluster_spec.num_tasks('worker'), tf.int32,
                                  shared_name='done_queue' + str(i))
             done_ops.append(queue.enqueue(task_index))
     return done_ops

def fill_done_queue(cluster_spec, server, done_ops, task_index):
     with tf.Session(server.target) as sess:
          for i in range(cluster_spec.num_tasks("ps")):
              sess.run(done_ops[i])
              # print("worker:%d sending done to ps:%d" % (task_index, i))

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
    # run options
    with_distrib = False 
    if 'with_distrib' in jdata:
        with_distrib = jdata['with_distrib']
    run_opt = RunOptions(args, with_distrib)
    run_opt.print_welcome()
    run_opt.print_citation()
    run_opt.print_summary()

    if run_opt.is_distrib :
        # distributed training
        if run_opt.my_job_name == "ps":
            queue = create_done_queue(run_opt.cluster_spec, run_opt.my_task_index)
            wait_done_queue(run_opt.cluster_spec, run_opt.server, queue, run_opt.my_task_index)
            #server.join()
        elif run_opt.my_job_name == "worker":
            done_ops = connect_done_queue(run_opt.cluster_spec, run_opt.my_task_index)
            _do_work(jdata, run_opt)
            fill_done_queue(run_opt.cluster_spec, run_opt.server, done_ops, run_opt.my_task_index)
        else :
            raise RuntimeError("unknown job name")
    else :
        # serial training
        _do_work(jdata, run_opt)

def _do_work(jdata, run_opt):
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
    data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt)
    tot_numb_batches = sum(data.get_nbatches())
    lr = LearingRate (jdata, tot_numb_batches)
    # init the model
    model = NNPModel (jdata, run_opt = run_opt)
    # build the model with stats from the first system
    model.build (data, lr)
    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    cur_batch = 0
    model.train (data, stop_batch)
    end_time = time.time()
    run_opt.message("finished training\nwall time: %.3f s" % (end_time-start_time))

if __name__ == '__main__':
    _main()
    
