#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import argparse
import json
from deepmd.env import tf
from deepmd.compat import convert_input_v0_v1

lib_path = os.path.dirname(os.path.realpath(__file__)) + "/../lib/"
sys.path.append (lib_path)

from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DataSystem, DeepmdDataSystem
from deepmd.Trainer import NNPTrainer
from deepmd.common import data_requirement

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

def train (args) :
    # load json database
    fp = open (args.INPUT, 'r')
    jdata = json.load (fp)
    if not 'model' in jdata.keys():
       jdata = convert_input_v0_v1(jdata, 
                                   warning = True, 
                                   dump = 'input_v1_compat.json')
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
    # init the model
    model = NNPTrainer (jdata, run_opt = run_opt)
    rcut = model.model.get_rcut()
    type_map = model.model.get_type_map()
    # init params and run options
    assert('training' in jdata)
    systems = j_must_have(jdata['training'], 'systems')
    set_pfx = j_must_have(jdata['training'], 'set_prefix')
    numb_sys = len(systems)
    seed = None
    if 'seed' in jdata['training'].keys() : seed = jdata['training']['seed']
    if seed is not None:
       seed = seed % (2**32)
    np.random.seed (seed)
    batch_size = j_must_have(jdata['training'], 'batch_size')
    test_size = j_must_have(jdata['training'], 'numb_test')
    stop_batch = j_must_have(jdata['training'], 'stop_batch')
    if len(type_map) == 0:
       # empty type_map
       ipt_type_map = None
    else:
       ipt_type_map = type_map
    data = DeepmdDataSystem(systems, batch_size, test_size, rcut, set_prefix=set_pfx, run_opt=run_opt, type_map = ipt_type_map)
    data.add_dict(data_requirement)
    # build the model with stats from the first system
    model.build (data)
    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    cur_batch = 0
    model.train (data, stop_batch)
    end_time = time.time()
    run_opt.message("finished training\nwall time: %.3f s" % (end_time-start_time))

