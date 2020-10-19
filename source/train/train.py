#!/usr/bin/env python3

import os
import sys
import time
import json
import numpy as np
from deepmd.env import tf
from deepmd.compat import convert_input_v0_v1
from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DeepmdDataSystem
from deepmd.Trainer import NNPTrainer
from deepmd.common import data_requirement, expand_sys_str, j_loader
from deepmd.DataModifier import DipoleChargeModifier
from deepmd.argcheck import normalize

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
    jdata = j_loader(args.INPUT)

    if not 'model' in jdata.keys():
       jdata = convert_input_v0_v1(jdata, 
                                   warning = True, 
                                   dump = 'input_v1_compat.json')
    
    jdata = normalize(jdata)
    with open(args.output, 'w') as fp:
        json.dump(jdata, fp, indent=4)

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
    if type(systems) == str:
       systems = expand_sys_str(systems)
    set_pfx = j_must_have(jdata['training'], 'set_prefix')
    seed = None
    if 'seed' in jdata['training'].keys() : seed = jdata['training']['seed']
    if seed is not None:
       seed = seed % (2**32)
    np.random.seed (seed)
    batch_size = j_must_have(jdata['training'], 'batch_size')
    test_size = j_must_have(jdata['training'], 'numb_test')
    stop_batch = j_must_have(jdata['training'], 'stop_batch')
    sys_probs = jdata['training'].get('sys_probs')
    auto_prob_style = jdata['training'].get('auto_prob_style', 'prob_sys_size')
    if len(type_map) == 0:
       # empty type_map
       ipt_type_map = None
    else:
       ipt_type_map = type_map
    # data modifier
    modifier = None
    modi_data = jdata['model'].get("modifier", None)
    if modi_data is not None:
       if modi_data['type'] == 'dipole_charge':
          modifier = DipoleChargeModifier(modi_data['model_name'],
                                          modi_data['model_charge_map'],
                                          modi_data['sys_charge_map'],
                                          modi_data['ewald_h'],
                                          modi_data['ewald_beta'])
       else:
          raise RuntimeError('unknown modifier type ' + str(modi_data['type']))
    # init data
    data = DeepmdDataSystem(systems, 
                            batch_size, 
                            test_size, 
                            rcut, 
                            set_prefix=set_pfx, 
                            type_map = ipt_type_map, 
                            modifier = modifier)
    data.print_summary(run_opt, 
                       sys_probs = sys_probs, 
                       auto_prob_style = auto_prob_style)
    data.add_dict(data_requirement)
    # build the model with stats from the first system
    model.build (data, stop_batch)
    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    model.train (data)
    end_time = time.time()
    run_opt.message("finished training\nwall time: %.3f s" % (end_time-start_time))

