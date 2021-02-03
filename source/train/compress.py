import re
import json
import copy
import argparse
import numpy as np
from deepmd.env import tf
from .train import train
from .freeze import freeze
from .transform import transform
from deepmd.common import j_loader
from deepmd.utils.argcheck import normalize

def compress(args):
    jdata = j_loader(args.INPUT)
    if not 'model' in jdata.keys():
       jdata = convert_input_v0_v1(jdata, 
                                   warning = True, 
                                   dump = 'input_v1_compat.json')
    
    jdata = normalize(jdata)
    jdata['model']['descriptor']['compress'] = True
    jdata['model']['descriptor']['model_file'] = args.input
    jdata['model']['descriptor']['table_info'] = args.table_info
    
    # check the descriptor info of the input file
    assert jdata['model']['descriptor']['type'] == 'se_a', 'Model compression error: descriptor type must be se_a!'
    assert jdata['model']['descriptor']['resnet_dt'] == False, 'Model compression error: descriptor resnet_dt must be false!'

    # stage 1: training or refining the model with tabulation
    print('\n\n# DEEPMD: stage 1: train or refine the model with tabulation')
    args_train = copy.deepcopy(args)
    args_train.INPUT = 'compress.json'
    args_train.output = 'compress.json'
    args_train.init_model = None
    args_train.restart = None
    jdata['training']['stop_batch'] = jdata['training']['save_freq'] # be careful here, if we want refine the model
    with open(args_train.INPUT, 'w') as fp:
        json.dump(jdata, fp, indent=4)
    train(args_train)

    # stage 2: freeze the model
    print('\n\n# DEEPMD: stage 2: freeze the model')
    args_frz = copy.deepcopy(args)
    args_frz.nodes = None
    freeze(args_frz)

    # stage 3: transform the model
    print('\n\n# DEEPMD: stage 3: transform the model')
    args_transform = copy.deepcopy(args)
    args_transform.old_model = args.input
    args_transform.raw_model = args.output
    args_transform.output = args.output
    transform(args_transform)
