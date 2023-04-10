import argparse
import glob
import os
import shutil

import dpdata
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default='.',
        help="input data path to explore",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default='',
        help="output data path",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default='10',
        help="sparse-system threshold",
    )
    parsed_args = parser.parse_args(args=args)
    return parsed_args 

def divide_train_valid(sys_dir):
    global index
    sys = dpdata.LabeledSystem(sys_dir, fmt="deepmd/npy")
    sys.shuffle()
    if sys.get_nframes() < threshold:
        return
    frames = sys.get_nframes()
    train_frames = int(frames * 0.9)
    valid_frames = frames - train_frames
    train_index = np.arange(frames)[:train_frames]
    valid_index = np.arange(frames)[train_frames:]
    train_system = sys.sub_system(train_index)
    valid_system = sys.sub_system(valid_index)
    train_system.to_deepmd_npy(os.path.join(output, "data", "train", 'sys.%.6d'% index))
    valid_system.to_deepmd_npy(os.path.join(output, "data", "valid", 'sys.%.6d'% index))
    index +=1
        
if __name__ == '__main__':
    args = parse_args()
    dict_args = vars(args)
    input = dict_args['input']
    output = dict_args['output']
    threshold = dict_args['threshold']
    index = 0
    
    Path(os.path.join(output, "data")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, "sys_dir")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, "workspace")).mkdir(parents=True, exist_ok=True)
    
    # all system directories
    input_systems = []
    dirs = glob.glob(input)
    for di in dirs:
        for root, dirs, files in os.walk(di):
            for name in files:    
                if 'type.raw' in name and dirs:
                    input_systems.append(root)
                continue
            
    # divide frames in single system into train system and valid system       
    for sys_dir in input_systems:
        divide_train_valid(sys_dir)
    
    # merge sparse-frame systems 
    sparse_systems = {}
    for sys_dir in input_systems:
        sys = dpdata.LabeledSystem(sys_dir, fmt="deepmd/npy")
        formula = sys.formula
        frames = sys.get_nframes()
        if frames < threshold:
            if formula in sparse_systems.keys():
                sparse_systems[formula].append(sys_dir)
            else:
                sparse_systems[formula] = [sys_dir]
                    
    for formula in sparse_systems.keys():
        multi_sys = dpdata.MultiSystems()   
        for sys_dir in sparse_systems[formula]:
            multi_sys.append(dpdata.LabeledSystem(sys_dir, fmt="deepmd/npy"))
        multi_sys.to_deepmd_npy(os.path.join(output, "data", "sparse"))
        sys = dpdata.LabeledSystem(os.path.join(output, "data", "sparse", formula), fmt="deepmd/npy")
        divide_train_valid(os.path.join(output, "data", "sparse", formula))
        
    if Path(os.path.join(output, "data", "sparse")).exists():
        shutil.rmtree(os.path.join(output, "data", "sparse"))
       
    # gen_list.py 
    output_systems = []
    dirs = glob.glob(output)
    for di in dirs:
        for root, dirs, files in os.walk(di):
            for name in files:    
                if 'type.raw' in name and dirs:
                    output_systems.append(root)
                continue
    gen_list = open(os.path.join(output, "sys_dir", "list.txt"), "w")
    gen_list.writelines(["{}\n".format(i) for i in sorted(output_systems)])
    gen_list.close()
    
    train_systems = []
    dirs = glob.glob(os.path.join(output, "data", "train"))
    for di in dirs:
        for root, dirs, files in os.walk(di):
            for name in files:    
                if 'type.raw' in name and dirs:
                    train_systems.append(root)
                continue
    train_list = open(os.path.join(output, "sys_dir", "train.txt"), "w")
    train_list.writelines(["{}\n".format(i) for i in sorted(train_systems)])
    train_list.close()
    
    valid_systems = []
    dirs = glob.glob(os.path.join(output, "data", "valid"))
    for di in dirs:
        for root, dirs, files in os.walk(di):
            for name in files:    
                if 'type.raw' in name and dirs:
                    valid_systems.append(root)
                continue
    valid_list = open(os.path.join(output, "sys_dir", "valid.txt"), "w")
    valid_list.writelines(["{}\n".format(i) for i in sorted(valid_systems)])
    valid_list.close()
    
    # count.py
    gen_list = open(os.path.join(output, 'sys_dir', 'list.txt'), 'r')
    data_sys = [i.strip() for i in gen_list.readlines()]
    count_list = open(os.path.join(output, 'sys_dir', 'count.txt'), 'w')
    count_sys_list = []
    all_frames = 0
    for single_system in tqdm(data_sys):
        try:
            temps = dpdata.LabeledSystem(single_system, fmt='deepmd/npy')
        except:
            print('not a system in{}'.format(single_system))
            continue
        temp_type = temps.get_atom_numbs()
        temp_nframes = temps.get_nframes()
        all_frames += temp_nframes
        temp_natoms = temps.get_natoms()
        count_sys_list.append('{}__NF_{}_TYPE_{}_NA_{}\n'.format(single_system, temp_nframes, temp_type, temp_natoms))
    count_list.writelines(count_sys_list)
    count_list.write('all frames: {}\n'.format(all_frames))
    count_list.close()
    
    shutil.make_archive(output, 'zip')
    # shutil.rmtree(output)
    # print("Output DPA dataset directory: {}".format(output + ".zip"))