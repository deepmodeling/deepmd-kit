#!/usr/bin/env python3

import os

def get_threads_env () :
    num_intra_threads = None
    if 'OMP_NUM_THREADS' in os.environ : 
        num_intra_threads = int(os.environ['OMP_NUM_THREADS'])
    else :
        num_intra_threads = 0
    return num_intra_threads

class RunOptions (object) :    
    def __init__ (self, 
                  args = None):
        # default set
        self.restart = None
        self.init_model = None
        self.init_mode = "init_from_scratch"
        self.num_intra_threads = get_threads_env()
        self.num_inter_threads = 0
        
        if args is not None :
            if (args.init_model is not None) and (args.restart is not None) :
                raise RuntimeError ("--init-model and --restart should not be set at the same time")
            if args.init_model is not None :
                self.init_model = os.path.abspath(args.init_model)
                self.init_mode = "init_from_model"
            if args.restart is not None: 
                self.restart = os.path.abspath(args.restart)
                self.init_mode = "restart"
            if args.inter_threads is not None :
                self.num_inter_threads = args.inter_threads
