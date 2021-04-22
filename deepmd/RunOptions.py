import os,sys
from deepmd.env import tf
from deepmd.env import get_tf_default_nthreads
import numpy as np
import deepmd.cluster.Slurm as Slurm
import deepmd.cluster.Local as Local

if "-DHIGH_PREC" == "-DHIGH_PREC" :
    global_tf_float_precision = tf.float64
    global_np_float_precision = np.float64
    global_ener_float_precision = np.float64
    global_float_prec = 'double'
else :
    global_tf_float_precision = tf.float32
    global_np_float_precision = np.float32
    global_ener_float_precision = np.float64
    global_float_prec = 'float'

def global_cvt_2_tf_float(xx) :
    return tf.cast(xx, global_tf_float_precision)
def global_cvt_2_ener_float(xx) :
    return tf.cast(xx, global_ener_float_precision)

global_install_prefix='/tmp/pip-req-build-g166qwb9/_skbuild/linux-x86_64-3.7/cmake-install'
global_git_summ='v1.2.2-145-g80f435d'
global_git_hash='80f435d'
global_git_date='2021-02-04 13:29:57 +0800'
global_git_branch='devel'
global_tf_include_dir='/root/anaconda3/lib/python3.7/site-packages/tensorflow/include;/root/anaconda3/lib/python3.7/site-packages/tensorflow/include'
global_tf_libs=''

def _is_slurm() :
    return "SLURM_JOB_NODELIST" in os.environ

def _is_distributed(MPI) :
    return MPI.COMM_WORLD.Get_size() > 1

def distributed_task_config (MPI,
                             node_name, 
                             node_list_, 
                             gpu_list = None, 
                             default_port = 2222) :
    """
    create configuration for distributed tensorflow session.
    inputs: 
        MPI             mpi4py.MPI
        node_name       str, the name of the this node
        node_list       [str], the list of nodes of the current mpirun
        gpu_list        [int], the list of GPUs on each node
        default_port    int, the default port for socket communication
    outputs:
        cluster         map, cluster_spec
        job             str, the job name of this task
        task_idx        int, the index of this task
        socket          str, hostname:port socket of this task
        device          str, the device for this task
    """
    # setup cluster
    node_list = list(set(node_list_))
    node_list.sort()
    node_color = node_list.index(node_name)
    world_idx = MPI.COMM_WORLD.Get_rank()
    node_comm = MPI.COMM_WORLD.Split(node_color, world_idx)
    node_task_idx = node_comm.Get_rank()
    node_numb_task = node_comm.Get_size()
    socket_list = []
    for ii in node_list :
        for jj in range(node_numb_task) :
            socket_list.append(ii + ":" + str(default_port+jj))
    ps_map = socket_list[0:1]
    worker_map = socket_list[1:]
    if node_color == 0 and node_task_idx == 0 :
        my_job = 'ps'
        my_socket = ps_map[0]
        my_task_idx = ps_map.index(my_socket)
    else :
        my_job = 'worker'
        my_socket = node_name + ":" + str(default_port+node_task_idx)
        assert(my_socket in worker_map)
        my_task_idx = worker_map.index(my_socket)    
    # setup gpu/cpu devices
    if gpu_list is not None :
        numb_gpu = len(gpu_list)
        gpu_idx = node_numb_task - node_task_idx - 1
        if gpu_idx >= numb_gpu :
            # my_device = "cpu:%d" % node_task_idx
            my_device = "cpu:0"
        else :
            my_device = "gpu:%d" % gpu_idx            
    else :
        # my_device = "cpu:%d" % node_task_idx
        my_device = "cpu:0"
    # return results
    cluster = {"worker": worker_map, "ps" : ps_map}
    return cluster, my_job, my_task_idx, my_socket, my_device


class RunOptions (object) :    
    def __init__ (self, 
                  args, 
                  try_distrib = False):        
        # distributed tasks
        if try_distrib :
            self._try_init_mpi()
        else :
            self.is_distrib = False
            self._init_serial()
        self.verbose = self.is_chief

        # model init options
        # default set
        self.restart = None
        self.init_model = None
        self.init_mode = "init_from_scratch"
        if args is not None :
            if (args.init_model is not None) and (args.restart is not None) :
                raise RuntimeError ("--init-model and --restart should not be set at the same time")
            if args.init_model is not None :
                self.init_model = os.path.abspath(args.init_model)
                self.init_mode = "init_from_model"
            if args.restart is not None: 
                self.restart = os.path.abspath(args.restart)
                self.init_mode = "restart"

    def message (self, msg) :
        if self.verbose :
            lines = msg.split('\n')
            for ii in lines :
                print ("# DEEPMD: " + str(ii))
            sys.stdout.flush()

    def print_welcome(self) :
        # http://patorjk.com/software/taag. Font:Big"
        msg = ""
        msg += " _____               _____   __  __  _____           _     _  _   \n"
        msg += "|  __ \             |  __ \ |  \/  ||  __ \         | |   (_)| |  \n"
        msg += "| |  | |  ___   ___ | |__) || \  / || |  | | ______ | | __ _ | |_ \n"
        msg += "| |  | | / _ \ / _ \|  ___/ | |\/| || |  | ||______|| |/ /| || __|\n"
        msg += "| |__| ||  __/|  __/| |     | |  | || |__| |        |   < | || |_ \n"
        msg += "|_____/  \___| \___||_|     |_|  |_||_____/         |_|\_\|_| \__|\n"
        self.message(msg)        

    def print_citation(self) :
        msg = ""
        msg += "Please read and cite:\n"
        msg += "Wang, Zhang, Han and E, Comput.Phys.Comm. 228, 178-184 (2018)\n"
        self.message(msg)        

    def print_build(self): 
        msg = ''
        msg += 'source code %s at brach %s commit at %s' % (global_git_hash, global_git_branch, global_git_date)
        self.message(msg)

    def print_summary(self) :
        msg = ""
        msg += "---Summary of the training---------------------------------------\n"
        msg += 'installed to:         %s\n' % global_install_prefix
        msg += 'source :              %s\n' % global_git_summ
        msg += 'source brach:         %s\n' % global_git_branch
        msg += 'source commit:        %s\n' % global_git_hash
        msg += 'source commit at:     %s\n' % global_git_date
        msg += 'build float prec:     %s\n' % global_float_prec
        msg += 'build with tf inc:    %s\n' % global_tf_include_dir
        for idx,ii in enumerate(global_tf_libs.split(';')) :
            if idx == 0 :
                msg += 'build with tf lib:    %s\n' % ii
            else :
                msg += '                      %s\n' % ii
        if self.is_distrib:
            msg += "distributed\n" 
            msg += "ps list:              %s\n" % str(self.cluster['ps'])
            msg += "worker list:          %s\n" % str(self.cluster['worker'])
            msg += "chief on:             %s\n" % self.nodename
        else :
            msg += "running on:           %s\n" % self.nodename
        if self.gpus is None:
            msg += "CUDA_VISIBLE_DEVICES: unset\n"
        else:
            msg += "CUDA_VISIBLE_DEVICES: %s\n" % self.gpus
        intra, inter = get_tf_default_nthreads()
        msg += "num_intra_threads:    %d\n" % intra
        msg += "num_inter_threads:    %d\n" % inter
        msg += "-----------------------------------------------------------------\n"
        self.message(msg)

    def _try_init_mpi(self):
        try :
            from mpi4py import MPI
        except ImportError :
            raise RuntimeError("cannot import mpi4py module, cannot do distributed simulation")
        else :            
            self.is_distrib = _is_distributed(MPI)
            if self.is_distrib :
                self._init_distributed(MPI)
            else :
                self._init_serial()

    def _init_distributed(self, MPI) :
        # Run options for distributed training        
        if _is_slurm() : 
            nodename, nodelist, gpus = Slurm.get_resource()
        else :
            nodename, nodelist, gpus = Local.get_resource()
        self.nodename = nodename
        self.gpus = gpus
        self.cluster, \
        self.my_job_name, \
        self.my_task_index, \
        self.my_socket, \
        self.my_device \
            = distributed_task_config(MPI, nodename, nodelist, gpus)
        self.is_chief = (self.my_job_name == 'worker' and self.my_task_index == 0)
        self.num_ps = len(self.cluster['ps'])
        self.num_workers = len(self.cluster['worker'])
        self.cluster_spec = tf.train.ClusterSpec(self.cluster)
        self.server = tf.train.Server(server_or_cluster_def = self.cluster_spec,
                                      job_name = self.my_job_name,
                                      task_index = self.my_task_index)

    def _init_serial(self) :
        # Run options for serial training        
        nodename, nodelist, gpus = Local.get_resource()
        self.nodename = nodename
        self.gpus = gpus
        self.cluster = None
        self.my_job_name = nodename
        self.my_task_index = 0
        self.my_socket = None
        if gpus is not None and len(gpus) > 0:
            self.my_device = "gpu:%d" % gpus[0]
            # self.my_device = "gpu:0"
        else :
            self.my_device = "cpu:0"
        self.is_chief = True
        self.num_ps = None
        self.num_workers = None
        self.cluster_spec = None
        self.server = None
