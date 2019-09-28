import argparse

from .train import train
from .freeze import freeze
from .config import config
from .test import test

def main () :    
    parser = argparse.ArgumentParser(
	description="DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')    

    # parser_cfig = subparsers.add_parser('config', help='fast configuration of parameter file for smooth model')
    # parser_cfig.add_argument("-o", "--output", type=str, default = "input.json", 
    #                          help="the output json file")    
    
    default_num_inter_threads = 0
    parser_train = subparsers.add_parser('train', help='train a model')
    parser_train.add_argument('INPUT', 
                              help='the input parameter file in json format')
    parser_train.add_argument('-t','--inter-threads', type = int, default = default_num_inter_threads,
                              help=
                              'With default value %d. ' % default_num_inter_threads + 
                              'Setting the "inter_op_parallelism_threads" key for the tensorflow, '  +
                              'the "intra_op_parallelism_threads" will be set by the env variable OMP_NUM_THREADS')
    parser_train.add_argument('--init-model', type = str, 
                              help=
                              'Initialize the model by the provided checkpoint.')
    parser_train.add_argument('--restart', type = str, 
                              help=
                              'Restart the training from the provided checkpoint.')
    
    parser_frz = subparsers.add_parser('freeze', help='freeze the model')
    parser_frz.add_argument("-d", "--folder", type=str, default = ".", 
                            help="path to checkpoint folder")
    parser_frz.add_argument("-o", "--output", type=str, default = "frozen_model.pb", 
                            help="name of graph, will output to the checkpoint folder")
    parser_frz.add_argument("-n", "--nodes", type=str, 
                            help="the frozen nodes, if not set, determined from the model type")

    parser_tst = subparsers.add_parser('test', help='test the model')
    parser_tst.add_argument("-m", "--model", default="frozen_model.pb", type=str, 
                            help="Frozen model file to import")
    parser_tst.add_argument("-s", "--system", default=".", type=str, 
                            help="The system dir")
    parser_tst.add_argument("-S", "--set-prefix", default="set", type=str, 
                            help="The set prefix")
    parser_tst.add_argument("-n", "--numb-test", default=100, type=int, 
                            help="The number of data for test")
    parser_tst.add_argument("-r", "--rand-seed", type=int, 
                            help="The random seed")
    parser_tst.add_argument("--shuffle-test", action = 'store_true', 
                            help="Shuffle test data")
    parser_tst.add_argument("-d", "--detail-file", type=str, 
                            help="The file containing details of energy force and virial accuracy")

    args = parser.parse_args()

    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'train' :
        train(args)
    elif args.command == 'freeze' :
        freeze(args)
    elif args.command == 'config' :
        config(args)
    elif args.command == 'test' :
        test(args)
    else :
        raise RuntimeError('unknown command ' + args.command)
