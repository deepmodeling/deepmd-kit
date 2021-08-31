"""DeePMD-Kit entry point module."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from deepmd import __version__
from deepmd.entrypoints import (
    compress,
    config,
    doc_train_input,
    freeze,
    test,
    train_dp,
    transfer,
    make_model_devi,
    convert,
)
from deepmd.loggers import set_log_handles

__all__ = ["main", "parse_args", "get_ll"]


def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level


def parse_args(args: Optional[List[str]] = None):
    """DeePMD-Kit commandline options argument parser.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = argparse.ArgumentParser(
        description="DeePMD-kit: A deep learning package for many-body potential energy"
        " representation and molecular dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # * logging options parser *********************************************************
    # with use of the parent argument this options will be added to every parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-v",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
        "and 3=DEBUG",
    )
    parser_log.add_argument(
        "-l",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
        "only be output to console",
    )
    # * mpi logging parser *************************************************************
    parser_mpi_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_mpi_log.add_argument(
        "-m",
        "--mpi-log",
        type=str,
        default="master",
        choices=("master", "collect", "workers"),
        help="Set the manner of logging when running with MPI. 'master' logs only on "
        "main process, 'collect' broadcasts logs from workers to master and 'workers' "
        "means each process will output its own log",
    )

    # * config script ******************************************************************
    parser_cfig = subparsers.add_parser(
        "config",
        parents=[parser_log],
        help="fast configuration of parameter file for smooth model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_cfig.add_argument(
        "-o", "--output", type=str, default="input.json", help="the output json file"
    )

    # * transfer script ****************************************************************
    parser_transfer = subparsers.add_parser(
        "transfer", parents=[parser_log], help="pass parameters to another model"
    )
    parser_transfer.add_argument(
        "-r",
        "--raw-model",
        default="raw_frozen_model.pb",
        type=str,
        help="the model receiving parameters",
    )
    parser_transfer.add_argument(
        "-O",
        "--old-model",
        default="old_frozen_model.pb",
        type=str,
        help="the model providing parameters",
    )
    parser_transfer.add_argument(
        "-o",
        "--output",
        default="frozen_model.pb",
        type=str,
        help="the model after passing parameters",
    )

    # * config parser ******************************************************************
    parser_train = subparsers.add_parser(
        "train",
        parents=[parser_log, parser_mpi_log],
        help="train a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format"
    )
    parser_train.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )
    parser_train.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided checkpoint.",
    )
    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.json",
        help="The output file of the parameters used in training.",
    )
    parser_train.add_argument(
        "-f",
        "--init-frz-model",
        type=str,
        default=None,
        help="Initialize the training from the frozen model.",
    )

    # * freeze script ******************************************************************
    parser_frz = subparsers.add_parser(
        "freeze",
        parents=[parser_log],
        help="freeze the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_frz.add_argument(
        "-c",
        "--checkpoint-folder",
        type=str,
        default=".",
        help="path to checkpoint folder",
    )
    parser_frz.add_argument(
        "-o",
        "--output",
        type=str,
        default="frozen_model.pb",
        help="name of graph, will output to the checkpoint folder",
    )
    parser_frz.add_argument(
        "-n",
        "--node-names",
        type=str,
        default=None,
        help="the frozen nodes, if not set, determined from the model type",
    )

    # * test script ********************************************************************
    parser_tst = subparsers.add_parser(
        "test",
        parents=[parser_log],
        help="test the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_tst.add_argument(
        "-m",
        "--model",
        default="frozen_model.pb",
        type=str,
        help="Frozen model file to import",
    )
    parser_tst.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system dir. Recursively detect systems in this directory",
    )
    parser_tst.add_argument(
        "-S", "--set-prefix", default="set", type=str, help="The set prefix"
    )
    parser_tst.add_argument(
        "-n", "--numb-test", default=100, type=int, help="The number of data for test"
    )
    parser_tst.add_argument(
        "-r", "--rand-seed", type=int, default=None, help="The random seed"
    )
    parser_tst.add_argument(
        "--shuffle-test", action="store_true", default=False, help="Shuffle test data"
    )
    parser_tst.add_argument(
        "-d",
        "--detail-file",
        type=str,
        default=None,
        help="File where details of energy force and virial accuracy will be written",
    )
    parser_tst.add_argument(
        "-a",
        "--atomic",
        action="store_true",
        default=False,
        help="Test the accuracy of atomic label, i.e. energy / tensor (dipole, polar)",
    )

    # * compress model *****************************************************************
    # Compress a model, which including tabulating the embedding-net.
    # The table is composed of fifth-order polynomial coefficients and is assembled
    # from two sub-tables. The first table takes the step(parameter) as it's uniform
    # step, while the second table takes 10 * step as it\s uniform step
    #  The range of the first table is automatically detected by deepmd-kit, while the
    # second table ranges from the first table's upper boundary(upper) to the
    # extrapolate(parameter) * upper.
    parser_compress = subparsers.add_parser(
        "compress",
        parents=[parser_log, parser_mpi_log],
        help="compress a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_compress.add_argument(
        "-i",
        "--input",
        default="frozen_model.pb",
        type=str,
        help="The original frozen model, which will be compressed by the code",
    )
    parser_compress.add_argument(
        "-o",
        "--output",
        default="frozen_model_compressed.pb",
        type=str,
        help="The compressed model",
    )
    parser_compress.add_argument(
        "-s",
        "--step",
        default=0.01,
        type=float,
        help="Model compression uses fifth-order polynomials to interpolate the embedding-net. " 
        "It introduces two tables with different step size to store the parameters of the polynomials. "
        "The first table covers the range of the training data, while the second table is an extrapolation of the training data. "
        "The domain of each table is uniformly divided by a given step size. "
        "And the step(parameter) denotes the step size of the first table and the second table will "
        "use 10 * step as it's step size to save the memory. "
        "Usually the value ranges from 0.1 to 0.001. " 
        "Smaller step means higher accuracy and bigger model size",
    )
    parser_compress.add_argument(
        "-e",
        "--extrapolate",
        default=5,
        type=int,
        help="The domain range of the first table is automatically detected by the code: [d_low, d_up]. "
        "While the second table ranges from the first table's upper boundary(d_up) to the extrapolate(parameter) * d_up: [d_up, extrapolate * d_up]",
    )
    parser_compress.add_argument(
        "-f",
        "--frequency",
        default=-1,
        type=int,
        help="The frequency of tabulation overflow check(Whether the input environment "
        "matrix overflow the first or second table range). "
        "By default do not check the overflow",
    )
    parser_compress.add_argument(
        "-c",
        "--checkpoint-folder",
        type=str,
        default="model-compression",
        help="path to checkpoint folder",
    )
    parser_compress.add_argument(
        "-t",
        "--training-script",
        type=str,
        default=None,
        help="The training script of the input frozen model",
    )

    # * print docs script **************************************************************
    parsers_doc = subparsers.add_parser(
        "doc-train-input",
        parents=[parser_log],
        help="print the documentation (in rst format) of input training parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parsers_doc.add_argument(
        "--out-type", 
        default="rst", 
        type=str, 
        help="The output type"
    )

    # * make model deviation ***********************************************************
    parser_model_devi = subparsers.add_parser(
        "model-devi",
        parents=[parser_log],
        help="calculate model deviation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_model_devi.add_argument(
        "-m",
        "--models",
        default=["graph.000.pb", "graph.001.pb", "graph.002.pb", "graph.003.pb"],
        nargs="+",
        type=str,
        help="Frozen models file to import",
    )
    parser_model_devi.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system directory, not support recursive detection.",
    )
    parser_model_devi.add_argument(
        "-S", "--set-prefix", default="set", type=str, help="The set prefix"
    )
    parser_model_devi.add_argument(
        "-o",
        "--output", 
        default="model_devi.out", 
        type=str, 
        help="The output file for results of model deviation"
    )
    parser_model_devi.add_argument(
        "-f",
        "--frequency",
        default=1,
        type=int,
        help="The trajectory frequency of the system"
    )

    # * convert models
    # supported: 1.2->2.0, 1.3->2.0
    parser_transform = subparsers.add_parser(
        'convert-from',
        parents=[parser_log],
        help='convert lower model version to supported version',
    )
    parser_transform.add_argument(
        'FROM',
        type = str,
        choices = ['1.2', '1.3'],
        help="The original model compatibility",
    )
    parser_transform.add_argument(
        '-i',
        "--input-model",
        default = "frozen_model.pb",
        type=str, 
		help = "the input model",
    )
    parser_transform.add_argument(
        "-o",
        "--output-model",
        default = "convert_out.pb",
        type=str, 
		help='the output model',
    )
    # --version
    parser.add_argument('--version', action='version', version='DeePMD-kit v%s' % __version__)

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args


def main():
    """DeePMD-Kit entry point.

    Raises
    ------
    RuntimeError
        if no command was input
    """
    args = parse_args()

    # do not set log handles for None, it is useless
    # log handles for train will be set separatelly
    # when the use of MPI will be determined in `RunOptions`
    if args.command not in (None, "train"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "train":
        train_dp(**dict_args)
    elif args.command == "freeze":
        freeze(**dict_args)
    elif args.command == "config":
        config(**dict_args)
    elif args.command == "test":
        test(**dict_args)
    elif args.command == "transfer":
        transfer(**dict_args)
    elif args.command == "compress":
        compress(**dict_args)
    elif args.command == "doc-train-input":
        doc_train_input(**dict_args)
    elif args.command == "model-devi":
        make_model_devi(**dict_args)
    elif args.command == "convert-from":
        convert(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
