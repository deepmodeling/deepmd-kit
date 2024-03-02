# SPDX-License-Identifier: LGPL-3.0-or-later
"""The entry points for DeePMD-kit.

If only printing the help message, this module does not call
the main DeePMD-kit module to avoid the slow import of TensorFlow.
"""
import argparse
import logging
import os
import textwrap
from collections import (
    defaultdict,
)
from typing import (
    Dict,
    List,
    Optional,
    Type,
)

from deepmd.backend.backend import (
    Backend,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"


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


class RawTextArgumentDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """This formatter is used to print multile-line help message with default value."""


BACKENDS: Dict[str, Type[Backend]] = Backend.get_backends_by_feature(
    Backend.Feature.ENTRY_POINT
)
BACKEND_TABLE: Dict[str, str] = {kk: vv.name.lower() for kk, vv in BACKENDS.items()}


class BackendOption(argparse.Action):
    """Map backend alias to unique name."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, BACKEND_TABLE[values])


def main_parser() -> argparse.ArgumentParser:
    """DeePMD-Kit commandline options argument parser.

    Returns
    -------
    argparse.ArgumentParser
        main parser of DeePMD-kit
    """
    parser = argparse.ArgumentParser(
        description="DeePMD-kit: A deep learning package for many-body potential energy"
        " representation and molecular dynamics",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        Use --tf or --pt to choose the backend:
            dp --tf train input.json
            dp --pt train input.json
        """
        ),
    )

    # default backend is TF for compatibility
    default_backend = os.environ.get("DP_BACKEND", "tensorflow").lower()
    if default_backend not in BACKEND_TABLE.keys():
        raise ValueError(
            f"Unknown backend {default_backend}. "
            "Please set DP_BACKEND to either tensorflow or pytorch."
        )

    parser_backend = parser.add_mutually_exclusive_group()
    parser_backend.add_argument(
        "-b",
        "--backend",
        choices=list(BACKEND_TABLE.keys()),
        action=BackendOption,
        default=default_backend,
        help=(
            "The backend of the model. Default can be set by environment variable "
            "DP_BACKEND."
        ),
    )

    BACKEND_ALIAS: Dict[str, List[str]] = defaultdict(list)
    for alias, backend in BACKEND_TABLE.items():
        BACKEND_ALIAS[backend].append(alias)
    for backend, alias in BACKEND_ALIAS.items():
        parser_backend.add_argument(
            *[f"--{aa}" for aa in alias],
            action="store_const",
            dest="backend",
            const=backend,
            help=f"Alias for --backend {backend}",
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

    # * transfer script ****************************************************************
    parser_transfer = subparsers.add_parser(
        "transfer",
        parents=[parser_log],
        help="(Supported backend: TensorFlow) pass parameters to another model",
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
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp train input.json
            dp train input.json --restart model.ckpt
            dp train input.json --init-model model.ckpt
        """
        ),
    )
    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format"
    )
    parser_train_subgroup = parser_train.add_mutually_exclusive_group()
    parser_train_subgroup.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided path prefix of checkpoint files.",
    )
    parser_train_subgroup.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided path prefix of checkpoint files.",
    )
    parser_train_subgroup.add_argument(
        "-f",
        "--init-frz-model",
        type=str,
        default=None,
        help="Initialize the training from the frozen model.",
    )
    parser_train_subgroup.add_argument(
        "-t",
        "--finetune",
        type=str,
        default=None,
        help="Finetune the frozen pretrained model.",
    )
    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.json",
        help="The output file of the parameters used in training.",
    )
    parser_train.add_argument(
        "--skip-neighbor-stat",
        action="store_true",
        help="Skip calculating neighbor statistics. Sel checking, automatic sel, and model compression will be disabled.",
    )
    parser_train.add_argument(
        # -m has been used by mpi-log
        "--model-branch",
        type=str,
        default="",
        help="(Supported backend: PyTorch) Model branch chosen for fine-tuning if multi-task. If not specified, it will re-init the fitting net.",
    )
    parser_train.add_argument(
        "--force-load",
        action="store_true",
        help="(Supported backend: PyTorch) Force load from ckpt, other missing tensors will init from scratch",
    )

    # * freeze script ******************************************************************
    parser_frz = subparsers.add_parser(
        "freeze",
        parents=[parser_log],
        help="freeze the model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp freeze
            dp freeze -o graph.pb
        """
        ),
    )
    parser_frz.add_argument(
        "-c",
        "--checkpoint-folder",
        "--checkpoint",
        type=str,
        default=".",
        help="Path to checkpoint, either a folder containing checkpoint or the checkpoint prefix",
    )
    parser_frz.add_argument(
        "-o",
        "--output",
        type=str,
        default="frozen_model",
        help="Filename (prefix) of the output model file. TensorFlow backend: suffix is .pb; PyTorch backend: suffix is .pth",
    )
    parser_frz.add_argument(
        "-n",
        "--node-names",
        type=str,
        default=None,
        help="(Supported backend: TensorFlow) the frozen nodes, if not set, determined from the model type",
    )
    parser_frz.add_argument(
        "-w",
        "--nvnmd-weight",
        type=str,
        default=None,
        help="(Supported backend: TensorFlow) the name of weight file (.npy), if set, save the model's weight into the file",
    )
    parser_frz.add_argument(
        "--united-model",
        action="store_true",
        default=False,
        help="(Supported backend: TensorFlow) When in multi-task mode, freeze all nodes into one united model",
    )
    parser_frz.add_argument(
        "--head",
        default=None,
        type=str,
        help="(Supported backend: PyTorch) Task head to freeze if in multi-task mode.",
    )

    # * test script ********************************************************************
    parser_tst = subparsers.add_parser(
        "test",
        parents=[parser_log],
        help="test the model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp test -m graph.pb -s /path/to/system -n 30
        """
        ),
    )
    parser_tst.add_argument(
        "-m",
        "--model",
        default="frozen_model",
        type=str,
        help="Frozen model file (prefix) to import. TensorFlow backend: suffix is .pb; PyTorch backend: suffix is .pth.",
    )
    parser_tst_subgroup = parser_tst.add_mutually_exclusive_group()
    parser_tst_subgroup.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system dir. Recursively detect systems in this directory",
    )
    parser_tst_subgroup.add_argument(
        "-f",
        "--datafile",
        default=None,
        type=str,
        help="The path to file of test list.",
    )
    parser_tst.add_argument(
        "-S",
        "--set-prefix",
        default="set",
        type=str,
        help="(Supported backend: TensorFlow) The set prefix",
    )
    parser_tst.add_argument(
        "-n",
        "--numb-test",
        default=0,
        type=int,
        help="The number of data for test. 0 means all data.",
    )
    parser_tst.add_argument(
        "-r",
        "--rand-seed",
        type=int,
        default=None,
        help="(Supported backend: TensorFlow) The random seed",
    )
    parser_tst.add_argument(
        "--shuffle-test", action="store_true", default=False, help="Shuffle test data"
    )
    parser_tst.add_argument(
        "-d",
        "--detail-file",
        type=str,
        default=None,
        help="The prefix to files where details of energy, force and virial accuracy/accuracy per atom will be written",
    )
    parser_tst.add_argument(
        "-a",
        "--atomic",
        action="store_true",
        default=False,
        help="(Supported backend: TensorFlow) Test the accuracy of atomic label, i.e. energy / tensor (dipole, polar)",
    )
    parser_tst.add_argument(
        "-i",
        "--input_script",
        type=str,
        help="(Supported backend: PyTorch) The input script of the model",
    )
    parser_tst.add_argument(
        "--head",
        default=None,
        type=str,
        help="(Supported backend: PyTorch) Task head to test if in multi-task mode.",
    )

    # * compress model *****************************************************************
    # Compress a model, which including tabulating the embedding-net.
    # The table is composed of fifth-order polynomial coefficients and is assembled
    # from two sub-tables. The first table takes the step(parameter) as it's uniform
    # step, while the second table takes 10 * step as it\s uniform step
    #  The range of the first table is automatically detected by deepmd-kit, while the
    # second table ranges from the first table's upper boundary(upper) to the
    # extrapolate(parameter) * upper.
    parser_compress = subparsers.add_parser(
        "compress",
        parents=[parser_log, parser_mpi_log],
        help="(Supported backend: TensorFlow) compress a model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp compress
            dp compress -i graph.pb -o compressed.pb
        """
        ),
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
        choices=["rst", "json"],
        type=str,
        help="The output type",
    )

    # * make model deviation ***********************************************************
    parser_model_devi = subparsers.add_parser(
        "model-devi",
        parents=[parser_log],
        help="calculate model deviation",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp model-devi -m graph.000.pb graph.001.pb graph.002.pb graph.003.pb -s ./data -o model_devi.out
        """
        ),
    )
    parser_model_devi.add_argument(
        "-m",
        "--models",
        default=["graph.000", "graph.001", "graph.002", "graph.003"],
        nargs="+",
        type=str,
        help="Frozen models file (prefix) to import. TensorFlow backend: suffix is .pb; PyTorch backend: suffix is .pth.",
    )
    parser_model_devi.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system directory. Recursively detect systems in this directory.",
    )
    parser_model_devi.add_argument(
        "-S", "--set-prefix", default="set", type=str, help="The set prefix"
    )
    parser_model_devi.add_argument(
        "-o",
        "--output",
        default="model_devi.out",
        type=str,
        help="The output file for results of model deviation",
    )
    parser_model_devi.add_argument(
        "-f",
        "--frequency",
        default=1,
        type=int,
        help="The trajectory frequency of the system",
    )
    parser_model_devi.add_argument(
        "--real_error",
        action="store_true",
        default=False,
        help="Calculate the RMS real error of the model. The real data should be given in the systems.",
    )
    parser_model_devi.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Print the force model deviation of each atom.",
    )
    parser_model_devi.add_argument(
        "--relative",
        type=float,
        help="Calculate the relative model deviation of force. The level parameter for computing the relative model deviation of the force should be given.",
    )
    parser_model_devi.add_argument(
        "--relative_v",
        type=float,
        help="Calculate the relative model deviation of virial. The level parameter for computing the relative model deviation of the virial should be given.",
    )

    # * convert models
    parser_transform = subparsers.add_parser(
        "convert-from",
        parents=[parser_log],
        help="(Supported backend: TensorFlow) convert lower model version to supported version",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp convert-from -i graph.pb -o graph_new.pb
            dp convert-from auto -i graph.pb -o graph_new.pb
            dp convert-from 1.0 -i graph.pb -o graph_new.pb
        """
        ),
    )
    parser_transform.add_argument(
        "FROM",
        nargs="?",
        default="auto",
        type=str,
        choices=["auto", "0.12", "1.0", "1.1", "1.2", "1.3", "2.0", "pbtxt"],
        help="The original model compatibility",
    )
    parser_transform.add_argument(
        "-i",
        "--input-model",
        default="frozen_model.pb",
        type=str,
        help="the input model",
    )
    parser_transform.add_argument(
        "-o",
        "--output-model",
        default="convert_out.pb",
        type=str,
        help="the output model\nIf OUTPUT_MODEL ends with '.pbtxt', the provided model will be converted to pbtxt format, without version conversion.",
    )

    # neighbor_stat
    parser_neighbor_stat = subparsers.add_parser(
        "neighbor-stat",
        parents=[parser_log],
        help="Calculate neighbor statistics",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp neighbor-stat -s data -r 6.0 -t O H
        """
        ),
    )
    parser_neighbor_stat.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system dir. Recursively detect systems in this directory",
    )
    parser_neighbor_stat.add_argument(
        "-r",
        "--rcut",
        type=float,
        required=True,
        help="cutoff radius",
    )
    parser_neighbor_stat.add_argument(
        "-t",
        "--type-map",
        type=str,
        nargs="+",
        required=True,
        help="type map",
    )
    parser_neighbor_stat.add_argument(
        "--mixed-type",
        "--one-type",
        action="store_true",
        default=False,
        help="treat all types as a single type. Used with se_atten descriptor.",
    )

    # --version
    parser.add_argument(
        "--version", action="version", version="DeePMD-kit v%s" % __version__
    )

    # * train nvnmd script ******************************************************************
    parser_train_nvnmd = subparsers.add_parser(
        "train-nvnmd",
        parents=[parser_log],
        help="(Supported backend: TensorFlow) train nvnmd model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp train-nvnmd input_cnn.json -s s1
            dp train-nvnmd input_qnn.json -s s2
            dp train-nvnmd input_cnn.json -s s1 --restart model.ckpt
            dp train-nvnmd input_cnn.json -s s2 --init-model model.ckpt
        """
        ),
    )
    parser_train_nvnmd.add_argument(
        "INPUT", help="the input parameter file in json format"
    )
    parser_train_nvnmd.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided path prefix of checkpoint files.",
    )
    parser_train_nvnmd.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided prefix of checkpoint files.",
    )
    parser_train_nvnmd.add_argument(
        "-s",
        "--step",
        default="s1",
        type=str,
        choices=["s1", "s2"],
        help="steps to train model of NVNMD: s1 (train CNN), s2 (train QNN)",
    )
    parser_train_nvnmd.add_argument(
        "--skip-neighbor-stat",
        action="store_true",
        help="Skip calculating neighbor statistics. Sel checking, automatic sel, and model compression will be disabled.",
    )

    # gui
    parser_gui = subparsers.add_parser(
        "gui",
        parents=[parser_log],
        help="Serve DP-GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_gui.add_argument(
        "-p",
        "--port",
        type=int,
        default=6042,
        help="The port to serve DP-GUI on.",
    )
    parser_gui.add_argument(
        "--bind_all",
        action="store_true",
        help=(
            "Serve on all public interfaces. This will expose your DP-GUI instance "
            "to the network on both IPv4 and IPv6 (where available)."
        ),
    )

    # convert_backend
    parser_convert_backend = subparsers.add_parser(
        "convert-backend",
        parents=[parser_log],
        help="Convert model to another backend.",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            dp convert-backend model.pb model.pth
            dp convert-backend model.pb model.dp
        """
        ),
    )
    parser_convert_backend.add_argument("INPUT", help="The input model file.")
    parser_convert_backend.add_argument("OUTPUT", help="The output model file.")
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args


def main():
    """DeePMD-kit new entry point.

    Raises
    ------
    RuntimeError
        if no command was input
    """
    args = parse_args()

    if args.backend not in BACKEND_TABLE:
        raise ValueError(f"Unknown backend {args.backend}")

    if args.command in (
        "test",
        "doc-train-input",
        "model-devi",
        "neighbor-stat",
        "gui",
        "convert-backend",
    ):
        # common entrypoints
        from deepmd.entrypoints.main import main as deepmd_main
    elif args.command in (
        "train",
        "freeze",
        "transfer",
        "compress",
        "convert-from",
        "train-nvnmd",
    ):
        deepmd_main = BACKENDS[args.backend]().entry_point_hook
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")

    deepmd_main(args)
