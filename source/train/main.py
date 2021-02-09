"""DeePMD-Kit entry point module."""

import argparse

from .config import config
from .doc import doc_train_input
from .freeze import freeze
from .test import test
from .train import train
from .transform import transform


def main():
    """DeePMD-Kit entry point.

    Raises
    ------
    RuntimeError
        if no command was input
    """
    parser = argparse.ArgumentParser(
        description="DeePMD-kit: A deep learning package for many-body potential energy"
        " representation and molecular dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # * config script ******************************************************************
    parser_cfig = subparsers.add_parser(
        "config", help="fast configuration of parameter file for smooth model"
    )
    parser_cfig.add_argument(
        "-o",
        "--output",
        type=str,
        default="input.json",
        help="the output json file"
    )

    # * transform script ***************************************************************
    parser_transform = subparsers.add_parser(
        "transform", help="pass parameters to another model"
    )
    parser_transform.add_argument(
        "-r",
        "--raw-model",
        default="raw_frozen_model.pb",
        type=str,
        help="the model receiving parameters",
    )
    parser_transform.add_argument(
        "-o",
        "--old-model",
        default="old_frozen_model.pb",
        type=str,
        help="the model providing parameters",
    )
    parser_transform.add_argument(
        "-n",
        "--output",
        default="frozen_model.pb",
        type=str,
        help="the model after passing parameters",
    )

    # * config parser ******************************************************************
    parser_train = subparsers.add_parser("train", help="train a model")
    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format"
    )
    parser_train.add_argument(
        "-im"
        "--init-model",
        type=str,
        default=False,
        help="Initialize the model by the provided checkpoint.",
    )
    parser_train.add_argument(
        "-r"
        "--restart",
        type=str,
        default=False,
        help="Restart the training from the provided checkpoint."
    )
    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.json",
        help="The output file of the parameters used in training.",
    )

    # * freeze script ******************************************************************
    parser_frz = subparsers.add_parser("freeze", help="freeze the model")
    parser_frz.add_argument(
        "-f",
        "--folder",
        type=str,
        default=".",
        help="path to checkpoint folder"
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
        "--nodes",
        type=str,
        help="the frozen nodes, if not set, determined from the model type",
    )

    # * test script ********************************************************************
    parser_tst = subparsers.add_parser("test", help="test the model")
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
        "-S",
        "--set-prefix",
        default="set",
        type=str,
        help="The set prefix"
    )
    parser_tst.add_argument(
        "-n",
        "--numb-test",
        default=100,
        type=int,
        help="The number of data for test"
    )
    parser_tst.add_argument(
        "-r",
        "--rand-seed",
        type=int,
        help="The random seed")
    parser_tst.add_argument(
        "-st",
        "--shuffle-test",
        action="store_true",
        help="Shuffle test data"
    )
    parser_tst.add_argument(
        "-d",
        "--detail-file",
        type=str,
        help="The file containing details of energy force and virial accuracy",
    )
    parser_tst.add_argument(
        "-a",
        "--atomic-energy",
        action="store_true",
        help="Test the accuracy of atomic energy",
    )

    # * print docs script **************************************************************
    subparsers.add_parser(
        "doc-train-input",
        help="print the documentation (in rst format) of input training parameters.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    if args.command == "train":
        train(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "config":
        config(args)
    elif args.command == "test":
        test(args)
    elif args.command == "transform":
        transform(args)
    elif args.command == "doc-train-input":
        doc_train_input(args)
    else:
        raise RuntimeError(f"unknown command {args.command}s)
