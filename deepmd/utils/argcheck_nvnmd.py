# SPDX-License-Identifier: LGPL-3.0-or-later
from dargs import (
    Argument,
)


def nvnmd_args():
    doc_version = (
        "configuration the nvnmd version (0 | 1), 0 for 4 types, 1 for 32 types"
    )
    doc_max_nnei = "configuration the max number of neighbors, 128|256 for version 0, 128 for version 1"
    doc_net_size_file = (
        "configuration the number of nodes of fitting_net, just can be set as 128"
    )
    doc_map_file = "A file containing the mapping tables to replace the calculation of embedding nets"
    doc_config_file = "A file containing the parameters about how to implement the model in certain hardware"
    doc_weight_file = "a *.npy file containing the weights of the model"
    doc_enable = "enable the nvnmd training"
    doc_restore_descriptor = (
        "enable to restore the parameter of embedding_net from weight.npy"
    )
    doc_restore_fitting_net = (
        "enable to restore the parameter of fitting_net from weight.npy"
    )
    doc_quantize_descriptor = "enable the quantizatioin of descriptor"
    doc_quantize_fitting_net = "enable the quantizatioin of fitting_net"
    args = [
        Argument("version", int, optional=False, default=0, doc=doc_version),
        Argument("max_nnei", int, optional=False, default=128, doc=doc_max_nnei),
        Argument("net_size", int, optional=False, default=128, doc=doc_net_size_file),
        Argument("map_file", str, optional=False, default="none", doc=doc_map_file),
        Argument(
            "config_file", str, optional=False, default="none", doc=doc_config_file
        ),
        Argument(
            "weight_file", str, optional=False, default="none", doc=doc_weight_file
        ),
        Argument("enable", bool, optional=False, default=False, doc=doc_enable),
        Argument(
            "restore_descriptor",
            bool,
            optional=False,
            default=False,
            doc=doc_restore_descriptor,
        ),
        Argument(
            "restore_fitting_net",
            bool,
            optional=False,
            default=False,
            doc=doc_restore_fitting_net,
        ),
        Argument(
            "quantize_descriptor",
            bool,
            optional=False,
            default=False,
            doc=doc_quantize_descriptor,
        ),
        Argument(
            "quantize_fitting_net",
            bool,
            optional=False,
            default=False,
            doc=doc_quantize_fitting_net,
        ),
    ]

    doc_nvnmd = "The nvnmd options."
    return Argument("nvnmd", dict, args, [], optional=True, doc=doc_nvnmd)
