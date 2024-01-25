# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.tf.utils.convert import (
    convert_10_to_21,
    convert_012_to_21,
    convert_12_to_21,
    convert_13_to_21,
    convert_20_to_21,
    convert_pb_to_pbtxt,
    convert_pbtxt_to_pb,
    convert_to_21,
)


def convert(
    *,
    FROM: str,
    input_model: str,
    output_model: str,
    **kwargs,
):
    if output_model[-6:] == ".pbtxt":
        if input_model[-6:] != ".pbtxt":
            convert_pb_to_pbtxt(input_model, output_model)
        else:
            raise RuntimeError("input model is already pbtxt")
    else:
        if FROM == "auto":
            convert_to_21(input_model, output_model)
        elif FROM == "0.12":
            convert_012_to_21(input_model, output_model)
        elif FROM == "1.0":
            convert_10_to_21(input_model, output_model)
        elif FROM in ["1.1", "1.2"]:
            # no difference between 1.1 and 1.2
            convert_12_to_21(input_model, output_model)
        elif FROM == "1.3":
            convert_13_to_21(input_model, output_model)
        elif FROM == "2.0":
            convert_20_to_21(input_model, output_model)
        elif FROM == "pbtxt":
            convert_pbtxt_to_pb(input_model, output_model)
        else:
            raise RuntimeError("unsupported model version " + FROM)
