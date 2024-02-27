# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import textwrap
from typing import (
    Optional,
)

from google.protobuf import (
    text_format,
)
from packaging.specifiers import (
    SpecifierSet,
)
from packaging.version import parse as parse_version

from deepmd.tf import (
    __version__,
)
from deepmd.tf.env import (
    tf,
)

log = logging.getLogger(__name__)


def detect_model_version(input_model: str):
    """Detect DP graph version.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    """
    convert_pb_to_pbtxt(input_model, "frozen_model.pbtxt")
    version = None
    with open("frozen_model.pbtxt") as fp:
        file_content = fp.read()
    if file_content.find("DescrptNorot") > -1:
        version = parse_version("0.12")
    elif (
        file_content.find("fitting_attr/dfparam") > -1
        and file_content.find("fitting_attr/daparam") == -1
    ):
        version = parse_version("1.0")
    elif file_content.find("model_attr/model_version") == -1:
        name_dsea = file_content.find('name: "DescrptSeA"')
        post_dsea = file_content[name_dsea:]
        post_dsea2 = post_dsea[:300].find(r"}")
        search_double = post_dsea[:post_dsea2]
        if search_double.find("DT_DOUBLE") == -1:
            version = parse_version("1.2")
        else:
            version = parse_version("1.3")
    elif file_content.find('string_val: "1.0"') > -1:
        version = parse_version("2.0")
    elif file_content.find('string_val: "1.1"') > -1:
        version = parse_version("2.1")
    return version


def convert_to_21(input_model: str, output_model: str, version: Optional[str] = None):
    """Convert DP graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    version : str
        version of the input graph, if not specified, it will be detected automatically
    """
    if version is None:
        version = detect_model_version(input_model)
    else:
        convert_pb_to_pbtxt(input_model, "frozen_model.pbtxt")
    if version is None:
        raise ValueError(
            "The version of the DP graph %s cannot be detected. Please do the conversion manually."
            % (input_model)
        )
    if version in SpecifierSet("<1.0"):
        convert_dp012_to_dp10("frozen_model.pbtxt")
    if version in SpecifierSet("<1.1"):
        convert_dp10_to_dp11("frozen_model.pbtxt")
    if version in SpecifierSet("<1.3"):
        convert_dp12_to_dp13("frozen_model.pbtxt")
    if version in SpecifierSet("<2.0"):
        convert_dp13_to_dp20("frozen_model.pbtxt")
    if version in SpecifierSet("<2.1"):
        convert_dp20_to_dp21("frozen_model.pbtxt")
    convert_pbtxt_to_pb("frozen_model.pbtxt", output_model)
    if os.path.isfile("frozen_model.pbtxt"):
        os.remove("frozen_model.pbtxt")
    log.info(
        "the converted output model (%s support) is saved in %s",
        __version__,
        output_model,
    )


def convert_13_to_21(input_model: str, output_model: str):
    """Convert DP 1.3 graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_to_21(input_model, output_model, version="1.3")


def convert_12_to_21(input_model: str, output_model: str):
    """Convert DP 1.2 graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_to_21(input_model, output_model, version="1.2")


def convert_10_to_21(input_model: str, output_model: str):
    """Convert DP 1.0 graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_to_21(input_model, output_model, version="1.0")


def convert_012_to_21(input_model: str, output_model: str):
    """Convert DP 0.12 graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_to_21(input_model, output_model, version="0.12")


def convert_20_to_21(input_model: str, output_model: str):
    """Convert DP 2.0 graph to 2.1 graph.

    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_to_21(input_model, output_model, version="2.0")


def convert_pb_to_pbtxt(pbfile: str, pbtxtfile: str):
    """Convert DP graph to graph text.

    Parameters
    ----------
    pbfile : str
        filename of the input graph
    pbtxtfile : str
        filename of the output graph text
    """
    with tf.gfile.GFile(pbfile, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        tf.train.write_graph(graph_def, "./", pbtxtfile, as_text=True)


def convert_pbtxt_to_pb(pbtxtfile: str, pbfile: str):
    """Convert DP graph text to graph.

    Parameters
    ----------
    pbtxtfile : str
        filename of the input graph text
    pbfile : str
        filename of the output graph
    """
    with tf.gfile.GFile(pbtxtfile, "r") as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, "./", pbfile, as_text=False)


def convert_dp012_to_dp10(file: str):
    """Convert DP 0.12 graph text to 1.0 graph text.

    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file) as fp:
        file_content = fp.read()
    # note: atom_energy must be put before energy,
    # otherwise atom_energy_test -> atom_o_energy
    file_content = (
        file_content.replace("DescrptNorot", "DescrptSeA")
        .replace("ProdForceNorot", "ProdForceSeA")
        .replace("ProdVirialNorot", "ProdVirialSeA")
        .replace("t_rcut", "descrpt_attr/rcut")
        .replace("t_ntypes", "descrpt_attr/ntypes")
        .replace("atom_energy_test", "o_atom_energy")
        .replace("atom_virial_test", "o_atom_virial")
        .replace("energy_test", "o_energy")
        .replace("force_test", "o_force")
        .replace("virial_test", "o_virial")
    )
    file_content += textwrap.dedent(
        """\
      node {
        name: "fitting_attr/dfparam"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_INT32
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {
              }
              int_val: 0
            }
          }
        }
      }
      """
    )
    file_content += textwrap.dedent(
        """\
      node {
        name: "model_attr/model_type"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_STRING
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {
              }
              string_val: "ener"
            }
          }
        }
      }
      """
    )
    file_content += textwrap.dedent(
        """\
      node {
        name: "model_attr/tmap"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_STRING
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {
              }
              string_val: ""
            }
          }
        }
      }
      """
    )
    with open(file, "w") as fp:
        fp.write(file_content)


def convert_dp10_to_dp11(file: str):
    """Convert DP 1.0 graph text to 1.1 graph text.

    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file, "a") as f:
        f.write(
            textwrap.dedent(
                """\
          node {
            name: "fitting_attr/daparam"
            op: "Const"
            attr {
              key: "dtype"
              value {
                type: DT_INT32
              }
            }
            attr {
              key: "value"
              value {
                tensor {
                  dtype: DT_INT32
                  tensor_shape {
                  }
                  int_val: 0
                }
              }                                                                                                                                                 }
          }
          """
            )
        )


def convert_dp12_to_dp13(file: str):
    """Convert DP 1.2 graph text to 1.3 graph text.

    Parameters
    ----------
    file : str
        filename of the graph text
    """
    file_data = ""
    with open(file, encoding="utf-8") as f:
        ii = 0
        lines = f.readlines()
        while ii < len(lines):
            line = lines[ii]
            file_data += line
            ii += 1
            if "name" in line and (
                "DescrptSeA" in line
                or "ProdForceSeA" in line
                or "ProdVirialSeA" in line
            ):
                while not ("attr" in lines[ii] and "{" in lines[ii]):
                    file_data += lines[ii]
                    ii += 1
                file_data += "  attr {\n"
                file_data += '    key: "T"\n'
                file_data += "    value {\n"
                file_data += "      type: DT_DOUBLE\n"
                file_data += "    }\n"
                file_data += "  }\n"
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def convert_dp13_to_dp20(fname: str):
    """Convert DP 1.3 graph text to 2.0 graph text.

    Parameters
    ----------
    fname : str
        filename of the graph text
    """
    with open(fname) as fp:
        file_content = fp.read()
    file_content += textwrap.dedent(
        """\
      node {
        name: "model_attr/model_version"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_STRING
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {
              }
              string_val: "1.0"
            }
          }
        }
      }
      """
    )
    file_content = file_content.replace("DescrptSeA", "ProdEnvMatA").replace(
        "DescrptSeR", "ProdEnvMatR"
    )
    with open(fname, "w") as fp:
        fp.write(file_content)


def convert_dp20_to_dp21(fname: str):
    with open(fname) as fp:
        file_content = fp.read()
    old_model_version_node = textwrap.dedent(
        """\
      node {
        name: "model_attr/model_version"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_STRING
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {
              }
              string_val: "1.0"
            }
          }
        }
      }
      """
    )
    new_model_version_node = textwrap.dedent(
        """\
      node {
        name: "model_attr/model_version"
        op: "Const"
        attr {
          key: "dtype"
          value {
            type: DT_STRING
          }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {
              }
              string_val: "1.1"
            }
          }
        }
      }
      """
    )
    file_content = (
        file_content.replace(old_model_version_node, new_model_version_node)
        .replace("TabulateFusion", "TabulateFusionSeA")
        .replace("TabulateFusionGrad", "TabulateFusionSeAGrad")
        .replace("TabulateFusionGradGrad", "TabulateFusionSeAGradGrad")
    )
    with open(fname, "w") as fp:
        fp.write(file_content)
