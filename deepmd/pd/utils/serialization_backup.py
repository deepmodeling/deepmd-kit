# SPDX-License-Identifier: LGPL-3.0-or-later

import paddle

from deepmd.pd.model.model.model import (
    BaseModel,
)


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary.

    Parameters
    ----------
    model_file : str
        The model file to be serialized.

    Returns
    -------
    dict
        The serialized model data.
    """
    raise NotImplementedError("Paddle do not support jit.export yet.")


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    if not model_file.endswith(".json"):
        raise ValueError("Paddle backend only supports converting .json file")
    model: paddle.nn.Layer = BaseModel.deserialize(data["model"])
    model.eval()
    # JIT will happy in this way...
    if "min_nbor_dist" in data.get("@variables", {}):
        model.register_buffer(
            "buffer_min_nbor_dist",
            paddle.to_tensor(
                float(data["@variables"]["min_nbor_dist"]),
            ),
        )
    paddle.set_flags(
        {
            "FLAGS_save_cf_stack_op": 1,
            "FLAGS_prim_enable_dynamic": 1,
            "FLAGS_enable_pir_api": 1,
        }
    )
    from paddle.static import (
        InputSpec,
    )

    # import numpy as np
    # coord = (
    #     np.array(
    #         [
    #             12.83,
    #             2.56,
    #             2.18,
    #             12.09,
    #             2.87,
    #             2.7,
    #             00.25,
    #             3.32,
    #             1.68,
    #             3.36,
    #             3.00,
    #             1.81,
    #             3.51,
    #             2.51,
    #             2.60,
    #             4.27,
    #             3.22,
    #             1.56,
    #         ]
    #     )
    #     .astype("float64")
    #     .reshape([-1, 6, 3])
    # )
    # atype = np.array([0, 1, 1, 0, 1, 1]).astype("int64").reshape([-1, 6])
    # box = (
    #     np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
    #     .astype("float64")
    #     .reshape([-1, 9])
    # )
    # out = model.forward(
    #     paddle.to_tensor(coord),
    #     paddle.to_tensor(atype),
    #     paddle.to_tensor(box),
    #     do_atomic_virial=True,
    # )
    # for k, v in out.items():
    #     print(k, v.shape, v.dtype)
    # exit()
    # output of forward
    # atom_energy: fetch_name_0 (1, 6, 1) float64
    # atom_virial: fetch_name_1 (1, 6, 1, 9) float64
    # energy: fetch_name_2 (1, 1) float64
    # force: fetch_name_3 (1, 6, 3) float64
    # mask: fetch_name_4 (1, 6) int32
    # virial: fetch_name_5 (1, 9) float64
    model.forward = paddle.jit.to_static(
        model.forward,
        full_graph=True,
        input_spec=[
            InputSpec([1, -1, 3], dtype="float64", name="coord"),
            InputSpec([1, -1], dtype="int64", name="atype"),
            InputSpec([1, 9], dtype="float64", name="box"),
            None,
            None,
            True,
        ],
    )

    extended_coord = paddle.load(
        "/workspace/hesensen/deepmd_partx/deepmd-kit-tmp/examples/water/se_e2_a/extended_coord.pddata"
    )
    extended_atype = paddle.load(
        "/workspace/hesensen/deepmd_partx/deepmd-kit-tmp/examples/water/se_e2_a/extended_atype.pddata"
    )
    nlist = paddle.load(
        "/workspace/hesensen/deepmd_partx/deepmd-kit-tmp/examples/water/se_e2_a/nlist.pddata"
    )
    outs = model.forward_lower(
        extended_coord,
        extended_atype,
        nlist,
        do_atomic_virial=True,
    )
    # for k, v in outs.items():
    #     print(k, v.shape, v.dtype)
    # exit()
    # output of forward_lower
    # fetch_name_0: atom_energy [1, 192, 1] paddle.float64
    # fetch_name_1: energy [1, 1] paddle.float64
    # fetch_name_2: extended_force [1, 5184, 3] paddle.float64
    # fetch_name_3: extended_virial [1, 5184, 1, 9] paddle.float64
    # fetch_name_4: virial [1, 9] paddle.float64
    model.forward_lower = paddle.jit.to_static(
        model.forward_lower,
        full_graph=True,
        input_spec=[
            InputSpec([1, -1, 3], dtype="float64", name="coord"),
            InputSpec([1, -1], dtype="int32", name="atype"),
            InputSpec([1, -1, -1], dtype="int32", name="nlist"),
            None,
            None,
            None,
            True,
            None,
        ],
    )
    paddle.jit.save(
        model,
        model_file.split(".json")[0],
    )
