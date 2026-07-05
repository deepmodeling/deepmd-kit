# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

import pytest
from tensorflow.core.protobuf import (
    saved_model_pb2,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)


def _saved_model_ops(model_dir: Path) -> set[str]:
    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString((model_dir / "saved_model.pb").read_bytes())
    ops = set()
    for meta_graph in saved_model.meta_graphs:
        ops.update(node.op for node in meta_graph.graph_def.node)
        for func in meta_graph.graph_def.library.function:
            ops.update(node.op for node in func.node_def)
    return ops


def test_savedmodel_export_contains_xla_call_module(tmp_path, monkeypatch) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("orbax.checkpoint")

    import jax.numpy as jnp

    from deepmd.jax.jax2tf import (
        serialization,
    )

    class DummyModel:
        def call_common_lower(
            self,
            coord,
            atype,
            nlist,
            mapping,
            fparam,
            aparam,
            do_atomic_virial: bool = False,
        ):
            del nlist, mapping, fparam, aparam, do_atomic_virial
            return {
                "coord_x": coord[..., :1]
                + jnp.asarray(atype[..., None], dtype=coord.dtype) * 0.0
            }

        def get_nnei(self) -> int:
            return 1

        def get_rcut(self) -> float:
            return 1.0

        def get_dim_fparam(self) -> int:
            return 0

        def get_dim_aparam(self) -> int:
            return 0

        def get_sel(self) -> list[int]:
            return [1]

        def mixed_types(self) -> bool:
            return True

        def model_output_def(self) -> ModelOutputDef:
            return ModelOutputDef(FittingOutputDef([OutputVariableDef("coord_x", [1])]))

        def get_type_map(self) -> list[str]:
            return ["O"]

        def get_sel_type(self) -> list[int]:
            return []

        def is_aparam_nall(self) -> bool:
            return False

        def model_output_type(self) -> list[str]:
            return ["coord_x"]

        def get_min_nbor_dist(self) -> None:
            return None

        def has_message_passing(self) -> bool:
            return False

        def has_default_fparam(self) -> bool:
            return False

        def get_default_fparam(self) -> None:
            return None

        def has_chg_spin_ebd(self) -> bool:
            return False

        def get_dim_chg_spin(self) -> int:
            return 0

        def has_default_chg_spin(self) -> bool:
            return False

        def get_default_chg_spin(self) -> None:
            return None

    monkeypatch.setattr(
        serialization.BaseModel,
        "deserialize",
        staticmethod(lambda data: DummyModel()),
    )

    model_dir = tmp_path / "dummy.savedmodel"
    serialization.deserialize_to_file(
        str(model_dir),
        {"model": {"type": "dummy"}, "model_def_script": {"type": "dummy"}},
    )

    assert "XlaCallModule" in _saved_model_ops(model_dir)
