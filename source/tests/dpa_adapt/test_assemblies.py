# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import json

import numpy as np
import pytest

from dpa_adapt import AssemblyDatasetBuilder, ComponentSpec, PoolMask, SiteSelector, SubstitutionSpec
from dpa_adapt.data.assemblies import GROUP_ID_KEY, POOL_MASK_KEY, WEIGHT_KEY
from dpa_adapt.data.errors import DPADataError


def _component(offset: float = 0.0, *, role: str = "state") -> ComponentSpec:
    return ComponentSpec.from_arrays(
        coords=[[offset, 0.0, 0.0], [0.0, 1.0 + offset, 0.0], [0.0, 0.0, 1.0]],
        symbols=["Ni", "O", "H"],
        box=np.eye(3) * 12.0,
        weight=0.5,
        pool_mask=PoolMask.exclude_indices([2]),
        role=role,
        block="oer_adsorbates",
        source=f"{role}.vasp",
        metadata={"anchor_atom": 0},
    )


def test_assembly_builder_writes_minimal_deepmd_tensors_and_manifest(tmp_path) -> None:
    builder = AssemblyDatasetBuilder(property_name="overpotential", type_map=["Ni", "O", "H"])
    sub = SubstitutionSpec(
        sites=SiteSelector.element("Ni"),
        composition={"Ni": 0.8, "Fe": 0.2},
        seed=123,
    )
    group = builder.group(
        key="Ni0.8Fe0.2O2H1",
        label=291.9,
        metadata={"substitution": sub.to_dict()},
    )
    group.add_component(_component(0.0, role="O*"))
    group.add_component(_component(0.1, role="OH*"))
    group.add_component(_component(0.2, role="OOH*"))

    result = builder.write_deepmd_npy(tmp_path)
    system = tmp_path / result["systems"][0]
    set_dir = system / "set.000"

    assert (tmp_path / "manifest.json").is_file()
    assert sorted(p.name for p in set_dir.iterdir()) == sorted(
        [
            "box.npy",
            "coord.npy",
            f"{GROUP_ID_KEY}.npy",
            "overpotential.npy",
            f"{POOL_MASK_KEY}.npy",
            f"{WEIGHT_KEY}.npy",
        ]
    )
    assert np.load(set_dir / "coord.npy").shape == (3, 9)
    assert np.load(set_dir / "box.npy").shape == (3, 9)
    assert np.load(set_dir / "overpotential.npy").shape == (3, 1)
    assert np.load(set_dir / f"{GROUP_ID_KEY}.npy").tolist() == [0, 0, 0]
    assert np.load(set_dir / f"{WEIGHT_KEY}.npy").tolist() == [0.5, 0.5, 0.5]
    assert np.load(set_dir / f"{POOL_MASK_KEY}.npy").tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert (system / "type.raw").read_text().splitlines() == ["0", "1", "2"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == "dpa_adapt.assembly.v1"
    assert manifest["tensor_fields"] == {
        "conditions": None,
        "group_id": "group_id",
        "label": "overpotential",
        "pool_mask": "pool_mask",
        "weight": "weight",
    }
    g0 = manifest["groups"][0]
    assert g0["key"] == "Ni0.8Fe0.2O2H1"
    assert g0["metadata"]["substitution"]["sites"] == {"mode": "element", "value": "Ni"}
    assert [c["role"] for c in g0["components"]] == ["O*", "OH*", "OOH*"]
    assert g0["components"][0]["pool_mask_excluded"] == [2]


def test_conditions_write_fparam_but_schema_stays_in_manifest(tmp_path) -> None:
    builder = AssemblyDatasetBuilder(property_name="cloud_point", type_map=["C", "H"])
    builder.set_condition_schema(
        [
            {"name": "log_mn", "source": "Mn", "transform": "log10(x)/6"},
            {"name": "pH", "default": 7.0},
        ]
    )
    group = builder.group(key="polymer_0", label=32.1, conditions={"log_mn": 0.67, "pH": 7.0})
    group.add_component(
        ComponentSpec.from_arrays(
            coords=[[0, 0, 0], [0, 0, 1]],
            symbols=["C", "H"],
            weight=1.0,
            role="repeat_unit_A",
            block="repeat_units",
        )
    )
    builder.write_deepmd_npy(tmp_path)
    fparam = np.load(tmp_path / "systems" / "polymer_0" / "set.000" / "fparam.npy")
    assert fparam.tolist() == [[0.67, 7.0]]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tensor_fields"]["conditions"] == "fparam"
    assert [item["name"] for item in manifest["condition_schema"]] == ["log_mn", "pH"]
    assert manifest["groups"][0]["components"][0]["block"] == "repeat_units"


def test_writer_rejects_mismatched_component_symbol_order(tmp_path) -> None:
    builder = AssemblyDatasetBuilder(property_name="property")
    group = builder.group(key="bad", label=1.0)
    group.add_component(ComponentSpec.from_arrays([[0, 0, 0], [0, 0, 1]], ["C", "H"]))
    group.add_component(ComponentSpec.from_arrays([[0, 0, 0], [0, 0, 1]], ["H", "C"]))
    with pytest.raises(DPADataError, match="identical symbol order"):
        builder.write_deepmd_npy(tmp_path)
