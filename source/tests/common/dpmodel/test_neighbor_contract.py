# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeighborContract unit tests."""

from __future__ import (
    annotations,
)

import unittest

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.dpa2 import (
    DescrptDPA2,
)
from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.dpmodel.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.utils.neighbor_contract import (
    GRAPH_NATIVE_CONSTRUCTION_SEL,
    NeighborContract,
    ensure_construction_sel,
)


class TestNeighborContract(unittest.TestCase):
    def test_graph_has_no_capacity(self) -> None:
        contract = NeighborContract.graph()
        self.assertTrue(contract.is_graph)
        self.assertFalse(contract.requires_capacity)
        self.assertIsNone(contract.capacity)
        with self.assertRaises(ValueError):
            contract.legacy_sel()

    def test_dense_auto_requires_capacity(self) -> None:
        contract = NeighborContract.from_legacy_sel("auto")
        self.assertTrue(contract.is_dense)
        self.assertTrue(contract.requires_capacity)
        self.assertIsNone(contract.capacity)

    def test_dense_explicit_capacity(self) -> None:
        contract = NeighborContract.from_legacy_sel([12, 24])
        self.assertEqual(contract.legacy_sel(), [12, 24])
        self.assertFalse(contract.requires_capacity)

    def test_metadata_roundtrip_and_legacy_adapter(self) -> None:
        graph = NeighborContract.graph()
        restored = NeighborContract.from_dict(graph.to_dict())
        self.assertEqual(restored, graph)

        legacy = NeighborContract.from_metadata({"sel": [8, 16], "nnei": 24})
        self.assertTrue(legacy.is_dense)
        self.assertEqual(legacy.legacy_sel(), [8, 16])

        from_contract = NeighborContract.from_metadata(
            {"neighbor_contract": graph.to_dict()}
        )
        self.assertTrue(from_contract.is_graph)

    def test_merge_rejects_mixed_representation(self) -> None:
        with self.assertRaises(ValueError):
            NeighborContract.graph().merge(NeighborContract.dense([4]))

    def test_graph_merge_preserves_requires_capacity(self) -> None:
        # Hybrid: graph without sel + graph sel:auto must keep discovery.
        left = NeighborContract.graph()
        right = NeighborContract(representation="graph", requires_capacity=True)
        merged = left.merge(right)
        self.assertTrue(merged.is_graph)
        self.assertTrue(merged.requires_capacity)
        self.assertIsNone(merged.capacity)
        # Symmetric
        self.assertTrue(right.merge(left).requires_capacity)
        # Both False stays False
        self.assertFalse(left.merge(NeighborContract.graph()).requires_capacity)

    def test_dense_merge_preserves_discovery_when_sibling_needs_it(self) -> None:
        # Hybrid dense: sel:auto + explicit sel must still require discovery.
        auto = NeighborContract.dense(None, requires_capacity=True)
        explicit = NeighborContract.dense([8, 16], requires_capacity=False)
        merged = auto.merge(explicit)
        self.assertTrue(merged.is_dense)
        self.assertTrue(merged.requires_capacity)
        self.assertEqual(merged.capacity, (8, 16))
        # Explicit + explicit: no discovery
        both = NeighborContract.dense([4]).merge(NeighborContract.dense([8]))
        self.assertFalse(both.requires_capacity)
        self.assertEqual(both.capacity, (8,))

    def test_ensure_construction_sel(self) -> None:
        prepared = ensure_construction_sel({"type": "dpa1", "sel": "auto"})
        self.assertEqual(prepared["sel"], GRAPH_NATIVE_CONSTRUCTION_SEL)
        prepared = ensure_construction_sel({"type": "dpa1"})
        self.assertEqual(prepared["sel"], GRAPH_NATIVE_CONSTRUCTION_SEL)


class TestDescriptorContracts(unittest.TestCase):
    def test_dpa1_graph_from_jdata_without_sel(self) -> None:
        jdata = {
            "type": "dpa1",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "tebd_input_mode": "concat",
        }
        contract = DescrptDPA1.neighbor_contract_from_jdata(jdata)
        self.assertTrue(contract.is_graph)
        self.assertFalse(contract.requires_capacity)
        prepared = DescrptDPA1.prepare_jdata_for_neighbor_contract(jdata, contract)
        self.assertEqual(prepared["sel"], GRAPH_NATIVE_CONSTRUCTION_SEL)

    def test_dpa1_dense_when_tebd_ineligible(self) -> None:
        # Force a non-eligible tebd mode string that from_legacy still densifies.
        # Unknown modes are not graph-eligible.
        jdata = {"type": "dpa1", "sel": "auto", "tebd_input_mode": "other"}
        contract = DescrptDPA1.neighbor_contract_from_jdata(jdata)
        self.assertTrue(contract.is_dense)
        self.assertTrue(contract.requires_capacity)

    def test_dpa4c_always_graph(self) -> None:
        contract = DescrptDPA4C.neighbor_contract_from_jdata(
            {"type": "dpa4c", "rcut": 6.0}
        )
        self.assertTrue(contract.is_graph)
        prepared = DescrptDPA4C.prepare_jdata_for_neighbor_contract(
            {"type": "dpa4c", "rcut": 6.0}, contract
        )
        self.assertNotIn("sel", prepared)

    def test_plugin_dispatch(self) -> None:
        contract = BaseDescriptor.neighbor_contract_from_jdata(
            {
                "type": "se_atten_v2",
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "tebd_input_mode": "strip",
            }
        )
        self.assertTrue(contract.is_graph)
        self.assertFalse(contract.requires_capacity)

    def test_dpa1_auto_sel_still_requires_capacity(self) -> None:
        contract = DescrptDPA1.neighbor_contract_from_jdata(
            {
                "type": "dpa1",
                "tebd_input_mode": "concat",
                "sel": "auto",
            }
        )
        self.assertTrue(contract.is_graph)
        self.assertTrue(contract.requires_capacity)


class TestPrepareNeighborsSkipsUpdateSel(unittest.TestCase):
    def test_graph_dpa1_does_not_call_update_sel(self) -> None:
        jdata = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa1",
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "tebd_input_mode": "concat",
            },
            "fitting": {"type": "ener"},
        }
        train_data = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel"
            ) as update_sel,
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.UpdateSel"
            ) as update_sel_cls,
        ):
            update_sel_cls.return_value.get_min_nbor_dist.return_value = 0.5
            updated, min_dist = DPModelCommon.prepare_neighbors(
                train_data, ["O", "H"], jdata
            )
            update_sel.assert_not_called()
            self.assertEqual(
                updated["descriptor"]["sel"], GRAPH_NATIVE_CONSTRUCTION_SEL
            )
            self.assertEqual(min_dist, 0.5)

    def test_dense_still_calls_update_sel(self) -> None:
        jdata = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "sel": "auto",
                "neuron": [2, 4, 8],
            },
            "fitting": {"type": "ener"},
        }
        train_data = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel",
                return_value=({"type": "se_e2_a", "sel": [4, 8]}, 0.8),
            ) as update_sel,
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.neighbor_contract_from_jdata",
                return_value=NeighborContract.dense(None, requires_capacity=True),
            ),
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.prepare_jdata_for_neighbor_contract",
                side_effect=lambda j, c: dict(j),
            ),
        ):
            updated, min_dist = DPModelCommon.prepare_neighbors(
                train_data, ["O", "H"], jdata
            )
            update_sel.assert_called_once()
            self.assertEqual(updated["descriptor"]["sel"], [4, 8])
            self.assertEqual(min_dist, 0.8)


class TestHybridUpdateSelClassmethod(unittest.TestCase):
    def test_update_sel_is_classmethod(self) -> None:
        self.assertTrue(isinstance(DescrptHybrid.__dict__["update_sel"], classmethod))

    def test_hybrid_update_sel_dispatch(self) -> None:
        jdata = {
            "type": "hybrid",
            "list": [
                {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "tebd_input_mode": "concat",
                    "sel": "auto",
                },
                {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "tebd_input_mode": "concat",
                },
            ],
        }
        train_data = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "deepmd.dpmodel.descriptor.hybrid.BaseDescriptor.update_sel",
            side_effect=lambda td, tm, child: (
                {
                    **child,
                    "sel": [4] if child.get("sel") == "auto" else child.get("sel", 1),
                },
                0.3,
            ),
        ):
            # Must be callable on the class without binding an instance.
            updated, min_dist = DescrptHybrid.update_sel(train_data, ["O", "H"], jdata)
        self.assertEqual(updated["list"][0]["sel"], [4])
        self.assertEqual(min_dist, 0.3)

    def test_hybrid_graph_merge_keeps_requires_capacity(self) -> None:
        jdata = {
            "type": "hybrid",
            "list": [
                {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "tebd_input_mode": "concat",
                },
                {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "tebd_input_mode": "concat",
                    "sel": "auto",
                },
            ],
        }
        contract = DescrptHybrid.neighbor_contract_from_jdata(jdata)
        self.assertTrue(contract.is_graph)
        self.assertTrue(contract.requires_capacity)

    def test_dpa2_auto_nsel_keeps_capacity_discovery(self) -> None:
        jdata = {
            "type": "dpa2",
            "repinit": {
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "nsel": "auto",
                "tebd_input_mode": "concat",
                "set_davg_zero": True,
            },
            "repformer": {
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "nsel": "auto",
                "set_davg_zero": True,
            },
        }
        contract = DescrptDPA2.neighbor_contract_from_jdata(jdata)
        self.assertTrue(contract.is_graph)
        self.assertTrue(contract.requires_capacity)

    def test_dpa2_prepare_jdata_does_not_inject_top_level_sel(self) -> None:
        jdata = {
            "type": "dpa2",
            "repinit": {
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "nsel": 20,
                "tebd_input_mode": "concat",
                "set_davg_zero": True,
            },
            "repformer": {
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "nsel": 10,
                "set_davg_zero": True,
            },
        }
        contract = DescrptDPA2.neighbor_contract_from_jdata(jdata)
        prepared = DescrptDPA2.prepare_jdata_for_neighbor_contract(jdata, contract)
        self.assertNotIn("sel", prepared)
        self.assertEqual(prepared["repinit"]["nsel"], 20)


class TestPrepareNeighborsMinNborDist(unittest.TestCase):
    def test_skip_capacity_returns_min_nbor_dist_without_top_level_rcut(self) -> None:
        # DPA2-like nested cutoffs under repinit/repformer only.
        jdata = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa2",
                "repinit": {"rcut": 6.0, "rcut_smth": 0.5, "nsel": 20},
                "repformer": {"rcut": 4.0, "rcut_smth": 0.5, "nsel": 10},
            },
            "fitting": {"type": "ener"},
        }
        train_data = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.neighbor_contract_from_jdata",
                return_value=NeighborContract.graph(),
            ),
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.prepare_jdata_for_neighbor_contract",
                side_effect=lambda j, c: dict(j),
            ),
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel"
            ) as update_sel,
            unittest.mock.patch(
                "deepmd.dpmodel.model.dp_model.UpdateSel"
            ) as update_sel_cls,
        ):
            update_sel_cls.return_value.get_min_nbor_dist.return_value = 0.42
            updated, min_dist = DPModelCommon.prepare_neighbors(
                train_data, ["O", "H"], jdata
            )
            update_sel.assert_not_called()
            update_sel_cls.return_value.get_min_nbor_dist.assert_called_once_with(
                train_data
            )
            self.assertEqual(min_dist, 0.42)
            self.assertNotIn("rcut", updated["descriptor"])


if __name__ == "__main__":
    unittest.main()
