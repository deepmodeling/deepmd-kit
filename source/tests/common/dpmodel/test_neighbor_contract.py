# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeighborContract unit tests."""

from __future__ import annotations

import unittest
from unittest import mock

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
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
        train_data = mock.MagicMock()
        with (
            mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel"
            ) as update_sel,
            mock.patch("deepmd.dpmodel.model.dp_model.UpdateSel") as update_sel_cls,
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
        train_data = mock.MagicMock()
        with (
            mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel",
                return_value=({"type": "se_e2_a", "sel": [4, 8]}, 0.8),
            ) as update_sel,
            mock.patch(
                "deepmd.dpmodel.model.dp_model.BaseDescriptor.neighbor_contract_from_jdata",
                return_value=NeighborContract.dense(None, requires_capacity=True),
            ),
            mock.patch(
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


if __name__ == "__main__":
    unittest.main()
