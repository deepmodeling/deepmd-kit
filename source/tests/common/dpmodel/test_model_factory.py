# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the shared dpmodel-driven model factory dispatch."""

import unittest

from deepmd.dpmodel.model.model_factory import (
    get_model,
    get_standard_model,
)


class _RegisteredModel:
    @classmethod
    def get_model(cls, data: dict) -> tuple[str, dict]:
        return "registered", data


class _BaseModel:
    @classmethod
    def get_class_by_type(cls, model_type: str) -> type[_RegisteredModel]:
        if model_type != "registered":
            raise KeyError(model_type)
        return _RegisteredModel


class _Descriptor:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def get_ntypes(self) -> int:
        return self.kwargs["ntypes"]

    def mixed_types(self) -> bool:
        return True

    def get_dim_emb(self) -> int:
        return 7

    def get_dim_out(self) -> int:
        return 11


class _DescriptorBase:
    @classmethod
    def get_class_by_type(cls, descriptor_type: str) -> type[_Descriptor]:
        if descriptor_type != "descriptor":
            raise KeyError(descriptor_type)
        return _Descriptor


class _Fitting:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FittingBase:
    @classmethod
    def get_class_by_type(cls, fitting_type: str) -> type[_Fitting]:
        if fitting_type != "dipole":
            raise KeyError(fitting_type)
        return _Fitting


class _StandardModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _StandardModelBase:
    @classmethod
    def get_class_by_type(cls, model_type: str) -> type[_StandardModel]:
        if model_type != "dipole":
            raise KeyError(model_type)
        return _StandardModel


def _factory(name: str):
    def factory(data: dict) -> tuple[str, dict]:
        return name, data

    return factory


class TestModelFactory(unittest.TestCase):
    """Verify common routing and backend extension points."""

    def test_standard_routes(self) -> None:
        standard = _factory("standard")
        spin = _factory("spin")
        zbl = _factory("zbl")

        self.assertEqual(
            get_model(
                {},
                base_model=_BaseModel,
                standard_model_factory=standard,
                spin_model_factory=spin,
                zbl_model_factory=zbl,
            )[0],
            "standard",
        )
        self.assertEqual(
            get_model(
                {"spin": {}, "use_srtab": "table"},
                base_model=_BaseModel,
                standard_model_factory=standard,
                spin_model_factory=spin,
                zbl_model_factory=zbl,
            )[0],
            "spin",
        )
        self.assertEqual(
            get_model(
                {"use_srtab": "table"},
                base_model=_BaseModel,
                standard_model_factory=standard,
                spin_model_factory=spin,
                zbl_model_factory=zbl,
            )[0],
            "zbl",
        )

    def test_unsupported_standard_variant(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "Spin model is not implemented yet"
        ):
            get_model(
                {"spin": {}},
                base_model=_BaseModel,
                standard_model_factory=_factory("standard"),
            )

    def test_unsupported_zbl_variant(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "ZBL model is not implemented yet"
        ):
            get_model(
                {"use_srtab": "table"},
                base_model=_BaseModel,
                standard_model_factory=_factory("standard"),
            )

    def test_explicit_factory_precedes_registry(self) -> None:
        result = get_model(
            {"type": "custom"},
            base_model=_BaseModel,
            standard_model_factory=_factory("standard"),
            model_factories={"custom": _factory("custom")},
        )
        self.assertEqual(result[0], "custom")

    def test_registry_fallback(self) -> None:
        data = {"type": "registered"}
        self.assertEqual(
            get_model(
                data,
                base_model=_BaseModel,
                standard_model_factory=_factory("standard"),
            ),
            ("registered", data),
        )

    def test_standard_construction_is_shared_and_non_mutating(self) -> None:
        data = {
            "type_map": ["O", "H"],
            "descriptor": {"type": "descriptor", "custom": 3},
            "fitting_net": {"type": "dipole", "custom": 5},
            "atom_exclude_types": [1],
            "pair_exclude_types": [[0, 1]],
        }
        expected = {
            "type_map": ["O", "H"],
            "descriptor": {"type": "descriptor", "custom": 3},
            "fitting_net": {"type": "dipole", "custom": 5},
            "atom_exclude_types": [1],
            "pair_exclude_types": [[0, 1]],
        }
        model = get_standard_model(
            data,
            descriptor_base=_DescriptorBase,
            fitting_base=_FittingBase,
            model_base=_StandardModelBase,
            backend_name="test",
        )

        self.assertEqual(data, expected)
        self.assertEqual(model.kwargs["descriptor"].kwargs["ntypes"], 2)
        self.assertEqual(model.kwargs["descriptor"].kwargs["type_map"], ["O", "H"])
        self.assertEqual(model.kwargs["fitting"].kwargs["ntypes"], 2)
        self.assertEqual(model.kwargs["fitting"].kwargs["dim_descrpt"], 11)
        self.assertEqual(model.kwargs["fitting"].kwargs["embedding_width"], 7)
        self.assertEqual(model.kwargs["atom_exclude_types"], [1])
        self.assertEqual(model.kwargs["pair_exclude_types"], [[0, 1]])


if __name__ == "__main__":
    unittest.main()
