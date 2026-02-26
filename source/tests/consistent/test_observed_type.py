# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest


class TestDpmodelGetObservedTypeList(unittest.TestCase):
    """Test dpmodel's get_observed_type_list() metadata parsing."""

    def _make_model_with_script(self, script: str):
        """Create a minimal mock that has model_def_script attribute."""
        from deepmd.dpmodel.model.base_model import (
            make_base_model,
        )

        BaseModel = make_base_model()

        class FakeModel:
            model_def_script = script

        # Bind the method from BaseBaseModel
        fake = FakeModel()
        fake.get_observed_type_list = BaseModel.get_observed_type_list.__get__(
            fake, FakeModel
        )
        return fake

    def test_with_observed_type_in_info(self) -> None:
        script = json.dumps(
            {
                "info": {"observed_type": ["H", "O"]},
                "type_map": ["O", "H"],
            }
        )
        model = self._make_model_with_script(script)
        result = model.get_observed_type_list()
        self.assertEqual(result, ["H", "O"])

    def test_without_info(self) -> None:
        script = json.dumps({"type_map": ["O", "H"]})
        model = self._make_model_with_script(script)
        result = model.get_observed_type_list()
        self.assertEqual(result, [])

    def test_info_without_observed_type(self) -> None:
        script = json.dumps({"info": {}, "type_map": ["O", "H"]})
        model = self._make_model_with_script(script)
        result = model.get_observed_type_list()
        self.assertEqual(result, [])

    def test_empty_script(self) -> None:
        model = self._make_model_with_script("")
        result = model.get_observed_type_list()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
