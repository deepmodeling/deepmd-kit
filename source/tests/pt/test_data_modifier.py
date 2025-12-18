# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test data modification functionality.

This module tests the data modification functionality, specifically
testing the BaseModifier implementations and their effects on training and
validation data. It includes:

1. Test modifier implementations (random_tester and zero_tester)
2. Tests to verify data modification is applied correctly
3. Tests to ensure data modification is only applied once

The tests use parameterized testing with different batch sizes for training
and validation data.
"""

import json
import os
import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.modifier.base_modifier import (
    BaseModifier,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.argcheck import (
    modifier_args_plugin,
)
from deepmd.utils.data import (
    DeepmdData,
)

from ..consistent.common import (
    parameterized,
)


@modifier_args_plugin.register("random_tester")
def modifier_random_tester() -> list:
    """Return empty argument list for random_tester modifier.

    This function registers the argument schema for the random_tester modifier.

    Returns
    -------
    list: Empty list indicating no additional arguments required
    """
    return []


@modifier_args_plugin.register("zero_tester")
def modifier_zero_tester() -> list:
    """Return empty argument list for zero_tester modifier.

    This function registers the argument schema for the zero_tester modifier.

    Returns
    -------
    list: Empty list indicating no additional arguments required
    """
    return []


@BaseModifier.register("random_tester")
class ModifierRandomTester(BaseModifier):
    def __new__(cls) -> BaseModifier:
        """Create a new instance of ModifierRandomTester.

        Returns
        -------
        BaseModifier: New instance of the modifier
        """
        return super().__new__(cls)

    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        super().__init__()
        self.modifier_type = "random_tester"
        # Use a fixed seed for deterministic behavior
        self.rng = np.random.default_rng(12345)  # Fixed seed for reproducibility

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Implementation of abstractmethod."""
        return {}

    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Multiply by a deterministic factor for testing."""
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] = data["energy"] * self.rng.random()
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] = data["force"] * self.rng.random()
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] = data["virial"] * self.rng.random()


@BaseModifier.register("zero_tester")
class ModifierZeroTester(BaseModifier):
    def __new__(cls) -> BaseModifier:
        """Create a new instance of ModifierZeroTester.

        Returns
        -------
        BaseModifier: New instance of the modifier
        """
        return super().__new__(cls)

    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        super().__init__()
        self.modifier_type = "zero_tester"

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Implementation of abstractmethod."""
        return {}

    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Zero out energy, force, and virial data."""
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= data["energy"]
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= data["force"]
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] -= data["virial"]


@parameterized(
    (1, 2),  # training data batch_size
    (1, 2),  # validation data batch_size
)
class TestDataModifier(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        input_json = str(Path(__file__).parent / "water/se_e2_a.json")
        training_data = [
            str(Path(__file__).parent / "water/data/data_0"),
            str(Path(__file__).parent / "water/data/data_1"),
        ]
        validation_data = [str(Path(__file__).parent / "water/data/data_1")]
        with open(input_json, encoding="utf-8") as f:
            config = json.load(f)
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = training_data
        config["training"]["training_data"]["batch_size"] = self.param[0]
        config["training"]["validation_data"]["systems"] = validation_data
        config["training"]["validation_data"]["batch_size"] = self.param[1]
        self.config = config

        self.training_nframes = self.get_dataset_nframes(training_data)
        self.validation_nframes = self.get_dataset_nframes(validation_data)

    @staticmethod
    def get_dataset_nframes(dataset: list[str]) -> int:
        """Calculate total number of frames in a dataset.

        Args:
            dataset: List of dataset paths

        Returns
        -------
        int: Total number of frames across all datasets
        """
        nframes = 0
        for _data in dataset:
            _dpdata = DeepmdData(_data)
            nframes += _dpdata.nframes
        return nframes

    @staticmethod
    def get_sampled_data(trainer, nbatch: int, is_train: bool):
        """
        Collect all data from trainer and organize by IDs for easy comparison.

        Args:
            trainer: The trainer object
            nbatch: Number of batches to iterate through
            is_train: Whether to get training data (True) or validation data (False)

        Returns
        -------
        dict: A nested dictionary organized by system_id and frame_id
            Format: {system_id: {frame_id: label_dict}}
        """
        output = {}
        # Keep track of all unique frames we've collected
        collected_frames = set()

        # Continue collecting data until we've gone through all batches
        for _ in range(nbatch):
            _, label_dict, log_dict = trainer.get_data(is_train=is_train)

            system_id = log_dict["sid"]
            frame_ids = log_dict["fid"]

            # Initialize system entry if not exists
            if system_id not in output:
                output[system_id] = {}

            # Store label data for each frame ID
            for idx, frame_id in enumerate(frame_ids):
                # Skip if we already have this frame
                frame_key = (system_id, frame_id)
                if frame_key in collected_frames:
                    continue

                # Create a copy of label_dict for this specific frame
                frame_data = {}
                for key, value in label_dict.items():
                    # If value is a tensor/array with batch dimension, extract the specific frame
                    if (
                        hasattr(value, "shape")
                        and len(value.shape) > 0
                        and value.shape[0] == len(frame_ids)
                    ):
                        # Handle batched data - extract the data for this frame
                        frame_data[key] = value[idx]
                    else:
                        # For scalar values or non-batched data, just copy as is
                        frame_data[key] = value

                output[system_id][frame_id] = frame_data
                collected_frames.add(frame_key)

        return output

    @staticmethod
    def check_sampled_data(
        ref_data: dict[int, dict], test_data: dict[int, dict], label_kw: str
    ):
        """Compare sampled data between reference and test datasets.

        Args:
            ref_data: Reference data dictionary organized by system and frame IDs
            test_data: Test data dictionary organized by system and frame IDs
            label_kw: Key of the label to compare (e.g., "energy", "force")

        Raises
        ------
        AssertionError: If the data doesn't match between reference and test
        """
        for sid in ref_data.keys():
            ref_sys = ref_data[sid]
            test_sys = test_data[sid]
            for fid in ref_sys.keys():
                # compare common elements
                try:
                    ref_label = to_numpy_array(ref_sys[fid][label_kw])
                    test_label = to_numpy_array(test_sys[fid][label_kw])
                except KeyError:
                    continue
                np.testing.assert_allclose(ref_label, test_label)

    def test_init_modify_data(self):
        """Ensure modify_data applied."""
        tmp_config = self.config.copy()
        # add tester data modifier
        tmp_config["model"]["modifier"] = {"type": "zero_tester"}

        # data modification is finished in __init__
        trainer = get_trainer(tmp_config)

        # training data
        training_data = trainer.get_data(is_train=True)
        # validation data
        validation_data = trainer.get_data(is_train=False)

        for dataset in [training_data, validation_data]:
            for kw in ["energy", "force"]:
                data = to_numpy_array(dataset[1][kw])
                np.testing.assert_allclose(data, np.zeros_like(data))

    def test_full_modify_data(self):
        """Ensure modify_data only applied once."""
        tmp_config = self.config.copy()
        # add tester data modifier
        tmp_config["model"]["modifier"] = {"type": "random_tester"}

        # data modification is finished in __init__
        trainer = get_trainer(tmp_config)

        # training data
        training_data_before = self.get_sampled_data(
            trainer, self.training_nframes, True
        )
        # validation data
        validation_data_before = self.get_sampled_data(
            trainer, self.validation_nframes, False
        )

        trainer.run()

        # training data
        training_data_after = self.get_sampled_data(
            trainer, self.training_nframes, True
        )
        # validation data
        validation_data_after = self.get_sampled_data(
            trainer, self.validation_nframes, False
        )

        for label_kw in ["energy", "force"]:
            self.check_sampled_data(training_data_before, training_data_after, label_kw)
            self.check_sampled_data(
                validation_data_before, validation_data_after, label_kw
            )

    def tearDown(self) -> None:
        """Clean up test artifacts after each test.

        Removes model files and other artifacts created during testing.
        """
        for f in os.listdir("."):
            if f.startswith("frozen_model") and f.endswith(".pth"):
                os.remove(f)
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
