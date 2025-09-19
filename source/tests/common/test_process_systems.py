# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest

import h5py

from deepmd.utils.data_system import (
    process_systems,
)


class TestProcessSystems(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_hdf5_file(self, filename: str, systems: list[str]) -> str:
        """Create an HDF5 file with multiple systems."""
        h5_path = os.path.join(self.temp_dir, filename)

        with h5py.File(h5_path, "w") as f:
            for sys_name in systems:
                sys_group = f.create_group(sys_name)

                # Add required type information
                sys_group.create_dataset("type.raw", data=[0, 1])
                sys_group.create_dataset("type_map.raw", data=[b"H", b"O"])

                # Create a data set
                set_group = sys_group.create_group("set.000")

                # Add minimal required data
                natoms = 2
                nframes = 1

                coords = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
                set_group.create_dataset("coord.npy", data=coords)

                boxes = [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]
                set_group.create_dataset("box.npy", data=boxes)

                energies = [1.0]
                set_group.create_dataset("energy.npy", data=energies)

                forces = [[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]
                set_group.create_dataset("force.npy", data=forces)

        return h5_path

    def _create_regular_system(self, dirname: str) -> str:
        """Create a regular directory-based system."""
        sys_path = os.path.join(self.temp_dir, dirname)
        os.makedirs(sys_path, exist_ok=True)

        # Create type.raw to make it a valid system
        with open(os.path.join(sys_path, "type.raw"), "w") as f:
            f.write("0\n1\n")

        return sys_path

    def test_hdf5_list_expansion(self) -> None:
        """Test that HDF5 files in lists are properly expanded."""
        # Create two HDF5 files with multiple systems each
        h5_file1 = self._create_hdf5_file("multi1.h5", ["sys_a", "sys_b"])
        h5_file2 = self._create_hdf5_file("multi2.h5", ["water", "ice"])

        # Process the list of HDF5 files
        input_systems = [h5_file1, h5_file2]
        result = process_systems(input_systems)

        # Should expand to 4 systems total
        expected_systems = [
            f"{h5_file1}#sys_a",
            f"{h5_file1}#sys_b",
            f"{h5_file2}#water",
            f"{h5_file2}#ice",
        ]

        self.assertEqual(len(result), 4)
        for expected in expected_systems:
            self.assertIn(expected, result)

    def test_mixed_systems_list(self) -> None:
        """Test mixed list with HDF5 files and regular directories."""
        h5_file = self._create_hdf5_file("mixed.h5", ["system1", "system2"])
        regular_dir = self._create_regular_system("regular_system")

        input_systems = [h5_file, regular_dir]
        result = process_systems(input_systems)

        # Should expand HDF5 but keep regular directory as-is
        expected_systems = [f"{h5_file}#system1", f"{h5_file}#system2", regular_dir]

        self.assertEqual(len(result), 3)
        for expected in expected_systems:
            self.assertIn(expected, result)

    def test_explicit_hdf5_system_preserved(self) -> None:
        """Test that explicitly specified HDF5 systems are not expanded."""
        h5_file = self._create_hdf5_file("explicit.h5", ["sys1", "sys2"])

        # Use explicit system specification with #
        input_systems = [f"{h5_file}#sys1"]
        result = process_systems(input_systems)

        # Should not expand, keep as explicit
        self.assertEqual(result, [f"{h5_file}#sys1"])

    def test_regular_list_unchanged(self) -> None:
        """Test that lists without HDF5 files are unchanged."""
        regular1 = self._create_regular_system("sys1")
        regular2 = self._create_regular_system("sys2")

        input_systems = [regular1, regular2]
        result = process_systems(input_systems)

        # Should remain unchanged
        self.assertEqual(result, input_systems)

    def test_invalid_hdf5_file_handled(self) -> None:
        """Test that invalid HDF5 files are handled gracefully."""
        # Create a text file with .h5 extension
        fake_h5 = os.path.join(self.temp_dir, "fake.h5")
        with open(fake_h5, "w") as f:
            f.write("This is not an HDF5 file")

        input_systems = [fake_h5]
        result = process_systems(input_systems)

        # Should treat as regular system since it can't be read as HDF5
        self.assertEqual(result, [fake_h5])

    def test_single_system_hdf5_not_expanded(self) -> None:
        """Test that single-system HDF5 files are not expanded."""
        # Create an HDF5 file that looks like a single system (has type.raw and set.* at root)
        h5_file = os.path.join(self.temp_dir, "single_system.h5")

        with h5py.File(h5_file, "w") as f:
            # Add type information at root level (single system structure)
            f.create_dataset("type.raw", data=[0, 1])
            f.create_dataset("type_map.raw", data=[b"H", b"O"])

            # Create sets at root level
            for set_idx in range(2):
                set_name = f"set.{set_idx:03d}"
                set_group = f.create_group(set_name)

                coords = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
                set_group.create_dataset("coord.npy", data=coords)

                boxes = [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]
                set_group.create_dataset("box.npy", data=boxes)

                energies = [1.0]
                set_group.create_dataset("energy.npy", data=energies)

                forces = [[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]
                set_group.create_dataset("force.npy", data=forces)

        # Process as list (this is the main use case)
        input_systems = [h5_file]
        result = process_systems(input_systems)

        # Should NOT expand single-system HDF5 file
        self.assertEqual(result, [h5_file])

    def test_backward_compatibility(self) -> None:
        """Test that existing functionality is preserved."""
        # Test single string (existing behavior)
        regular_dir = self._create_regular_system("test_system")
        result = process_systems(regular_dir)
        self.assertIn(regular_dir, result)

        # Test regular list (existing behavior)
        regular_list = [regular_dir]
        result = process_systems(regular_list)
        self.assertEqual(result, regular_list)


if __name__ == "__main__":
    unittest.main()
