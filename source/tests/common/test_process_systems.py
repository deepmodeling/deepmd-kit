# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest

import h5py

from deepmd.utils.data_system import (
    _is_hdf5_file,
    _is_hdf5_format,
    _is_hdf5_multisystem,
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
            f"{h5_file1}#/sys_a",
            f"{h5_file1}#/sys_b",
            f"{h5_file2}#/water",
            f"{h5_file2}#/ice",
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
        expected_systems = [f"{h5_file}#/system1", f"{h5_file}#/system2", regular_dir]

        self.assertEqual(len(result), 3)
        for expected in expected_systems:
            self.assertIn(expected, result)

    def test_explicit_hdf5_system_preserved(self) -> None:
        """Test that explicitly specified HDF5 systems are not expanded."""
        h5_file = self._create_hdf5_file("explicit.h5", ["sys1", "sys2"])

        # Use explicit system specification with #
        input_systems = [f"{h5_file}#/sys1"]
        result = process_systems(input_systems)

        # Should not expand, keep as explicit
        self.assertEqual(result, [f"{h5_file}#/sys1"])

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

    def test_is_hdf5_file_edge_cases(self) -> None:
        """Test edge cases for _is_hdf5_file function."""
        # Test non-existent file
        self.assertFalse(_is_hdf5_file("/non/existent/file.h5"))

        # Test file with .h5 extension but not HDF5 format
        fake_h5 = os.path.join(self.temp_dir, "fake.h5")
        with open(fake_h5, "w") as f:
            f.write("This is not an HDF5 file")
        self.assertTrue(_is_hdf5_file(fake_h5))  # True due to .h5 extension

        # Test file without extension but HDF5 format
        real_h5_no_ext = os.path.join(self.temp_dir, "no_extension")
        with h5py.File(real_h5_no_ext, "w") as f:
            f.create_dataset("test", data=[1, 2, 3])
        self.assertTrue(_is_hdf5_file(real_h5_no_ext))

        # Test with HDF5 path containing #
        h5_file = self._create_hdf5_file("test.h5", ["sys1"])
        self.assertTrue(_is_hdf5_file(f"{h5_file}#/sys1"))

    def test_is_hdf5_format_edge_cases(self) -> None:
        """Test edge cases for _is_hdf5_format function."""
        # Test non-existent file
        self.assertFalse(_is_hdf5_format("/non/existent/file"))

        # Test non-HDF5 file
        text_file = os.path.join(self.temp_dir, "text.txt")
        with open(text_file, "w") as f:
            f.write("plain text")
        self.assertFalse(_is_hdf5_format(text_file))

        # Test valid HDF5 file
        h5_file = os.path.join(self.temp_dir, "valid.h5")
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        self.assertTrue(_is_hdf5_format(h5_file))

    def test_is_hdf5_multisystem_edge_cases(self) -> None:
        """Test edge cases for _is_hdf5_multisystem function."""
        # Test non-existent file
        self.assertFalse(_is_hdf5_multisystem("/non/existent/file.h5"))

        # Test non-HDF5 file
        fake_h5 = os.path.join(self.temp_dir, "fake.h5")
        with open(fake_h5, "w") as f:
            f.write("not hdf5")
        self.assertFalse(_is_hdf5_multisystem(fake_h5))

        # Test empty HDF5 file
        empty_h5 = os.path.join(self.temp_dir, "empty.h5")
        with h5py.File(empty_h5, "w") as f:
            pass
        self.assertFalse(_is_hdf5_multisystem(empty_h5))

        # Test file with single system-like group (should be False)
        single_group = os.path.join(self.temp_dir, "single_group.h5")
        with h5py.File(single_group, "w") as f:
            grp = f.create_group("system1")
            grp.create_dataset("type.raw", data=[0, 1])
            grp.create_group("set.000")
        self.assertFalse(_is_hdf5_multisystem(single_group))

        # Test file with multiple system-like groups (should be True)
        multi_group = os.path.join(self.temp_dir, "multi_group.h5")
        with h5py.File(multi_group, "w") as f:
            for i in range(2):
                grp = f.create_group(f"system{i}")
                grp.create_dataset("type.raw", data=[0, 1])
                grp.create_group("set.000")
        self.assertTrue(_is_hdf5_multisystem(multi_group))

        # Test file with groups that don't look like systems
        non_system_groups = os.path.join(self.temp_dir, "non_system.h5")
        with h5py.File(non_system_groups, "w") as f:
            grp1 = f.create_group("group1")
            grp1.create_dataset("random_data", data=[1, 2, 3])
            grp2 = f.create_group("group2")
            grp2.create_dataset("other_data", data=[4, 5, 6])
        self.assertFalse(_is_hdf5_multisystem(non_system_groups))

    def test_hdf5_file_read_error_handling(self) -> None:
        """Test error handling when HDF5 files cannot be read."""
        # Create a corrupted file with .h5 extension
        corrupted_h5 = os.path.join(self.temp_dir, "corrupted.h5")
        with open(corrupted_h5, "wb") as f:
            f.write(b"corrupted\x00\x01\x02data")

        # Should handle gracefully and treat as regular system
        input_systems = [corrupted_h5]
        result = process_systems(input_systems)
        self.assertEqual(result, [corrupted_h5])

    def test_empty_systems_list(self) -> None:
        """Test with empty systems list."""
        result = process_systems([])
        self.assertEqual(result, [])

    def test_non_hdf5_files_with_hdf5_extensions(self) -> None:
        """Test files with .hdf5 extension that aren't valid HDF5."""
        fake_hdf5 = os.path.join(self.temp_dir, "fake.hdf5")
        with open(fake_hdf5, "w") as f:
            f.write("not an hdf5 file")

        input_systems = [fake_hdf5]
        result = process_systems(input_systems)
        self.assertEqual(result, [fake_hdf5])

    def test_hdf5_with_mixed_group_types(self) -> None:
        """Test HDF5 file with mix of system and non-system groups."""
        mixed_h5 = os.path.join(self.temp_dir, "mixed.h5")
        with h5py.File(mixed_h5, "w") as f:
            # Valid system group
            sys_grp = f.create_group("water_system")
            sys_grp.create_dataset("type.raw", data=[0, 1])
            sys_grp.create_dataset("type_map.raw", data=[b"H", b"O"])
            sys_grp.create_group("set.000")

            # Invalid group (no type.raw)
            invalid_grp = f.create_group("metadata")
            invalid_grp.create_dataset("info", data=[1, 2, 3])

            # Another valid system group
            sys_grp2 = f.create_group("ice_system")
            sys_grp2.create_dataset("type.raw", data=[0, 1])
            sys_grp2.create_dataset("type_map.raw", data=[b"H", b"O"])
            sys_grp2.create_group("set.000")

        input_systems = [mixed_h5]
        result = process_systems(input_systems)

        # Should expand to include only valid systems
        expected = [f"{mixed_h5}#/water_system", f"{mixed_h5}#/ice_system"]
        self.assertEqual(len(result), 2)
        for expected_sys in expected:
            self.assertIn(expected_sys, result)

    def test_patterns_parameter(self) -> None:
        """Test process_systems with patterns parameter."""
        # Test that patterns parameter still works for single string input
        regular_dir = self._create_regular_system("pattern_test")
        result = process_systems(regular_dir, patterns=["*"])
        # This should call rglob_sys_str instead of expand_sys_str
        # The exact behavior depends on the directory structure, but it should not crash
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
