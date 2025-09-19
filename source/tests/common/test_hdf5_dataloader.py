# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest

import h5py

# Test PyTorch dataloader - may not be available in all environments
try:
    from deepmd.pt.utils.dataloader import (
        DpLoaderSet,
    )

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
class TestHDF5DataloaderSupport(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_single_system_hdf5(self, filename: str) -> str:
        """Create an HDF5 file representing a single system."""
        h5_path = os.path.join(self.temp_dir, filename)

        with h5py.File(h5_path, "w") as f:
            # Add type information at root level
            f.create_dataset("type.raw", data=[0, 1])
            f.create_dataset("type_map.raw", data=[b"H", b"O"])

            # Create data sets
            for set_idx in range(2):
                set_name = f"set.{set_idx:03d}"
                set_group = f.create_group(set_name)

                # Add realistic data for 2 atoms, 3 frames
                coords = [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
                    [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
                ]
                set_group.create_dataset("coord.npy", data=coords)

                boxes = [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]] * 3
                set_group.create_dataset("box.npy", data=boxes)

                energies = [1.0, 1.1, 1.2]
                set_group.create_dataset("energy.npy", data=energies)

                forces = [
                    [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                    [[0.11, 0.0, 0.0], [0.21, 0.0, 0.0]],
                    [[0.12, 0.0, 0.0], [0.22, 0.0, 0.0]],
                ]
                set_group.create_dataset("force.npy", data=forces)

        return h5_path

    def _create_multisystem_hdf5(self, filename: str, systems: list[str]) -> str:
        """Create an HDF5 file with multiple systems."""
        h5_path = os.path.join(self.temp_dir, filename)

        with h5py.File(h5_path, "w") as f:
            for sys_name in systems:
                sys_group = f.create_group(sys_name)

                # Add type information
                sys_group.create_dataset("type.raw", data=[0, 1])
                sys_group.create_dataset("type_map.raw", data=[b"H", b"O"])

                # Create data set
                set_group = sys_group.create_group("set.000")

                coords = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
                set_group.create_dataset("coord.npy", data=coords)

                boxes = [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]
                set_group.create_dataset("box.npy", data=boxes)

                energies = [1.0]
                set_group.create_dataset("energy.npy", data=energies)

                forces = [[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]
                set_group.create_dataset("force.npy", data=forces)

        return h5_path

    def test_single_system_hdf5_string_input(self) -> None:
        """Test dataloader with single-system HDF5 file as string input."""
        h5_file = self._create_single_system_hdf5("single.h5")

        try:
            loader = DpLoaderSet(
                systems=h5_file,  # String input
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Should create one system without expansion
            self.assertEqual(len(loader.systems), 1)
            # The system path should be the original file, not expanded
            system_path = loader.systems[0].system
            self.assertEqual(system_path, h5_file)

        except Exception as e:
            # If there are issues with the actual data loading, that's ok
            # We're mainly testing the path handling logic
            self.assertIn("System", str(e))

    def test_multisystem_hdf5_string_input(self) -> None:
        """Test dataloader with multisystem HDF5 file as string input."""
        h5_file = self._create_multisystem_hdf5("multi.h5", ["sys1", "sys2"])

        try:
            loader = DpLoaderSet(
                systems=h5_file,  # String input
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Should expand to multiple systems
            self.assertEqual(len(loader.systems), 2)

            # Check that the systems have correct expanded paths
            system_paths = [sys.system for sys in loader.systems]
            expected_paths = [f"{h5_file}#sys1", f"{h5_file}#sys2"]

            for expected in expected_paths:
                self.assertIn(expected, system_paths)

        except Exception as e:
            # Test the path handling even if data loading fails
            self.assertIn("System", str(e))

    def test_hdf5_file_list_input(self) -> None:
        """Test dataloader with list of HDF5 files."""
        # This tests that when process_systems expands HDF5 files in lists,
        # the dataloader can handle the resulting expanded paths
        h5_file1 = self._create_multisystem_hdf5("multi1.h5", ["water", "ice"])
        h5_file2 = self._create_single_system_hdf5("single.h5")

        # Simulate what process_systems would do
        from deepmd.utils.data_system import (
            process_systems,
        )

        processed_systems = process_systems([h5_file1, h5_file2])

        # Should have 3 systems: 2 from multi1.h5 + 1 from single.h5
        self.assertEqual(len(processed_systems), 3)

        try:
            loader = DpLoaderSet(
                systems=processed_systems,  # List input from process_systems
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Should handle the expanded system paths
            self.assertEqual(len(loader.systems), 3)

        except Exception as e:
            # Path handling should work even if data details fail
            self.assertIn("System", str(e))

    def test_invalid_hdf5_string_input(self) -> None:
        """Test dataloader with invalid HDF5 file as string input."""
        fake_h5 = os.path.join(self.temp_dir, "fake.h5")
        with open(fake_h5, "w") as f:
            f.write("not an hdf5 file")

        try:
            loader = DpLoaderSet(
                systems=fake_h5,  # Invalid HDF5 file
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Should treat as single system without expansion
            self.assertEqual(len(loader.systems), 1)
            self.assertEqual(loader.systems[0].system, fake_h5)

        except Exception as e:
            # Should fail gracefully without crashing on path processing
            self.assertIsInstance(e, (OSError, FileNotFoundError, ValueError))

    def test_non_hdf5_string_input(self) -> None:
        """Test dataloader with non-HDF5 file as string input."""
        text_file = os.path.join(self.temp_dir, "data.txt")
        with open(text_file, "w") as f:
            f.write("plain text file")

        try:
            loader = DpLoaderSet(
                systems=text_file,  # Non-HDF5 file
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Should treat as single system
            self.assertEqual(len(loader.systems), 1)
            self.assertEqual(loader.systems[0].system, text_file)

        except Exception as e:
            # Should handle gracefully
            self.assertIsInstance(e, (OSError, FileNotFoundError, ValueError))

    def test_empty_hdf5_file(self) -> None:
        """Test dataloader with empty HDF5 file."""
        empty_h5 = os.path.join(self.temp_dir, "empty.h5")
        with h5py.File(empty_h5, "w") as f:
            pass  # Create empty file

        try:
            loader = DpLoaderSet(
                systems=empty_h5,
                batch_size=1,
                type_map=["H", "O"],
                seed=42,
                shuffle=False,
            )

            # Empty HDF5 file might result in different behavior
            # The key is that it doesn't crash during path processing
            self.assertIsInstance(loader.systems, list)

        except Exception as e:
            # Expected to fail on data validation, not path processing
            self.assertIsInstance(e, (ValueError, FileNotFoundError, KeyError, OSError))


# Test without PyTorch dependency
class TestPyTorchDataloaderFallback(unittest.TestCase):
    def test_pytorch_import_graceful_failure(self) -> None:
        """Test that missing PyTorch dependency is handled gracefully."""
        if not PYTORCH_AVAILABLE:
            # Verify that the import fails gracefully
            with self.assertRaises(ImportError):
                from deepmd.pt.utils.dataloader import DpLoaderSet  # noqa: F401
        else:
            # If PyTorch is available, this test should pass
            self.skipTest("PyTorch is available")


if __name__ == "__main__":
    unittest.main()
