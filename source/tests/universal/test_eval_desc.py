# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import dpdata
    DPDATA_AVAILABLE = True
except ImportError:
    DPDATA_AVAILABLE = False

try:
    from deepmd.entrypoints.eval_desc import eval_desc
    EVAL_DESC_AVAILABLE = True
except ImportError:
    EVAL_DESC_AVAILABLE = False

try:
    from ..infer.case import get_cases
    CASES_AVAILABLE = True
except ImportError:
    CASES_AVAILABLE = False


@unittest.skipIf(not DPDATA_AVAILABLE, "dpdata not available")
@unittest.skipIf(not EVAL_DESC_AVAILABLE, "eval_desc not available")
@unittest.skipIf(not CASES_AVAILABLE, "test cases not available")
class TestEvalDesc(unittest.TestCase):
    """Test the eval-desc CLI functionality."""
    
    def setUp(self) -> None:
        """Set up test data and temporary directories."""
        # Get a test case with a model
        self.case = get_cases()["se_e2_a"]
        self.model_path = self.case.get_model(".pb")
        
        # Create test system data
        self.coords = np.array([
            12.83, 2.56, 2.18,
            12.09, 2.87, 2.74,
            00.25, 3.32, 1.68,
            3.36, 3.00, 1.81,
            3.51, 2.51, 2.60,
            4.27, 3.22, 1.56,
        ])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
        
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.system_dir = os.path.join(self.test_dir, "system")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        # Create test system using manual method (like in test_dp_test.py)
        self._create_test_system()
        
    def _create_test_system(self) -> None:
        """Create test system manually following test_dp_test.py pattern."""
        # Create system directory structure
        os.makedirs(self.system_dir, exist_ok=True)
        set_dir = os.path.join(self.system_dir, "set.000")
        os.makedirs(set_dir, exist_ok=True)
        
        # Save system data as .npy files
        np.save(os.path.join(set_dir, "coord.npy"), self.coords.reshape(1, 6, 3))
        np.save(os.path.join(set_dir, "box.npy"), self.box.reshape(1, 3, 3))
        np.save(os.path.join(set_dir, "type.npy"), np.array(self.atype))
        np.save(os.path.join(set_dir, "energy.npy"), np.array([0.0]))  # dummy energy
        np.save(os.path.join(set_dir, "force.npy"), np.zeros((1, 6, 3)))  # dummy forces
        
        # Create type.raw and type_map.raw
        with open(os.path.join(self.system_dir, "type.raw"), 'w') as f:
            f.write(' '.join(map(str, self.atype)))
        with open(os.path.join(self.system_dir, "type_map.raw"), 'w') as f:
            f.write('O\nH\n')
        
    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up model file."""
        if hasattr(cls, 'case'):
            try:
                os.remove(cls.case.get_model(".pb"))
            except (AttributeError, FileNotFoundError):
                pass
        
    def test_eval_desc_single_system(self) -> None:
        """Test evaluating descriptors for a single system."""
        # Run eval_desc
        eval_desc(
            model=self.model_path,
            system=self.system_dir,
            datafile=None,
            output=self.output_dir,
        )
        
        # Check output
        output_file = Path(self.output_dir) / "system.npy"
        self.assertTrue(output_file.exists(), "Descriptor output file should exist")
        
        # Load and check descriptors
        descriptors = np.load(output_file)
        self.assertIsInstance(descriptors, np.ndarray, "Descriptors should be numpy array")
        self.assertEqual(len(descriptors.shape), 2, "Descriptors should be 2D array")
        self.assertEqual(descriptors.shape[0], 1, "Should have 1 frame")
        self.assertGreater(descriptors.shape[1], 0, "Should have descriptor features")
        
    def test_eval_desc_with_datafile(self) -> None:
        """Test evaluating descriptors using a datafile with system paths."""
        # Create a second system
        system2_dir = os.path.join(self.test_dir, "system2")
        os.makedirs(system2_dir, exist_ok=True)
        set_dir2 = os.path.join(system2_dir, "set.000")
        os.makedirs(set_dir2, exist_ok=True)
        
        # Copy system data
        np.save(os.path.join(set_dir2, "coord.npy"), self.coords.reshape(1, 6, 3))
        np.save(os.path.join(set_dir2, "box.npy"), self.box.reshape(1, 3, 3))
        np.save(os.path.join(set_dir2, "type.npy"), np.array(self.atype))
        np.save(os.path.join(set_dir2, "energy.npy"), np.array([0.0]))
        np.save(os.path.join(set_dir2, "force.npy"), np.zeros((1, 6, 3)))
        
        with open(os.path.join(system2_dir, "type.raw"), 'w') as f:
            f.write(' '.join(map(str, self.atype)))
        with open(os.path.join(system2_dir, "type_map.raw"), 'w') as f:
            f.write('O\nH\n')
        
        # Create datafile
        datafile_path = os.path.join(self.test_dir, "systems.txt")
        with open(datafile_path, 'w') as f:
            f.write(f"{self.system_dir}\n")
            f.write(f"{system2_dir}\n")
        
        # Run eval_desc with datafile
        eval_desc(
            model=self.model_path,
            system=None,
            datafile=datafile_path,
            output=self.output_dir,
        )
        
        # Check outputs for both systems
        output_file1 = Path(self.output_dir) / "system.npy"
        output_file2 = Path(self.output_dir) / "system2.npy"
        
        self.assertTrue(output_file1.exists(), "First system output should exist")
        self.assertTrue(output_file2.exists(), "Second system output should exist")
        
        # Check descriptor shapes are consistent
        desc1 = np.load(output_file1)
        desc2 = np.load(output_file2)
        self.assertEqual(desc1.shape, desc2.shape, "Descriptors should have same shape")
        
    def test_eval_desc_custom_output_dir(self) -> None:
        """Test evaluating descriptors with custom output directory."""
        custom_output = os.path.join(self.test_dir, "custom_desc")
        
        # Run eval_desc with custom output
        eval_desc(
            model=self.model_path,
            system=self.system_dir,
            datafile=None,
            output=custom_output,
        )
        
        # Check output in custom directory
        output_file = Path(custom_output) / "system.npy"
        self.assertTrue(output_file.exists(), "Output should be in custom directory")
        
    def test_eval_desc_error_no_system(self) -> None:
        """Test that eval_desc raises error when no valid system is found."""
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        
        with self.assertRaises(RuntimeError) as context:
            eval_desc(
                model=self.model_path,
                system=nonexistent_dir,
                datafile=None,
                output=self.output_dir,
            )
        
        self.assertIn("Did not find valid system", str(context.exception))


@unittest.skipIf(DPDATA_AVAILABLE or EVAL_DESC_AVAILABLE or CASES_AVAILABLE, "All dependencies available")
class TestEvalDescSkipped(unittest.TestCase):
    """Placeholder test when dependencies are not available."""
    
    def test_dependencies_not_available(self) -> None:
        """Test that shows which dependencies are missing."""
        missing = []
        if not DPDATA_AVAILABLE:
            missing.append("dpdata")
        if not EVAL_DESC_AVAILABLE:
            missing.append("eval_desc")
        if not CASES_AVAILABLE:
            missing.append("test cases")
        
        self.skipTest(f"Missing dependencies: {', '.join(missing)}")


if __name__ == "__main__":
    unittest.main()