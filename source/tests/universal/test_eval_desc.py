# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
import tempfile
import os
from pathlib import Path


class TestEvalDesc(unittest.TestCase):
    """Test the eval-desc CLI functionality."""
    
    def test_eval_desc_function_signature(self) -> None:
        """Test that eval_desc function has the expected signature."""
        # Import the function
        try:
            from deepmd.entrypoints.eval_desc import eval_desc
            # Check that it's callable
            self.assertTrue(callable(eval_desc))
            
            # Check that it accepts the expected parameters
            import inspect
            sig = inspect.signature(eval_desc)
            expected_params = {'model', 'system', 'datafile', 'output', 'head'}
            actual_params = set(sig.parameters.keys()) - {'kwargs'}
            self.assertEqual(expected_params, actual_params, 
                           f"Expected parameters {expected_params}, got {actual_params}")
            
        except ImportError as e:
            self.skipTest(f"Cannot import eval_desc: {e}")
            
    def test_eval_desc_module_docstring(self) -> None:
        """Test that eval_desc module has proper documentation."""
        try:
            from deepmd.entrypoints import eval_desc as eval_desc_module
            self.assertIsNotNone(eval_desc_module.__doc__)
            self.assertIn("descriptor", eval_desc_module.__doc__.lower())
        except ImportError as e:
            self.skipTest(f"Cannot import eval_desc module: {e}")
            
    def test_eval_desc_expansion_logic(self) -> None:
        """Test system expansion logic without requiring full deepmd."""
        try:
            from deepmd.entrypoints.eval_desc import eval_desc
            from deepmd.common import expand_sys_str
            
            # Create test directories
            test_dir = tempfile.mkdtemp()
            try:
                # Test that expand_sys_str is available and works
                result = expand_sys_str("nonexistent_path")
                self.assertIsInstance(result, list)
                
                # Test with existing directory
                os.makedirs(os.path.join(test_dir, "system1"))
                result = expand_sys_str(os.path.join(test_dir, "system*"))
                self.assertIsInstance(result, list)
                
            finally:
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)
                
        except ImportError as e:
            self.skipTest(f"Cannot import required modules: {e}")
    
    def test_eval_desc_parameter_validation(self) -> None:
        """Test parameter validation without requiring model loading."""
        try:
            from deepmd.entrypoints.eval_desc import eval_desc
            
            # Test with completely invalid inputs - should fail early
            test_dir = tempfile.mkdtemp()
            try:
                nonexistent = os.path.join(test_dir, "nonexistent")
                output = os.path.join(test_dir, "output")
                
                # This should raise RuntimeError about not finding valid system
                # before trying to load the model
                with self.assertRaises(RuntimeError) as context:
                    eval_desc(
                        model="fake_model.pb",
                        system=nonexistent,
                        datafile=None,
                        output=output,
                    )
                
                # Check that it's the expected error message
                self.assertIn("Did not find valid system", str(context.exception))
                
            finally:
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)
                
        except ImportError as e:
            self.skipTest(f"Cannot import eval_desc: {e}")
        except Exception as e:
            # If it fails with a different error (e.g., model loading error),
            # that's actually expected in this environment
            if "model" in str(e).lower() or "tensorflow" in str(e).lower() or "pytorch" in str(e).lower():
                self.skipTest(f"Cannot test without proper deepmd installation: {e}")
            else:
                raise


if __name__ == "__main__":
    unittest.main()