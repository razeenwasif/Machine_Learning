import unittest
import cudf
import cupy as cp
import numpy as np

# Import the GPU functions to be tested
from comparison import (
    run_exact_comp_gpu,
    run_jaccard_gpu,
    run_dice_gpu
)

class TestGpuComparisonFunctions(unittest.TestCase):
    """
    Test suite for the GPU-accelerated comparison functions.
    """

    def assertGpuResults(self, result_series, expected_values, places=5):
        """Helper function to check GPU results against expected values."""
        self.assertIsInstance(result_series, cudf.Series)
        self.assertEqual(len(result_series), len(expected_values))
        
        # Move result to CPU for comparison
        result_list = result_series.to_arrow().to_pylist()
        
        for res, exp in zip(result_list, expected_values):
            self.assertAlmostEqual(res, exp, places=places)
            self.assertGreaterEqual(res, 0.0)
            self.assertLessEqual(res, 1.0)

    def test_exact_comp_gpu(self):
        """Test the exact comparison GPU kernel."""
        s1 = cudf.Series(['apple', 'banana', 'orange', 'grape', ''])
        s2 = cudf.Series(['apple', 'banaa', 'orange', 'grapefruit', ''])
        expected = [1.0, 0.0, 1.0, 0.0, 1.0]
        
        result = run_exact_comp_gpu(s1, s2)
        self.assertGpuResults(result, expected)
        print("TestExactCompGPU: PASSED")

    def test_jaccard_gpu(self):
        """Test the Jaccard similarity GPU kernel for q-grams."""
        s1 = cudf.Series(['peter', 'crate', 'charlie'])
        s2 = cudf.Series(['peter', 'trace', 'charles'])
        expected = [1.0, 0.14286, 0.5]

        result = run_jaccard_gpu(s1, s2)
        self.assertGpuResults(result, expected)
        print("TestJaccardGPU: PASSED")

    def test_dice_gpu(self):
        """Test the Dice similarity GPU kernel for q-grams."""
        s1 = cudf.Series(['peter', 'crate', 'charlie'])
        s2 = cudf.Series(['peter', 'trace', 'charles'])
        expected = [1.0, 0.25, 0.66667]

        result = run_dice_gpu(s1, s2)
        self.assertGpuResults(result, expected)
        print("TestDiceGPU: PASSED")

    def test_empty_and_short_strings(self):
        """Test edge cases like empty and short strings for q-gram measures."""
        s1 = cudf.Series(['', 'a', 'ab', 'abc'])
        s2 = cudf.Series(['', 'b', 'ab', 'xyz'])
        # Jaccard: 1.0, 0.0, 1.0, 0.0
        jaccard_result = run_jaccard_gpu(s1, s2)
        self.assertGpuResults(jaccard_result, [1.0, 0.0, 1.0, 0.0])
        # Dice: 1.0, 0.0, 1.0, 0.0
        dice_result = run_dice_gpu(s1, s2)
        self.assertGpuResults(dice_result, [1.0, 0.0, 1.0, 0.0])
        print("TestEmptyAndShortStrings: PASSED")

if __name__ == '__main__':
    print("=================================================")
    print("       RUNNING GPU COMPARISON UNIT TESTS         ")
    print("=================================================")
    unittest.main()


