import unittest
import numpy as np
from ft import save_heatmap_with_timestamp
import os

class TestUtils(unittest.TestCase):

    def test_heatmap_generation(self):
        data = np.random.rand(10, 10)
        filename = save_heatmap_with_timestamp(data)
        self.assertTrue(os.path.isfile(filename))
        os.remove(filename)  # Clean up after test

if __name__ == '__main__':
    unittest.main()
