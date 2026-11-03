"""Regression checks for water/background behavior before depth estimation."""
import unittest

import numpy as np

from myutils.data import postprocessing_pred


class WaterMaskTests(unittest.TestCase):
    def test_dry_scene_stays_dry(self):
        mask = np.zeros((12, 15), dtype=np.uint8)
        np.testing.assert_array_equal(postprocessing_pred(mask), mask)

    def test_fully_flooded_scene_stays_water(self):
        mask = np.ones((12, 15), dtype=np.uint8)
        np.testing.assert_array_equal(postprocessing_pred(mask), mask)

    def test_largest_water_region_excludes_background_and_small_island(self):
        mask = np.zeros((15, 20), dtype=np.uint8)
        mask[1:5, 2:8] = 1
        mask[10:12, 14:16] = 1
        expected = mask.copy()
        expected[10:12, 14:16] = 0
        np.testing.assert_array_equal(postprocessing_pred(mask), expected)

    def test_water_at_top_left_is_not_inverted(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:4, :4] = 1
        np.testing.assert_array_equal(postprocessing_pred(mask), mask)


if __name__ == '__main__':
    unittest.main()
