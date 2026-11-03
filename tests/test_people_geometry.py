"""Check aligned crops and unavailable waterline behavior independently of SMPL."""
import unittest
import numpy as np

from scripts.prepare_people_candidates import square_crop
from scripts.people_depth import estimate_fraction


class PeopleGeometryTests(unittest.TestCase):
    def test_crop_stays_square_at_image_edges(self):
        for box in ([0, 10, 70, 150], [450, 250, 499, 399], [30, 30, 100, 180]):
            x1, y1, x2, y2 = square_crop(box, 500, 400)
            self.assertEqual(x2 - x1, y2 - y1)
            self.assertTrue(0 <= x1 < x2 <= 500 and 0 <= y1 < y2 <= 400)

    def test_no_waterline_leaves_fraction_unavailable(self):
        points = np.column_stack([np.zeros(80), np.linspace(-.9, .9, 80)])
        for mask in (np.zeros((224, 224), np.uint8), np.ones((224, 224), np.uint8)):
            self.assertIsNone(estimate_fraction(points, points, mask))

    def test_horizontal_synthetic_waterline(self):
        # A numerical boundary check; this is not a real mesh inference test.
        points = np.column_stack([np.zeros(100), np.linspace(-.9, .9, 100)])
        mask = np.zeros((224, 224), np.uint8)
        mask[112:, :] = 1
        fraction = estimate_fraction(points, points, mask)
        self.assertIsNotNone(fraction)
        self.assertTrue(.4 < fraction < .6)

    def test_three_dimensional_template_uses_vertical_coordinate(self):
        projected = np.column_stack([np.zeros(100), np.linspace(-.9, .9, 100)])
        template = np.column_stack([projected, np.zeros(100)])
        mask = np.zeros((224, 224), np.uint8)
        mask[112:, :] = 1
        self.assertEqual(estimate_fraction(template, projected, mask),
                         estimate_fraction(projected, projected, mask))


if __name__ == '__main__':
    unittest.main()
