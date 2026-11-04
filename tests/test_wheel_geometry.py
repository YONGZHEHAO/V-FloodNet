"""Check reference geometry, affine invariance, and unavailable cases."""
import unittest
import numpy as np
from scripts.wheel_geometry import fit_reference, reference_submergence


class WheelGeometryTests(unittest.TestCase):
    def test_known_quarter_submergence(self):
        angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
        points = np.column_stack([100+40*np.cos(angles), 100+40*np.sin(angles)])
        self.assertAlmostEqual(reference_submergence(fit_reference(points), [[30, 120], [170, 120]]), .25, places=5)

    def test_affine_oblique_projection_preserves_normalized_chord(self):
        angles = np.linspace(-np.pi, .2, 30)
        points = np.column_stack([40*np.cos(angles), 40*np.sin(angles)])
        waterline = np.array([[-50, 16], [50, 16]])
        affine = np.array([[.55, .25], [.12, 1.1]])
        offset = np.array([250, 300])
        projected = points @ affine.T + offset
        projected_line = waterline @ affine.T + offset
        self.assertAlmostEqual(reference_submergence(fit_reference(projected), projected_line), .3, places=4)

    def test_collinear_landmarks_are_unavailable(self):
        with self.assertRaisesRegex(ValueError, "collinear"):
            fit_reference([[i, i] for i in range(10)])

    def test_too_short_visible_arc_is_unavailable(self):
        angles = np.linspace(-1.4, -.8, 12)
        with self.assertRaisesRegex(ValueError, "too short"):
            fit_reference(np.column_stack([100+40*np.cos(angles), 100+40*np.sin(angles)]))

    def test_waterline_outside_reference_is_not_zero_water_depth(self):
        angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
        ellipse = fit_reference(np.column_stack([100+40*np.cos(angles), 100+40*np.sin(angles)]))
        with self.assertRaisesRegex(ValueError, "does not cut"):
            reference_submergence(ellipse, [[0, 180], [200, 180]])


if __name__ == "__main__":
    unittest.main()
