"""Ellipse/waterline geometry for reviewed wheel-reference annotations.

The normalized chord position assumes a local affine view of a circular
reference in a vertical wheel plane. It is not a calibrated water depth.
"""
import math
import numpy as np


def fit_reference(points, minimum_arc_degrees=120):
    import cv2
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("Provide at least eight finite, visible boundary points")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Boundary points are collinear")
    (cx, cy), (diameter_a, diameter_b), angle = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    radii = np.array([diameter_a/2, diameter_b/2], dtype=float)
    if not np.isfinite([cx, cy, *radii, angle]).all() or min(radii) <= 0:
        raise ValueError("Invalid fitted ellipse")
    if min(radii) / max(radii) < .15:
        raise ValueError("Reference is too narrow or poorly constrained")
    theta = math.radians(angle)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    normalized = ((points - [cx, cy]) @ rotation) / radii
    residual = float(np.sqrt(np.mean((np.linalg.norm(normalized, axis=1)-1)**2)))
    angles = np.sort(np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2*math.pi))
    arc = math.degrees(2*math.pi - np.max(np.diff(np.r_[angles, angles[0]+2*math.pi])))
    if arc < minimum_arc_degrees:
        raise ValueError("Visible boundary arc is too short to constrain the hidden part")
    if residual > .08:
        raise ValueError("Boundary points do not follow a consistent ellipse")
    covariance = rotation @ np.diag(radii**2) @ rotation.T
    return {"center_xy": [float(cx), float(cy)], "radii": radii.tolist(), "angle_degrees": float(angle),
            "shape_matrix": covariance.tolist(), "radial_fit_rms": residual, "observed_arc_degrees": arc}


def reference_submergence(ellipse, waterline):
    line = np.asarray(waterline, dtype=float)
    if line.shape != (2, 2) or not np.isfinite(line).all():
        raise ValueError("Waterline must contain two finite image points")
    delta = line[1] - line[0]
    if np.linalg.norm(delta) < 1e-6 or abs(delta[0]) < 1e-6:
        raise ValueError("Waterline must have distinct points and a left-to-right extent")
    normal = np.array([-delta[1], delta[0]], dtype=float)
    if normal[1] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    shape = np.asarray(ellipse["shape_matrix"])
    radius = float(np.sqrt(normal @ shape @ normal))
    if 2*radius < 24:
        raise ValueError("Projected reference diameter along the waterline normal is too small")
    signed_offset = float(normal @ (np.asarray(ellipse["center_xy"]) - line[0]))
    fraction = (1 + signed_offset/radius) / 2
    if not 0 < fraction < 1:
        raise ValueError("Waterline does not cut this reference; use a qualitative category")
    return float(fraction)


def estimate_annotation(annotation, samples=400):
    points = np.asarray(annotation["boundary_points_xy"], dtype=float)
    line = np.asarray(annotation["waterline_xy"], dtype=float)
    ellipse = fit_reference(points)
    fraction = reference_submergence(ellipse, line)
    point_sigma = float(annotation["point_uncertainty_px"])
    water_sigma = float(annotation["waterline_uncertainty_px"])
    if not (0 < point_sigma <= 20 and 0 < water_sigma <= 30):
        raise ValueError("Provide positive finite annotation uncertainty in pixels")
    rng = np.random.default_rng(20261104)
    values = []
    for _ in range(samples):
        trial_points = points + rng.normal(0, point_sigma, points.shape)
        trial_line = line + rng.normal(0, water_sigma, line.shape)
        try:
            values.append(reference_submergence(fit_reference(trial_points), trial_line))
        except ValueError:
            pass
    if len(values) < .5*samples:
        raise ValueError("Annotation perturbations frequently produce invalid fits; reference is unstable")
    low, high = np.quantile(values, [.05, .95]).tolist()
    return {"reference_submergence_fraction": fraction, "ellipse": ellipse,
            "fraction_definition": "submerged height / reference diameter under local affine projection; not area or metric water depth",
            "annotation_sensitivity_p05_p95": [low, high], "valid_perturbation_fits": len(values),
            "perturbation_samples": samples,
            "sensitivity_is_accuracy_interval": False, "metric_depth_cm": None,
            "accuracy_validated": False, "status": "manual_geometry_estimate_unvalidated",
            "quality_flags": ["local_affine_projection_assumption", "manually_reviewed_landmarks", "unknown_reference_dimensions"]
                             + (["sensitive_to_annotation"] if high-low > .25 else [])}
