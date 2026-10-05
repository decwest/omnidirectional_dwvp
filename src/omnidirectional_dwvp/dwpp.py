"""DWPP window selection from dwpp/simulator/controllers.py (MIT).

Copyright (c) 2024 Fumiya Ohnishi. See LICENSE and docs/provenance.md.
The 1e-3 curvature/tie thresholds and candidate ordering are unchanged.
"""
import math


def optimal_velocity_in_window(
    window: tuple[float, float, float, float], curvature: float
) -> tuple[float, float]:
    """Faithful forward-motion port of computeOptimalVelocityWithinDynamicWindow."""
    v_hi, v_lo, w_hi, w_lo = window
    if abs(curvature) < 1e-3:
        if w_lo <= 0.0 <= w_hi:
            return v_hi, 0.0
        return v_hi, w_lo if abs(w_lo) <= abs(w_hi) else w_hi

    candidates = (
        (v_lo, curvature * v_lo),
        (v_hi, curvature * v_hi),
        (w_lo / curvature, w_lo),
        (w_hi / curvature, w_hi),
    )
    valid = [
        (v, w)
        for v, w in candidates
        if v_lo <= v <= v_hi and w_lo <= w <= w_hi
    ]
    if valid:
        return max(valid, key=lambda command: command[0])

    corners = ((v_lo, w_lo), (v_lo, w_hi), (v_hi, w_lo), (v_hi, w_hi))
    denominator = math.sqrt(curvature * curvature + 1.0)
    best = corners[0]
    closest = math.inf
    for corner in corners:
        distance = abs(curvature * corner[0] - corner[1]) / denominator
        if distance < closest or (abs(distance - closest) <= 1e-3 and corner[0] > best[0]):
            closest = distance
            best = corner
    return best
