"""IROS reference geometry from dwpp@28bc0207 (MIT); hardware scale explicit."""
import math
import numpy as np


def append_heading_to_path(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty((0, 3))
    if len(path_xy) == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], 0.0]])

    diffs = np.diff(path_xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])

    return np.c_[path_xy, headings]


def right_angle_polyline_curve(segment_length: float = 0.5, points_per_segment: int = 50) -> np.ndarray:
    """
    90度の折れ線経路 (N x 3) を生成する。
    1辺目: (0, 0) -> (segment_length, 0), 姿勢 0 [rad]
    2辺目: (segment_length, 0) -> (segment_length, segment_length), 姿勢 pi/2 [rad]
    """
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    x1 = np.linspace(0.0, segment_length, points_per_segment + 1)
    y1 = np.zeros_like(x1)
    theta1 = np.zeros_like(x1)

    x2 = np.full(points_per_segment + 1, segment_length)
    y2 = np.linspace(0.0, segment_length, points_per_segment + 1)
    theta2 = np.full(points_per_segment + 1, np.pi / 2.0)

    # 折れ点の重複を避けるため2区間目の先頭を除外
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    theta = np.concatenate([theta1, theta2[1:]])

    return np.c_[x, y, theta]


def right_angle_polyline_curve_last_segment_heading_minus_pi(
    segment_length: float = 0.5,
    points_per_segment: int = 50
) -> np.ndarray:
    """
    path1(90度折れ線)をベースに、最後の1点のみ姿勢角を -pi に設定した経路を生成する。
    それ以外の点の姿勢角は経路の接線方向に合わせる。
    """
    base_path = right_angle_polyline_curve(
        segment_length=segment_length,
        points_per_segment=points_per_segment,
    )
    path = append_heading_to_path(base_path[:, :2])
    path[-1, 2] = -np.pi

    return path


def _resample_xy_by_arclength(x: np.ndarray, y: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s[-1])

    if total_length <= 1e-12:
        return (
            np.linspace(float(x[0]), float(x[-1]), num_points),
            np.linspace(float(y[0]), float(y[-1]), num_points),
        )

    s_new = np.linspace(0.0, total_length, num_points)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new


def one_minus_cos_curve(
    amplitude: float = 1.0,
    length_x: float = 10.0,
    num_points: int = 200,
    cycles: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
    resample_arclength: bool = True,
) -> np.ndarray:
    """
    y = A(1 - cos(k(x-x0))) 形状の経路を N x 3 (x, y, theta) で生成する。
    theta は接線方向（arctan2(dy, dx)）を用いる。
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if length_x <= 0.0:
        raise ValueError("length_x must be > 0")

    a = float(amplitude)
    l = float(length_x)
    k = 2.0 * math.pi * float(cycles) / l

    x = np.linspace(float(x0), float(x0) + l, num_points, dtype=float)
    u = x - float(x0)
    y = float(y0) + a * (1.0 - np.cos(k * u))

    if abs(theta0) > 0.0:
        c = math.cos(theta0)
        s = math.sin(theta0)
        x_shift = x - float(x0)
        y_shift = y - float(y0)
        x = float(x0) + c * x_shift - s * y_shift
        y = float(y0) + s * x_shift + c * y_shift

    if resample_arclength:
        x, y = _resample_xy_by_arclength(x, y, num_points)

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)

    return np.c_[x, y, theta]


def make_paths():
    corner = right_angle_polyline_curve_last_segment_heading_minus_pi(1.0, 100)
    curve = one_minus_cos_curve(amplitude=.75, length_x=1.5, num_points=501, cycles=1.5, resample_arclength=True)
    constant = corner.copy()
    constant[:, 2] = 0.0
    independent = curve.copy()
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1))]
    s = arc / arc[-1]
    independent[:, 2] = (3*s*s - 2*s*s*s) * (np.pi / 2)
    return {"iros_docking": corner, "iros_curve": curve,
            "constant_heading_corner": constant, "independent_heading_curve": independent}


def straight_path(length=4.0, spacing=0.005, heading=0.0):
    """Straight path from the origin with tangent heading, including its endpoint."""
    if not all(math.isfinite(v) for v in (length, spacing, heading)) or min(length, spacing) <= 0:
        raise ValueError("length and spacing must be finite and positive")
    s = np.r_[np.arange(0.0, length, spacing), length]
    return np.c_[s * math.cos(heading), s * math.sin(heading), np.full_like(s, heading)]


def orientation_ramp_path(transition_length, length=4.0, spacing=0.005, start=1.0):
    """Straight x-axis path with independent yaw 0 to pi/2; zero length is a step.

    The step uses duplicate XY poses at start so projected reference yaw has a
    discontinuity instead of a one-sample artificial ramp. At exactly start,
    segment projection chooses the incoming (zero-yaw) segment.
    """
    if not all(math.isfinite(v) for v in (transition_length, start)) or transition_length < 0:
        raise ValueError("transition length must be finite and nonnegative")
    if start <= 0 or start + transition_length >= length:
        raise ValueError("transition must lie inside the path")
    path = straight_path(length, spacing)
    if transition_length == 0:
        x = np.r_[path[path[:, 0] < start, 0], start, start, path[path[:, 0] > start, 0]]
        yaw = np.where(x >= start, math.pi / 2, 0.0)
        yaw[np.flatnonzero(x == start)[0]] = 0.0
        return np.c_[x, np.zeros_like(x), yaw]
    path[:, 2] = np.clip((path[:, 0] - start) / transition_length, 0.0, 1.0) * math.pi / 2
    return path
