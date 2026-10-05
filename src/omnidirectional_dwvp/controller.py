"""Common reference, speed regulation and feasible boxes for VP and DWVP."""
from dataclasses import dataclass
import math
import numpy as np
from .config import Config
from .dwpp import optimal_velocity_in_window
from .geometry import wrap, surface_distance
from .solver import calc_ray_box_intersection_alpha_range, calc_closest_alpha_to_box_from_ray


def dynamic_box(current, config):
    lower = np.maximum(current - config.acceleration * config.dt, config.lower)
    upper = np.minimum(current + config.acceleration * config.dt, config.upper)
    if np.any(lower > upper):
        raise ValueError("current velocity has no reachable physical interval")
    return lower, upper


def regulated_box(lower, upper, cap, config):
    """Shrink physical x/y box; retain reachable endpoints after an abrupt cap."""
    cap = float(np.clip(cap, 0.0, config.nominal_speed))
    ratio = 0.0 if config.box_speed <= 1e-12 else cap / config.box_speed
    regulated_lower = lower.copy()
    regulated_upper = upper.copy()
    for i in (0, 1):
        limit_lo, limit_hi = config.lower[i] * ratio, config.upper[i] * ratio
        if lower[i] > limit_hi:
            regulated_lower[i] = regulated_upper[i] = lower[i]
        elif upper[i] < limit_lo:
            regulated_lower[i] = regulated_upper[i] = upper[i]
        else:
            regulated_lower[i] = max(lower[i], limit_lo)
            regulated_upper[i] = min(upper[i], limit_hi)
    return regulated_lower, regulated_upper


def desired_vector(pose, target, config):
    delta = target[:2] - pose[:2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    body = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
    distance = float(np.linalg.norm(body))
    speed = config.translation_speed
    translation = np.zeros(2) if distance <= 1e-6 else speed * body / distance
    time = config.min_orientation_time
    if speed > 1e-12:
        time = max(time, config.orientation_time_weight * distance / speed)
    return np.r_[translation, wrap(target[2] - pose[2]) / time]


def ray_command(desired, lower, upper, prefer_large_alpha):
    """Geometric selector; control always requests the largest minimizing alpha."""
    if np.linalg.norm(desired) <= 1e-12:
        return np.clip(np.zeros(3), lower, upper), "zero"
    intersects, alpha_min, alpha_max = calc_ray_box_intersection_alpha_range(desired, lower, upper)
    if intersects:
        alpha = alpha_max if prefer_large_alpha else alpha_min
        mode = "intersection"
    else:
        alpha = calc_closest_alpha_to_box_from_ray(desired, lower, upper, prefer_large_alpha)
        mode = "projection"
    return np.clip(alpha * desired, lower, upper), mode


def uniformly_scale(vector, lower, upper):
    """Largest scale in [0, 1] inside a box containing zero, including zero limits."""
    scale = 1.0
    for value, lo, hi in zip(vector, lower, upper):
        if value > 0:
            scale = min(scale, hi / value)
        elif value < 0:
            scale = min(scale, lo / value)
    return scale * vector


def speed_cap(pose, path, arc, nearest, config, obstacles):
    nominal = config.nominal_speed
    cost_cap = nominal
    obstacle_distance = surface_distance(pose[:2], obstacles)
    # RPP recovers distance to the obstacle surface (not footprint clearance)
    # from the inflation cost. Explicit distances avoid grid quantization here.
    inflated = obstacle_distance <= config.inflation_radius
    if config.use_cost_regulation and inflated and obstacle_distance < config.cost_scaling_dist:
        cost_cap = max(config.regulated_min_speed,
                       nominal * config.cost_scaling_gain * max(0.0, obstacle_distance) / config.cost_scaling_dist)
    cap = max(cost_cap, config.regulated_min_speed)
    remaining = float(arc[-1] - arc[nearest])
    if config.approach_distance > 0 and remaining < config.approach_distance:
        euclidean = float(np.linalg.norm(path[-1, :2] - pose[:2]))
        cap = min(cap, max(config.min_approach_speed, cap * euclidean / config.approach_distance))
    return min(nominal, cap), obstacle_distance


@dataclass
class Command:
    desired: np.ndarray
    command: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    speed_cap: float
    lookahead: float
    nearest: int
    mode: str
    obstacle_distance: float


def compute_command(pose, current, path, arc, method, config, obstacles=()):
    if method not in {"vp", "vp_scaled", "vp_scaled_accel", "dwvp", "dwpp"}:
        raise ValueError(f"unknown controller: {method}")
    nearest = int(np.argmin(np.sum((path[:, :2] - pose[:2])**2, axis=1)))
    lookahead = config.fixed_lookahead
    if lookahead is None:
        lookahead = float(np.clip(config.lookahead_time * np.linalg.norm(current[:2]),
                                  config.lookahead_min, config.lookahead_max))
    # Humble RPP: Euclidean preview circle on the retained path, with the
    # first outer pose's yaw and circle/segment interpolation of x,y only.
    distances = np.linalg.norm(path[nearest:, :2] - pose[:2], axis=1)
    outer = np.flatnonzero(distances >= lookahead)
    target_index = nearest + int(outer[0]) if len(outer) else len(path) - 1
    target = path[target_index].copy()
    if target_index > nearest and len(outer):
        start = path[target_index - 1, :2] - pose[:2]
        segment = path[target_index, :2] - path[target_index - 1, :2]
        a = float(segment @ segment)
        b = 2 * float(start @ segment)
        c = float(start @ start - lookahead * lookahead)
        if a > 1e-20:
            t = float(np.clip((-b + math.sqrt(max(0., b*b - 4*a*c))) / (2*a), 0., 1.))
            target[:2] = pose[:2] + start + t * segment
    desired = desired_vector(pose, target, config)
    cap, obstacle_distance = speed_cap(pose, path, arc, nearest, config, obstacles)
    lower, upper = dynamic_box(current, config)
    lower, upper = regulated_box(lower, upper, cap, config)
    if method == "dwpp":
        lower[0] = max(0.0, lower[0])
        lower[1] = upper[1] = 0.0
    goal_distance = float(np.linalg.norm(pose[:2] - path[-1, :2]))
    if goal_distance <= config.goal_xy or config.box_speed <= 1e-12:
        goal_yaw = terminal_heading(path) if method == "dwpp" else path[-1, 2]
        yaw_error = float(wrap(goal_yaw - pose[2]))
        target_w = 0.0 if abs(yaw_error) <= config.goal_yaw else math.copysign(
            min(abs(yaw_error) / config.min_orientation_time, math.sqrt(2 * config.aw * abs(yaw_error))), yaw_error)
        desired = np.array([0.0, 0.0, target_w])
        command = np.clip(desired, lower, upper)
        mode = "terminal"
    elif method == "dwpp":
        delta = target[:2] - pose[:2]
        lateral = -math.sin(pose[2]) * delta[0] + math.cos(pose[2]) * delta[1]
        distance2 = float(delta @ delta)
        curvature = 2.0 * lateral / distance2 if distance2 > 0.001 else 0.0
        v, w = optimal_velocity_in_window((upper[0], lower[0], upper[2], lower[2]), curvature)
        desired = np.array([config.vx_max, 0.0, curvature * config.vx_max])
        command = np.array([v, 0.0, w])
        mode = "intersection" if abs(w - curvature * v) <= 1e-10 else "projection"
    elif method == "vp":
        command = np.clip(desired, lower, upper)
        mode = "clipping"
    elif method in ("vp_scaled", "vp_scaled_accel"):
        # First use only the regulated velocity box, without acceleration limits.
        velocity_lower, velocity_upper = regulated_box(config.lower, config.upper, cap, config)
        target = uniformly_scale(desired, velocity_lower, velocity_upper)
        if method == "vp_scaled_accel":
            step = config.acceleration * config.dt
            target = current + uniformly_scale(target - current, -step, step)
        # A sudden cap can be unreachable in one cycle; share the same reachable
        # regulated window as VP/DWVP, including its retained braking endpoints.
        command = np.clip(target, lower, upper)
        mode = "scaled_acceleration" if method == "vp_scaled_accel" else "scaled_velocity"
    else:
        command, mode = ray_command(desired, lower, upper, True)
    return Command(desired, command, lower, upper, cap, lookahead, nearest, mode, obstacle_distance)


def terminal_heading(path):
    """PP uses the final nonzero positional tangent, never the supplied yaw."""
    for delta in np.diff(path[:, :2], axis=0)[::-1]:
        if float(delta @ delta) > 1e-20:
            return math.atan2(delta[1], delta[0])
    return 0.0
