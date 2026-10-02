"""Seeded trials with separate demands, controller commands and applied velocities."""
from dataclasses import dataclass
from time import perf_counter_ns
import math
import numpy as np
from .config import Config
from .metrics import project_reference
from .controller import compute_command, dynamic_box
from .geometry import integrate, wrap, surface_distance, swept_clearance


@dataclass
class Result:
    arrays: dict
    metrics: dict
    performance: dict


def _direction_angles(desired, commands, scale):
    a, b = desired / scale, commands / scale
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-12
    angles = np.full(len(desired), np.nan)
    angles[valid] = np.rad2deg(np.arccos(np.clip(np.sum(a[valid] * b[valid], axis=1) / denom[valid], -1, 1)))
    return angles


def simulate(path, method="dwvp", config=Config(), obstacles=(), seed=0):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("path must contain at least two finite [x,y,yaw] poses")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    pose = np.array([0.0, 0.0, 0.0])
    current = np.zeros(3)
    rng = np.random.default_rng(seed)
    poses, applied, demands, commands, boxes, caps, looks, durations, modes, clearances = [pose.copy()], [], [], [], [], [], [], [], [], []
    reached = False
    first_reach = None
    physical_violation, acceleration_violation, demand_violation = [], [], []
    mode_code = {"clipping": 0, "intersection": 1, "projection": 2, "terminal": 3, "zero": 4}
    for step in range(math.ceil(config.timeout / config.dt)):
        within_goal = (np.linalg.norm(pose[:2] - path[-1, :2]) <= config.goal_xy and
                       abs(wrap(pose[2] - path[-1, 2])) <= config.goal_yaw)
        if within_goal and first_reach is None:
            first_reach = step * config.dt
        if within_goal and np.max(np.abs(current)) <= 1e-3:
            reached = True
            break
        observed = pose + rng.normal(size=3) * np.array([config.noise_xy, config.noise_xy, config.noise_yaw])
        started = perf_counter_ns()
        output = compute_command(observed, current, path, arc, method, config, obstacles)
        durations.append((perf_counter_ns() - started) / 1000)
        lo, hi = dynamic_box(current, config)
        command = output.command
        physical_violation.append(bool(np.any(command < config.lower - 1e-10) or np.any(command > config.upper + 1e-10)))
        acceleration_violation.append(bool(np.any(command < lo - 1e-10) or np.any(command > hi + 1e-10)))
        demand_violation.append(bool(np.any(output.desired < lo - 1e-10) or np.any(output.desired > hi + 1e-10)))
        accepted = np.clip(command, lo, hi)
        next_pose = integrate(pose, accepted, config.dt)
        if not np.all(np.isfinite(np.r_[next_pose, accepted, command, output.desired])):
            raise FloatingPointError("non-finite command or state")
        clearances.append(swept_clearance(pose, next_pose, accepted, config.dt, obstacles, config.robot_radius))
        poses.append(next_pose)
        applied.append(accepted)
        demands.append(output.desired)
        commands.append(command)
        boxes.append(np.stack((output.lower, output.upper)))
        caps.append(output.speed_cap)
        looks.append(output.lookahead)
        modes.append(mode_code[output.mode])
        current, pose = accepted, next_pose
    positions = np.asarray(poses)
    applied = np.asarray(applied).reshape((-1, 3))
    commands = np.asarray(commands).reshape((-1, 3))
    demands = np.asarray(demands).reshape((-1, 3))
    reference, errors, yaw_errors, reference_segment, reference_fraction = project_reference(positions, path)
    acceleration = np.diff(np.vstack((np.zeros(3), applied)), axis=0) / config.dt
    jerk = np.diff(np.vstack((np.zeros(3), acceleration)), axis=0) / config.dt
    angle = _direction_angles(demands, applied, config.axis_scale)
    arrays = dict(path=path, poses=positions, commands=commands, applied=applied, measured=applied.copy(),
                  demands=demands, boxes=np.asarray(boxes), speed_caps=np.asarray(caps), lookahead=np.asarray(looks),
                  modes=np.asarray(modes), times=np.arange(len(positions))*config.dt,
                  clearance=np.asarray(clearances), direction_angle_deg=angle,
                  position_errors=errors, yaw_errors=yaw_errors, reference_poses=reference,
                  reference_segment=reference_segment, reference_fraction=reference_fraction)
    finite_angle = angle[np.isfinite(angle)]
    metrics = dict(success=reached, timeout=not reached, steps=len(applied), duration_s=len(applied)*config.dt,
                   first_goal_time_s=first_reach, travel_time_s=len(applied)*config.dt if reached else None,
                   mean_position_error_m=float(np.mean(errors)), max_position_error_m=float(np.max(errors)),
                   mean_heading_error_deg=float(np.rad2deg(np.mean(yaw_errors))),
                   max_heading_error_deg=float(np.rad2deg(np.max(yaw_errors))),
                   final_position_error_m=float(np.linalg.norm(positions[-1, :2]-path[-1, :2])),
                   final_heading_error_deg=float(abs(np.rad2deg(wrap(positions[-1, 2]-path[-1, 2])))),
                   velocity_violation_pct=100*float(np.mean(physical_violation)) if applied.size else 0.,
                   acceleration_violation_pct=100*float(np.mean(acceleration_violation)) if applied.size else 0.,
                   unconstrained_demand_violation_pct=100*float(np.mean(demand_violation)) if applied.size else 0.,
                   mean_direction_distortion_deg=float(np.mean(finite_angle)) if len(finite_angle) else 0.,
                   max_direction_distortion_deg=float(np.max(finite_angle)) if len(finite_angle) else 0.,
                   jerk_xy_rms_m_s3=float(np.sqrt(np.mean(np.sum(jerk[:, :2]**2, axis=1)))) if jerk.size else 0.,
                   jerk_yaw_rms_rad_s3=float(np.sqrt(np.mean(jerk[:, 2]**2))) if jerk.size else 0.,
                   min_clearance_m=float(min(clearances)) if obstacles and clearances else None,
                   collision=bool(obstacles and clearances and min(clearances) <= 0),
                   max_translation_speed_m_s=float(np.max(np.linalg.norm(applied[:, :2], axis=1))) if applied.size else 0.,
                   max_abs_yaw_rate_rad_s=float(np.max(np.abs(applied[:, 2]))) if applied.size else 0.,
                   projection_steps=int(np.sum(np.asarray(modes)==2)),
                   mean_near_obstacle_speed_m_s=None)
    if obstacles and len(applied):
        near=np.array([surface_distance(p[:2], obstacles) < config.cost_scaling_dist for p in positions[:-1]])
        if near.any(): metrics["mean_near_obstacle_speed_m_s"] = float(np.mean(np.linalg.norm(applied[near,:2],axis=1)))
    velocity_excess = np.maximum(np.maximum(config.lower - commands, commands - config.upper), 0.)
    accel_excess = np.maximum(np.abs(acceleration) - config.acceleration, 0.)
    cap_excess = np.maximum(np.linalg.norm(applied[:, :2], axis=1) - np.asarray(caps), 0.)
    metrics.update(max_normalized_velocity_excess=float(np.max(velocity_excess/config.axis_scale)) if velocity_excess.size else 0.,
                   max_normalized_acceleration_excess=float(np.max(accel_excess/np.maximum(config.acceleration,1e-12))) if accel_excess.size else 0.,
                   constraint_violation_duration_s=float(np.sum(np.logical_or(physical_violation, acceleration_violation)))*config.dt,
                   max_speed_cap_excess_m_s=float(np.max(cap_excess)) if len(cap_excess) else 0.,
                   speed_cap_excess_duration_s=float(np.sum(cap_excess>1e-10))*config.dt)
    performance = {"controller_mean_us": float(np.mean(durations)) if durations else 0.,
                   "controller_p95_us": float(np.percentile(durations,95)) if durations else 0.,
                   "controller_max_us": float(max(durations)) if durations else 0.}
    return Result(arrays, metrics, performance)
