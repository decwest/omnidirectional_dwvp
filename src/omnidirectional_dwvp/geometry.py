"""Constant-twist planar kinematics and explicitly defined circular obstacles."""
from dataclasses import dataclass
import math
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def integrate(pose, velocity, dt):
    """Exact SE(2) step for constant BODY-frame vx, vy and yaw rate."""
    vx, vy, w = velocity
    theta = pose[2]
    if abs(w) < 1e-10:
        local = velocity[:2] * dt
    else:
        a = w * dt
        local = np.array([vx * np.sin(a) - vy * (1 - np.cos(a)),
                          vx * (1 - np.cos(a)) + vy * np.sin(a)]) / w
    c, s = np.cos(theta), np.sin(theta)
    return pose + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1], w * dt])


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


def surface_distance(xy, obstacles):
    if not obstacles:
        return float("inf")
    return min(float(np.hypot(xy[0] - o.x, xy[1] - o.y)) - o.radius for o in obstacles)


def swept_clearance(start, end, velocity, dt, obstacles, robot_radius):
    """Conservative swept-circle bound: chord clearance minus arc sagitta."""
    if not obstacles:
        return float("inf")
    delta = end[:2] - start[:2]
    denom = float(delta @ delta)
    w = abs(float(velocity[2]))
    sagitta = 0.0 if w < 1e-10 else float(np.linalg.norm(velocity[:2])) / w * (1 - math.cos(w * dt / 2))
    values = []
    for o in obstacles:
        center = np.array([o.x, o.y])
        t = 0.0 if denom < 1e-20 else float(np.clip((center - start[:2]) @ delta / denom, 0, 1))
        values.append(float(np.linalg.norm(start[:2] + t * delta - center)) - o.radius - robot_radius - sagitta)
    return min(values)
