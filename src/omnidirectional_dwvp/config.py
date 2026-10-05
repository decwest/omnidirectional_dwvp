"""Explicit experiment settings; adapted from dwpp_python's frozen config pattern."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Config:
    frequency: float = 30.0
    timeout: float = 120.0
    vx_min: float = -0.22
    vx_max: float = 0.22
    vy_min: float = -0.22
    vy_max: float = 0.22
    w_min: float = -0.60
    w_max: float = 0.60
    ax: float = 0.22
    ay: float = 0.22
    aw: float = 0.60
    desired_linear_vel: float = 0.32
    vp_translation_speed: float | None = None
    lookahead_min: float = 0.11
    lookahead_max: float = 0.33
    lookahead_time: float = 1.5
    fixed_lookahead: float | None = None
    orientation_time_weight: float = 1.0
    min_orientation_time: float = 0.20
    goal_xy: float = 0.02
    goal_yaw: float = math.pi / 180.0
    use_cost_regulation: bool = False
    cost_scaling_dist: float = 0.60
    cost_scaling_gain: float = 1.0
    regulated_min_speed: float = 0.05
    approach_distance: float = 0.60
    min_approach_speed: float = 0.05
    robot_radius: float = 0.22
    inflation_radius: float = 0.70
    inflation_factor: float = 3.0
    noise_xy: float = 0.0
    noise_yaw: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.frequency, self.timeout, self.min_orientation_time) <= 0:
            raise ValueError("frequency, timeout and minimum orientation time must be positive")
        if min(self.ax, self.ay, self.aw, self.desired_linear_vel, self.approach_distance,
               self.noise_xy, self.noise_yaw, self.robot_radius) < 0:
            raise ValueError("rates, nominal speed, distances and noise must be nonnegative")
        if self.lookahead_min <= 0 or self.lookahead_max < self.lookahead_min:
            raise ValueError("lookahead bounds must be ordered and positive")
        if self.fixed_lookahead is not None and self.fixed_lookahead <= 0:
            raise ValueError("fixed lookahead must be positive")
        if self.vp_translation_speed is not None and self.vp_translation_speed < 0:
            raise ValueError("VP translation speed must be nonnegative")
        if self.cost_scaling_dist <= 0 or self.inflation_factor <= 0:
            raise ValueError("cost distance and inflation factor must be positive")
        if np.any(self.lower > 0) or np.any(self.upper < 0):
            raise ValueError("physical intervals must contain zero")

    @property
    def dt(self): return 1.0 / self.frequency
    @property
    def lower(self): return np.array([self.vx_min, self.vy_min, self.w_min])
    @property
    def upper(self): return np.array([self.vx_max, self.vy_max, self.w_max])
    @property
    def acceleration(self): return np.array([self.ax, self.ay, self.aw])
    @property
    def axis_scale(self): return np.maximum(np.maximum(np.abs(self.lower), np.abs(self.upper)), 1e-12)
    @property
    def box_speed(self): return float(np.hypot(max(abs(self.vx_min), abs(self.vx_max)), max(abs(self.vy_min), abs(self.vy_max))))
    @property
    def nominal_speed(self): return min(self.box_speed, self.desired_linear_vel)

    @property
    def translation_speed(self):
        """Demand magnitude; default is feasible in every planar direction."""
        if self.vp_translation_speed is not None:
            return self.vp_translation_speed
        return min(-self.vx_min, self.vx_max, -self.vy_min, self.vy_max)
