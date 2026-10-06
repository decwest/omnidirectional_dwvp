"""Evaluation windows and aggregates for the straight-path studies."""
import math
import numpy as np
from .geometry import wrap
from .solver import calc_ray_box_intersection_alpha_range

TOLERANCE = 1e-10
COMMON_METRICS = ('eval_max_position_error_m', 'eval_mean_position_error_m',
                  'eval_position_error_integral_m_s', 'eval_max_heading_error_deg',
                  'eval_mean_heading_error_deg', 'eval_heading_error_integral_deg_s',
                  'command_constraint_violation_pct', 'travel_time_s')


def constraint_violation_counts(velocities, previous, config):
    """Count physical velocity and one-step acceleration excess separately.

    Use the demand metric's velocity tolerance, including for the acceleration
    step. The union counts a cycle once even if both constraints are exceeded.
    """
    velocity = np.any((velocities < config.lower - TOLERANCE) |
                      (velocities > config.upper + TOLERANCE), axis=1)
    acceleration = np.any(np.abs(velocities - previous) >
                          config.acceleration * config.dt + TOLERANCE, axis=1)
    return dict(velocity_violation_steps=int(velocity.sum()),
                acceleration_violation_steps=int(acceleration.sum()),
                both_violation_steps=int((velocity & acceleration).sum()),
                violation_steps=int((velocity | acceleration).sum()))


def time_error_metrics(times, errors, mask):
    """Trapezoids on adjacent in-window samples; never bridge excluded intervals.

    Keep the existing spatial window and sample endpoints (no interpolation at
    its boundary). A singleton has zero integral and an undefined time mean.
    """
    intervals = mask[:-1] & mask[1:]
    dt = np.diff(times)[intervals]
    duration = float(dt.sum())
    integral = float((.5 * (np.abs(errors[:-1]) + np.abs(errors[1:])))[intervals] @ dt)
    return duration, integral, integral / duration if duration > 0 else None


def heading_braking_prediction(a, config, evaluation_end):
    """Freeze omega and T at the first braking command with the final target.

    Omega is the applied velocity immediately BEFORE that command, not the
    velocity when yaw crosses the final reference. The latter is already reduced
    by braking and cannot test the proposed stopping-angle argument.
    """
    result = dict(heading_braking_time_s=None, heading_braking_omega_rad_s=None,
                  heading_braking_T_s=None, heading_braking_remaining_deg=None,
                  predicted_heading_overshoot_deg=None)
    p, path, u = a['poses'], a['path'], a['applied']
    previous = np.r_[0., u[:-1, 2]]
    remaining = wrap(path[-1, 2] - p[:-1, 2])
    candidates = np.flatnonzero((previous > TOLERANCE) & (u[:, 2] < previous - TOLERANCE)
                               & (remaining >= 0.) & (p[:-1, 0] >= 0.)
                               & (p[:-1, 0] <= evaluation_end))
    for i in candidates:
        # Reconstruct only the preview target index, using the unchanged
        # Euclidean lookahead rule, to exclude braking within the ramp itself.
        nearest = int(np.argmin(np.sum((path[:, :2] - p[i, :2])**2, axis=1)))
        outer = np.flatnonzero(np.linalg.norm(path[nearest:, :2] - p[i, :2], axis=1) >= a['lookahead'][i])
        target = nearest + int(outer[0]) if len(outer) else len(path) - 1
        if abs(wrap(path[target, 2] - path[-1, 2])) > TOLERANCE:
            continue
        omega = float(previous[i])
        time = max(config.min_orientation_time,
                   config.orientation_time_weight * float(a['lookahead'][i]) / config.translation_speed)
        predicted = max(0., omega**2 / (2 * config.aw) - omega * time)
        result.update(heading_braking_time_s=float(a['times'][i]), heading_braking_omega_rad_s=omega,
                      heading_braking_T_s=time, heading_braking_remaining_deg=float(np.rad2deg(remaining[i])),
                      predicted_heading_overshoot_deg=float(np.rad2deg(predicted)))
        break
    return result


def evaluate(result, config, scenario, initial_pose, evaluation_end, ramp_start):
    a = result.arrays
    p, u = a['poses'], a['applied']
    distance = np.r_[0., np.cumsum(np.linalg.norm(u[:, :2], axis=1) * config.dt)]
    a['travel_distance'] = distance
    mask = (p[:, 0] >= -1e-10) & (p[:, 0] <= evaluation_end)
    m = dict(result.metrics)
    m['evaluation_end_m'] = evaluation_end
    m['evaluation_complete'] = bool(np.any(p[:, 0] >= evaluation_end))
    m['evaluation_samples'] = int(mask.sum())
    m['eval_max_heading_error_deg'] = float(np.rad2deg(a['yaw_errors'][mask].max())) if mask.any() else None
    m['eval_sample_mean_heading_error_deg'] = float(np.rad2deg(a['yaw_errors'][mask].mean())) if mask.any() else None
    signed_yaw = wrap(p[:, 2] - a['reference_poses'][:, 2])
    a['signed_yaw_errors'] = signed_yaw
    m['eval_max_heading_lead_deg'] = max(0., float(np.rad2deg(signed_yaw[mask].max()))) if mask.any() else None
    m['eval_max_heading_lag_deg'] = max(0., float(np.rad2deg(-signed_yaw[mask].min()))) if mask.any() else None
    m['eval_max_position_error_m'] = float(a['position_errors'][mask].max()) if mask.any() else None
    m['eval_sample_mean_position_error_m'] = float(a['position_errors'][mask].mean()) if mask.any() else None
    duration, integral, mean = time_error_metrics(a['times'], a['position_errors'], mask)
    m.update(eval_duration_s=duration, eval_position_error_integral_m_s=integral,
             eval_mean_position_error_m=mean)
    _, integral, mean = time_error_metrics(a['times'], np.rad2deg(a['yaw_errors']), mask)
    m.update(eval_heading_error_integral_deg_s=integral, eval_mean_heading_error_deg=mean)
    m['command_constraint_violation_pct'] = (100 * m['constraint_violation_duration_s'] / m['duration_s']
                                              if m['duration_s'] > 0 else 0.)
    m['goal_overshoot_m'] = max(0., float(p[:, 0].max() - a['path'][-1, 0]))
    m['eval_max_heading_change_deg'] = float(np.rad2deg(np.abs(wrap(p[mask, 2] - initial_pose[2])).max())) if mask.any() else None
    angles = a['direction_angle_deg'][mask[:-1]]
    angles = angles[np.isfinite(angles)]
    m['eval_mean_direction_distortion_deg'] = float(angles.mean()) if len(angles) else None
    m['eval_max_direction_distortion_deg'] = float(angles.max()) if len(angles) else None
    m['eval_projection_steps'] = int(np.sum(a['modes'][mask[:-1]] == 2))
    # Clipped VP never calls the ray solver, but its ray can also miss its box.
    # Evaluate this geometry for BOTH methods; keep solver-mode counts separate.
    intersects = np.array([calc_ray_box_intersection_alpha_range(d, box[0], box[1])[0]
                           for d, box in zip(a['demands'], a['boxes'])], dtype=bool)
    tracking = (a['modes'] != 3) & (a['modes'] != 4)
    a['ray_intersects_box'] = intersects
    m['nonintersection_steps'] = int(np.sum(~intersects & tracking))
    m['eval_nonintersection_steps'] = int(np.sum(~intersects & tracking & mask[:-1]))
    e0 = abs(initial_pose[1])
    m.update(crossing_m=None, crossed=None, first_2pct_time_s=None, first_2pct_distance_m=None,
             settling_2pct_time_s=None, settling_2pct_distance_m=None)
    if e0 and mask.any():
        indices = np.flatnonzero(mask)
        e = p[indices, 1]
        crossing = max(0., float(np.max(-np.sign(initial_pose[1]) * e)))
        inside = np.abs(e) <= .02 * e0
        m.update(crossing_m=crossing, crossed=crossing > 1e-6)
        if inside.any():
            first = indices[np.flatnonzero(inside)[0]]
            m.update(first_2pct_time_s=float(a['times'][first]), first_2pct_distance_m=float(distance[first]))
        if inside[-1] and m['evaluation_complete']:
            outside = np.flatnonzero(~inside)
            settled = indices[outside[-1] + 1] if len(outside) else indices[0]
            m.update(settling_2pct_time_s=float(a['times'][settled]), settling_2pct_distance_m=float(distance[settled]))
    ell = scenario.get('transition_length')
    m['post_transition_heading_overshoot_deg'] = None
    m['transition_heading_lag_deg'] = None
    m.update(min_transition_speed_m_s=None, predicted_speed_m_s=None, min_minus_predicted_speed_m_s=None,
             transition_max_yaw_rate_rad_s=None, transition_samples=0)
    if ell is not None:
        changing = mask & (p[:, 0] >= ramp_start) & (p[:, 0] <= ramp_start + ell)
        # A step has no finite changing interval: use its first outgoing sample.
        if ell == 0:
            indices = np.flatnonzero(mask & (p[:, 0] > ramp_start))
            changing[:] = False
            if len(indices):
                changing[indices[0]] = True
        if changing.any():
            m['transition_heading_lag_deg'] = max(0., float(np.rad2deg(-signed_yaw[changing].min())))
        # Same pre-goal evaluation window as the other tracking errors. All
        # configured ramps turn positively through pi/2, so positive is overshoot.
        after = mask & (p[:, 0] > ramp_start + ell)
        if after.any():
            m['post_transition_heading_overshoot_deg'] = max(0., float(np.rad2deg(
                wrap(p[after, 2] - a['path'][-1, 2])).max()))
        transition = (p[:-1, 0] >= ramp_start - a['lookahead']) & (p[:-1, 0] <= ramp_start + ell)
        m['transition_samples'] = int(transition.sum())
        predicted = config.w_max * ell / (math.pi / 2)
        m['predicted_speed_m_s'] = predicted
        if transition.any():
            minimum = float(np.linalg.norm(u[transition, :2], axis=1).min())
            m.update(min_transition_speed_m_s=minimum, min_minus_predicted_speed_m_s=minimum-predicted,
                     transition_max_yaw_rate_rad_s=float(np.abs(u[transition, 2]).max()))
    commands = a['commands']
    previous = np.vstack((np.zeros(3), u[:-1])) if len(u) else np.empty((0, 3))
    m['control_steps'] = len(u)
    for label, velocities in (('demand', a['demands']), ('command', commands)):
        m.update({f'{label}_{key}': value for key, value in
                  constraint_violation_counts(velocities, previous, config).items()})
    m['lookahead_min_m'] = float(a['lookahead'].min()) if len(u) else None
    m['lookahead_max_m'] = float(a['lookahead'].max()) if len(u) else None
    raw_lookahead = config.lookahead_time * np.linalg.norm(previous[:, :2], axis=1)
    m['lookahead_upper_active_steps'] = (int(np.sum(raw_lookahead > config.lookahead_max + TOLERANCE))
                                        if config.fixed_lookahead is None else 0)
    m['lookahead_at_upper_steps'] = (int(np.sum(a['lookahead'] >= config.lookahead_max - TOLERANCE))
                                    if config.fixed_lookahead is None else 0)
    acceleration = (commands - previous) / config.dt
    velocity_excess = np.maximum(np.maximum(config.lower - commands, commands - config.upper), 0.)
    acceleration_excess = np.maximum(np.abs(acceleration) - config.acceleration, 0.)
    box_excess = np.maximum(np.maximum(a['boxes'][:, 0] - commands, commands - a['boxes'][:, 1]), 0.)
    m['max_dynamic_window_excess'] = float(box_excess.max()) if len(u) else 0.
    m['dynamic_window_violation_duration_s'] = float(np.sum(np.any(box_excess > TOLERANCE, axis=1)) * config.dt)
    for i, axis in enumerate(('vx', 'vy', 'w')):
        m[f'max_{axis}_excess'] = float(velocity_excess[:, i].max()) if len(u) else 0.
        m[f'{axis}_violation_duration_s'] = float(np.sum(velocity_excess[:, i] > TOLERANCE) * config.dt)
        m[f'max_a{axis}_excess'] = float(acceleration_excess[:, i].max()) if len(u) else 0.
        m[f'a{axis}_violation_duration_s'] = float(np.sum(acceleration_excess[:, i] > TOLERANCE) * config.dt)
    m['velocity_violation_duration_s'] = float(np.sum(np.any(velocity_excess > TOLERANCE, axis=1)) * config.dt)
    m['acceleration_violation_duration_s'] = float(np.sum(np.any(acceleration_excess > TOLERANCE, axis=1)) * config.dt)
    m['success'] = bool(m['success'] and not m['collision'])
    m['status'] = 'collision' if m['collision'] else 'success' if m['success'] else 'timeout'
    return m


AGGREGATE_METRICS = COMMON_METRICS + ('crossing_m', 'settling_2pct_time_s', 'settling_2pct_distance_m',
                     'eval_duration_s', 'transition_heading_lag_deg',
                     'eval_max_heading_lead_deg', 'eval_max_heading_lag_deg', 'post_transition_heading_overshoot_deg',
                     'duration_s', 'min_transition_speed_m_s',
                     'eval_mean_direction_distortion_deg', 'eval_projection_steps', 'eval_nonintersection_steps',
                     'jerk_xy_rms_m_s3', 'jerk_yaw_rms_rad_s3')


def aggregate(rows, keys):
    """All observed runs remain in statistics; each metric records its finite n.

    Missing metrics are counted, never imputed as zero. SD is the sample SD;
    repeated zero-noise seeds are deterministic copies, not independent data.
    """
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    output = []
    for identity, group in groups.items():
        item = dict(zip(keys, identity))
        item.update(n=len(group), success_count=sum(r['success'] for r in group),
                    timeout_count=sum(r['status'] == 'timeout' for r in group),
                    failure_count=sum(r['status'] not in ('success', 'timeout') for r in group),
                    evaluation_complete_count=sum(r.get('evaluation_complete', False) for r in group))
        for metric in AGGREGATE_METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None and math.isfinite(r[metric])]
            item[metric + '_n'] = len(values)
            item[metric + '_mean'] = float(np.mean(values)) if values else None
            item[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output
