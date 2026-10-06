"""Travel-time recovery summaries for Test 4(c), without changing trial metrics."""
from .access_metrics import aggregate


def noise_lookahead_estimate(config, noise_xy):
    """Transverse velocity jitter V sigma / L versus one-cycle allowance ay dt."""
    return config.translation_speed * noise_xy / (config.ay * config.dt)


def preview_noise_summary(rows):
    return aggregate([r for r in rows if r['test'] == 'preview-noise'],
                     ('part', 'scenario', 'method', 'parameter', 'value', 'noise_xy_m', 'noise_yaw_deg'))


def travel_time_recovery(summary, config):
    """Smallest sampled fixed L within 1.1 times either zero-noise reference.

    Require every seed to succeed and have finite travel time in both groups;
    a fast successful subset of a failing condition is not recovery.
    """
    nominal_parameter = 'lookahead_time' if config.fixed_lookahead is None else 'fixed_lookahead'
    nominal_value = config.lookahead_time if config.fixed_lookahead is None else config.fixed_lookahead
    nominal = {r['scenario']: r for r in summary if r['noise_xy_m'] == r['noise_yaw_deg'] == 0
               and (r['parameter'], r['value']) == (nominal_parameter, nominal_value)}
    fixed = [r for r in summary if r['parameter'] == 'fixed_lookahead']
    baseline = {(r['scenario'], r['value']): r for r in fixed
                if r['noise_xy_m'] == 0 and r['noise_yaw_deg'] == 0}
    results = []
    for reference_kind, scene, xy, yaw in (
            (kind, *identity) for kind in ('nominal_preview', 'same_fixed_lookahead')
            for identity in dict.fromkeys((r['scenario'], r['noise_xy_m'], r['noise_yaw_deg']) for r in fixed)):
        recovered = None
        reference = None
        for row in sorted((r for r in fixed if (r['scenario'], r['noise_xy_m'], r['noise_yaw_deg']) == (scene, xy, yaw)),
                          key=lambda r: r['value']):
            zero = nominal.get(scene) if reference_kind == 'nominal_preview' else baseline.get((scene, row['value']))
            if zero is None or not all(r['n'] == r['success_count'] == r['travel_time_s_n'] for r in (row, zero)):
                continue
            if row['travel_time_s_mean'] <= 1.1 * zero['travel_time_s_mean'] + 1e-10:
                recovered, reference = row, zero
                break
        results.append(dict(reference=reference_kind, scenario=scene, noise_xy_m=xy, noise_yaw_deg=yaw,
                            estimate_lookahead_m=noise_lookahead_estimate(config, xy),
                            minimum_fixed_lookahead_m=recovered['value'] if recovered else None,
                            travel_time_s_mean=recovered['travel_time_s_mean'] if recovered else None,
                            zero_noise_travel_time_s_mean=reference['travel_time_s_mean'] if reference else None,
                            success_count=recovered['success_count'] if recovered else None,
                            n=recovered['n'] if recovered else None))
    return results
