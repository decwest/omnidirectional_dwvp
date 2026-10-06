"""Four reproducible straight-path studies and an optional preview/noise grid, with explicit outcomes for every condition."""
from dataclasses import asdict, replace
from time import perf_counter
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
import numpy as np
from .config import Config
from .geometry import Obstacle
from .paths import straight_path, orientation_ramp_path
from .simulation import Result, simulate
from .access_metrics import evaluate, aggregate
from .studies import ROOT, code_hash, sha, git_revision, cpu_model, write_csv


def atomic_json(path, value):
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numerical_hash():
    names = ('config.py', 'controller.py', 'dwpp.py', 'geometry.py', 'solver.py',
             'paths.py', 'simulation.py', 'metrics.py', 'access_metrics.py')
    return sha({n: hashlib.sha256((ROOT / 'src/omnidirectional_dwvp' / n).read_bytes()).hexdigest() for n in names})


def make_condition(test, part, scenario, method, config, seed, settings, parameter='', value=None):
    return dict(test=test, part=part, scenario=scenario, method=method, config=asdict(config), seed=seed,
                parameter=parameter, value=value,
                evaluation_end=min(settings['evaluation_end'], settings['path_length'] - config.approach_distance - settings['path_spacing']) if config.approach_distance > 0 else settings['evaluation_end'],
                path_length=settings['path_length'], path_spacing=settings['path_spacing'],
                ramp_start=settings['ramp_start'], obstacles=settings['obstacles'] if scenario['kind']=='obstacles' else [])


def plan_conditions(config, settings, seed, selected):
    conditions = []
    def add(test, part, scenario, methods, cfg=config, parameter='', value=None, seeds=None):
        if test not in selected:
            return
        for method in methods:
            for trial_seed in ([seed] if seeds is None else seeds):
                conditions.append(make_condition(test, part, scenario, method, cfg, trial_seed, settings, parameter, value))
    representatives = [dict(name='offset', kind='offset', offset=settings['representative_offset']),
                       dict(name='gradual', kind='ramp', transition_length=settings['representative_gradual']),
                       dict(name='rapid', kind='ramp', transition_length=settings['representative_rapid'])]
    both = ('vp', 'vp_scaled', 'dwvp')
    heading_methods = ('vp', 'vp_scaled', 'vp_scaled_accel', 'dwvp')
    for e in settings['offsets']:
        add('test1', 'nominal', dict(name=f'offset_{e:g}', kind='offset', offset=e), ('dwpp', *both))
    for ell in settings['transition_lengths']:
        add('test2', 'nominal', dict(name=f'ramp_{ell:g}', kind='ramp', transition_length=ell), heading_methods)
    if settings['include_step']:
        add('test2', 'step', dict(name='ramp_0', kind='ramp', transition_length=0.), heading_methods)
    for parameter, values, lengths in (
            ('acceleration_scale', settings['acceleration_scales'], settings['acceleration_transition_lengths']),
            ('angular_acceleration_scale', settings['angular_acceleration_scales'], [settings['representative_rapid']])):
        for ell in lengths:
            for value in values:
                cfg = replace(config, aw=config.aw*value,
                              ax=config.ax*value if parameter=='acceleration_scale' else config.ax,
                              ay=config.ay*value if parameter=='acceleration_scale' else config.ay)
                add('test2', 'acceleration', dict(name=f'ramp_{ell:g}', kind='ramp', transition_length=ell),
                    heading_methods, cfg, parameter, value)
    obstacle = dict(name='obstacles', kind='obstacles')
    if config.approach_distance <= 0 or any(x <= 0 for x in settings['approach_distances']):
        raise ValueError('Straight-path studies require goal approach regulation')
    for cost in (False, True):
        add('test3', 'nominal', obstacle, ('rpp', 'dwvp'), replace(config, use_cost_regulation=cost),
            'regulation', f'cost={int(cost)},approach=1')
    lookaheads = [('fixed_lookahead', x) for x in settings['fixed_lookaheads']]
    lookaheads += [('lookahead_time', x) for x in settings['lookahead_times']]
    for parameter, value in lookaheads:
        for scene in representatives:
            add('test4', 'b', scene, ('dwpp', *both) if scene['kind']=='offset' else both,
                replace(config, **{parameter: value}), parameter, value)
    for parameter, values in (('acceleration_scale', settings['acceleration_scales']),
                              ('angular_acceleration_scale', settings['angular_acceleration_scales'])):
        for value in values:
            cfg = replace(config, aw=config.aw*value,
                          ax=config.ax*value if parameter=='acceleration_scale' else config.ax,
                          ay=config.ay*value if parameter=='acceleration_scale' else config.ay)
            for scene in representatives:
                add('acceleration-sweep', 'acceleration', scene, ('dwpp', *both) if scene['kind']=='offset' else both, cfg, parameter, value)
    for parameter, key in (('cost_scaling_dist', 'cost_distances'), ('cost_scaling_gain', 'cost_gains'), ('approach_distance', 'approach_distances')):
        for value in settings[key]:
            add('regulation-sweep', 'd', obstacle, both, replace(config, use_cost_regulation=True, **{parameter: value}), parameter, value)
    seeds = range(seed, seed + settings['noise_seeds'])
    for xy, yaw_deg in settings['noise_levels']:
        cfg = replace(config, noise_xy=xy, noise_yaw=float(np.deg2rad(yaw_deg)))
        for scene in representatives:
            add('test4', 'c', scene, ('dwpp', *both) if scene['kind']=='offset' else both, cfg, 'noise', xy, seeds)
        for parameter, value in lookaheads:
            for scene in representatives:
                add('preview-noise', 'g', scene, ('dwvp',), replace(cfg, **{parameter: value}), parameter, value, seeds)
    return conditions, representatives


def reuse_key(condition):
    """Match a saved round-2 condition while allowing only the preview cap to change."""
    return sha({**condition, 'config': {k: v for k, v in condition['config'].items()
                                      if k != 'lookahead_max'}})


def reusable_lookahead(history, old_config, new_config):
    """A tighter inactive cap preserves the full deterministic/noisy state history."""
    before, after = asdict(old_config), asdict(new_config)
    old_cap, new_cap = before.pop('lookahead_max'), after.pop('lookahead_max')
    return (before == after and new_cap <= old_cap and
            (new_config.fixed_lookahead is not None or
             bool(np.all(history['lookahead'] <= new_cap))))


def execute_trial(condition, output, numerical_source_hash, force=False, reuse=None):
    cfg = Config(**condition['config'])
    scene = condition['scenario']
    path = (orientation_ramp_path(scene['transition_length'], condition['path_length'], condition['path_spacing'], condition['ramp_start'])
            if scene['kind']=='ramp' else straight_path(condition['path_length'], condition['path_spacing']))
    initial = [0., scene.get('offset', 0.), 0.]
    obstacles = tuple(Obstacle(*o) for o in condition['obstacles'])
    spec = dict(method=condition['method'], config=asdict(cfg), seed=condition['seed'], initial_pose=initial,
                path_sha256=hashlib.sha256(path.tobytes()).hexdigest(), obstacles=condition['obstacles'],
                numerical_source_sha256=numerical_source_hash, scenario=scene,
                evaluation_end=condition['evaluation_end'], ramp_start=condition['ramp_start'],
                numpy_version=np.__version__, python_version=sys.version.split()[0])
    identity = sha(spec)
    directory = output / 'trials' / identity
    metadata = directory / 'trial.json'
    cache = False
    if not force and metadata.exists():
        saved = json.loads(metadata.read_text())
        cache = saved['metrics']['status']=='error' or (directory / 'trajectory.npz').exists()
    if not cache:
        directory.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        try:
            reused = None
            if reuse is not None and condition['test'] != 'test3':
                root, original = reuse
                old_id = original['summary']['trial_id']
                old_path = root/'trials'/old_id/'trajectory.npz'
                if old_path.exists():
                    old_metadata = json.loads((old_path.parent/'trial.json').read_text())
                    expected = {**original['spec'], 'config': asdict(cfg),
                                'numerical_source_sha256': numerical_source_hash}
                    assert expected == spec, 'Reuse must preserve every other simulation input'
                    with np.load(old_path) as history:
                        if reusable_lookahead(history, Config(**original['spec']['config']), cfg):
                            result = Result({k: history[k] for k in history}, old_metadata['metrics'], old_metadata['performance'])
                            reused = dict(trial_id=old_id, trajectory_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                                          max_lookahead_m=float(history['lookahead'].max()))
            if reused is None:
                result = simulate(path, condition['method'], cfg, obstacles, condition['seed'], initial_pose=initial)
            metrics = evaluate(result, cfg, scene, initial, condition['evaluation_end'], condition['ramp_start'])
            trajectory_temp = directory / f'trajectory.{os.getpid()}.npz'
            if reused is None:
                np.savez_compressed(trajectory_temp, **result.arrays)
            else:
                shutil.copyfile(old_path, trajectory_temp)
            trajectory_temp.replace(directory / 'trajectory.npz')
            saved = dict(spec=spec, trial_id=identity, metrics=metrics, performance=result.performance)
            if reused is not None:
                saved['trajectory_reuse'] = reused
        except Exception as exc:
            saved = dict(spec=spec, trial_id=identity, metrics=dict(success=False, timeout=False, status='error',
                         collision=False, error=f'{type(exc).__name__}: {exc}'), traceback=traceback.format_exc(), performance={})
        saved['wall_time_s'] = perf_counter() - started
        atomic_json(metadata, saved)
    row = dict(test=condition['test'], part=condition['part'], scenario=scene['name'], method=condition['method'],
               parameter=condition['parameter'], value=condition['value'], seed=condition['seed'],
               noise_xy_m=cfg.noise_xy, noise_yaw_deg=float(np.rad2deg(cfg.noise_yaw)), trial_id=identity,
               offset_m=scene.get('offset'), transition_length_m=scene.get('transition_length'))
    row.update(saved['metrics'])
    for axis, limit, acceleration in zip(('vx', 'vy', 'w'), cfg.axis_scale, cfg.acceleration):
        row[f'acceleration_time_{axis}_s'] = float(limit / acceleration) if acceleration > 0 else None
    row['lookahead_time_s'] = cfg.lookahead_time
    row['fixed_lookahead_m'] = cfg.fixed_lookahead
    return row, spec, cache


def run(study, output, config, settings, seed=0, force=False, config_path=None, workers=1, baseline=None,
        reuse_from=None):
    from concurrent.futures import ProcessPoolExecutor
    from .access_plotting import figures
    from .access_report import report
    start = perf_counter()
    selected = tuple(f'test{i}' for i in range(1, 5)) if study=='all' else (study,)
    output.mkdir(parents=True, exist_ok=True)
    conditions, representatives = plan_conditions(config, settings, seed, selected)
    source = code_hash()
    numerical_source = numerical_hash()
    previous = json.loads(reuse_from.read_text()) if reuse_from is not None else None
    reuse_index = {reuse_key(t['condition']): t for t in previous['trials']} if previous else {}
    def reuse_for(condition):
        trial = reuse_index.get(reuse_key(condition))
        return (output, trial) if trial is not None else None
    manifest = dict(schema_version=4, study=study, suite='access_v2', source_sha256=source, numerical_source_sha256=numerical_source,
                    git_revision=git_revision(), python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    cpu_model=cpu_model(), workers=workers, config=asdict(config), settings=settings, seed=seed,
                    dependency_lock_sha256=hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest(),
                    config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path else None,
                    planned_conditions=list(conditions), trials=[], complete=False,
                    notes=['All observations, including failed/timeout runs, remain in aggregates with finite counts.',
                           'Noise is independent Gaussian pose observation noise at every control cycle; plant state is exact.',
                           'Sample standard deviation (ddof=1); zero-noise seeds repeat deterministic trials.',
                           'Error evaluation ends before each condition-specific goal-slowdown boundary. Full durations include terminal settling.',
                           'PP uses position only, including the final positional tangent for terminal yaw.',
                           'Raw mixed-unit solver objective is unchanged; reported direction angles use per-axis normalization.'])
    target = output/('manifest.json' if study=='all' else f'manifest_{study}.json')
    if previous:
        manifest['trajectory_reuse_baseline_sha256'] = hashlib.sha256(reuse_from.read_bytes()).hexdigest()
    atomic_json(target, manifest)
    rows = []
    last_log = perf_counter()
    def accept(condition, result):
        nonlocal last_log
        row, spec, cached = result
        row['condition_id'] = sha(condition)
        rows.append(row)
        manifest['trials'].append(dict(condition_id=row['condition_id'], condition=condition, spec=spec, summary=row))
        if perf_counter()-last_log >= 20 or len(rows) % 100 == 0:
            print(f"{len(rows)}/{len(manifest['planned_conditions'])} conditions; {sum(r['success'] for r in rows)} success; elapsed {perf_counter()-start:.1f}s", flush=True)
            atomic_json(target, manifest)
            last_log = perf_counter()
        return row
    # Time matching is sequential and every search candidate is retained.
    if 'test4' in selected:
        rapid = representatives[2]
        def matched(method, scale, parameter):
            cfg = replace(config, vx_min=config.vx_min*scale, vx_max=config.vx_max*scale,
                          vy_min=config.vy_min*scale, vy_max=config.vy_max*scale,
                          vp_translation_speed=config.translation_speed*scale, desired_linear_vel=config.desired_linear_vel*scale)
            c = make_condition('test4', 'a', rapid, method, cfg, seed, settings, parameter, scale)
            manifest['planned_conditions'].append(c)
            return accept(c, execute_trial(c, output, numerical_source, force, reuse_for(c)))
        reference = matched('dwvp', 1., 'match_reference')
        manifest['time_matches'] = {}
        for method in ('vp', 'vp_scaled'):
            high = matched(method, 1., 'match_search')
            low = matched(method, settings['match_min_scale'], 'match_search')
            candidates = [high, low]
            lo, hi = settings['match_min_scale'], 1.
            target_time = reference.get('travel_time_s')
            bracket = target_time is not None and high.get('travel_time_s') is not None and high['travel_time_s'] <= target_time <= low.get('duration_s', 0)
            if bracket:
                for _ in range(settings['match_iterations']):
                    candidate = matched(method, (lo+hi)/2, 'match_search')
                    candidates.append(candidate)
                    if candidate.get('success') and abs(candidate['travel_time_s']-target_time) <= settings['match_tolerance_s']+1e-10:
                        break
                    if candidate.get('duration_s', float('inf')) > target_time:
                        lo = candidate['value']
                    else:
                        hi = candidate['value']
            successful = [r for r in candidates if r['success']]
            best = min(successful, key=lambda r: abs(r['travel_time_s']-target_time)) if successful and target_time is not None else None
            manifest['time_matches'][method] = dict(bracketed=bracket, target_time_s=target_time,
                                        best_trial_id=best['trial_id'] if best else None,
                                        best_scale=best['value'] if best else None,
                                        difference_s=best['travel_time_s']-target_time if best else None,
                                        matched=bool(best and abs(best['travel_time_s']-target_time)<=settings['match_tolerance_s']+1e-10))
        # Keep the previous clipped-VP entry for existing report consumers.
        manifest['time_match'] = manifest['time_matches']['vp']
    # Scientific worker processes run simulations only; result acceptance/order is deterministic.
    batch = conditions
    if workers == 1:
        for c in batch:
            accept(c, execute_trial(c, output, numerical_source, force, reuse_for(c)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            from itertools import repeat
            for c, result in zip(batch, pool.map(execute_trial, batch, repeat(output), repeat(numerical_source),
                                               repeat(force), map(reuse_for, batch), chunksize=1)):
                accept(c, result)
    for test in selected:
        directory = output/test
        directory.mkdir(exist_ok=True)
        chosen = [r for r in rows if r['test']==test]
        write_csv(directory/'summary.csv', chosen)
        if test in ('test4', 'preview-noise'):
            for part, filename in ((('c', 'noise_summary.csv'),) if test=='test4' else (('g', 'nominal_selection.csv'),)):
                grouped = aggregate([r for r in chosen if r['part']==part],
                                    ('part', 'scenario', 'method', 'parameter', 'value', 'noise_xy_m', 'noise_yaw_deg'))
                write_csv(directory/filename, grouped)
        issues = [r for r in chosen if not r['success'] or r.get('constraint_violation_duration_s', 0)>0]
        if issues:
            write_csv(directory/'issues.csv', issues)
    manifest['condition_entries'] = len(rows)
    manifest['distinct_trajectories'] = len({r['trial_id'] for r in rows})
    manifest['success_count'] = sum(r['success'] for r in rows)
    manifest['timeout_count'] = sum(r['status']=='timeout' for r in rows)
    manifest['failure_count'] = sum(r['status'] not in ('success', 'timeout') for r in rows)
    manifest['simulation_wall_time_s'] = perf_counter()-start
    figures(output, rows, settings, config)
    if baseline is not None:
        from .access_report import compare_baseline
        manifest['method_comparison'] = compare_baseline(output, manifest, baseline)
    report(output, rows, manifest)
    manifest['complete'] = True
    manifest['total_wall_time_s'] = perf_counter()-start
    atomic_json(target, manifest)
    print(f"Finished {len(rows)} conditions / {manifest['distinct_trajectories']} trajectories in {manifest['total_wall_time_s']:.1f}s: "
          f"{manifest['success_count']} success, {manifest['timeout_count']} timeout, {manifest['failure_count']} failure", flush=True)
    return manifest
