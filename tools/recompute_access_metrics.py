"""Re-evaluate every Access trajectory over its full run, without simulation."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from omnidirectional_dwvp.access_metrics import aggregate, evaluate
from omnidirectional_dwvp import access_plotting as plotting
from omnidirectional_dwvp.access_report import report
from omnidirectional_dwvp.access_studies import atomic_json, numerical_hash
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.simulation import Result
from omnidirectional_dwvp.studies import ROOT, code_hash, sha, write_csv
from refresh_test2_figures import archived_acceleration_figures


# Only quantities whose evaluation support changes may differ. Everything else,
# including ramp-local lag, constraint counts, travel times and search, must agree.
INTERVAL_METRICS = {
    'evaluation_end_m', 'evaluation_complete', 'evaluation_samples',
    'eval_duration_s', 'eval_max_position_error_m', 'eval_mean_position_error_m',
    'eval_position_error_integral_m_s', 'eval_sample_mean_position_error_m',
    'eval_max_heading_error_deg', 'eval_mean_heading_error_deg',
    'eval_heading_error_integral_deg_s', 'eval_sample_mean_heading_error_deg',
    'eval_max_heading_lead_deg', 'eval_max_heading_lag_deg',
    'eval_max_heading_change_deg', 'eval_mean_direction_distortion_deg',
    'eval_max_direction_distortion_deg', 'eval_projection_steps',
    'eval_nonintersection_steps', 'crossing_m', 'crossed',
    'first_2pct_time_s', 'first_2pct_distance_m',
    'settling_2pct_time_s', 'settling_2pct_distance_m',
    'post_transition_heading_overshoot_deg',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_run_condition(condition):
    return {**condition, 'evaluation_end': None}


def archived_figures(root, rows, settings, config, optional_rows):
    """Refresh every existing archived panel while preserving its filename."""
    # These retained regulation panels use the unchanged full-run near-obstacle
    # speed, so their historical summary is sufficient to regenerate them.
    with (root/'regulation_sweep_previous.csv').open() as stream:
        regulation = list(csv.DictReader(stream))
    for parameter, xlabel in (('approach_distance', 'Approach distance [m]'),
                              ('cost_scaling_dist', 'Cost distance [m]'),
                              ('cost_scaling_gain', 'Cost gain')):
        curves = []
        for method in ('vp', 'vp_scaled', 'dwvp'):
            group = sorted((r for r in regulation if r['parameter']==parameter and r['method']==method
                            and r['mean_near_obstacle_speed_m_s']),
                           key=lambda r: float(r['value']))
            if group:
                curves.append((method, [float(r['value']) for r in group],
                               [float(r['mean_near_obstacle_speed_m_s']) for r in group]))
        plotting.panel(root/'test4'/('obstacles_'+parameter), curves, xlabel, 'Near-obstacle mean speed [m/s]')
    out = root/'test2/previous_acceleration_figures'
    archived_acceleration_figures(out, rows, out)
    for stem in sorted(p.stem for p in out.glob('ramp_*.pdf')):
        prefix = f"ramp_{settings['representative_rapid']:g}_acceleration_".replace('.', 'p')
        scale, quantity = stem.removeprefix(prefix).split('_', 1)
        group = [r for r in rows if r['test']=='test2' and r['parameter']=='acceleration_scale'
                 and r['transition_length_m']==settings['representative_rapid']
                 and r['value']==float(scale.replace('p', '.')) and r['method'] in plotting.HEADING_METHODS]
        curves = []
        for r in group:
            with np.load(root/'trials'/r['trial_id']/'trajectory.npz') as a:
                if quantity in ('heading', 'signed_heading'):
                    x = a['times']
                    y = np.rad2deg(a['yaw_errors'] if quantity=='heading' else a['signed_yaw_errors'])
                else:
                    x = a['times'][:-1]
                    y = np.linalg.norm(a['applied'][:, :2], axis=1) if quantity=='speed' else a['applied'][:, 2]
                curves.append((r['method'], x, y))
        labels = dict(heading='Orientation error [deg]', signed_heading='Signed orientation error [deg]',
                      speed='Translation speed [m/s]', yaw='Angular velocity [rad/s]')
        plotting.panel(out/stem, curves, 'Time [s]', labels[quantity],
                       reference=0. if quantity=='signed_heading' else None)
    out = root/'test4/previous_acceleration_figures'
    for path in sorted(out.glob('*.pdf')):
        scene, parameter = path.stem.split('_', 1)
        metric, label = (('crossing_m', 'Crossing [m]') if scene=='offset' else
                         ('eval_max_heading_error_deg', 'Max. orientation error [deg]'))
        curves = []
        for method in ('dwpp', 'vp', 'vp_scaled', 'dwvp'):
            group = sorted((r for r in optional_rows if r['scenario']==scene and r['parameter']==parameter
                            and r['method']==method), key=lambda r: r['value'])
            if group:
                curves.append((method, [r['value'] for r in group], [r[metric] for r in group]))
        assert curves, path
        plotting.panel(path.with_suffix(''), curves,
                       'Acceleration multiplier' if parameter=='acceleration_scale' else 'Angular acceleration multiplier', label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Immutable pre-full-run manifest; created if absent inside the repository.')
    args = parser.parse_args()
    root, baseline_path = args.output.resolve(), args.baseline.resolve()
    assert root.is_dir() and root.is_relative_to(ROOT) and not args.output.is_symlink()
    assert baseline_path.is_relative_to(ROOT) and baseline_path != root/'manifest.json'
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_bytes((root/'manifest.json').read_bytes())
    baseline = json.loads(baseline_path.read_text())
    assert baseline['complete'] and not baseline.get('full_run_evaluation')
    current = json.loads((root/'manifest.json').read_text())
    assert Counter(t['summary']['trial_id'] for t in current['trials']) == Counter(t['summary']['trial_id'] for t in baseline['trials'])
    manifest = deepcopy(baseline)
    source, numerical = code_hash(), numerical_hash()
    metrics, hashes, metadata_updates, changes = {}, {}, {}, []
    start, last_log = perf_counter(), perf_counter()

    def reaggregate(trial, location, publish=True):
        identity, spec = trial['summary']['trial_id'], trial['spec']
        directory = location/'trials'/identity
        assert identity == sha(spec)
        if identity not in metrics:
            metadata = json.loads((directory/'trial.json').read_text())
            assert metadata['spec'] == spec and metadata['metrics']['status'] != 'error'
            path = directory/'trajectory.npz'
            hashes[str(path.relative_to(ROOT))] = digest(path)
            with np.load(path) as stored:
                arrays = {k: stored[k] for k in stored}
            metrics[identity] = evaluate(Result(arrays, metadata['metrics'], metadata['performance']),
                                         Config(**spec['config']), spec['scenario'], spec['initial_pose'],
                                         None, spec['ramp_start'])
            metadata['metrics'] = metrics[identity]
            metadata['metrics_source_sha256'] = source
            if publish:
                metadata_updates[identity] = metadata
        return metrics[identity]

    for trial, original in zip(manifest['trials'], baseline['trials']):
        new_metrics = reaggregate(trial, root)
        trial['summary'].update(new_metrics)
        for key, old in original['summary'].items():
            new = trial['summary'][key]
            numeric = isinstance(old, (int, float)) and isinstance(new, (int, float))
            equal = abs(old-new) <= 1e-10 if numeric else old == new
            if not equal:
                assert key in INTERVAL_METRICS, (key, old, new)
                changes.append(dict(condition_id=original['condition_id'], metric=key, before=old, after=new,
                                    difference=new-old if numeric else None))
        trial['condition'] = full_run_condition(trial['condition'])
        trial['condition_id'] = sha(trial['condition'])
        trial['summary']['condition_id'] = trial['condition_id']
        if perf_counter()-last_log > 20:
            print(f'Re-evaluated {len(metrics)} saved trajectories; {perf_counter()-start:.1f}s', flush=True)
            last_log = perf_counter()
    optional_root = root.with_name(root.name+'_acceleration_sweep')
    optional_rows = []
    if (root/'test4/previous_acceleration_figures').exists():
        optional = json.loads((optional_root/'manifest_acceleration-sweep.json').read_text())
        for trial in optional['trials']:
            optional_rows.append({**trial['summary'], **reaggregate(trial, optional_root, publish=False)})
    assert manifest['time_matches'] == baseline['time_matches']
    for path, expected in hashes.items():
        assert digest(ROOT/path) == expected
    for identity, metadata in metadata_updates.items():
        atomic_json(root/'trials'/identity/'trial.json', metadata)
    write_csv(root/'full_run_changes.csv', changes)
    rows = [t['summary'] for t in manifest['trials']]
    for test in ('test1', 'test2', 'test3', 'test4', 'preview-noise'):
        selected = [r for r in rows if r['test'] == test]
        write_csv(root/test/'summary.csv', selected)
        issues = [r for r in selected if not r['success'] or r.get('constraint_violation_duration_s', 0) > 0]
        if issues:
            write_csv(root/test/'issues.csv', issues)
    keys = ('part', 'scenario', 'method', 'parameter', 'value', 'noise_xy_m', 'noise_yaw_deg')
    for test, filename in (('test4', 'noise_summary.csv'), ('preview-noise', 'nominal_selection.csv')):
        write_csv(root/test/filename, aggregate([r for r in rows if r['test']==test and r['part']=='c'], keys))
    manifest.setdefault('simulation_source_sha256', baseline['source_sha256'])
    manifest.setdefault('simulation_numerical_source_sha256', baseline['numerical_source_sha256'])
    manifest['settings']['evaluation_end'] = None
    manifest['planned_conditions'] = [t['condition'] for t in manifest['trials']]
    manifest['notes'] = [n for n in manifest['notes'] if not n.startswith('Error evaluation ends')]
    manifest['notes'].append('Errors include every saved pose from start through the goal criterion or timeout; historical trial specifications retain their original cutoff solely as simulation provenance.')
    manifest.update(source_sha256=source, numerical_source_sha256=numerical,
                    config_file_sha256=digest(ROOT/'configs/access_v2.yaml'))
    audit = dict(baseline_manifest=str(baseline_path.relative_to(ROOT)), baseline_sha256=digest(baseline_path),
                 baseline_source_sha256=baseline['source_sha256'], source_sha256=source,
                 numerical_source_sha256=numerical, interval='start_to_goal_or_timeout', simulation_runs=0,
                 conditions=len(rows), distinct_trajectories=len(metadata_updates),
                 additional_archived_trajectories=len(metrics)-len(metadata_updates),
                 unexpected_existing_metric_changes=0, changed_metrics=dict(Counter(c['metric'] for c in changes)),
                 unchanged_by_construction=['travel_time_s', 'duration_s', 'status', 'success',
                     'command_constraint_violation_pct', 'mean_near_obstacle_speed_m_s', 'transition_heading_lag_deg', 'time_matches'],
                 trajectory_sha256=hashes)
    manifest['full_run_evaluation'] = {k: v for k, v in audit.items() if k != 'trajectory_sha256'}
    existing_figures = {str(p.relative_to(root)) for p in root.rglob('*') if p.suffix in ('.pdf', '.png')}
    regenerated = []
    save = plotting.save
    def recorded_save(fig, stem):
        save(fig, stem)
        regenerated.extend(str(stem.with_suffix(ext).relative_to(root)) for ext in ('.pdf', '.png'))
    plotting.save = recorded_save
    plotting.figures(root, rows, manifest['settings'], Config(**manifest['config']))
    archived_figures(root, rows, manifest['settings'], Config(**manifest['config']), optional_rows)
    assert set(regenerated) == existing_figures, (existing_figures-set(regenerated), set(regenerated)-existing_figures)
    regenerated += ['manifest.json', 'REPORT.md', 'validation.json', 'full_run_evaluation.json', 'full_run_changes.csv',
                    'regenerated_files.txt', 'full_run_checks.json', 'issues.csv', 'lookahead_ranges.csv', 'test2/overshoot_prediction.csv',
                    'test4/noise_summary.csv', 'preview-noise/nominal_selection.csv', 'preview-noise/travel_time_recovery.csv']
    regenerated += [f'{test}/summary.csv' for test in ('test1', 'test2', 'test3', 'test4', 'preview-noise')]
    regenerated += ['test4/issues.csv', 'preview-noise/issues.csv']
    (root/'regenerated_files.txt').write_text('\n'.join(sorted(regenerated))+'\n')
    audit['regenerated_files'] = sorted(regenerated)
    manifest['full_run_evaluation']['regenerated_files'] = sorted(regenerated)
    atomic_json(root/'full_run_checks.json', dict(pytest='pending', validation='pending'))
    report(root, rows, manifest)
    atomic_json(root/'full_run_evaluation.json', audit)
    atomic_json(root/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('trajectory_sha256', 'regenerated_files')}, indent=2))
    print(f'Refreshed {len(regenerated)} artifacts in {perf_counter()-start:.1f}s; no simulation executed.')


if __name__ == '__main__':
    main()
