"""Verify restored round-2 results while preserving the earlier audit files."""
import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

from omnidirectional_dwvp.access_report import lookahead_ranges, report
from omnidirectional_dwvp.access_studies import atomic_json, reuse_key
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.controller import compute_command
from omnidirectional_dwvp.geometry import Obstacle
from omnidirectional_dwvp.paths import orientation_ramp_path, straight_path
from omnidirectional_dwvp.studies import code_hash, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    return (abs(left-right) <= 1e-10 if isinstance(left, (int, float)) and
            isinstance(right, (int, float)) else left == right)


def controller_regression(snapshot):
    """Compare every command field to the entry-state implementation."""
    before = types.ModuleType('omnidirectional_dwvp._before_followup')
    before.__package__ = 'omnidirectional_dwvp'
    source = snapshot/'src/omnidirectional_dwvp/controller.py'
    exec(compile(source.read_text(), str(source), 'exec'), before.__dict__)
    rng = np.random.default_rng(20261006)
    checks = 0
    for method in ('vp', 'vp_scaled', 'vp_scaled_accel', 'dwvp', 'dwpp'):
        for index in range(100):
            cfg = Config(lookahead_time=.75, use_cost_regulation=bool(index % 2))
            cfg = replace(cfg, ax=cfg.ax*(.25 if index % 3 else 2.))
            path = orientation_ramp_path(.3) if index % 2 else straight_path()
            pose = np.r_[rng.uniform([0., -.5], [4., .5]), rng.uniform(-np.pi, np.pi)]
            if index % 10 == 0:
                pose[:2] = path[-1, :2]
            current = rng.uniform(cfg.lower, cfg.upper)
            if method == 'dwpp':
                current[:2] = [abs(current[0]), 0.]
            obstacles = (Obstacle(1.6, -.45, .1), Obstacle(2.4, .55, .1))
            left = before.compute_command(pose, current, path, path[:, 0], method, cfg, obstacles)
            right = compute_command(pose, current, path, path[:, 0], method, cfg, obstacles)
            for key in vars(left):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            checks += 1
    return dict(commands_checked=checks, methods=5, all_fields_exact=True)


def test3_breakdown(root, trial):
    row = trial['summary']
    cfg = Config(**trial['spec']['config'])
    with np.load(root/'trials'/row['trial_id']/'trajectory.npz') as history:
        velocities = history['demands'] if row['method'] == 'rpp' else history['commands']
        previous = np.vstack((np.zeros(3), history['applied'][:-1]))
        acceleration = np.any(np.abs(velocities-previous) > cfg.acceleration*cfg.dt+1e-10, axis=1)
        # Startup is the initial uninterrupted run of acceleration exceedances.
        startup = np.zeros(len(acceleration), dtype=bool)
        stop = np.flatnonzero(~acceleration)
        startup[:int(stop[0]) if len(stop) else len(startup)] = True
        terminal = (history['modes'] == 3) & ~startup
        cap_change = np.r_[False, np.abs(np.diff(history['speed_caps'])) > 1e-10]
        regulation = cap_change & ~startup & ~terminal
        other = ~startup & ~terminal & ~regulation
        result = {k: row[k] for k in ('condition_id', 'method', 'value', 'control_steps',
                                      'mean_near_obstacle_speed_m_s', 'travel_time_s',
                                      'unconstrained_demand_violation_pct', 'command_constraint_violation_pct')}
        prefix = 'demand' if row['method'] == 'rpp' else 'command'
        result.update({key: row[f'{prefix}_{key}'] for key in
                       ('velocity_violation_steps', 'acceleration_violation_steps', 'both_violation_steps', 'violation_steps')})
        for key, mask in (('startup', startup), ('regulation', regulation), ('terminal', terminal), ('other', other)):
            result[f'{key}_acceleration_steps'] = int(np.sum(acceleration & mask))
        assert sum(result[f'{key}_acceleration_steps'] for key in ('startup', 'regulation', 'terminal', 'other')) == result['acceleration_violation_steps']
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    root, snapshot = args.output, args.snapshot
    old = json.loads(args.baseline.read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    previous_root = snapshot/'results/access_v2'
    previous_manifest = json.loads((previous_root/'manifest.json').read_text())
    assert old['complete'] and manifest['complete'] and previous_manifest['complete']
    assert old['config']['lookahead_max'] == manifest['config']['lookahead_max'] == .33
    assert manifest['source_sha256'] == code_hash()
    hashes = json.loads((args.baseline.parent/'trajectory_sha256.json').read_text())
    previous_hashes = json.loads((snapshot/'trajectory_sha256.json').read_text())
    for identity, expected in (hashes | previous_hashes).items():
        assert digest(root/'trials'/identity/'trajectory.npz') == expected
    protected = json.loads((snapshot/'protected.json').read_text())
    for path, expected in protected.items():
        assert digest(Path(path)) == expected
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True) == (snapshot/'git_branch.txt').read_text()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True) == (snapshot/'git_head.txt').read_text()
    for name in ('comment_round2.json', 'comment_round2_conditions.csv', 'comment_round2_changes.csv'):
        assert digest(root/name) == digest(previous_root/name)
    originals = {reuse_key(t['condition']): t for t in old['trials']}
    current = {reuse_key(t['condition']): t for t in manifest['trials']}
    prior = {t['condition_id']: t for t in previous_manifest['trials']}
    conditions, reused = [], {}
    for key, trial in current.items():
        row = trial['summary']
        before = originals.get(key)
        changed = []
        if before:
            assert before['condition'] == trial['condition']
            changed = [metric for metric, left in before['summary'].items()
                       if metric not in ('condition_id', 'trial_id') and not equal(left, row[metric])]
        metadata = json.loads((root/'trials'/row['trial_id']/'trial.json').read_text())
        provenance = metadata.get('trajectory_reuse')
        if provenance:
            assert digest(root/'trials'/row['trial_id']/'trajectory.npz') == hashes[provenance['trial_id']] == provenance['trajectory_sha256']
            reused[row['trial_id']] = provenance
        conditions.append({**{k: row[k] for k in ('test', 'part', 'scenario', 'method', 'parameter', 'value', 'seed', 'condition_id', 'trial_id')},
                           'outcome': ('changed' if changed else 'unchanged') if before else 'new',
                           'trajectory_reused': bool(provenance), 'changed_metrics': ';'.join(changed)})
    restoration = []
    with (previous_root/'comment_round2_changes.csv').open() as stream:
        for item in csv.DictReader(stream):
            key = reuse_key(prior[item['condition_id']]['condition'])
            trial = current[key]
            metric = item['metric']
            target = originals[key]['summary'][metric]
            assert str(target) == item['before'] or (target is None and item['before'] == '')
            value = trial['summary'][metric]
            restoration.append(dict(previous_condition_id=item['condition_id'], condition_id=trial['condition_id'],
                                    metric=metric, original=target, previous=item['after'], restored_value=value,
                                    restored=equal(target, value)))
    outcomes = Counter(r['outcome'] for r in conditions)
    assert outcomes['changed'] == 0 and outcomes['new'] == 2
    assert {r['method'] for r in conditions if r['outcome'] == 'new'} == {'rpp'}
    assert all(r['restored'] for r in restoration)
    changed_ids = {r['previous_condition_id'] for r in restoration}
    assert len(changed_ids) == 962
    summary = dict(source_sha256=code_hash(), baseline_sha256=digest(args.baseline),
                   previous_manifest_sha256=digest(previous_root/'manifest.json'),
                   matched_conditions=outcomes['unchanged'], changed_conditions=outcomes['changed'], new_conditions=outcomes['new'],
                   previously_changed_conditions=len(changed_ids), restored_conditions=len(changed_ids),
                   restored_metric_entries=len(restoration),
                   reused_conditions=sum(r['trajectory_reused'] for r in conditions), reused_trajectories=len(reused),
                   recomputed_trajectories=manifest['distinct_trajectories']-len(reused),
                   original_trajectories_preserved=len(hashes), previous_trajectories_preserved=len(previous_hashes),
                   protected_files_unchanged=len(protected), controller_regression=controller_regression(snapshot),
                   lookahead_ranges=lookahead_ranges([t['summary'] for t in manifest['trials']]),
                   test3_breakdown=[test3_breakdown(root, t) for t in manifest['trials'] if t['condition']['test'] == 'test3'])
    write_csv(root/'comment_round2_followup_conditions.csv', conditions)
    write_csv(root/'comment_round2_followup_restoration.csv', restoration)
    manifest['comment_round2_followup'] = summary
    report(root, [t['summary'] for t in manifest['trials']], manifest)
    atomic_json(root/'comment_round2_followup.json', {**summary, 'reused_trajectories_provenance': reused,
                'original_trajectory_sha256': hashes, 'previous_trajectory_sha256': previous_hashes,
                'protected_sha256': protected})
    atomic_json(root/'manifest.json', manifest)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
