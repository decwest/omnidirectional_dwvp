"""Re-evaluate saved Access trajectories without running or changing a controller."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from omnidirectional_dwvp.access_metrics import aggregate, evaluate
from omnidirectional_dwvp.access_plotting import figures
from omnidirectional_dwvp.access_report import report
from omnidirectional_dwvp.access_studies import atomic_json, numerical_hash
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.simulation import Result
from omnidirectional_dwvp.studies import code_hash, sha, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Immutable copy of the pre-alignment manifest (created if absent).')
    args = parser.parse_args()
    root = args.output
    assert root.is_dir() and not root.is_symlink()
    target = root/'manifest.json'
    assert args.baseline.resolve() != target.resolve()
    if not args.baseline.exists():
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_bytes(target.read_bytes())
    baseline = json.loads(args.baseline.read_text())
    manifest = json.loads(target.read_text())
    assert manifest['complete'] and baseline['complete']
    before = {t['condition_id']: t for t in baseline['trials']}
    assert set(before) == {t['condition_id'] for t in manifest['trials']}
    source, numerical = code_hash(), numerical_hash()
    metrics, files, updates, changes = {}, {}, {}, []
    start = perf_counter()
    last_log = start
    for trial in manifest['trials']:
        original = before[trial['condition_id']]
        assert trial['condition'] == original['condition'] and trial['spec'] == original['spec']
        identity = trial['summary']['trial_id']
        assert identity == original['summary']['trial_id'] == sha(trial['spec'])
        directory = root/'trials'/identity
        if identity not in metrics:
            metadata = json.loads((directory/'trial.json').read_text())
            assert metadata['spec'] == trial['spec']
            assert metadata['metrics']['status'] != 'error', 'Cannot refresh a missing trajectory'
            path = directory/'trajectory.npz'
            files[identity] = digest(path)
            with np.load(path) as stored:
                arrays = {k: stored[k] for k in stored}
            spec = trial['spec']
            metrics[identity] = evaluate(Result(arrays, metadata['metrics'], metadata['performance']),
                                         Config(**spec['config']), spec['scenario'], spec['initial_pose'],
                                         spec['evaluation_end'], spec['ramp_start'])
            assert digest(path) == files[identity]
            metadata['metrics'] = metrics[identity]
            metadata['metrics_source_sha256'] = source
            updates[identity] = metadata
        trial['summary'].update(metrics[identity])
        # Compare every pre-existing summary value, including status/identities.
        for key, old in original['summary'].items():
            new = trial['summary'][key]
            numeric = isinstance(old, (int, float)) and isinstance(new, (int, float))
            equal = abs(old-new) <= 1e-10 if numeric else old == new
            if not equal:
                changes.append(dict(condition_id=trial['condition_id'], metric=key, before=old, after=new,
                                    difference=new-old if numeric else None))
        if perf_counter()-last_log > 20:
            print(f'Re-evaluated {len(metrics)} saved trajectories; {perf_counter()-start:.1f}s', flush=True)
            last_log = perf_counter()
    allowed = {'eval_mean_position_error_m', 'eval_mean_heading_error_deg'}
    unexpected = [c for c in changes if c['metric'] not in allowed]
    assert not unexpected, f'Unexpected changes to existing metrics: {unexpected[:3]}'
    for trial in manifest['trials']:
        for kind in ('position_error_m', 'heading_error_deg'):
            np.testing.assert_allclose(trial['summary']['eval_sample_mean_'+kind],
                                       before[trial['condition_id']]['summary']['eval_mean_'+kind], atol=1e-12)
    audit = dict(baseline_sha256=digest(args.baseline), baseline_source_sha256=baseline['source_sha256'],
                 baseline_numerical_source_sha256=baseline['numerical_source_sha256'],
                 source_sha256=source, numerical_source_sha256=numerical,
                 conditions=len(manifest['trials']), distinct_trajectories=len(metrics),
                 unexpected_existing_metric_changes=len(unexpected),
                 changed_metrics=dict(Counter(c['metric'] for c in changes)),
                 max_absolute_changes={k: max((abs(c['difference']) for c in changes if c['metric']==k), default=0.)
                                       for k in sorted(allowed)},
                 nonzero_overshoot_before=sum(t['summary'].get('post_transition_heading_overshoot_deg') not in (None, 0.) for t in baseline['trials']),
                 trajectory_sha256=files)
    # Publish only after all scientific values have passed the comparison.
    for identity, metadata in updates.items():
        atomic_json(root/'trials'/identity/'trial.json', metadata)
    with (root/'metrics_alignment_changes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('condition_id', 'metric', 'before', 'after', 'difference'))
        writer.writeheader()
        writer.writerows(changes)
    rows = [t['summary'] for t in manifest['trials']]
    for test in ('test1', 'test2', 'test3', 'test4'):
        selected = [r for r in rows if r['test'] == test]
        write_csv(root/test/'summary.csv', selected)
        issues = [r for r in selected if not r['success'] or r.get('constraint_violation_duration_s', 0) > 0]
        if issues:
            write_csv(root/test/'issues.csv', issues)
    write_csv(root/'test4'/'noise_summary.csv', aggregate([r for r in rows if r['part'] == 'e'],
              ('part', 'scenario', 'method', 'parameter', 'value', 'noise_xy_m', 'noise_yaw_deg')))
    manifest.setdefault('simulation_source_sha256', baseline['source_sha256'])
    manifest.setdefault('simulation_numerical_source_sha256', baseline['numerical_source_sha256'])
    manifest.update(source_sha256=source, numerical_source_sha256=numerical,
                    metrics_alignment={k: v for k, v in audit.items() if k != 'trajectory_sha256'})
    figures(root, rows, manifest['settings'], Config(**manifest['config']))
    report(root, rows, manifest)
    atomic_json(root/'metrics_alignment.json', audit)
    atomic_json(target, manifest)
    print(json.dumps({k: v for k, v in audit.items() if k != 'trajectory_sha256'}, indent=2))
    print(f'Refreshed metrics and figures in {perf_counter()-start:.1f}s; no simulation executed.')


if __name__ == '__main__':
    main()
