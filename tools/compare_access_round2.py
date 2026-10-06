"""Audit the round-2 refresh against its saved manifest and trajectory hashes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from omnidirectional_dwvp.access_report import report
from omnidirectional_dwvp.access_studies import atomic_json, reuse_key
from omnidirectional_dwvp.studies import code_hash, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    old = json.loads(args.baseline.read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert old['complete'] and manifest['complete']
    assert manifest['source_sha256'] == code_hash()
    hashes = json.loads((args.baseline.parent/'trajectory_sha256.json').read_text())
    for identity, expected in hashes.items():
        assert digest(root/'trials'/identity/'trajectory.npz') == expected
    protected = json.loads((args.baseline.parent/'protected.json').read_text())
    for path, expected in protected.items():
        assert digest(Path(path)) == expected
    previous = {reuse_key(t['condition']): t for t in old['trials']}
    changes, conditions, retained, reused = [], [], set(), {}
    for trial in manifest['trials']:
        row = trial['summary']
        metadata = json.loads((root/'trials'/row['trial_id']/'trial.json').read_text())
        provenance = metadata.get('trajectory_reuse')
        if provenance:
            assert digest(root/'trials'/row['trial_id']/'trajectory.npz') == hashes[provenance['trial_id']] == provenance['trajectory_sha256']
            reused[row['trial_id']] = provenance
        before = previous.get(reuse_key(trial['condition']))
        labels = {k: row[k] for k in ('test', 'part', 'scenario', 'method', 'parameter', 'value', 'seed', 'condition_id', 'trial_id')}
        changed = []
        if before:
            retained.add(before['condition_id'])
            for metric, left in before['summary'].items():
                if metric in ('condition_id', 'trial_id'):
                    continue
                right = row[metric]
                numeric = isinstance(left, (int, float)) and isinstance(right, (int, float))
                equal = abs(left-right) <= 1e-10 if numeric else left == right
                if equal:
                    continue
                changed.append(metric)
                changes.append({**labels, 'metric': metric, 'before': left, 'after': right,
                                'difference': right-left if numeric else None})
            assert not (provenance and changed), 'Reused trajectories must retain all metrics'
        conditions.append({**labels, 'previous_condition_id': before['condition_id'] if before else None,
                           'previous_trial_id': before['summary']['trial_id'] if before else None,
                           'outcome': ('changed' if changed else 'unchanged') if before else 'new',
                           'trajectory_reused': bool(provenance), 'changed_metrics': ';'.join(changed)})
    write_csv(root/'comment_round2_conditions.csv', conditions)
    fields = [*labels, 'metric', 'before', 'after', 'difference']
    with (root/'comment_round2_changes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(changes)
    # Keep the removed sweep's values available without including them in all.
    write_csv(root/'regulation_sweep_previous.csv',
              [t['summary'] for t in old['trials'] if t['condition']['part'] == 'd'])
    outcomes = Counter(r['outcome'] for r in conditions)
    summary = dict(baseline_sha256=digest(args.baseline), source_sha256=code_hash(),
                   matched_conditions=outcomes['changed']+outcomes['unchanged'],
                   unchanged_conditions=outcomes['unchanged'], changed_conditions=outcomes['changed'],
                   new_conditions=outcomes['new'], reused_conditions=sum(r['trajectory_reused'] for r in conditions),
                   reused_trajectories=len(reused), recomputed_trajectories=manifest['distinct_trajectories']-len(reused),
                   original_trajectories_preserved=len(hashes), protected_files_unchanged=len(protected),
                   controller_regression=json.loads((args.baseline.parent/'controller_regression.json').read_text()),
                   changed_groups=dict(Counter('/'.join((r['test'], r['part'], r['method']))
                                               for r in conditions if r['outcome'] == 'changed')))
    audit = {**summary, 'reused_trajectories_provenance': reused, 'original_trajectory_sha256': hashes,
             'excluded_conditions': [{k: t['summary'][k] for k in ('condition_id', 'test', 'part', 'scenario', 'method', 'parameter', 'value', 'seed')}
                                     for t in old['trials'] if t['condition_id'] not in retained],
             'previous_provenance': {k: old[k] for k in ('method_comparison', 'metrics_alignment') if k in old}}
    manifest['comment_round2'] = summary
    report(root, [t['summary'] for t in manifest['trials']], manifest)
    atomic_json(root/'comment_round2.json', audit)
    atomic_json(root/'manifest.json', manifest)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
