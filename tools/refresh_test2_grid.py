"""Replot saved Test 2 data and separate the optional acceleration sweep, without simulation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from omnidirectional_dwvp.access_metrics import aggregate
from omnidirectional_dwvp.access_plotting import figures
from omnidirectional_dwvp.access_report import report
from omnidirectional_dwvp.access_studies import atomic_json, numerical_hash, plan_conditions
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.studies import ROOT, code_hash, sha, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root, baseline):
    baseline.mkdir(parents=True)
    shutil.copyfile(root/'manifest.json', baseline/'manifest.json')
    paths = list((ROOT/'results/paper').rglob('*'))
    paths += [ROOT/p for p in ('src/omnidirectional_dwvp/solver.py', 'fixtures/solver_cases.json',
                              'src/omnidirectional_dwvp/controller.py', 'src/omnidirectional_dwvp/simulation.py',
                              'configs/access_v2.yaml', 'uv.lock')]
    paths += list((root/'trials').glob('*/trajectory.npz')) + list((root/'trials').glob('*/trial.json'))
    paths += [p for p in root.glob('*') if p.is_file() and p.name.startswith(('comment_round2', 'metrics_alignment'))]
    paths += [root/f'test{i}/summary.csv' for i in (1, 2, 3)]
    atomic_json(baseline/'preserved_sha256.json',
                {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()})
    atomic_json(baseline/'git.json', {cmd: subprocess.check_output(['git', *cmd.split()], cwd=ROOT, text=True).strip()
                                    for cmd in ('branch --show-current', 'rev-parse HEAD')})


def relabel(trial):
    """Change suite metadata only; keep the trajectory specification and every metric."""
    trial = deepcopy(trial)
    condition = trial['condition']
    if condition['test']=='test4' and condition['part']=='c':
        condition.update(test='acceleration-sweep', part='acceleration')
    elif condition['test']=='test4' and condition['part']=='e':
        condition['part'] = 'c'
    identity = sha(condition)
    trial['condition_id'] = identity
    trial['summary'].update(test=condition['test'], part=condition['part'], condition_id=identity)
    return trial


def updated_manifest(original, trials, study):
    manifest = deepcopy(original)
    if study!='all':
        for key in ('comment_round2_followup', 'comment_round2', 'metrics_alignment', 'method_comparison',
                    'time_match', 'time_matches'):
            manifest.pop(key, None)
    manifest.update(study=study, source_sha256=code_hash(), trials=trials,
                    planned_conditions=[t['condition'] for t in trials], condition_entries=len(trials),
                    distinct_trajectories=len({t['summary']['trial_id'] for t in trials}),
                    success_count=sum(t['summary']['success'] for t in trials),
                    timeout_count=sum(t['summary']['status']=='timeout' for t in trials),
                    failure_count=sum(t['summary']['status'] not in ('success', 'timeout') for t in trials))
    selected = tuple(f'test{i}' for i in range(1, 5)) if study=='all' else (study,)
    expected, _ = plan_conditions(Config(**manifest['config']), manifest['settings'], manifest['seed'], selected)
    assert {sha(c) for c in expected} == {t['condition_id'] for t in trials if t['condition']['part']!='a'}
    return manifest


def archive_figures(directory, patterns):
    for pattern in patterns:
        for path in directory.glob(pattern):
            archive = directory/'previous_acceleration_figures'/path.name
            archive.parent.mkdir(exist_ok=True)
            if archive.exists():
                raise FileExistsError(f'Refusing to replace archived figure: {archive}')
            path.rename(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert args.output.is_dir() and not args.output.is_symlink()
    root = args.output.resolve()
    assert root.is_relative_to(ROOT)
    baseline = root/'trials/test2_grid_baseline'
    if not baseline.exists():
        snapshot(root, baseline)
    original = json.loads((baseline/'manifest.json').read_text())
    assert original['complete'] and original['numerical_source_sha256']==numerical_hash()
    assert any(t['condition']['test']=='test4' and t['condition']['part']=='e' for t in original['trials']), \
        'Expected the saved pre-grid manifest, with Test 4 noise in part e'
    preserved = json.loads((baseline/'preserved_sha256.json').read_text())
    for path, expected in preserved.items():
        assert digest(ROOT/path)==expected, path
    trials = [relabel(t) for t in original['trials']]
    ignored = {'test', 'part', 'condition_id'}
    for before, after in zip(original['trials'], trials):
        assert before['spec']==after['spec']
        assert {k: v for k, v in before['summary'].items() if k not in ignored} == {
            k: v for k, v in after['summary'].items() if k not in ignored}
    optional_trials = [t for t in trials if t['condition']['test']=='acceleration-sweep']
    main_trials = [t for t in trials if t['condition']['test']!='acceleration-sweep']
    manifest = updated_manifest(original, main_trials, 'all')
    optional = updated_manifest(original, optional_trials, 'acceleration-sweep')
    optional_root = root.with_name(root.name+'_acceleration_sweep')
    assert not optional_root.is_symlink()
    optional_root.mkdir(exist_ok=True)
    audit = dict(source_sha256=code_hash(), previous_source_sha256=original['source_sha256'],
                 baseline_manifest=str((baseline/'manifest.json').relative_to(ROOT)),
                 baseline_manifest_sha256=digest(baseline/'manifest.json'),
                 preserved_sha256_file=str((baseline/'preserved_sha256.json').relative_to(ROOT)),
                 preserved_sha256_file_sha256=digest(baseline/'preserved_sha256.json'),
                 preserved_conditions=len(trials), unchanged_metric_conditions=len(trials),
                 main_conditions=len(main_trials), optional_conditions=len(optional_trials),
                 relabeled_conditions=sum(a['condition_id']!=b['condition_id'] for a, b in zip(original['trials'], trials)),
                 simulation_runs=0, preserved_files=len(preserved),
                 optional_output=str(optional_root.relative_to(ROOT)))
    manifest['test2_grid_refresh'] = audit
    identities = {t['summary']['trial_id'] for t in optional_trials}
    for identity in list(identities):
        metadata = json.loads((root/'trials'/identity/'trial.json').read_text())
        if metadata.get('trajectory_reuse'):
            identities.add(metadata['trajectory_reuse']['trial_id'])
    for identity in identities:
        destination = optional_root/'trials'/identity
        if not destination.exists():
            shutil.copytree(root/'trials'/identity, destination)
        for name in ('trial.json', 'trajectory.npz'):
            assert digest(destination/name)==digest(root/'trials'/identity/name)
    (optional_root/'acceleration-sweep').mkdir(exist_ok=True)
    optional_rows = [t['summary'] for t in optional_trials]
    write_csv(optional_root/'acceleration-sweep/summary.csv', optional_rows)
    figures(optional_root, optional_rows, optional['settings'], Config(**optional['config']))
    report(optional_root, optional_rows, optional)
    atomic_json(optional_root/'manifest_acceleration-sweep.json', optional)
    rows = [t['summary'] for t in main_trials]
    test4 = [r for r in rows if r['test']=='test4']
    write_csv(root/'test4/summary.csv', test4)
    write_csv(root/'test4/noise_summary.csv', aggregate([r for r in test4 if r['part']=='c'],
              ('part', 'scenario', 'method', 'parameter', 'value', 'noise_xy_m', 'noise_yaw_deg')))
    issues = [r for r in test4 if not r['success'] or r.get('constraint_violation_duration_s', 0)>0]
    if issues:
        write_csv(root/'test4/issues.csv', issues)
    archive_figures(root/'test2', ('acceleration_scale_ramp_*.*', 'angular_acceleration_scale_ramp_*.*',
                                  'ramp_0p3_acceleration_0p5_*.*', 'ramp_0p3_acceleration_0p25_heading.*',
                                  'ramp_0p3_acceleration_1_heading.*'))
    archive_figures(root/'test4', ('*_acceleration_scale.*', '*_angular_acceleration_scale.*'))
    figures(root, [r for r in rows if r['test']=='test2' and r['part']=='acceleration'],
            manifest['settings'], Config(**manifest['config']))
    report(root, rows, manifest)
    for path, expected in preserved.items():
        assert digest(ROOT/path)==expected, path
    for cmd, expected in json.loads((baseline/'git.json').read_text()).items():
        assert subprocess.check_output(['git', *cmd.split()], cwd=ROOT, text=True).strip()==expected
    atomic_json(root/'test2_grid_refresh.json', audit)
    atomic_json(root/'manifest.json', manifest)
    print(json.dumps(audit, indent=2))


if __name__=='__main__':
    main()
