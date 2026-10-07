"""Audit complete saved runs, content identity, constraints and grid coverage offline."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
from omnidirectional_dwvp.access_studies import numerical_hash, plan_conditions, selected_studies
from omnidirectional_dwvp.access_metrics import COMMON_METRICS, aggregate
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.geometry import wrap
from omnidirectional_dwvp.studies import ROOT, code_hash, sha


def validate_grid_refresh(root, manifest):
    """Check the combined main/optional results against the immutable saved run."""
    audit = json.loads((root/'test2_grid_refresh.json').read_text())
    assert audit==manifest['test2_grid_refresh'] and audit['source_sha256']==code_hash()
    baseline = ROOT/audit['baseline_manifest']
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==audit['baseline_manifest_sha256']
    original = json.loads(baseline.read_text())
    assert original['source_sha256']==audit['previous_source_sha256']
    assert original['numerical_source_sha256']==manifest['numerical_source_sha256']
    optional = json.loads((ROOT/audit['optional_output']/'manifest_acceleration-sweep.json').read_text())
    combined = manifest['trials']+optional['trials']
    def scientific_rows(trials):
        return Counter(sha({k: v for k, v in t['summary'].items() if k not in ('test', 'part', 'condition_id')})
                       for t in trials)
    assert scientific_rows(combined)==scientific_rows(original['trials'])
    assert Counter(sha(t['spec']) for t in combined)==Counter(sha(t['spec']) for t in original['trials'])
    assert len(combined)==audit['preserved_conditions']==audit['unchanged_metric_conditions']
    assert len(optional['trials'])==audit['optional_conditions']==100
    assert len(manifest['trials'])==audit['main_conditions'] and audit['simulation_runs']==0
    preserved = ROOT/audit['preserved_sha256_file']
    assert hashlib.sha256(preserved.read_bytes()).hexdigest()==audit['preserved_sha256_file_sha256']
    for path, expected in json.loads(preserved.read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected, path
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--study', default='all', choices=('all','test1','test2','test3','test4','preview-noise','regulation-sweep','acceleration-sweep'))
    args=parser.parse_args()
    start=perf_counter()
    root=args.output
    assert root.is_dir() and not root.is_symlink()
    manifest=json.loads((root/('manifest.json' if args.study=='all' else f'manifest_{args.study}.json')).read_text())
    assert manifest['complete']
    assert manifest['source_sha256']==code_hash()
    assert manifest['numerical_source_sha256']==numerical_hash()
    grid_refresh = validate_grid_refresh(root, manifest) if 'test2_grid_refresh' in manifest else None
    previous_source = grid_refresh['previous_source_sha256'] if grid_refresh else manifest['source_sha256']
    full_run = validate_full_run(root, manifest) if 'full_run_evaluation' in manifest else None
    alignment = None
    if 'metrics_alignment' in manifest:
        alignment=json.loads((root/'metrics_alignment.json').read_text())
        assert alignment['source_sha256']==manifest['source_sha256']
        assert alignment['unexpected_existing_metric_changes']==0
    round2 = None
    audit_name = ('comment_round2_followup' if 'comment_round2_followup' in manifest else 'comment_round2')
    if audit_name in manifest:
        round2=json.loads((root/f'{audit_name}.json').read_text())
        assert round2['source_sha256']==previous_source
        for identity, expected_hash in round2['original_trajectory_sha256'].items():
            assert hashlib.sha256((root/'trials'/identity/'trajectory.npz').read_bytes()).hexdigest()==expected_hash
        if audit_name=='comment_round2_followup':
            assert round2['restored_conditions']==round2['previously_changed_conditions']==962
            assert round2['changed_conditions']==0 and round2['new_conditions']==2
            with (root/'comment_round2_followup_restoration.csv').open() as stream:
                restored=list(csv.DictReader(stream))
            assert len({r['previous_condition_id'] for r in restored})==962
            assert all(r['restored']=='True' for r in restored)
    planned=Counter(sha(c) for c in manifest['planned_conditions'])
    saved=Counter(t['condition_id'] for t in manifest['trials'])
    assert planned==saved
    assert set(saved.values())=={1}
    rows=[]
    selected=selected_studies(args.study)
    expected,_=plan_conditions(Config(**manifest['config']),manifest['settings'],manifest['seed'],selected)
    assert Counter(sha(c) for c in expected)==Counter(t['condition_id'] for t in manifest['trials'] if t['condition']['part']!='a')
    for test in selected:
        with (root/test/'summary.csv').open() as stream:
            data=list(csv.DictReader(stream))
        assert Counter(r['condition_id'] for r in data)==Counter(t['condition_id'] for t in manifest['trials'] if t['condition']['test']==test)
        typed={t['condition_id']:t['summary'] for t in manifest['trials'] if t['condition']['test']==test}
        for row in data:
            assert all(value==('' if typed[row['condition_id']].get(key) is None else str(typed[row['condition_id']][key])) for key,value in row.items())
        rows.extend(data)
    assert not any(r['part']=='f' for r in rows)
    assert (any(r['part']=='d' for r in rows)) == (args.study=='regulation-sweep')
    assert not any(r['part']=='g' for r in rows)
    noise=[r for r in rows if (r['test']=='test4' and r['part']=='c') or r['test']=='preview-noise']
    groups={}
    for row in noise:
        key=tuple(row[k] for k in ('scenario','method','parameter','value','noise_xy_m','noise_yaw_deg'))
        groups.setdefault(key,[]).append(int(row['seed']))
    expected_groups=(50 if 'test4' in selected else 0)
    if 'preview-noise' in selected:
        expected_groups+=3*len(manifest['settings']['noise_levels'])*(len(manifest['settings']['fixed_lookaheads'])+len(manifest['settings']['lookahead_times']))
    assert len(groups)==expected_groups
    assert all(sorted(v)==list(range(manifest['seed'],manifest['seed']+manifest['settings']['noise_seeds'])) for v in groups.values())
    if 'preview-noise' in selected:
        validate_preview_noise(root, manifest)
    if 'test1' in selected:
        assert len([r for r in rows if r['test']=='test1'])==16
    if 'test2' in selected:
        assert len([r for r in rows if r['test']=='test2'])==176
        assert len([r for r in rows if r['test']=='test2' and r['part']=='step'])==4
        assert len([r for r in rows if r['test']=='test2' and r['part']=='acceleration'])==136
    if 'test3' in selected:
        assert len([r for r in rows if r['test']=='test3'])==4
        assert {r['method'] for r in rows if r['test']=='test3'}=={'rpp','dwvp'}
    if 'test4' in selected:
        previews=len(manifest['settings']['fixed_lookaheads'])+len(manifest['settings']['lookahead_times'])
        assert {p:sum(r['test']=='test4' and r['part']==p for r in rows) for p in 'bcde'}==dict(b=10*previews,c=1000,d=0,e=0)
        assert not any(r['test']=='test4' and r['parameter'] in ('acceleration_scale','angular_acceleration_scale') for r in rows)
        assert set(manifest['time_matches'])=={'vp','vp_scaled'}
        for method, match in manifest['time_matches'].items():
            assert match['bracketed'] and match['matched']
            best=next(t['summary'] for t in manifest['trials'] if t['summary']['trial_id']==match['best_trial_id'])
            assert best['method']==method and best['success']
            assert abs(best['travel_time_s']-match['target_time_s'])<=manifest['settings']['match_tolerance_s']+1e-10
    assert all(t['condition']['config']['approach_distance']>0 for t in manifest['trials'])
    if args.study=='regulation-sweep':
        assert len(rows)==36
    if args.study=='acceleration-sweep':
        assert len(rows)==100 and {r['part'] for r in rows}=={'acceleration'}
    checked=set()
    max_velocity_excess=max_acceleration_excess=max_dynamic_window_excess=0.
    for trial in manifest['trials']:
        row=trial['summary'];identity=row['trial_id']
        assert all(k in row for k in COMMON_METRICS)
        cfg=Config(**trial['spec']['config'])
        for axis,limit,acceleration in zip(('vx','vy','w'),cfg.axis_scale,cfg.acceleration):
            np.testing.assert_allclose(row[f'acceleration_time_{axis}_s'],limit/acceleration)
        assert row['lookahead_time_s']==cfg.lookahead_time and row['fixed_lookahead_m']==cfg.fixed_lookahead
        if identity in checked:
            continue
        checked.add(identity)
        metadata=json.loads((root/'trials'/identity/'trial.json').read_text())
        assert sha(metadata['spec'])==identity
        assert metadata['spec']==trial['spec']
        assert trial['condition']['config']==trial['spec']['config']
        assert trial['spec']['numerical_source_sha256']==manifest['numerical_source_sha256'] or alignment or full_run
        reuse=metadata.get('trajectory_reuse')
        if reuse:
            trajectory_hash=hashlib.sha256((root/'trials'/identity/'trajectory.npz').read_bytes()).hexdigest()
            assert trajectory_hash==reuse['trajectory_sha256']
            assert hashlib.sha256((root/'trials'/reuse['trial_id']/'trajectory.npz').read_bytes()).hexdigest()==trajectory_hash
            if round2:
                assert round2['reused_trajectories_provenance'][identity]==reuse
        for metric in COMMON_METRICS:
            assert metadata['metrics'][metric]==row[metric]
        if alignment:
            assert hashlib.sha256((root/'trials'/identity/'trajectory.npz').read_bytes()).hexdigest()==alignment['trajectory_sha256'][identity]
            assert metadata['metrics_source_sha256']==manifest['source_sha256']
        if row['status']=='error':
            assert metadata.get('traceback')
            continue
        with np.load(root/'trials'/identity/'trajectory.npz') as a:
            commands=a['commands'];applied=a['applied'];cfg=metadata['spec']['config']
            assert a['poses'].shape==(len(commands)+1,3)
            assert np.isfinite(a['poses']).all() and np.isfinite(commands).all()
            np.testing.assert_array_equal(a['poses'][0],metadata['spec']['initial_pose'])
            assert hashlib.sha256(a['path'].tobytes()).hexdigest()==metadata['spec']['path_sha256']
            mask=np.ones(len(a['poses']),dtype=bool)
            assert row['evaluation_samples']==len(a['poses'])
            assert row['evaluation_end_m'] is None and trial['condition']['evaluation_end'] is None
            assert row['evaluation_interval']=='start_to_goal_or_timeout'
            assert row['evaluation_complete']==row['success']
            np.testing.assert_allclose(a['times'][-1],row['duration_s'])
            if row['success']:
                np.testing.assert_allclose(row['travel_time_s'],a['times'][-1])
            else:
                assert row['status']=='timeout' and row['travel_time_s'] is None
            validate_goal_and_settling(a,row,trial['spec'])
            np.testing.assert_allclose(row['eval_max_position_error_m'],a['position_errors'][mask].max())
            intervals=mask[:-1]&mask[1:]
            times=a['times']
            duration=np.diff(times)[intervals].sum()
            np.testing.assert_allclose(row['eval_duration_s'],duration)
            for errors, integral_key, mean_key in (
                    (a['position_errors'],'eval_position_error_integral_m_s','eval_mean_position_error_m'),
                    (np.rad2deg(a['yaw_errors']),'eval_heading_error_integral_deg_s','eval_mean_heading_error_deg')):
                integral=sum(.5*(errors[i]+errors[i+1])*(times[i+1]-times[i]) for i in np.flatnonzero(intervals))
                np.testing.assert_allclose(row[integral_key],integral,atol=1e-12)
                if duration>0:
                    np.testing.assert_allclose(row[mean_key],integral/duration,atol=1e-12)
                else:
                    assert row[mean_key] is None
            signed=wrap(a['poses'][:,2]-a['reference_poses'][:,2])
            np.testing.assert_allclose(a['signed_yaw_errors'],signed,atol=1e-14)
            np.testing.assert_allclose(a['yaw_errors'],np.abs(signed),atol=1e-14)
            np.testing.assert_allclose(row['eval_max_heading_lead_deg'],max(0.,float(np.rad2deg(signed[mask]).max())))
            np.testing.assert_allclose(row['eval_max_heading_lag_deg'],max(0.,float(-np.rad2deg(signed[mask]).min())))
            np.testing.assert_allclose(row['eval_max_heading_error_deg'],max(row['eval_max_heading_lead_deg'],row['eval_max_heading_lag_deg']))
            ell=trial['condition']['scenario'].get('transition_length')
            if ell is not None:
                changing=mask & (a['poses'][:,0]>=trial['condition']['ramp_start']) & (a['poses'][:,0]<=trial['condition']['ramp_start']+ell)
                if ell==0:
                    outgoing=np.flatnonzero(mask & (a['poses'][:,0]>trial['condition']['ramp_start']))
                    changing[:]=False
                    if len(outgoing): changing[outgoing[0]]=True
                if changing.any():
                    np.testing.assert_allclose(row['transition_heading_lag_deg'],max(0.,-np.rad2deg(signed[changing]).min()))
                else:
                    assert row['transition_heading_lag_deg'] is None
                after=mask & (a['poses'][:,0]>trial['condition']['ramp_start']+ell)
                expected_overshoot=max(0.,float(np.rad2deg(wrap(a['poses'][after,2]-a['path'][-1,2])).max())) if after.any() else None
                if expected_overshoot is None:
                    assert row['post_transition_heading_overshoot_deg'] is None
                else:
                    np.testing.assert_allclose(row['post_transition_heading_overshoot_deg'],expected_overshoot)
            lower=np.array([cfg['vx_min'],cfg['vy_min'],cfg['w_min']])
            upper=np.array([cfg['vx_max'],cfg['vy_max'],cfg['w_max']])
            accel=np.array([cfg['ax'],cfg['ay'],cfg['aw']])
            previous=np.vstack((np.zeros(3),applied[:-1]))
            preview=(np.full(len(commands),cfg['fixed_lookahead']) if cfg['fixed_lookahead'] is not None else
                     np.clip(cfg['lookahead_time']*np.linalg.norm(previous[:,:2],axis=1),cfg['lookahead_min'],cfg['lookahead_max']))
            np.testing.assert_allclose(a['lookahead'],preview,atol=1e-14,rtol=0)
            assert row['control_steps']==len(commands)
            assert row['lookahead_min_m']==float(preview.min())
            assert row['lookahead_max_m']==float(preview.max())
            raw=cfg['lookahead_time']*np.linalg.norm(previous[:,:2],axis=1)
            assert row['lookahead_upper_active_steps']==(int(np.sum(raw>cfg['lookahead_max']+1e-10)) if cfg['fixed_lookahead'] is None else 0)
            assert row['lookahead_at_upper_steps']==(int(np.sum(preview>=cfg['lookahead_max']-1e-10)) if cfg['fixed_lookahead'] is None else 0)
            dv=np.maximum(np.maximum(lower-commands,commands-upper),0.)
            da=np.maximum(abs((commands-previous)*cfg['frequency'])-accel,0.)
            max_velocity_excess=max(max_velocity_excess,float(dv.max()))
            max_acceleration_excess=max(max_acceleration_excess,float(da.max()))
            db=np.maximum(np.maximum(a['boxes'][:,0]-commands,commands-a['boxes'][:,1]),0.)
            max_dynamic_window_excess=max(max_dynamic_window_excess,float(db.max()))
            np.testing.assert_allclose(row['max_dynamic_window_excess'],db.max())
            assert int(np.any(db>1e-10,axis=1).sum())==round(row['dynamic_window_violation_duration_s']*cfg['frequency'])
            np.testing.assert_array_equal(commands,applied)
            assert int(np.any(dv>1e-10,axis=1).sum())==round(row['velocity_violation_duration_s']*cfg['frequency'])
            assert int(np.any(da>1e-10,axis=1).sum())==round(row['acceleration_violation_duration_s']*cfg['frequency'])
            violations=np.any(dv>1e-10,axis=1)|np.any(da>1e-10,axis=1)
            np.testing.assert_allclose(row['command_constraint_violation_pct'],100*violations.mean())
            lo=np.maximum(previous-accel/cfg['frequency'],lower)
            hi=np.minimum(previous+accel/cfg['frequency'],upper)
            demand_violations=np.any(a['demands']<lo-1e-10,axis=1)|np.any(a['demands']>hi+1e-10,axis=1)
            np.testing.assert_allclose(row['unconstrained_demand_violation_pct'],100*demand_violations.mean())
            for label, velocities in (('demand',a['demands']),('command',commands)):
                v=np.any((velocities<lower-1e-10)|(velocities>upper+1e-10),axis=1)
                acc=np.any((velocities<previous-accel/cfg['frequency']-1e-10)|
                           (velocities>previous+accel/cfg['frequency']+1e-10),axis=1)
                for key, values in (('velocity_violation_steps',v),('acceleration_violation_steps',acc),
                                    ('both_violation_steps',v&acc),('violation_steps',v|acc)):
                    assert row[f'{label}_{key}']==int(values.sum())
            if metadata['spec']['method'] in ('dwpp','rpp'):
                assert np.all(commands[:,0]>=-1e-10) and np.all(commands[:,1]==0)
            if metadata['spec']['method']=='rpp':
                np.testing.assert_array_equal(commands,np.clip(a['demands'],a['boxes'][:,0],a['boxes'][:,1]))
                tracking=a['modes']!=3
                box_speed=np.hypot(max(abs(cfg['vx_min']),abs(cfg['vx_max'])),
                                   max(abs(cfg['vy_min']),abs(cfg['vy_max'])))
                np.testing.assert_allclose(a['demands'][tracking,0],cfg['vx_max']*a['speed_caps'][tracking]/box_speed,rtol=0,atol=1e-14)
    assert max_velocity_excess<=1e-10 and max_acceleration_excess<=1e-10 and max_dynamic_window_excess<=1e-10
    assert not any(r['status']=='error' for r in rows)
    report_lines=len((root/'REPORT.md').read_text().splitlines())
    pdfs=sorted(p for test in selected for p in (root/test).rglob('*.pdf'))
    pngs=sorted(p for test in selected for p in (root/test).rglob('*.png'))
    if 'test2' in selected:
        for stem in ('max_heading_vs_length','max_heading_vs_rate_ratio','ramp_0p3_speed','ramp_0p3_yaw','prediction_legend','rate_limit_legend'):
            assert (root/'test2'/f'{stem}.pdf').exists()
        for ell in manifest['settings']['acceleration_transition_lengths']:
            for quantity in ('heading','overshoot','heading_integral'):
                stem=f'acceleration_ratio_ramp_{ell:g}_{quantity}'.replace('.','p')
                for extension in ('pdf','png'):
                    assert (root/'test2'/f'{stem}.{extension}').exists()
        for scale in (.25, 1.):
            for quantity in ('speed','yaw','signed_heading'):
                stem=f'ramp_0.3_acceleration_{scale:g}_{quantity}'.replace('.','p')
                assert (root/'test2'/f'{stem}.pdf').exists()
        assert (root/'test2'/'acceleration_time_series_legend.pdf').exists()
        assert (root/'test2'/'acceleration_ratio_legend.pdf').exists()
        with (root/'test2'/'overshoot_prediction.csv').open() as stream:
            predictions=list(csv.DictReader(stream))
        expected_predictions=[r for r in rows if r['test']=='test2' and r['part']=='acceleration' and r['method'] in ('vp','vp_scaled','dwvp')]
        assert {r['condition_id'] for r in predictions}=={r['condition_id'] for r in expected_predictions}
        by_id={t['condition_id']:t for t in manifest['trials']}
        for prediction in predictions:
            trial=by_id[prediction['condition_id']]
            omega=float(prediction['heading_braking_omega_rad_s'])
            time=float(prediction['heading_braking_T_s'])
            cfg=Config(**trial['spec']['config'])
            np.testing.assert_allclose(float(prediction['predicted_heading_overshoot_deg']),np.rad2deg(max(0.,omega**2/(2*cfg.aw)-omega*time)))
            np.testing.assert_allclose(float(prediction['observed_heading_overshoot_deg']),trial['summary']['post_transition_heading_overshoot_deg'])
            with np.load(root/'trials'/prediction['trial_id']/'trajectory.npz') as history:
                i=int(round(float(prediction['heading_braking_time_s'])/cfg.dt))
                assert i>0 and history['applied'][i,2]<history['applied'][i-1,2]-1e-10
                np.testing.assert_allclose(omega,history['applied'][i-1,2])
                np.testing.assert_allclose(time,max(cfg.min_orientation_time,cfg.orientation_time_weight*history['lookahead'][i]/cfg.translation_speed))
        # Historical 0.33 m-cap values remain covered by controller regression
        # tests. Audit the current cap's overshoot directly from its trajectory.
    report=(root/'REPORT.md').read_text()
    assert '横ずれ' not in report and '速度上限超過' not in report and '[mm]' not in report
    table_headers=[line for line in report.splitlines() if line.startswith('|') and '最大位置 [m]' in line]
    assert len(table_headers)>=len(selected)
    assert all(all(label in line for label in ('平均位置 [m]','位置積分 [m·s]','最大姿勢 [°]','平均姿勢 [°]','姿勢積分 [°·s]','走行時間 [s]')) for line in table_headers)
    assert not any('2%' in line or '最大姿勢変化' in line for line in report.splitlines() if line.startswith('|'))
    if 'test1' in selected:
        section=report.split('## 試験1：')[1].split('## ')[0]
        assert '整定時間 [s]' in section
    if 'test4' in selected:
        section=report.split('## 試験4：')[1].split('\n## ')[0]
        assert '整定時間 [s]' in section and '(d)' not in section and '(e)' not in section
        for title in ('### (a) 走行時間を揃えた比較', '### (b) 前方注視', '### (c) 自己位置推定のノイズ'):
            assert title in section
    if 'test2' in selected:
        section=report.split('### 区間長さと加速度制約の格子')[1].split('\n### ')[0]
        assert 'T=0.75 s' in section and '比2' in section and 'vp / vp_scaled / dwvp' in section
        grid_rows=[line for line in section.splitlines() if line.startswith('| ')][1:]
        assert len(grid_rows)==30
        expected=[t['summary'] for t in manifest['trials'] if t['condition']['test']=='test2'
                  and t['condition']['parameter']=='acceleration_scale' and t['condition']['method'] in ('vp','vp_scaled','dwvp')]
        for line in grid_rows:
            cells=[cell.strip() for cell in line.split('|')[1:-1]]
            group=[r for r in expected if f"{r['transition_length_m']:.1f}"==cells[0]
                   and f"{r['acceleration_time_w_s']/r['lookahead_time_s']:.2f}"==cells[1]]
            by_method={r['method']:r for r in group}
            for cell,key in zip(cells[2:],('eval_max_heading_error_deg','eval_heading_error_integral_deg_s',
                                          'transition_heading_lag_deg','post_transition_heading_overshoot_deg')):
                assert cell==' / '.join(f"{by_method[m][key]:.2f}" for m in ('vp','vp_scaled','dwvp'))
    if 'test3' in selected:
        section=report.split('## 試験3：')[1].split('## ')[0]
        assert '近傍平均速度 [m/s]' in section and '指令制約違反 [%]' in section
        assert 'unconstrained_demand_violation_pct' in section
        assert '速度超過 [周期]' in section and '加速度超過 [周期]' in section
        for row in (t['summary'] for t in manifest['trials'] if t['condition']['test']=='test3'):
            line=next(line for line in section.splitlines() if line.startswith(f"| {row['value']} | {row['method']} |"))
            expected=row['unconstrained_demand_violation_pct'] if row['method']=='rpp' else row['command_constraint_violation_pct']
            assert line.endswith(f"| {expected:.3f} |")
    assert not any('cap' in p.stem for p in pdfs)
    assert [p.with_suffix('') for p in pdfs]==[p.with_suffix('') for p in pngs]
    fonts=set()
    for pdf in pdfs:
        labels=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        assert 'heading' not in labels.lower() and 'yaw rate' not in labels.lower(), pdf
        if pdf.is_relative_to(root/'test2'):
            assert 'Scaled VP (vel. and acc.)' not in labels, pdf
        output=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        lines=output.splitlines()[2:]
        assert lines and ('TimesNewRoman' in output or 'STIX' in output)
        assert 'Type 3' not in output
        assert all('yes' in line for line in lines)
        fonts.update(line.split()[0].split('+')[-1] for line in lines)
    summary=dict(conditions=len(rows),distinct_trial_specs=len(checked),selection_groups=len(groups),
                 statuses=dict(Counter(r['status'] for r in rows)),report_lines=report_lines,
                 pdf_png_pairs=len(pdfs),embedded_fonts=sorted(fonts),
                 max_velocity_excess=max_velocity_excess,max_acceleration_excess=max_acceleration_excess,
                 max_dynamic_window_excess=max_dynamic_window_excess,
                 baseline_comparison=manifest.get('method_comparison'),
                 grid_refresh=grid_refresh,
                 full_run_evaluation=({k:full_run[k] for k in ('conditions','distinct_trajectories','simulation_runs','unexpected_existing_metric_changes')} if full_run else None),
                 audit_wall_time_s=perf_counter()-start)
    (root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def validate_full_run(root, manifest):
    from recompute_access_metrics import INTERVAL_METRICS
    audit=json.loads((root/'full_run_evaluation.json').read_text())
    assert {k:v for k,v in audit.items() if k!='trajectory_sha256'}==manifest['full_run_evaluation']
    assert audit['source_sha256']==code_hash() and audit['numerical_source_sha256']==numerical_hash()
    baseline=ROOT/audit['baseline_manifest']
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==audit['baseline_sha256']
    original=json.loads(baseline.read_text())
    before={sha({**t['condition'],'evaluation_end':None}):t for t in original['trials']}
    assert set(before)=={t['condition_id'] for t in manifest['trials']}
    assert manifest['time_matches']==original['time_matches'] and manifest['time_match']==original['time_match']
    changes=[]
    for trial in manifest['trials']:
        old=before[trial['condition_id']]
        assert trial['spec']==old['spec'] and trial['condition']=={**old['condition'],'evaluation_end':None}
        for key,value in old['summary'].items():
            if key=='condition_id':
                continue
            new=trial['summary'][key]
            if isinstance(value,(int,float)) and isinstance(new,(int,float)):
                equal=abs(value-new)<=1e-10
            else:
                equal=value==new
            if not equal:
                assert key in INTERVAL_METRICS, (key,value,new)
                changes.append(key)
        metadata=json.loads((root/'trials'/trial['summary']['trial_id']/'trial.json').read_text())
        assert metadata['metrics_source_sha256']==code_hash()
        for key,value in metadata['metrics'].items():
            assert trial['summary'][key]==value
    assert dict(Counter(changes))==audit['changed_metrics']
    for path,expected in audit['trajectory_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    assert audit['simulation_runs']==audit['unexpected_existing_metric_changes']==0
    assert audit['conditions']==len(manifest['trials'])
    assert audit['distinct_trajectories']==len({t['summary']['trial_id'] for t in manifest['trials']})
    assert (root/'regenerated_files.txt').read_text().splitlines()==audit['regenerated_files']
    for filename in audit['regenerated_files']:
        assert (root/filename).is_file(),filename
    # Verify the generated old/new tables against both manifests.
    from omnidirectional_dwvp.access_comparison import comparison_lines
    expected='\n'.join(comparison_lines(root,[t['summary'] for t in manifest['trials']],manifest)).strip()
    assert expected in (root/'REPORT.md').read_text()
    return audit


def validate_goal_and_settling(a,row,spec):
    from omnidirectional_dwvp.controller import terminal_heading
    cfg=Config(**spec['config'])
    goal_yaw=terminal_heading(a['path']) if spec['method'] in ('dwpp','rpp') else a['path'][-1,2]
    within=(np.linalg.norm(a['poses'][:,:2]-a['path'][-1,:2],axis=1)<=cfg.goal_xy)
    within &= np.abs(wrap(a['poses'][:,2]-goal_yaw))<=cfg.goal_yaw
    previous=np.vstack((np.zeros(3),a['applied']))
    arrived=within & (np.max(np.abs(previous),axis=1)<=1e-3)
    assert not arrived[:-1].any()
    if row['success']:
        assert arrived[-1]
    else:
        np.testing.assert_allclose(row['duration_s'],np.ceil(cfg.timeout/cfg.dt)*cfg.dt)
    e0=spec['initial_pose'][1]
    if e0:
        crossing=max(0.,float((-np.sign(e0)*a['poses'][:,1]).max()))
        np.testing.assert_allclose(row['crossing_m'],crossing)
        inside=np.abs(a['poses'][:,1])<=.02*abs(e0)
        if inside[-1] and row['success']:
            outside=np.flatnonzero(~inside)
            index=outside[-1]+1 if len(outside) else 0
            np.testing.assert_allclose(row['settling_2pct_time_s'],a['times'][index])
        else:
            assert row['settling_2pct_time_s'] is None


def validate_preview_noise(root, manifest):
    """Recompute every grid aggregate and recovery threshold from saved rows."""
    directory=root/'preview-noise'
    assert directory.is_dir() and not directory.is_symlink()
    trials=[t['summary'] for t in manifest['trials'] if t['condition']['test']=='preview-noise']
    assert all(r['method']=='dwvp' and r['part']=='c' for r in trials)
    keys=('part','scenario','method','parameter','value','noise_xy_m','noise_yaw_deg')
    expected=aggregate(trials,keys)
    with (directory/'nominal_selection.csv').open() as stream:
        saved=list(csv.DictReader(stream))
    assert len(saved)==len(expected)
    for left,right in zip(saved,expected):
        assert set(left)==set(right)
        for key,value in right.items():
            if value is None:
                assert left[key]==''
            elif isinstance(value,(int,float)):
                np.testing.assert_allclose(float(left[key]),value,rtol=0,atol=1e-10)
            else:
                assert left[key]==value
    with (directory/'travel_time_recovery.csv').open() as stream:
        recovery=list(csv.DictReader(stream))
    fixed=[r for r in expected if r['parameter']=='fixed_lookahead']
    zero={(r['scenario'],r['value']):r for r in fixed if r['noise_xy_m']==r['noise_yaw_deg']==0}
    identities={(kind,r['scenario'],r['noise_xy_m'],r['noise_yaw_deg'])
                for kind in ('nominal_preview','same_fixed_lookahead') for r in fixed}
    assert len(recovery)==len(identities)
    assert {(r['reference'],r['scenario'],float(r['noise_xy_m']),float(r['noise_yaw_deg'])) for r in recovery}==identities
    cfg=Config(**manifest['config'])
    parameter,value=('lookahead_time',cfg.lookahead_time) if cfg.fixed_lookahead is None else ('fixed_lookahead',cfg.fixed_lookahead)
    nominal={r['scenario']:r for r in expected if r['noise_xy_m']==r['noise_yaw_deg']==0
             and (r['parameter'],r['value'])==(parameter,value)}
    report=(root/'REPORT.md').read_text()
    for row in recovery:
        identity=row['scenario'],float(row['noise_xy_m']),float(row['noise_yaw_deg'])
        candidates=[]
        for r in fixed:
            if (r['scenario'],r['noise_xy_m'],r['noise_yaw_deg'])!=identity:
                continue
            baseline=nominal.get(r['scenario']) if row['reference']=='nominal_preview' else zero[r['scenario'],r['value']]
            if baseline is None:
                continue
            if (all(g['n']==g['success_count']==g['travel_time_s_n'] for g in (r,baseline))
                    and r['travel_time_s_mean']<=1.1*baseline['travel_time_s_mean']+1e-10):
                candidates.append(r)
        np.testing.assert_allclose(float(row['estimate_lookahead_m']),cfg.translation_speed*identity[1]/(cfg.ay/cfg.frequency))
        if candidates:
            best=min(candidates,key=lambda r:r['value'])
            assert float(row['minimum_fixed_lookahead_m'])==best['value']
            for key in ('travel_time_s_mean','success_count','n'):
                np.testing.assert_allclose(float(row[key]),best[key])
            baseline=nominal[best['scenario']] if row['reference']=='nominal_preview' else zero[best['scenario'],best['value']]
            np.testing.assert_allclose(float(row['zero_noise_travel_time_s_mean']),baseline['travel_time_s_mean'])
        else:
            assert all(row[key]=='' for key in ('minimum_fixed_lookahead_m','travel_time_s_mean',
                                               'zero_noise_travel_time_s_mean','success_count','n'))
    for scene in {r['scenario'] for r in trials}:
        assert f'| {scene}: σ_xy [m] / σ_yaw [°]' in report
        for ext in ('pdf','png'):
            assert (directory/f'{scene}_travel_time_vs_fixed_lookahead.{ext}').exists()
    for ext in ('pdf','png'):
        assert (directory/f'noise_lookahead_legend.{ext}').exists()
    assert '## 任意試験：前方注視×ノイズ' not in report
    assert '1.1倍以内' in report and 'L≥Vσ_xy/(aΔt)' in report


if __name__=='__main__':
    main()
