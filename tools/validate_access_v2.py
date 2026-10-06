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
from omnidirectional_dwvp.access_studies import numerical_hash, plan_conditions
from omnidirectional_dwvp.access_metrics import COMMON_METRICS
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.geometry import wrap
from omnidirectional_dwvp.studies import code_hash, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--study', default='all', choices=('all','test1','test2','test3','test4','preview-noise','regulation-sweep'))
    args=parser.parse_args()
    start=perf_counter()
    root=args.output
    assert root.is_dir() and not root.is_symlink()
    manifest=json.loads((root/('manifest.json' if args.study=='all' else f'manifest_{args.study}.json')).read_text())
    assert manifest['complete']
    assert manifest['source_sha256']==code_hash()
    assert manifest['numerical_source_sha256']==numerical_hash()
    alignment = None
    if 'metrics_alignment' in manifest:
        alignment=json.loads((root/'metrics_alignment.json').read_text())
        assert alignment['source_sha256']==manifest['source_sha256']
        assert alignment['unexpected_existing_metric_changes']==0
    round2 = None
    audit_name = ('comment_round2_followup' if 'comment_round2_followup' in manifest else 'comment_round2')
    if audit_name in manifest:
        round2=json.loads((root/f'{audit_name}.json').read_text())
        assert round2['source_sha256']==manifest['source_sha256']
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
    selected=tuple(f'test{i}' for i in range(1,5)) if args.study=='all' else (args.study,)
    expected,_=plan_conditions(Config(**manifest['config']),manifest['settings'],manifest['seed'],selected)
    assert Counter(sha(c) for c in expected)==Counter(t['condition_id'] for t in manifest['trials'] if t['condition']['part']!='a')
    for test in selected:
        with (root/test/'summary.csv').open() as stream:
            data=list(csv.DictReader(stream))
        assert Counter(r['condition_id'] for r in data)==Counter(t['condition_id'] for t in manifest['trials'] if t['condition']['test']==test)
        rows.extend(data)
    assert not any(r['part']=='f' for r in rows)
    assert (any(r['part']=='d' for r in rows)) == (args.study=='regulation-sweep')
    assert (any(r['part']=='g' for r in rows)) == (args.study=='preview-noise')
    noise=[r for r in rows if r['part'] in ('e','g')]
    groups={}
    for row in noise:
        key=tuple(row[k] for k in ('scenario','method','parameter','value','noise_xy_m','noise_yaw_deg'))
        groups.setdefault(key,[]).append(int(row['seed']))
    assert len(groups)==(165 if args.study=='preview-noise' else 50 if 'test4' in selected else 0)
    assert all(sorted(v)==list(range(manifest['seed'],manifest['seed']+manifest['settings']['noise_seeds'])) for v in groups.values())
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
        assert {p:sum(r['test']=='test4' and r['part']==p for r in rows) for p in 'bcde'}==dict(b=110,c=100,d=0,e=1000)
        assert set(manifest['time_matches'])=={'vp','vp_scaled'}
        for method, match in manifest['time_matches'].items():
            assert match['bracketed'] and match['matched']
            best=next(t['summary'] for t in manifest['trials'] if t['summary']['trial_id']==match['best_trial_id'])
            assert best['method']==method and best['success']
            assert abs(best['travel_time_s']-match['target_time_s'])<=manifest['settings']['match_tolerance_s']+1e-10
    assert all(t['condition']['config']['approach_distance']>0 for t in manifest['trials'])
    if args.study=='regulation-sweep':
        assert len(rows)==36
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
        assert trial['spec']['numerical_source_sha256']==manifest['numerical_source_sha256'] or alignment
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
            mask=(a['poses'][:,0]>=-1e-10)&(a['poses'][:,0]<=trial['condition']['evaluation_end'])
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
    assert report_lines<=240
    pdfs=sorted(p for test in selected for p in (root/test).glob('*.pdf'))
    pngs=sorted(p for test in selected for p in (root/test).glob('*.png'))
    if 'test2' in selected:
        for stem in ('max_heading_vs_length','max_heading_vs_rate_ratio','ramp_0p3_speed','ramp_0p3_yaw','prediction_legend','rate_limit_legend'):
            assert (root/'test2'/f'{stem}.pdf').exists()
        for ell in manifest['settings']['acceleration_transition_lengths']:
            for quantity in ('heading','position','heading_integral'):
                stem=f'acceleration_scale_ramp_{ell:g}_{quantity}'.replace('.','p')
                assert (root/'test2'/f'{stem}.pdf').exists()
        for quantity in ('heading','position','heading_integral'):
            assert (root/'test2'/f'angular_acceleration_scale_ramp_0p3_{quantity}.pdf').exists()
        for scale in set(manifest['settings']['acceleration_time_series_scales']) | {min(manifest['settings']['acceleration_scales'])}:
            for quantity in ('speed','yaw','signed_heading','heading'):
                stem=f'ramp_0.3_acceleration_{scale:g}_{quantity}'.replace('.','p')
                assert (root/'test2'/f'{stem}.pdf').exists()
        assert (root/'test2'/'acceleration_time_series_legend.pdf').exists()
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
        section=report.split('## 試験4：')[1].split('## ')[0]
        assert '整定時間 [s]' in section and '(d)' not in section
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
                 audit_wall_time_s=perf_counter()-start)
    (root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
