from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import pytest
import yaml
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.controller import desired_vector, compute_command
from omnidirectional_dwvp.dwpp import optimal_velocity_in_window
from omnidirectional_dwvp.paths import straight_path, orientation_ramp_path
from omnidirectional_dwvp.simulation import simulate
from omnidirectional_dwvp.metrics import project_reference
from omnidirectional_dwvp.access_metrics import aggregate, evaluate, time_error_metrics, heading_braking_prediction
from omnidirectional_dwvp.access_studies import plan_conditions, execute_trial, numerical_hash

ROOT=Path(__file__).resolve().parents[1]


def test_demand_default_feasible_all_directions_and_legacy():
    c=Config()
    assert c.translation_speed==.22
    for angle in np.linspace(0,2*np.pi,100):
        target=np.array([np.cos(angle),np.sin(angle),.3])
        d=desired_vector(np.zeros(3),target,c)
        assert np.linalg.norm(d[:2])==pytest.approx(.22)
        assert np.all(d[:2]>=c.lower[:2]-1e-12) and np.all(d[:2]<=c.upper[:2]+1e-12)
        old=desired_vector(np.zeros(3),target,replace(c,vp_translation_speed=c.box_speed))
        np.testing.assert_allclose(old[:2], c.box_speed*target[:2])
        assert old[2]==pytest.approx(.3/max(c.min_orientation_time,1/c.box_speed))
    assert replace(c,vx_min=-.08).translation_speed==.08
    with pytest.raises(ValueError):
        Config(vp_translation_speed=-.1)


def test_dwpp_original_kernel_and_full_command_snapshots():
    data=json.loads((ROOT/'fixtures/dwpp_reference_cases.json').read_text())
    for case in data['kernel_cases']:
        np.testing.assert_allclose(optimal_velocity_in_window(case['window'],case['curvature']),case['expected'],atol=1e-14)
    path=straight_path()
    for case in data['command_cases']:
        config=Config(lookahead_time=.75,fixed_lookahead=case['fixed_lookahead'])
        out=compute_command(np.array(case['pose']),np.array([case['current'][0],0,case['current'][1]]),path,path[:,0],'dwpp',config)
        np.testing.assert_allclose(out.command[[0,2]],case['expected'],atol=1e-12)
        assert out.command[1]==0


@pytest.mark.parametrize('method', ['dwpp', 'rpp'])
def test_differential_methods_ignore_reference_yaw_including_terminal(method):
    path=straight_path(length=1.)
    changed=path.copy();changed[:,2]=np.linspace(.5,1.7,len(path))
    config=Config(lookahead_time=.75)
    a=simulate(path,method,config)
    b=simulate(changed,method,config)
    np.testing.assert_array_equal(a.arrays['applied'],b.arrays['applied'])
    assert a.metrics['success'] and b.metrics['success']
    assert np.all(a.arrays['applied'][:,0]>=0)
    assert np.all(a.arrays['applied'][:,1]==0)


def test_initial_pose_validation_copy_and_heading():
    pose=np.array([.2,.3,.4])
    r=simulate(straight_path(),config=Config(timeout=.1),initial_pose=pose)
    np.testing.assert_array_equal(r.arrays['poses'][0],pose)
    np.testing.assert_array_equal(pose,[.2,.3,.4])
    for invalid in ([1,2],[0,0,float('nan')]):
        with pytest.raises(ValueError,match='initial_pose'):
            simulate(straight_path(),initial_pose=invalid)


def test_straight_ramp_and_true_step_reference():
    p=straight_path()
    assert p.shape==(801,3)
    np.testing.assert_allclose(np.diff(p[:,0]),.005)
    np.testing.assert_allclose(straight_path(heading=np.pi/2)[-1],[0,4,np.pi/2],atol=1e-14)
    for ell in (2.,1.,.5,.25,.1):
        p=orientation_ramp_path(ell)
        for x,yaw in ((1.,0.),(1+ell/2,np.pi/4),(1+ell,np.pi/2)):
            assert p[np.argmin(abs(p[:,0]-x)),2]==pytest.approx(yaw)
    p=orientation_ramp_path(0.)
    poses=np.array([[.999,0.,0.],[1.001,0.,np.pi/2]])
    np.testing.assert_allclose(project_reference(poses,p)[2],0,atol=1e-12)
    for invalid in (-1,float('nan')):
        with pytest.raises(ValueError):
            orientation_ramp_path(invalid)


@pytest.mark.parametrize('method',['vp','dwvp'])
def test_ideal_omni_orbit_matches_exact_integral_with_timestep_convergence(method):
    L,e0=.33,.25
    errors=[]
    for frequency in (150.,600.):
        c=Config(frequency=frequency,timeout=2.,fixed_lookahead=L,ax=1e4,ay=1e4,aw=1e4,approach_distance=0.)
        r=simulate(straight_path(),method,c,initial_pose=[0.,e0,0.])
        x,e=r.arrays['poses'][1:,:2].T
        root0=math.sqrt(L*L-e0*e0)
        root=np.sqrt(L*L-e*e)
        exact_x=root0-root+L*np.log(e0/e*(L+root)/(L+root0))
        errors.append(float(np.max(np.abs(exact_x-x))))
    assert errors[1]<.001
    assert errors[1]<.3*errors[0]


def test_ray_selection_can_exceed_unit_scale():
    config=Config(ax=100.,ay=100.,aw=100.)
    pose=np.array([0.,0.,np.pi/4])
    path=straight_path()
    path[:,2]=np.pi/4
    output=compute_command(pose,np.zeros(3),path,path[:,0],'dwvp',config)
    assert np.linalg.norm(output.command[:2])>config.translation_speed


def test_noise_and_selection_grid_keep_all_twenty_seeds():
    profile=yaml.safe_load((ROOT/'configs/access_v2.yaml').read_text())
    settings=profile.pop('study')
    conditions,_=plan_conditions(Config(**profile),settings,0,tuple(f'test{i}' for i in range(1,5)))
    assert not any(c['part'] in ('d','f','g') for c in conditions)
    optional,_=plan_conditions(Config(**profile),settings,0,('preview-noise',))
    g=[c for c in optional if c['part']=='g']
    e=[c for c in conditions if c['part']=='e']
    assert len(g)==11*5*3*20 and len(e)==10*5*20
    assert set(c['seed'] for c in g)==set(range(20))
    assert len([c for c in conditions if c['test']=='test1'])==16
    assert len([c for c in conditions if c['test']=='test2'])==176
    assert len([c for c in conditions if c['test']=='test3'])==4
    assert {c['method'] for c in conditions if c['test']=='test3'}=={'rpp','dwvp'}
    assert all(c['config']['approach_distance']>0 for c in conditions)
    assert len([c for c in conditions if c['test']=='test2' and c['part']=='step'])==4
    assert set(c['part'] for c in conditions if c['test']=='test4')==set('bce')
    assert len(conditions)==1406
    optional,_=plan_conditions(Config(**profile),settings,0,('regulation-sweep',))
    assert len(optional)==36 and {c['part'] for c in optional}=={'d'}
    sweep=[c for c in conditions if c['test']=='test2' and c['part']=='acceleration']
    assert len(sweep)==136
    assert {c['method'] for c in sweep}=={'vp','vp_scaled','vp_scaled_accel','dwvp'}
    for c in sweep:
        assert c['config']['aw']==pytest.approx(profile['aw']*c['value'])
        scale=c['value'] if c['parameter']=='acceleration_scale' else 1.
        assert c['config']['ax']==pytest.approx(profile['ax']*scale)
        assert c['config']['ay']==pytest.approx(profile['ay']*scale)
        if c['parameter']=='angular_acceleration_scale':
            assert c['scenario']['transition_length']==.3


def test_aggregate_retains_timeout_metrics_and_missing_counts():
    rows=[dict(method='vp',success=True,status='success',evaluation_complete=True,duration_s=2.,crossing_m=.1),
          dict(method='vp',success=False,status='timeout',evaluation_complete=False,duration_s=120.,crossing_m=.3),
          dict(method='vp',success=False,status='error',evaluation_complete=False)]
    result=aggregate(rows,('method',))[0]
    assert (result['n'],result['success_count'],result['timeout_count'],result['failure_count'])==(3,1,1,1)
    assert result['duration_s_mean']==61.
    assert result['crossing_m_mean']==pytest.approx(.2)
    assert result['crossing_m_n']==2
    assert result['crossing_m_std']==pytest.approx(np.std([.1,.3],ddof=1))


def test_failed_trial_saved_and_cache_identity_includes_initial_pose(tmp_path,monkeypatch):
    from omnidirectional_dwvp import access_studies
    profile=yaml.safe_load((ROOT/'configs/access_v2.yaml').read_text());settings=profile.pop('study')
    conditions,_=plan_conditions(Config(**profile),settings,0,('test1',))
    def fail(*args,**kwargs):
        raise FloatingPointError('injected test failure')
    monkeypatch.setattr(access_studies,'simulate',fail)
    a=execute_trial(conditions[0],tmp_path,numerical_hash())
    assert a[0]['status']=='error' and not a[0]['success']
    assert execute_trial(conditions[0],tmp_path,numerical_hash())[2]
    b=execute_trial(conditions[3],tmp_path,numerical_hash())
    assert b[0]['trial_id']!=a[0]['trial_id']
    assert (tmp_path/'trials'/a[0]['trial_id']/'trial.json').exists()


def test_clipped_vp_nonintersection_is_geometric_not_solver_mode():
    config=Config(lookahead_time=.75)
    scene={'kind':'ramp','transition_length':.25}
    result=simulate(orientation_ramp_path(.25),'vp',config)
    metrics=evaluate(result,config,scene,[0.,0.,0.],3.25,1.)
    assert metrics['eval_projection_steps']==0
    assert metrics['eval_nonintersection_steps']>0
    assert np.any(~result.arrays['ray_intersects_box'])


def test_evaluation_window_precedes_swept_goal_regulation():
    profile=yaml.safe_load((ROOT/'configs/access_v2.yaml').read_text());settings=profile.pop('study')
    conditions,_=plan_conditions(Config(**profile),settings,0,('regulation-sweep',))
    condition=next(c for c in conditions if c['parameter']=='approach_distance' and c['value']==1.)
    assert condition['evaluation_end']==pytest.approx(2.995)


def test_dwvp_always_selects_fastest_point_inside_stopping_distance():
    config=Config(approach_distance=0.)
    path=straight_path()
    current=np.array([.22,0.,0.])
    # 0.05 m remains: outside goal tolerance, inside stopping distance 0.11 m.
    output=compute_command(np.array([3.95,0.,0.]),current,path,path[:,0],'dwvp',config)
    assert output.mode=='intersection'
    assert output.command[0]==pytest.approx(.22)
    regulated=compute_command(np.array([3.95,0.,0.]),current,path,path[:,0],'dwvp',replace(config,approach_distance=.6))
    assert regulated.command[0]<output.command[0]


def test_dwvp_nonintersection_tie_selects_larger_alpha():
    from omnidirectional_dwvp.controller import ray_command
    command,mode=ray_command(np.array([1.,0.,0.]),np.array([.1,.2,0.]),np.array([.3,.4,0.]),True)
    assert mode=='projection'
    np.testing.assert_allclose(command,[.3,.2,0.])


def test_position_error_uses_path_distance_and_reports_mean():
    config=Config(timeout=.2)
    result=simulate(straight_path(),'dwvp',config,initial_pose=[0.,.5,0.])
    metrics=evaluate(result,config,{'kind':'offset'},[0.,.5,0.],3.25,1.)
    errors=result.arrays['position_errors']
    assert metrics['eval_max_position_error_m']==pytest.approx(errors.max())
    integral=np.sum(.5*(errors[1:]+errors[:-1])*np.diff(result.arrays['times']))
    assert metrics['eval_position_error_integral_m_s']==pytest.approx(integral)
    assert metrics['eval_mean_position_error_m']==pytest.approx(integral/result.arrays['times'][-1])
    assert metrics['eval_sample_mean_position_error_m']==pytest.approx(errors.mean())


@pytest.mark.parametrize('vector, lower, upper, expected', [
    ([0.,0.,0.], [-1.,-1.,-1.], [1.,1.,1.], [0.,0.,0.]),
    ([.4,-.4,1.2], [-.2,-.4,-.6], [.2,.4,.6], [.2,-.2,.6]),
    ([-.4,.4,-1.2], [-.1,-.4,-.6], [.2,.4,.6], [-.1,.1,-.3]),
    ([.4,0.,1.2], [-1.,-1.,-1.], [0.,1.,1.], [0.,0.,0.]),
])
def test_uniform_scaling_handles_asymmetric_and_zero_limits(vector, lower, upper, expected):
    from omnidirectional_dwvp.controller import uniformly_scale
    np.testing.assert_allclose(uniformly_scale(np.array(vector), np.array(lower), np.array(upper)), expected)


@pytest.mark.parametrize('method, expected', [
    ('vp_scaled', [.1,-.08,.18]),
    ('vp_scaled_accel', [.08666666666666667,-.06,.18]),
])
def test_scaled_command_uses_regulated_velocity_box_before_acceleration(monkeypatch, method, expected):
    from omnidirectional_dwvp import controller
    config=Config(frequency=10.,vx_min=-.2,vx_max=.2,vy_min=-.4,vy_max=.4,ax=.2,ay=.4,aw=.6)
    monkeypatch.setattr(controller,'desired_vector',lambda *args:np.array([.4,-.4,1.2]))
    monkeypatch.setattr(controller,'speed_cap',lambda *args:(config.box_speed*.5,float('inf')))
    path=straight_path()
    out=compute_command(np.zeros(3),np.array([.08,-.04,.12]),path,path[:,0],method,config)
    np.testing.assert_allclose(out.command,expected,atol=1e-15)
    np.testing.assert_array_equal(out.desired,[.4,-.4,1.2])


@pytest.mark.parametrize('method', ['vp_scaled','vp_scaled_accel'])
def test_scaled_commands_remain_reachable_after_abrupt_regulation_and_at_goal(method):
    from omnidirectional_dwvp.controller import dynamic_box
    from omnidirectional_dwvp.geometry import Obstacle
    config=Config(use_cost_regulation=True)
    path=straight_path()
    rng=np.random.default_rng(82)
    for _ in range(40):
        current=rng.uniform(config.lower,config.upper)
        pose=np.r_[rng.uniform([1.,-.2],[3.8,.2]),rng.uniform(-np.pi,np.pi)]
        obstacles=(Obstacle(pose[0],pose[1]-.11,.1),)
        out=compute_command(pose,current,path,path[:,0],method,config,obstacles)
        lo,hi=dynamic_box(current,config)
        assert np.all(out.command>=lo-1e-12) and np.all(out.command<=hi+1e-12)
        assert np.all(out.command>=out.lower-1e-12) and np.all(out.command<=out.upper+1e-12)
    pose=np.array([3.99,0.,.2])
    current=np.array([.05,-.01,.1])
    out=compute_command(pose,current,path,path[:,0],method,config)
    vp=compute_command(pose,current,path,path[:,0],'vp',config)
    assert out.mode==vp.mode=='terminal'
    np.testing.assert_array_equal(out.command,vp.command)
    np.testing.assert_array_equal(out.desired,vp.desired)


def test_signed_heading_metrics_distinguish_lead_lag_and_post_transition_overshoot():
    config=Config(timeout=.1)
    result=simulate(orientation_ramp_path(.3),'vp',config)
    a=result.arrays
    a['poses'][:,0]=[.9,1.15,1.31,1.4]
    a['reference_poses'][:,2]=np.deg2rad([0.,45.,90.,90.])
    a['poses'][:,2]=np.deg2rad([10.,20.,100.,85.])
    a['yaw_errors']=np.deg2rad([10.,25.,10.,5.])
    m=evaluate(result,config,{'transition_length':.3},[0.,0.,0.],3.25,1.)
    np.testing.assert_allclose(np.rad2deg(a['signed_yaw_errors']),[10.,-25.,10.,-5.])
    assert m['eval_max_heading_lead_deg']==pytest.approx(10.)
    assert m['eval_max_heading_lag_deg']==pytest.approx(25.)
    assert m['post_transition_heading_overshoot_deg']==pytest.approx(10.)
    assert m['eval_max_heading_error_deg']==pytest.approx(25.)
    assert m['transition_heading_lag_deg']==pytest.approx(25.)
    # Post-transition lag must not contaminate the transition-only measure.
    a['poses'][-1,2]=np.deg2rad(20.)
    m=evaluate(result,config,{'transition_length':.3},[0.,0.,0.],3.25,1.)
    assert m['eval_max_heading_lag_deg']==pytest.approx(70.)
    assert m['transition_heading_lag_deg']==pytest.approx(25.)


def test_time_integral_handles_stationary_motion_nonuniform_samples_and_gaps():
    times=np.array([0.,1.,3.,4.,7.])
    errors=np.array([2.,-4.,6.,100.,8.])
    duration,integral,mean=time_error_metrics(times,errors,np.ones(5,dtype=bool))
    assert duration==7.
    assert integral==pytest.approx(3.+10.+53.+162.)
    assert mean==pytest.approx(228./7.)
    duration,integral,mean=time_error_metrics(times,errors,np.array([True,True,False,True,True]))
    assert (duration,integral,mean)==pytest.approx((4.,165.,41.25))
    assert time_error_metrics(times,errors,np.array([True,False,False,False,False]))==(0.,0.,None)


@pytest.mark.parametrize('scale, method, expected', [
    (.25,'vp',43.0),(.25,'vp_scaled',52.1),(.25,'dwvp',8.8),
    (.5,'vp',8.5),(.5,'vp_scaled',16.6),(.5,'dwvp',3.7),
    (1.,'vp',0.),(1.,'vp_scaled',0.),(1.,'dwvp',0.),
])
def test_saved_brief_overshoot_values_reproduced_by_unchanged_controllers(scale,method,expected):
    profile=yaml.safe_load((ROOT/'configs/access_v2.yaml').read_text());profile.pop('study')
    # The restored profile preserves the historical 0.33 m cap.
    cfg=replace(Config(**profile),lookahead_max=.33,ax=profile['ax']*scale,ay=profile['ay']*scale,aw=profile['aw']*scale)
    result=simulate(orientation_ramp_path(.3),method,cfg)
    metrics=evaluate(result,cfg,{'transition_length':.3},[0.,0.,0.],3.25,1.)
    assert metrics['post_transition_heading_overshoot_deg']==pytest.approx(expected,abs=.051)
    prediction=heading_braking_prediction(result.arrays,cfg,3.25)
    assert prediction['heading_braking_remaining_deg']>0
    assert prediction['heading_braking_T_s']>=.2
    if method=='vp_scaled' and scale==.25:
        assert prediction['heading_braking_omega_rad_s']==pytest.approx(.6)
        assert prediction['heading_braking_T_s']==pytest.approx(.5)
        assert prediction['predicted_heading_overshoot_deg']==pytest.approx(51.5662,abs=.001)


def test_baseline_comparison_keeps_existing_position_error_and_checks_missing_conditions(tmp_path):
    from omnidirectional_dwvp.access_report import compare_baseline
    spec=dict(method='vp',numerical_source_sha256='before')
    summary=dict(test='test1',part='nominal',scenario='offset',method='vp',parameter='',value=None,
                 seed=0,condition_id='existing',trial_id='before',eval_max_position_error_m=.25)
    old=dict(source_sha256='before',trials=[dict(spec=spec,summary=summary,condition_id='existing')])
    baseline=tmp_path/'baseline.json';baseline.write_text(json.dumps(old))
    current=dict(spec=dict(spec,numerical_source_sha256='after'),summary=dict(summary,trial_id='after'),condition_id='existing')
    comparison=compare_baseline(tmp_path,{'trials':[current]},baseline)
    assert comparison['changed_conditions']==0
    assert comparison['preserved_baseline_conditions']==1
    assert comparison['missing_baseline_conditions']==[]
    current['summary']['eval_max_position_error_m']=.3
    comparison=compare_baseline(tmp_path,{'trials':[current]},baseline)
    assert comparison['changed_conditions']==1
    assert comparison['reported_changes'][0]['metric']=='eval_max_position_error_m'


@pytest.mark.parametrize('offset', [-.05, 0., .05, .3])
def test_rpp_pp_curvature_without_curvature_speed_regulation(offset):
    cfg=Config(fixed_lookahead=.2,ax=100.,aw=100.)
    path=straight_path()
    out=compute_command(np.array([0.,offset,0.]),np.zeros(3),path,path[:,0],'rpp',cfg)
    distance=max(.2,abs(offset))
    curvature=-2*offset/distance**2
    np.testing.assert_allclose(out.desired,[.22,0.,curvature*.22],atol=1e-14)
    dwpp=compute_command(np.array([0.,offset,0.]),np.zeros(3),path,path[:,0],'dwpp',cfg)
    assert out.desired[2]/out.desired[0]==pytest.approx(dwpp.desired[2]/dwpp.desired[0])


def test_rpp_cost_and_approach_regulation_and_component_clipping():
    from omnidirectional_dwvp.geometry import Obstacle
    cfg=Config(use_cost_regulation=True,fixed_lookahead=.11)
    path=straight_path()
    pose=np.array([1.,.05,0.])
    current=np.zeros(3)
    obstacle=(Obstacle(1.,-.2,.1),)
    out=compute_command(pose,current,path,path[:,0],'rpp',cfg,obstacle)
    assert out.speed_cap==pytest.approx(cfg.nominal_speed*.15/cfg.cost_scaling_dist)
    assert out.desired[0]==pytest.approx(cfg.vx_max*.15/cfg.cost_scaling_dist)
    np.testing.assert_array_equal(out.command,np.clip(out.desired,out.lower,out.upper))
    assert out.command[0]<out.desired[0] and abs(out.command[2])<abs(out.desired[2])
    assert out.mode=='clipping' and out.command[1]==0.
    fast=compute_command(pose,np.array([.15,0.,0.]),path,path[:,0],'rpp',cfg,obstacle)
    np.testing.assert_array_equal(fast.desired,out.desired)
    assert fast.command[0]>=.15-cfg.ax*cfg.dt-1e-12  # Reachable braking after abrupt regulation.
    goal=compute_command(np.array([3.7,0.,0.]),current,path,path[:,0],'rpp',cfg)
    assert goal.desired[0]==pytest.approx(cfg.vx_max*.3/cfg.approach_distance)


def test_rpp_terminal_braking_and_tangent_match_dwpp():
    path=straight_path(heading=.4)
    path[:,2]=1.7
    cfg=Config()
    pose=path[-1].copy();pose[2]=.2
    current=np.array([.05,0.,.1])
    arc=np.arange(len(path))*.005
    rpp=compute_command(pose,current,path,arc,'rpp',cfg)
    dwpp=compute_command(pose,current,path,arc,'dwpp',cfg)
    assert rpp.mode==dwpp.mode=='terminal'
    np.testing.assert_array_equal(rpp.desired,dwpp.desired)
    np.testing.assert_array_equal(rpp.command,dwpp.command)
    assert rpp.desired[0]==0. and rpp.command[0]>0.


def test_rpp_records_unclipped_demand_violation():
    cfg=Config(timeout=2.,approach_distance=0.)
    result=simulate(straight_path(),'rpp',cfg)
    np.testing.assert_allclose(result.arrays['demands'][:,0],cfg.vx_max)
    assert result.metrics['unconstrained_demand_violation_pct']==pytest.approx(100*29/60)
    assert result.metrics['velocity_violation_pct']==result.metrics['acceleration_violation_pct']==0.
    metrics=evaluate(result,cfg,{'kind':'offset'},[0.,0.,0.],3.25,1.)
    assert metrics['demand_velocity_violation_steps']==0
    assert metrics['demand_acceleration_violation_steps']==metrics['demand_violation_steps']==29
    assert metrics['command_violation_steps']==0


def test_rpp_uses_regulated_x_limit_with_asymmetric_box_and_lower_nominal_cap():
    cfg=Config(vx_min=-.3,vx_max=.2,vy_min=-.4,vy_max=.4,desired_linear_vel=.25)
    path=straight_path()
    out=compute_command(np.zeros(3),np.zeros(3),path,path[:,0],'rpp',cfg)
    assert cfg.box_speed==.5
    assert out.desired[0]==pytest.approx(.1)
    # A stationary physical box enters terminal handling without division by zero.
    stopped=replace(cfg,vx_min=0.,vx_max=0.,vy_min=0.,vy_max=0.)
    out=compute_command(np.zeros(3),np.zeros(3),path,path[:,0],'rpp',stopped)
    assert out.mode=='terminal' and np.isfinite(out.command).all()


def test_constraint_breakdown_counts_overlap_once_and_uses_applied_previous_velocity():
    from omnidirectional_dwvp.access_metrics import constraint_violation_counts
    cfg=Config()
    previous=np.array([[.22,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    velocities=np.array([[.221,0.,0.],[.02,0.,0.],[0.,0.,.7],[0.,0.,0.]])
    assert constraint_violation_counts(velocities,previous,cfg)==dict(
        velocity_violation_steps=2,acceleration_violation_steps=2,both_violation_steps=1,violation_steps=3)


def test_saved_trajectory_reuse_rejects_active_cap_and_changed_inputs():
    from omnidirectional_dwvp.access_studies import reusable_lookahead
    cfg=Config(lookahead_max=.33)
    new=replace(cfg,lookahead_max=.165)
    assert reusable_lookahead({'lookahead':np.array([.11,.165])},cfg,new)
    assert not reusable_lookahead({'lookahead':np.array([.17])},cfg,new)
    assert not reusable_lookahead({'lookahead':np.array([.11])},cfg,replace(new,ax=.1))
    assert reusable_lookahead({'lookahead':np.array([.44])},replace(cfg,fixed_lookahead=.44),replace(new,fixed_lookahead=.44))
