from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.controller import desired_vector, dynamic_box, regulated_box, ray_command, compute_command
from omnidirectional_dwvp.geometry import integrate, Obstacle, swept_clearance
from omnidirectional_dwvp.paths import make_paths
from omnidirectional_dwvp.simulation import simulate


def test_projection_against_independent_numerical_oracle():
    rng=np.random.default_rng(20261003)
    for _ in range(500):
        lower=rng.uniform(-1, .8, 3)
        upper=lower+rng.uniform(.01, 1, 3)
        ray=rng.normal(size=3)
        if rng.random()<.4: ray[rng.integers(3)]=0
        cmd,_=ray_command(ray,lower,upper,True)
        assert np.all(cmd>=lower-1e-10) and np.all(cmd<=upper+1e-10)
        nonzero=np.abs(ray)>1e-12
        bound=max(1.,float(np.max(np.maximum(np.abs(lower[nonzero]),np.abs(upper[nonzero]))/np.abs(ray[nonzero])))*2)
        def objective(alpha): return np.sum((alpha*ray-np.clip(alpha*ray,lower,upper))**2)
        oracle=minimize_scalar(objective,bounds=(0,bound),method='bounded',options={'xatol':1e-12})
        # Distance of selected box point to its nearest ray point, evaluated independently.
        alpha=max(0.,float(cmd@ray/(ray@ray)))
        loss=float(np.sum((cmd-alpha*ray)**2))
        assert loss <= min(objective(0.),oracle.fun)+1e-8


def test_ray_scaling_preserves_command_and_not_a_speed_regulator():
    lo=np.array([.10,-.01,-.02]);hi=np.array([.12,.01,.02]);ray=np.array([.3,.2,1.])
    a,_=ray_command(ray,lo,hi,True)
    b,_=ray_command(ray*.05,lo,hi,True)
    np.testing.assert_allclose(a,b,atol=1e-10)


def test_abrupt_cap_decelerates_inside_original_window():
    config=Config()
    current=np.array([.2,-.18,.3])
    lo,hi=dynamic_box(current,config)
    rl,rh=regulated_box(lo,hi,.01,config)
    assert np.all(rl>=lo) and np.all(rh<=hi)
    assert rl[0]==rh[0]==lo[0]
    assert rl[1]==rh[1]==hi[1]
    assert rl[2]==lo[2] and rh[2]==hi[2]
    command,_=ray_command(np.array([1.,-1.,.3]),rl,rh,True)
    assert np.all(np.abs(command-current)<=config.acceleration*config.dt+1e-10)


def test_zero_goal_pure_yaw_and_signed_directions():
    c=Config()
    np.testing.assert_allclose(desired_vector(np.zeros(3),np.zeros(3),c),np.zeros(3))
    yaw=desired_vector(np.zeros(3),np.array([0.,0.,np.pi/2]),c)
    assert np.all(np.isfinite(yaw)) and yaw[2]>0 and np.all(yaw[:2]==0)
    path=np.array([[0.,0.,0.],[0.,0.,np.pi/2]])
    r=simulate(path,'dwvp',c)
    assert r.metrics['success']
    assert r.metrics['acceleration_violation_pct']==0
    assert np.max(np.abs(r.arrays['applied'][-1])) <= 1e-3
    for ray in (np.zeros(3),np.array([-1.,0.,0.]),np.array([0.,-1.,1.])):
        lo,hi=dynamic_box(np.zeros(3),c)
        command,_=ray_command(ray,lo,hi,True)
        assert np.all(np.isfinite(command)) and np.all(command>=lo) and np.all(command<=hi)


def test_true_zero_translation_limits_allow_rotation():
    c=replace(Config(),vx_min=0.,vx_max=0.,vy_min=0.,vy_max=0.)
    assert c.box_speed==0
    path=np.array([[0.,0.,0.],[0.,0.,.3]])
    r=simulate(path,'dwvp',c)
    assert r.metrics['success']
    assert np.all(r.arrays['applied'][:,:2]==0)


def test_identical_boxes_both_baselines_and_deterministic_trials():
    c=replace(Config(),noise_xy=.002,noise_yaw=.001)
    path=make_paths()['constant_heading_corner']
    a=simulate(path,'dwvp',c,seed=42)
    b=simulate(path,'dwvp',c,seed=42)
    assert a.metrics==b.metrics
    for key in a.arrays: np.testing.assert_array_equal(a.arrays[key],b.arrays[key])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))]
    vp=compute_command(np.array([.8,.0,0]),np.array([.12,.0,.1]),path,arc,'vp',c)
    dw=compute_command(np.array([.8,.0,0]),np.array([.12,.0,.1]),path,arc,'dwvp',c)
    np.testing.assert_array_equal(vp.lower,dw.lower)
    np.testing.assert_array_equal(vp.upper,dw.upper)
    np.testing.assert_array_equal(vp.desired,dw.desired)


def test_timeout_retained_and_nominal_cap():
    c=replace(Config(),timeout=.1)
    r=simulate(make_paths()['iros_curve'],'dwvp',c)
    assert r.metrics['timeout'] and not r.metrics['success'] and r.metrics['travel_time_s'] is None
    assert r.metrics['duration_s']==pytest.approx(.1)
    c=replace(Config(),desired_linear_vel=.08)
    r=simulate(make_paths()['constant_heading_corner'],'dwvp',c)
    assert r.metrics['max_translation_speed_m_s'] <= .08+1e-10


def test_exact_body_twist_and_swept_circle_collision():
    end=integrate(np.zeros(3),np.array([1.,0.,1.]),np.pi/2)
    np.testing.assert_allclose(end,[1.,1.,np.pi/2],atol=1e-12)
    clear=swept_clearance(np.zeros(3),np.array([2.,0.,0.]),np.array([2.,0.,0.]),1.,[Obstacle(1.,0.,.1)],.22)
    assert clear==pytest.approx(-.32)


def test_iros_actual_hardware_geometry():
    paths=make_paths()
    assert paths['iros_docking'].shape==(201,3)
    assert paths['iros_curve'].shape==(501,3)
    np.testing.assert_allclose(paths['iros_curve'][-1,:2],[1.5,1.5])
    np.testing.assert_allclose(paths['constant_heading_corner'][:,2],0.)
    assert paths['independent_heading_curve'][-1,2]==pytest.approx(np.pi/2)


def test_shared_plugin_fixtures():
    data=json.loads((Path(__file__).resolve().parents[1]/'fixtures/solver_cases.json').read_text())
    for item in data['cases']:
        c=Config(vx_min=item['physical_min'][0],vy_min=item['physical_min'][1],w_min=item['physical_min'][2],
                 vx_max=item['physical_max'][0],vy_max=item['physical_max'][1],w_max=item['physical_max'][2],
                 ax=item['acceleration'][0],ay=item['acceleration'][1],aw=item['acceleration'][2],
                 frequency=1/item['dt'],desired_linear_vel=10.)
        lo,hi=dynamic_box(np.array(item['current']),c)
        lo,hi=regulated_box(lo,hi,item['cap'],c)
        cmd,_=ray_command(np.array(item['desired']),lo,hi,item['prefer_large'])
        np.testing.assert_allclose(lo,item['expected_min'],atol=1e-9)
        np.testing.assert_allclose(hi,item['expected_max'],atol=1e-9)
        np.testing.assert_allclose(cmd,item['expected_command'],atol=1e-9)


def test_segment_projection_and_yaw_share_same_location():
    from omnidirectional_dwvp.metrics import project_reference
    path=np.array([[0.,0.,0.],[2.,0.,np.pi/2]])
    poses=np.array([[.5,.2,np.pi/8],[1.,-.3,np.pi/4]])
    reference,error,yaw_error,segment,fraction=project_reference(poses,path)
    np.testing.assert_allclose(reference[:,:2],[[.5,0.],[1.,0.]])
    np.testing.assert_allclose(error,[.2,.3])
    np.testing.assert_allclose(yaw_error,0.,atol=1e-12)
    np.testing.assert_array_equal(segment,[0,0])
    np.testing.assert_allclose(fraction,[.25,.5])


def test_segment_yaw_wrap_and_duplicate_points():
    from omnidirectional_dwvp.metrics import project_reference
    path=np.array([[0.,0.,np.deg2rad(170.)],[2.,0.,np.deg2rad(-170.)]])
    poses=np.array([[1.,.1,np.pi]])
    reference,error,yaw_error,_,_=project_reference(poses,path)
    assert abs(abs(reference[0,2])-np.pi)<1e-12
    np.testing.assert_allclose(yaw_error,0.,atol=1e-12)
    duplicate=np.vstack((path[0],path))
    assert np.all(np.isfinite(project_reference(poses,duplicate)[0]))


def test_cached_trial_reuses_only_identical_specification(tmp_path,monkeypatch):
    from omnidirectional_dwvp import studies
    c=replace(Config(),timeout=.1)
    path=make_paths()['constant_heading_corner']
    first,_,_=studies.run_trial(tmp_path,'p',path,'dwvp',c)
    def should_not_run(*args,**kwargs): raise RuntimeError('unexpected recalculation')
    monkeypatch.setattr(studies,'simulate',should_not_run)
    second,_,_=studies.run_trial(tmp_path,'p',path,'dwvp',c)
    assert first==second
    with pytest.raises(RuntimeError,match='recalculation'):
        studies.run_trial(tmp_path,'p',path,'dwvp',replace(c,lookahead_time=2.))


def test_sweep_axes_include_both_controllers(tmp_path,monkeypatch):
    from omnidirectional_dwvp import plotting
    import matplotlib.pyplot as plt
    inspected=[]
    def inspect_figure(fig,stem):
        for ax in fig.axes:
            lower,upper=ax.get_ylim()
            for line in ax.lines:
                y=np.asarray(line.get_ydata())
                assert np.all(y>=lower-1e-12) and np.all(y<=upper+1e-12)
        inspected.append(stem)
        plt.close(fig)
    monkeypatch.setattr(plotting,'save',inspect_figure)
    rows=[]
    for method,scale in [('vp',1.),('dwvp',3.)]:
        for value in (1.,2.):
            rows.append(dict(parameter='lookahead_time',path='iros_docking',method=method,value=value,success=True,
                             mean_position_error_m=scale*value,mean_heading_error_deg=scale*value,duration_s=scale*value))
    plotting.sweep_figures(tmp_path,rows)
    assert len(inspected)==1
