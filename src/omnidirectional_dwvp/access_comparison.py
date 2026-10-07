"""Old/new manuscript tables for the full-run evaluation change."""
import json
from .access_metrics import COMMON_METRICS, aggregate
from .access_noise import preview_noise_summary, travel_time_recovery
from .config import Config
from .studies import ROOT, sha


def comparison_lines(output, rows, manifest):
    audit = manifest['full_run_evaluation']
    baseline = json.loads((ROOT/audit['baseline_manifest']).read_text())
    old = {sha({**t['condition'], 'evaluation_end': None}): t['summary'] for t in baseline['trials']}
    lines = ['', '## Full-run evaluation: old → new manuscript values', '',
             'Old: saved samples with 0 ≤ x ≤ 3.25 m. New: all saved samples from t=0 through the simulator goal criterion, including terminal settling, or through timeout. Each cell gives old → new; — denotes undefined. Values are rounded only for display.',
             'Travel time, timeout/success flags, constraint shares and counts, Test 3 near-obstacle speed, Test 2 section-local lag, and the Test 4 matched-time search are unchanged by construction. Error maxima, means, integrals, crossing and settling were recomputed. All original NPZ bytes and trial IDs are preserved; condition IDs reflect the removal of the spatial cutoff.',
             'C = Clipped VP, S = Scaled VP, A = Scaled VP (vel. and acc.), D = DWVP, P = DWPP. Tuple order is stated for each table. Position units: m and m s; orientation units: deg and deg s; times: s.']
    labels = dict(vp='C', vp_scaled='S', vp_scaled_accel='A', dwvp='D', dwpp='P', rpp='RPP')
    def pair(left, right, digits=2):
        fmt = lambda v: '—' if v is None else f'{v:.{digits}f}'
        return fmt(left)+' → '+fmt(right)
    def cell(row, key, digits=2):
        return pair(old[row['condition_id']].get(key), row.get(key), digits)
    def table(title, headers, data):
        lines.extend(['', '### '+title, '', '| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|'])
        lines.extend('| '+' | '.join(map(str, row))+' |' for row in data)
    position = [('eval_max_position_error_m',4), ('eval_mean_position_error_m',4), ('eval_position_error_integral_m_s',4)]
    orientation = [('eval_max_heading_error_deg',2), ('eval_mean_heading_error_deg',2), ('eval_heading_error_integral_deg_s',2)]
    table('Test 1: each initial offset', ['Offset', 'Method', 'Position max', 'Position mean', 'Position integral',
          'Orientation max', 'Orientation mean', 'Orientation integral', 'Travel', 'Crossing', 'Settling'],
          [[r['offset_m'],labels[r['method']],*[cell(r,k,d) for k,d in position+orientation],cell(r,'travel_time_s'),
            cell(r,'crossing_m',5),cell(r,'settling_2pct_time_s')] for r in rows if r['test']=='test1'])
    heading_metrics = orientation + [('transition_heading_lag_deg',2), ('post_transition_heading_overshoot_deg',2), ('travel_time_s',2)]
    table('Test 2: default setting by section length (including the auxiliary method)',
          ['Length','Method','Orientation max','Orientation mean','Orientation integral','Lag','Overshoot','Travel'],
          [[r['transition_length_m'],labels[r['method']],*[cell(r,k,d) for k,d in heading_metrics]]
           for r in rows if r['test']=='test2' and r['part'] in ('nominal','step')])
    sweep = [r for r in rows if r['test']=='test2' and r['parameter']=='acceleration_scale' and r['method'] in ('vp','vp_scaled','dwvp')]
    grid = []
    for ell, scale in dict.fromkeys((r['transition_length_m'],r['value']) for r in sweep):
        group = {r['method']:r for r in sweep if (r['transition_length_m'],r['value'])==(ell,scale)}
        grid.append([f'{ell:g}',f"{group['vp']['acceleration_time_w_s']/group['vp']['lookahead_time_s']:.2f}",
                     *[' / '.join(cell(group[m],k,d) for m in ('vp','vp_scaled','dwvp')) for k,d in heading_metrics]])
    table('Test 2: length × ratio grid; each cell C / S / D',
          ['Length','Ratio','Orientation max','Orientation mean','Orientation integral','Lag','Overshoot','Travel'],grid)
    table('Test 3: obstacle speed regulation', ['Cost','Method','Position max','Position mean','Position integral',
          'Orientation max','Orientation mean','Orientation integral','Travel','Near speed','Constraint share [%]'],
          [[r['value'],labels[r['method']],*[cell(r,k,d) for k,d in position+orientation],cell(r,'travel_time_s'),
            cell(r,'mean_near_obstacle_speed_m_s',4),cell(r,'unconstrained_demand_violation_pct' if r['method']=='rpp' else 'command_constraint_violation_pct',3)]
           for r in rows if r['test']=='test3'])
    selected_ids = {v['best_trial_id'] for v in manifest['time_matches'].values()}
    matched = [r for r in rows if r['test']=='test4' and r['part']=='a' and
               (r['parameter']=='match_reference' or r['trial_id'] in selected_ids)]
    table('Test 4: selected matched-travel-time runs (not means over search candidates)',
          ['Method','Speed scale','Position max','Position mean','Position integral','Orientation max','Orientation mean',
           'Orientation integral','Lag','Overshoot','Travel'],
          [[labels[r['method']],pair(old[r['condition_id']]['value'],r['value'],6),
            *[cell(r,k,d) for k,d in position+heading_metrics]] for r in matched])
    fixed = [r for r in rows if r['test']=='test4' and r['parameter']=='fixed_lookahead']
    data=[]
    for length in sorted({r['value'] for r in fixed}):
        group={(r['scenario'],r['method']):r for r in fixed if r['value']==length}
        data.append([f'{length:g}',
                     *[' / '.join(cell(group[scene,m],key,d) for m in methods)
                       for scene,key,d,methods in (
                         ('offset','crossing_m',5,('dwpp','vp','vp_scaled','dwvp')),
                         ('offset','eval_max_heading_error_deg',2,('dwpp','vp','vp_scaled','dwvp')),
                         ('gradual','eval_max_heading_error_deg',2,('vp','vp_scaled','dwvp')),
                         ('rapid','eval_max_heading_error_deg',2,('vp','vp_scaled','dwvp')))]])
    table('Test 4: fixed lookahead; offset P / C / S / D, ramps C / S / D',
          ['L [m]','Offset crossing','Offset orientation max','Gradual orientation max','Rapid orientation max'],data)
    noise = [r for r in rows if r['test']=='test4' and r['part']=='c']
    keys=('scenario','method','noise_xy_m','noise_yaw_deg')
    grouped=aggregate(noise,keys)
    previous={tuple(g[k] for k in keys):g for g in aggregate([old[r['condition_id']] for r in noise],keys)}
    data=[]
    for scene,xy,yaw in dict.fromkeys((r['scenario'],r['noise_xy_m'],r['noise_yaw_deg']) for r in grouped):
        group={r['method']:r for r in grouped if (r['scenario'],r['noise_xy_m'],r['noise_yaw_deg'])==(scene,xy,yaw)}
        methods=('dwpp','vp','vp_scaled','dwvp') if scene=='offset' else ('vp','vp_scaled','dwvp')
        data.append([scene,f'{xy:g}/{yaw:g}',
                     ' / '.join(pair(previous[scene,m,xy,yaw]['travel_time_s_mean'],group[m]['travel_time_s_mean']) for m in methods),
                     ' / '.join(f"{group[m]['success_count']}/{group[m]['n']}" for m in methods)])
    table('Test 4: noise travel times; offset P / C / S / D, ramps C / S / D',
          ['Scenario','σxy / σorientation','Mean travel time','Successes (unchanged)'],data)
    grouped=preview_noise_summary(rows)
    prior=preview_noise_summary([old[r['condition_id']] for r in rows])
    keys=('scenario','parameter','value','noise_xy_m','noise_yaw_deg')
    previous={tuple(r[k] for k in keys):r for r in prior}
    previews=list(dict.fromkeys((r['parameter'],r['value']) for r in grouped))
    levels=sorted({(r['noise_xy_m'],r['noise_yaw_deg']) for r in grouped})
    for scene in ('offset','gradual','rapid'):
        group={(r['parameter'],r['value'],r['noise_xy_m'],r['noise_yaw_deg']):r for r in grouped if r['scenario']==scene}
        data=[]
        for xy,yaw in levels:
            data.append([f'{xy:g}/{yaw:g}',*[pair(previous[scene,p,v,xy,yaw]['travel_time_s_mean'],group[p,v,xy,yaw]['travel_time_s_mean']) for p,v in previews]])
        table(f'Test 4: preview-noise travel times, {scene}',
              ['σxy / σorientation',*[f"{'L' if p=='fixed_lookahead' else 'T'}={v:g}" for p,v in previews]],data)
    lines.extend(['', 'Preview-noise successes are unchanged: 0/20 at σxy=0.02 m with fixed L=0.055 m in each scenario; 20/20 at every other setting. — travel times exclude those failed runs.'])
    recovery=travel_time_recovery(grouped,Config(**manifest['config']))
    prior_recovery=travel_time_recovery(prior,Config(**baseline['config']))
    key=lambda r:(r['scenario'],r['noise_xy_m'],r['noise_yaw_deg'],r['reference'])
    old_recovery={key(r):r for r in prior_recovery}
    new_recovery={key(r):r for r in recovery}
    data=[]
    for scene,xy,yaw in dict.fromkeys((r['scenario'],r['noise_xy_m'],r['noise_yaw_deg']) for r in recovery):
        nominal=new_recovery[scene,xy,yaw,'nominal_preview']
        data.append([scene,f'{xy:g}/{yaw:g}',*[pair(old_recovery[scene,xy,yaw,ref]['minimum_fixed_lookahead_m'],
                     new_recovery[scene,xy,yaw,ref]['minimum_fixed_lookahead_m'],3) for ref in ('nominal_preview','same_fixed_lookahead')],
                     pair(old_recovery[scene,xy,yaw,'nominal_preview']['estimate_lookahead_m'],nominal['estimate_lookahead_m'],3)])
    table('Test 4: L = 30σ check (minimum sampled L within 1.1× zero-noise time, both 20/20 successes)',
          ['Scenario','σxy / σorientation','Nominal-preview reference','Same-L reference','30σ estimate [m]'],data)
    lines.extend(['', '### Validation and regenerated artifacts', '',
                  f"Re-aggregated {audit['conditions']} conditions / {audit['distinct_trajectories']} saved trajectories; simulation reruns: 0. SHA-256 checks and all metric differences are in `full_run_evaluation.json` and `full_run_changes.csv`. Historical comparison CSVs remain historical evidence.",
                  'The complete regenerated file list is below (paths relative to results/access_v2). Per-trial trial.json metadata were also refreshed; trajectory.npz files were not changed.', '', '```text',
                  *(output/'regenerated_files.txt').read_text().splitlines(), '```'])
    validation_path=output/'full_run_checks.json'
    if validation_path.exists():
        checks=json.loads(validation_path.read_text())
        lines.extend(['', 'pytest: '+checks['pytest']+'.', 'Validation: '+checks['validation']+'.'])
    else:
        lines.extend(['', 'pytest and independent artifact validation: pending; the final report is refreshed after both checks.'])
    return lines
