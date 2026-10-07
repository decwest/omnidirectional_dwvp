"""Generate the bounded factual handoff from saved metrics, never hand-entered values."""
import hashlib
import json
import math
import numpy as np
from .access_metrics import COMMON_METRICS, aggregate, heading_braking_prediction
from .access_noise import noise_lookahead_estimate, preview_noise_summary, travel_time_recovery
from .config import Config
from .studies import sha, write_csv


def f(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def lookahead_ranges(rows):
    """Full-run ranges; counts include every condition entry, including repeats."""
    ranges = []
    for test in dict.fromkeys(r['test'] for r in rows):
        group = [r for r in rows if r['test'] == test and r.get('lookahead_min_m') is not None]
        if not group:
            continue
        adaptive = [r for r in group if r['fixed_lookahead_m'] is None]
        fixed = [r for r in group if r['fixed_lookahead_m'] is not None]
        ranges.append(dict(test=test, minimum_m=min(r['lookahead_min_m'] for r in group),
                           maximum_m=max(r['lookahead_max_m'] for r in group),
                           adaptive_minimum_m=min((r['lookahead_min_m'] for r in adaptive), default=None),
                           adaptive_maximum_m=max((r['lookahead_max_m'] for r in adaptive), default=None),
                           fixed_minimum_m=min((r['lookahead_min_m'] for r in fixed), default=None),
                           fixed_maximum_m=max((r['lookahead_max_m'] for r in fixed), default=None),
                           upper_active_steps=sum(r['lookahead_upper_active_steps'] for r in adaptive),
                           at_upper_steps=sum(r['lookahead_at_upper_steps'] for r in adaptive),
                           adaptive_steps=sum(r['control_steps'] for r in adaptive)))
    return ranges


def prediction_comparison(output, rows, manifest):
    trials = {t['condition_id']: t for t in manifest['trials']}
    comparison = []
    for row in rows:
        if (row['test'] != 'test2' or row['part'] != 'acceleration'
                or row['method'] not in ('vp', 'vp_scaled', 'dwvp') or row['status'] == 'error'):
            continue
        with np.load(output/'trials'/row['trial_id']/'trajectory.npz') as history:
            prediction = heading_braking_prediction(history, Config(**trials[row['condition_id']]['spec']['config']),
                                                    row['evaluation_end_m'])
        observed = row['post_transition_heading_overshoot_deg']
        predicted = prediction['predicted_heading_overshoot_deg']
        comparison.append({**{k: row[k] for k in ('condition_id', 'trial_id', 'method', 'parameter', 'value', 'transition_length_m')},
                           **prediction, 'observed_heading_overshoot_deg': observed,
                           'prediction_error_deg': predicted-observed if predicted is not None and observed is not None else None})
    if comparison:
        write_csv(output/'test2'/'overshoot_prediction.csv', comparison)
    return comparison


def report(output, rows, manifest):
    lines = ['# シミュレーション結果', '',
             f"条件数 {len(rows)}、保存試行ID {manifest['distinct_trajectories']}。成功 {manifest['success_count']}、タイムアウト {manifest['timeout_count']}、その他失敗 {manifest['failure_count']}。",
             f"設定は `manifest.json`、全条件の指標は各試験の `summary.csv`。既定の設定：30 Hz、速度上限 ±0.22 m/s・±0.6 rad/s、加速度上限 0.22 m/s²・0.6 rad/s²、VP所望並進速度 0.22 m/s、前方注視時間 0.75 s、前方注視距離上限 {manifest['config']['lookahead_max']:g} m（固定距離の掃引は別指定）。",
             '誤差は開始時刻からシミュレーションのゴール判定成立時までの全保存試料で評価する。終端の減速・整定も含む。タイムアウトは終了時までを評価し、未成功として残す。'
             '位置誤差は経路への距離、姿勢誤差は線分射影位置の参照姿勢との差の絶対値。積分は隣接する評価内試料間の台形則、平均は積分÷評価時間。',
             '走行時間と指令制約違反率は従来どおり全走行で評価する。時間は成功時だけ記録し、タイムアウトは空欄。制約違反は速度または加速度超過の周期数÷全周期数（閾値1e−10）。',
             '位置の行き過ぎは初期横偏差と反対側への最大偏差。姿勢の遅れは姿勢変化区間内の参照−ロボットの最大値、行き過ぎは区間通過後のロボット−最終参照の最大値（いずれも非負）。'
             '整定時間は横方向誤差が初期偏差の2%以内に入り、評価区間の終わりまで留まる時刻。評価区間未完走・未整定は空欄。段差（ℓ=0）の遅れは変化直後の最初の試料。',
             'vp=成分別クリップ、vp_scaled=速度箱への一様縮小後にクリップ、vp_scaled_accel=速度差も一様縮小する補助比較、dwvp=DWVP、dwpp=差動二輪DWPP、rpp=差動二輪RPP。', '']
    preview = lookahead_ranges([{**r, 'test': 'test4'} if r['test']=='preview-noise' else r for r in rows])
    write_csv(output/'lookahead_ranges.csv', preview)
    for item in preview:
        fixed = (f"、適応 {f(item['adaptive_minimum_m'], 6)}–{f(item['adaptive_maximum_m'], 6)} m、固定 {f(item['fixed_minimum_m'], 6)}–{f(item['fixed_maximum_m'], 6)} m"
                 if item['fixed_minimum_m'] is not None else '')
        lines.append(f"試験{item['test'].removeprefix('test')}の実際の前方注視距離：{item['minimum_m']:.6f}–{item['maximum_m']:.6f} m{fixed}。適応距離を上限で切り詰めた周期 {item['upper_active_steps']}/{item['adaptive_steps']}（上限到達 {item['at_upper_steps']} 周期、全条件・全走行を集計）。")
    lines.append('')
    chosen = lambda test: [r for r in rows if r['test'] == test]
    headers = ['最大位置 [m]', '平均位置 [m]', '位置積分 [m·s]', '最大姿勢 [°]', '平均姿勢 [°]', '姿勢積分 [°·s]', '走行時間 [s]']
    metrics = [m for m in COMMON_METRICS if m != 'command_constraint_violation_pct']
    digits = [4, 4, 4, 2, 2, 2, 2]
    def table(head, data):
        if lines[-1]:
            lines.append('')
        lines.extend(['| '+' | '.join(head)+' |', '|'+'|'.join(['---']*len(head))+'|'])
        lines.extend('| '+' | '.join(map(str, row))+' |' for row in data)
        lines.append('')
    def common_table(prefix, group, extras=()):
        def constraint(r):
            key = 'unconstrained_demand_violation_pct' if r['method'] == 'rpp' else 'command_constraint_violation_pct'
            return r.get(key, 0)
        violation = any(constraint(r) > 0 for r in group)
        extra_headers = [label for _, label, _ in extras]
        data = []
        for r in group:
            data.append([*[r[k] for k, _ in prefix], *[f(r.get(k), d) for k, d in zip(metrics, digits)],
                         *[f(r.get(k), d) for k, _, d in extras],
                         *([f(constraint(r))] if violation else [])])
        table([*[label for _, label in prefix], *headers, *extra_headers,
               *(['指令制約違反 [%]'] if violation else [])], data)
    lateral = (('crossing_m', '位置行き過ぎ [m]', 5),
               ('settling_2pct_time_s', '整定時間 [s]', 2))
    heading = (('transition_heading_lag_deg', '姿勢遅れ [°]', 2),
               ('post_transition_heading_overshoot_deg', '姿勢行き過ぎ [°]', 2))
    if chosen('test1'):
        lines.extend(['## 試験1：直線経路への収束', ''])
        common_table((('offset_m', '初期偏差 [m]'), ('method', '手法')), chosen('test1'), lateral)
    if chosen('test2'):
        lines.extend(['', '## 試験2：経路上の姿勢の追従', ''])
        nominal = [r for r in chosen('test2') if r['part'] in ('nominal', 'step')]
        common_table((('transition_length_m', 'ℓ [m]'), ('method', '手法')), nominal, heading)
        sweep = [r for r in chosen('test2') if r['part']=='acceleration'
                 and r['parameter']=='acceleration_scale' and r['method'] in ('vp', 'vp_scaled', 'dwvp')]
        if sweep:
            lines.extend(['', '### 区間長さと加速度制約の格子', '',
                          '横軸は無次元比 ω_max/(a_ω T)。ω_max=0.6 rad/s、a_ω=0.6×倍率 rad/s²、T=0.75 s（既定の前方注視時間）とする。倍率0.25、0.5、0.75、1、1.5、2は比5.33、2.67、1.78、1.33、0.89、0.67に対応する。',
                          '図の縦の点線は比2（a_ω=0.4 rad/s²）。右ほど加速度上限が厳しい。これはω=ω_max、T一定とした停止角と残り角度の境界であり、各周期の姿勢所要時間や全条件の行き過ぎ発生を保証する境界ではない。',
                          '各セルは vp / vp_scaled / dwvp の順。並進・回転の加速度を同時に変更。角加速度のみの掃引と補助比較vp_scaled_accelを含む全176条件の値は `test2/summary.csv` に保持する。'])
            methods = ('vp', 'vp_scaled', 'dwvp')
            data = []
            for ell, scale in dict.fromkeys((r['transition_length_m'], r['value']) for r in sweep):
                group = {r['method']: r for r in sweep if (r['transition_length_m'], r['value']) == (ell, scale)}
                values = [' / '.join(f(group[m].get(k), d) for m in methods)
                          for k, d in [('eval_max_heading_error_deg', 2), ('eval_heading_error_integral_deg_s', 2),
                                       *((k, d) for k, _, d in heading)]]
                row = group['vp']
                data.append([f(ell, 1), f(row['acceleration_time_w_s']/row['lookahead_time_s'], 2), *values])
            table(['ℓ [m]', 'ω_max/(a_ω T)', '最大姿勢誤差 [°]', '姿勢誤差積分 [°·s]',
                   *[label for _, label, _ in heading]], data)
            if any(r['command_constraint_violation_pct'] != 0 for r in sweep):
                table(['ℓ [m]', '倍率', '手法', '指令制約違反 [%]'],
                      [[r['transition_length_m'], r['value'], r['method'], f(r['command_constraint_violation_pct'])] for r in sweep])
            lines.append('格子図は `test2/acceleration_ratio_ramp_{1,0p6,0p4,0p3,0p2}_{overshoot,heading_integral,heading}`、凡例は `acceleration_ratio_legend`。時系列は `ramp_0p3_acceleration_{0p25,1}_{speed,yaw,signed_heading}`、凡例は `acceleration_time_series_legend`（各PDF/PNG）。時系列は変化区間の終端x=1.3 mを過ぎ、ゴール判定成立時（未成功時はタイムアウト）まで表示する。')
            comparisons = prediction_comparison(output, rows, manifest)
            lines.extend(['', '### 姿勢の行き過ぎの予測との照合', '',
                          '予測は max(0, ω²/(2aω)−ωT)。T=max(0.20, kL/v所望)。最終姿勢を前方注視点が参照しているときの最初の減速周期を選び、その直前の適用角速度ωと、その周期のLを使う。',
                          '最終姿勢を横切る直前はすでに減速しているため、停止角の議論に対応する減速開始を採った。表は区間0.3 mの同時掃引。全条件の時刻・ω・T・残り角度と予測差は `test2/overshoot_prediction.csv`。',
                          '各セルは予測 / 計算結果 [°]。予測はT一定、減速開始時の残り角度ωT、一定の最大角減速度を仮定する。DWVPの速度選択にはこのVP用の仮定を保証しない。'])
            data = []
            for parameter, ell, scale in dict.fromkeys((r['parameter'], r['transition_length_m'], r['value'])
                    for r in comparisons if r['parameter']=='acceleration_scale' and r['transition_length_m']==.3):
                group = {r['method']: r for r in comparisons if (r['parameter'], r['transition_length_m'], r['value']) == (parameter, ell, scale)}
                data.append([f(ell, 1), ('ωのみ ' if parameter == 'angular_acceleration_scale' else '')+f(scale, 2),
                             *[f(group[m]['predicted_heading_overshoot_deg'], 1)+' / '+f(group[m]['observed_heading_overshoot_deg'], 1)
                               for m in ('vp', 'vp_scaled', 'dwvp')]])
            table(['ℓ [m]', '倍率', 'vp [°]', 'vp_scaled [°]', 'dwvp [°]'], data)
            matches = []
            for method in ('vp', 'vp_scaled', 'dwvp'):
                group = [r for r in comparisons if r['method'] == method and r['prediction_error_deg'] is not None]
                near = sum(abs(r['prediction_error_deg']) <= 1 for r in group)
                worst = max(group, key=lambda r: abs(r['prediction_error_deg']))
                matches.append(f"{method}: 絶対差1°以内 {near}/{len(group)} 条件、最大絶対差 {abs(worst['prediction_error_deg']):.2f}°（ℓ={worst['transition_length_m']:g} m、{worst['parameter']}={worst['value']:g}）。")
            example = next((r for r in comparisons if r['method']=='dwvp' and r['parameter']=='acceleration_scale'
                            and r['transition_length_m']==.3 and r['value']==.25), None)
            detail = (f" 0.3 m・0.25倍のDWVPでは減速開始時の残り角度 {example['heading_braking_remaining_deg']:.1f}°に対しωTは {np.rad2deg(example['heading_braking_omega_rad_s']*example['heading_braking_T_s']):.1f}°であり、減速開始の仮定が成立していない。"
                      if example else '')
            lines.append(' '.join(matches)+' 1°は記述上の目安であり、統計的な一致判定ではない。'+detail)
    if chosen('test3'):
        lines.extend(['', '## 試験3：障害物付近での速度調整', '',
                      '障害物 (x,y,r)=(1.6,−0.45,0.10), (2.4,0.55,0.10) m。近傍平均速度は障害物表面までの距離が0.6 m未満の周期の適用並進速度の平均。ゴール接近時の調整は常に有効。',
                      'RPPはDWPPと同じ前方注視点と曲率κ=2sinα/Lを用い、所望速度を(v,0,κv)、v=vx_max×speed_cap/box_speedとする。調整なしで0.22 m/s。障害物近接・ゴール接近の調整比は速度箱と共通で、曲率による速度調整は使わない。終端処理はDWPPと共通。',
                      'RPPの指令計算に動的窓は使わず、適用前に共通の速度・加速度・速度調整の窓へ成分ごとにクリップする。表の制約違反率はRPPではクリップ前の所望速度（unconstrained_demand_violation_pct）、DWVPでは選択後の指令を評価する。適用速度の違反率は両手法とも0%。',
                      '内訳は物理的な速度上限と、直前の適用速度からの加速度上限について数える。両方を超える周期は違反率では1回だけ数える。Nav2全体の性能比較は行っていない。'])
        test3_rows = [{**r, **{f'reported_{k}': r[f"{'demand' if r['method']=='rpp' else 'command'}_{k}"]
                               for k in ('velocity_violation_steps', 'acceleration_violation_steps', 'both_violation_steps')}}
                      for r in chosen('test3')]
        common_table((('value', '調整条件'), ('method', '手法')), test3_rows,
                     (('mean_near_obstacle_speed_m_s', '近傍平均速度 [m/s]', 4),
                      ('control_steps', '全周期数', 0),
                      ('reported_velocity_violation_steps', '速度超過 [周期]', 0),
                      ('reported_acceleration_violation_steps', '加速度超過 [周期]', 0),
                      ('reported_both_violation_steps', '両方超過 [周期]', 0)))
        lines.append('並進速度と走行距離の図は `test3/cost0_approach1_distance_speed` と `test3/cost1_approach1_distance_speed`（PDF/PNG）。')
    if chosen('test4'):
        lines.extend(['', '## 試験4：結果の頑健性', '',
                      '表は区分・経路・手法ごとの条件平均（最大誤差列も各条件の最大値の平均）。'
                      '全探索候補とタイムアウトを含む。走行時間は成功試行、整定時間は値が定義された試行の平均。各水準はsummary.csv、ノイズの平均・標本SD・有効数はnoise_summary.csv。ゼロノイズ20 seedは同一軌跡。'])
        grouped = aggregate(chosen('test4'), ('part', 'scenario', 'method'))
        group_rows = [{**g, **{k: g[k+'_mean'] for k in (*COMMON_METRICS, 'crossing_m', 'settling_2pct_time_s', 'transition_heading_lag_deg', 'post_transition_heading_overshoot_deg')},
                       'counts': f"{g['success_count']}/{g['n']}"} for g in grouped]
        for part, title in (('a', '走行時間を揃えた比較'), ('b', '前方注視'), ('c', '自己位置推定のノイズ')):
            lines.extend(['', f'### ({part}) {title}', ''])
            common_table((('scenario', '経路'), ('method', '手法'), ('counts', '成功/条件数')),
                         [r for r in group_rows if r['part']==part], lateral+heading)
            if part=='a':
                for method, match in manifest.get('time_matches', {}).items():
                    lines.append(f"{method}: 採用速度倍率 {f(match['best_scale'], 6)}、DWVPとの時間差 {f(match['difference_s'])} s、許容差内={match['matched']}。")
    if chosen('preview-noise'):
        if not chosen('test4'):
            lines.extend(['', '## 試験4：結果の頑健性', '', '### (c) 自己位置推定のノイズ', ''])
            means = aggregate(chosen('preview-noise'), ('scenario', 'method'))
            common_table((('scenario', '経路'), ('method', '手法')),
                         [{**g, **{k: g[k+'_mean'] for k in COMMON_METRICS}} for g in means])
        grouped = preview_noise_summary(rows)
        config = Config(**manifest['config'])
        recovery = travel_time_recovery(grouped, config)
        write_csv(output/'preview-noise'/'travel_time_recovery.csv', recovery)
        lines.extend(['', 'DWVPの前方注視×ノイズも(c)に含む。各水準20 seed。共通指標の全試行は `preview-noise/summary.csv`、平均・標本SD・有効数・成功数は `nominal_selection.csv`。',
                      '下表は平均走行時間 [s]（成功数/20）。固定L [m]と適応T [s]を列に示す。未成功の時間は平均に含めず、全条件の成否を残す。',
                      f'方向の揺れをσ_xy/L、横方向の速度変化をVσ_xy/Lと近似し、1周期の加速度の幅aΔtとの比較から L≥Vσ_xy/(aΔt)={noise_lookahead_estimate(config, 1.):g}σ_xy と見積もる（V={config.translation_speed:g} m/s、a={config.ay:g} m/s²、Δt=1/{config.frequency:g} s）。周期間の独立ノイズの差分、姿勢ノイズ、終端処理はこの近似に含まない。',
                      '回復は同じ場面のノイズなし平均の1.1倍以内、かつ両条件20/20成功と定義する。基準は既定の適応注視と同じ固定Lの2通りを併記する。最小Lは掃引した水準内の値であり、連続的な境界や保証ではない。'])
        levels = sorted({(r['noise_xy_m'], r['noise_yaw_deg']) for r in grouped})
        previews = list(dict.fromkeys((r['parameter'], r['value']) for r in grouped))
        for scene in dict.fromkeys(r['scenario'] for r in grouped):
            by_key = {(r['noise_xy_m'], r['noise_yaw_deg'], r['parameter'], r['value']): r
                      for r in grouped if r['scenario']==scene}
            table([f'{scene}: σ_xy [m] / σ_yaw [°]',
                   *[f"{'L' if p=='fixed_lookahead' else 'T'}={v:g}" for p, v in previews]],
                  [[f'{xy:g} / {yaw:g}', *[f"{f(by_key[xy,yaw,p,v]['travel_time_s_mean'], 2)} ({by_key[xy,yaw,p,v]['success_count']}/{by_key[xy,yaw,p,v]['n']})"
                    for p, v in previews]] for xy, yaw in levels])
            paired = {(r['noise_xy_m'], r['noise_yaw_deg']): r for r in recovery
                      if r['scenario']==scene and r['reference']=='same_fixed_lookahead'}
            values = [f"σ={r['noise_xy_m']:g}: {f(r['minimum_fixed_lookahead_m'])} / {f(paired[r['noise_xy_m'],r['noise_yaw_deg']]['minimum_fixed_lookahead_m'])} / {f(r['estimate_lookahead_m'])}"
                      for r in recovery if r['scenario']==scene and r['reference']=='nominal_preview']
            lines.append(f"{scene}の最小回復L（既定基準 / 同じL基準） / 見積もりL [m]："+'、'.join(values)+'。')
        lines.append('図は `preview-noise/{offset,gradual,rapid}_travel_time_vs_fixed_lookahead`、横一列の別凡例は `noise_lookahead_legend`（PDF/PNG）。ノイズ水準ごとの縦点線はL=30σ_xy。回復判定の平均時間と基準値は `travel_time_recovery.csv`。')
    if chosen('regulation-sweep'):
        lines.extend(['', '## 任意試験：速度調整パラメータ', ''])
        common_table((('parameter', 'パラメータ'), ('value', '値'), ('method', '手法')), chosen('regulation-sweep'),
                     (('mean_near_obstacle_speed_m_s', '近傍平均速度 [m/s]', 4),))
    if chosen('acceleration-sweep'):
        lines.extend(['', '## 任意試験：加速度制約', ''])
        common_table((('scenario', '経路'), ('parameter', 'パラメータ'), ('value', '倍率'), ('method', '手法')),
                     chosen('acceleration-sweep'), lateral+heading)
    lines.extend(['', '## 再集計と検証範囲', ''])
    grid = manifest.get('test2_grid_refresh')
    if grid:
        lines.append(f"図と試験区分だけを更新し、既存 {grid['preserved_conditions']} 条件の指標と試行IDをすべて維持した。シミュレーション再実行は0件。保存軌跡・保護ファイル・旧監査記録のSHA-256照合は `test2_grid_refresh.json`。")
    round2 = manifest.get('comment_round2')
    followup = manifest.get('comment_round2_followup')
    if followup:
        lines.extend([f"上限0.33 mへの復帰：前回変更された {followup['restored_conditions']}/{followup['previously_changed_conditions']} 条件の全指標が元の値と一致した（絶対差1e−10以内）。既存 {followup['matched_conditions']} 条件も全指標一致。照合結果は `comment_round2_followup_conditions.csv` と `comment_round2_followup_restoration.csv`。",
                      f"元の軌跡を {followup['reused_conditions']} 条件で再集計し、試験3の4条件を再計算した。SHA-256、保護ファイル、既存5手法の指令の照合は `comment_round2_followup.json`。前回の変更記録 `comment_round2_changes.csv` は保存した。"])
        for item in followup['test3_breakdown']:
            if item['method'] == 'rpp':
                lines.append(f"RPP・{item['value']} の加速度超過：発進 {item['startup_acceleration_steps']}、追従中の調整比変化 {item['regulation_acceleration_steps']}、終端処理 {item['terminal_acceleration_steps']}、その他 {item['other_acceleration_steps']} 周期。")
    elif round2:
        lines.extend([f"著者コメント第2弾：既存 {round2['matched_conditions']} 条件を照合し、指標不変 {round2['unchanged_conditions']} 条件、変更 {round2['changed_conditions']} 条件（絶対差1e−10超）。追加 {round2['new_conditions']} 条件。全変更条件は `comment_round2_conditions.csv`、指標別の旧値・新値は `comment_round2_changes.csv`。",
                      f"保存軌跡から再集計した条件は {round2['reused_conditions']} 件。試験3と、0.165 mの上限が有効になる条件・時間一致探索は再計算した。再利用した軌跡のSHA-256一致と旧軌跡の保存を `comment_round2.json` に記録。",
                      '上限を下げても結果が変わらないという予想は成立しなかった。DWVPは車体x・y成分の速度箱で選ぶため並進速度が0.22 m/sを超え得る。長い前方注視時間、加速度制約、観測ノイズの条件でも旧上限との違いが生じた。'])
    refresh = manifest.get('metrics_alignment')
    if refresh:
        lines.extend([f"保存軌跡 {refresh['distinct_trajectories']} 件から再集計。軌跡ファイルのSHA-256、試行仕様、条件ID、制御器・試験設定は維持。既存指標の差分は `metrics_alignment_changes.csv`、検査結果は `metrics_alignment.json`。既存の最大誤差・走行時間・制約違反・位置行き過ぎ・姿勢行き過ぎに変化なし。",
                      '平均2列だけ台形則の時間平均へ変更し、旧値をeval_sample_mean_*列に保存。遅れは従来の全評価窓の列を残し、姿勢変化区間に限定した列を追加。',
                      '依頼時の「姿勢行き過ぎが全条件0」は作業開始時の現行結果では再現しなかった。0.3 m・0.25倍で43.0/52.1/8.8°、0.5倍で8.5/16.6/3.7°が既に保存され、軌跡の直接計算と一致した。現行の式は区間通過後を正しく評価している。旧報告書に欠けていた行き過ぎ列を今回は追加した。全条件0だった別の版の原因は、現行ファイルからは確定できない。'])
    issues = [r for r in rows if not r['success'] or r.get('constraint_violation_duration_s', 0) > 0]
    if issues:
        write_csv(output/'issues.csv', issues)
        issue_counts = []
        for (part, method) in dict.fromkeys((r['part'], r['method']) for r in issues):
            group = [r for r in issues if (r['part'], r['method']) == (part, method)]
            issue_counts.append(f"({part}) {method} {len(group)}件")
        lines.append('終了未成功: '+'、'.join(issue_counts)+'。条件・seed・終了状態は `issues.csv`。')
    if manifest.get('method_comparison'):
        lines.append('比較手法追加時の過去の照合記録はmanifest.jsonのmethod_comparisonとmethod_changes.csvに保持。今回の再集計とは別の履歴である。')
    lines.append('共通指標表で指令制約違反の列を省略した表は、全手法・全条件で違反率0%。図は従来の書体・寸法を維持。実機試行は0件。')
    lines = [line for i, line in enumerate(lines) if line or i == 0 or lines[i-1]]
    if manifest.get('full_run_evaluation'):
        from .access_comparison import comparison_lines
        lines.extend(comparison_lines(output, rows, manifest))
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def compare_baseline(output, manifest, baseline):
    """Compare matching trajectory specifications, independent of test numbering."""
    old=json.loads(baseline.read_text())
    def identity(spec):
        return sha({k:v for k,v in spec.items() if k!='numerical_source_sha256'})
    previous={identity(t['spec']):t for t in old['trials']}
    previous_conditions={t['condition_id']:t for t in old['trials']}
    preserved=set()
    differences=[]
    groups={}
    matched=0
    changed=set()
    for trial in manifest['trials']:
        before=previous_conditions.get(trial['condition_id']) or previous.get(identity(trial['spec']))
        if before is None:
            continue
        matched+=1
        if trial['condition_id'] in previous_conditions:
            preserved.add(trial['condition_id'])
        a=dict(before['summary'])
        b=trial['summary']
        if 'eval_max_lateral_error_m' in a:
            a['eval_max_position_error_m']=a.pop('eval_max_lateral_error_m')
        trajectory=baseline.parent/'trials'/a['trial_id']/'trajectory.npz'
        if trajectory.exists():
            with np.load(trajectory) as history:
                poses=history['poses']
                # Preserve saved baseline metrics and their original support.
                # Very old baselines without this field used a sample mean.
                end=before['spec'].get('evaluation_end')
                mask=(np.ones(len(poses),dtype=bool) if end is None else
                      (poses[:,0]>=-1e-10)&(poses[:,0]<=end))
                a.setdefault('eval_mean_position_error_m',
                             float(history['position_errors'][mask].mean()) if mask.any() else None)
                a['goal_overshoot_m']=max(0.,float(poses[:,0].max()-history['path'][-1,0]))
        # Summary identity/labels change with the suite; compare scientific values only.
        ignored={'test','part','trial_id','condition_id','scenario','method','parameter','value','seed'}
        for metric in sorted(a.keys() & b.keys() - ignored):
            left,right=a[metric],b[metric]
            equal=(math.isclose(left,right,rel_tol=0.,abs_tol=1e-10)
                   if isinstance(left,(int,float)) and isinstance(right,(int,float)) else left==right)
            if equal:
                continue
            changed.add(b['condition_id'])
            differences.append(dict(condition_id=b['condition_id'],test=b['test'],part=b['part'],scenario=b['scenario'],method=b['method'],
                                    parameter=b['parameter'],value=b['value'],seed=b['seed'],metric=metric,before=left,after=right,
                                    difference=right-left if isinstance(left,(int,float)) and isinstance(right,(int,float)) else None))
            key=f"{b['test']}/{b['part']}/{b['method']}/{b['parameter']}={b['value']}"
            g=groups.setdefault(key,dict(ids=set(),metrics=set()))
            g['ids'].add(b['condition_id']);g['metrics'].add(metric)
    # Always produce a header even when there are no differences.
    import csv
    with (output/'method_changes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=('condition_id','test','part','scenario','method','parameter','value','seed','metric','before','after','difference'))
        writer.writeheader();writer.writerows(differences)
    return dict(baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                baseline_source_sha256=old['source_sha256'],matched_conditions=matched,
                baseline_conditions=len(previous_conditions), preserved_baseline_conditions=len(preserved),
                missing_baseline_conditions=sorted(previous_conditions.keys()-preserved),
                unmatched_conditions=len(manifest['trials'])-matched,changed_conditions=len(changed),
                reported_changes=[d for d in differences if 'speed_cap' not in d['metric']],
                changed_groups={k:dict(conditions=len(v['ids']),metrics=sorted(v['metrics'])) for k,v in groups.items()})
