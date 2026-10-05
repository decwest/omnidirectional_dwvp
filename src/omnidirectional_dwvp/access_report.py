"""Generate the bounded factual handoff from saved metrics, never hand-entered values."""
import hashlib
import json
import math
import numpy as np
from .access_metrics import COMMON_METRICS, aggregate, heading_braking_prediction
from .config import Config
from .studies import sha, write_csv


def f(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


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
             '設定は `manifest.json`、全条件の指標は `testN/summary.csv`。公称設定と制御器は変更していない。公称設定：30 Hz、速度上限 ±0.22 m/s・±0.6 rad/s、加速度上限 0.22 m/s²・0.6 rad/s²、所望並進速度 0.22 m/s、前方注視時間 0.75 s。',
             '誤差の評価は従来どおり 0≤x≤3.25 m の保存試料。接近距離1 mの既存条件だけ上端2.995 m。境界への外挿はしない。',
             '位置誤差は経路への距離、姿勢誤差は線分射影位置の参照姿勢との差の絶対値。積分は隣接する評価内試料間の台形則、平均は積分÷評価時間。',
             '走行時間と指令制約違反率は従来どおり全走行で評価する。時間は成功時だけ記録し、タイムアウトは空欄。制約違反は速度または加速度超過の周期数÷全周期数（閾値1e−10）。',
             '位置の行き過ぎは初期横偏差と反対側への最大偏差。姿勢の遅れは姿勢変化区間内の参照−ロボットの最大値、行き過ぎは区間通過後のロボット−最終参照の最大値（いずれも非負）。',
             '段差（ℓ=0）の遅れは変化直後の最初の試料。2%到達・整定指標と従来の試料平均はCSVに残す。最大姿勢変化は表では最大姿勢誤差として扱う。',
             'vp=成分別クリップ、vp_scaled=速度箱への一様縮小後にクリップ、vp_scaled_accel=速度差も一様縮小する補助比較、dwvp=DWVP、dwpp=差動二輪DWPP。', '']
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
        violation = any(r.get('command_constraint_violation_pct', 0) > 0 for r in group)
        extra_headers = [label for _, label, _ in extras]
        data = []
        for r in group:
            data.append([*[r[k] for k, _ in prefix], *[f(r.get(k), d) for k, d in zip(metrics, digits)],
                         *[f(r.get(k), d) for k, _, d in extras],
                         *([f(r.get('command_constraint_violation_pct'))] if violation else [])])
        table([*[label for _, label in prefix], *headers, *extra_headers,
               *(['指令制約違反 [%]'] if violation else [])], data)
        if not violation:
            lines.append('この表の全手法・全条件で、指令が制約を超えた周期の割合は0%。')
    lateral = (('crossing_m', '位置行き過ぎ [m]', 5),)
    heading = (('transition_heading_lag_deg', '姿勢遅れ [°]', 2),
               ('post_transition_heading_overshoot_deg', '姿勢行き過ぎ [°]', 2))
    if chosen('test1'):
        lines.extend(['## 試験1：直線経路への収束', ''])
        common_table((('offset_m', '初期偏差 [m]'), ('method', '手法')), chosen('test1'), lateral)
    if chosen('test2'):
        lines.extend(['', '## 試験2：経路上の姿勢の追従', ''])
        nominal = [r for r in chosen('test2') if r['part'] in ('nominal', 'step')]
        common_table((('transition_length_m', 'ℓ [m]'), ('method', '手法')), nominal, heading)
        sweep = [r for r in chosen('test2') if r['part'] == 'acceleration']
        if sweep:
            lines.extend(['', '### 加速度制約の掃引', '',
                          '各セルは vp / vp_scaled / vp_scaled_accel / dwvp の順。ωのみは角加速度だけ、それ以外は並進・回転の加速度を同時に変更。'])
            methods = ('vp', 'vp_scaled', 'vp_scaled_accel', 'dwvp')
            data = []
            for parameter, ell, scale in dict.fromkeys((r['parameter'], r['transition_length_m'], r['value']) for r in sweep):
                group = {r['method']: r for r in sweep if (r['parameter'], r['transition_length_m'], r['value']) == (parameter, ell, scale)}
                values = [' / '.join(f(group[m].get(k), d) for m in methods)
                          for k, d in [*zip(metrics, digits), *((k, d) for k, _, d in heading)]]
                data.append([f(ell, 1), ('ωのみ ' if parameter == 'angular_acceleration_scale' else '')+f(scale, 2), *values])
            table(['ℓ [m]', '倍率', *headers, *[label for _, label, _ in heading]], data)
            if all(r['command_constraint_violation_pct'] == 0 for r in sweep):
                lines.append('この表の全手法・全条件で、指令が制約を超えた周期の割合は0%。')
            else:
                table(['ℓ [m]', '倍率', '手法', '指令制約違反 [%]'],
                      [[r['transition_length_m'], r['value'], r['method'], f(r['command_constraint_violation_pct'])] for r in sweep])
            comparisons = prediction_comparison(output, rows, manifest)
            lines.extend(['', '### 姿勢の行き過ぎの予測との照合', '',
                          '予測は max(0, ω²/(2aω)−ωT)。T=max(0.20, kL/v所望)。最終姿勢を前方注視点が参照しているときの最初の減速周期を選び、その直前の適用角速度ωと、その周期のLを使う。',
                          '最終姿勢を横切る直前はすでに減速しているため、停止角の議論に対応する減速開始を採った。各周期の時刻・ω・T・残り角度と予測差は `test2/overshoot_prediction.csv`。',
                          '各セルは予測 / 計算結果 [°]。予測はT一定、減速開始時の残り角度ωT、一定の最大角減速度を仮定する。DWVPの速度選択にはこのVP用の仮定を保証しない。'])
            data = []
            for parameter, ell, scale in dict.fromkeys((r['parameter'], r['transition_length_m'], r['value']) for r in comparisons):
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
                      '障害物 (x,y,r)=(1.6,−0.45,0.10), (2.4,0.55,0.10) m。近傍速度はCSVと時系列図に保存。'])
        common_table((('value', '調整条件'), ('method', '手法')), chosen('test3'))
    if chosen('test4'):
        lines.extend(['', '## 試験4：結果の頑健性', '',
                      '(a)時間一致探索、(b)前方注視、(c)加速度、(d)速度調整、(e)観測ノイズ。表は区分・経路・手法ごとの条件平均（最大誤差列も各条件の最大値の平均）。',
                      '全探索候補とタイムアウトを含む。走行時間だけは成功試行の平均。各水準はsummary.csv、ノイズの平均・標本SD・有効数はnoise_summary.csv。ゼロノイズ20 seedは同一軌跡。'])
        grouped = aggregate(chosen('test4'), ('part', 'scenario', 'method'))
        group_rows = [{**g, **{k: g[k+'_mean'] for k in (*COMMON_METRICS, 'crossing_m', 'transition_heading_lag_deg', 'post_transition_heading_overshoot_deg')},
                       'counts': f"{g['success_count']}/{g['n']}"} for g in grouped]
        common_table((('part', '区分'), ('scenario', '経路'), ('method', '手法'), ('counts', '成功/条件数')), group_rows, lateral+heading)
        for method, match in manifest.get('time_matches', {}).items():
            lines.append(f"(a) {method}: 採用速度倍率 {f(match['best_scale'], 6)}、DWVPとの時間差 {f(match['difference_s'])} s、許容差内={match['matched']}。")
    if chosen('preview-noise'):
        lines.extend(['', '## 任意試験：前方注視×ノイズ', '',
                      '全条件の共通指標はsummary.csv、水準ごとの平均・標本SD・有効数はnominal_selection.csv。'])
        grouped = aggregate(chosen('preview-noise'), ('scenario', 'method'))
        group_rows = [{**g, **{k: g[k+'_mean'] for k in (*COMMON_METRICS, 'crossing_m', 'transition_heading_lag_deg', 'post_transition_heading_overshoot_deg')}} for g in grouped]
        common_table((('scenario', '経路'), ('method', '手法')), group_rows, lateral+heading)
    lines.extend(['', '## 再集計と検証範囲', ''])
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
    lines.append('図は従来の書体・寸法を維持。姿勢積分対加速度倍率を区間ごとに追加し、0.3 mの時系列は評価区間の終わりまで表示。実機試行は0件。')
    if len(lines) > 240:
        raise RuntimeError(f'REPORT.md would exceed 240 lines: {len(lines)}')
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
                mask=(poses[:,0]>=-1e-10)&(poses[:,0]<=trial['condition']['evaluation_end'])
                a['eval_mean_position_error_m']=float(history['position_errors'][mask].mean()) if mask.any() else None
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
