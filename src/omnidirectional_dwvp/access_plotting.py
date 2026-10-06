"""Single-panel figures with separate horizontal legends and embedded TrueType fonts."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from .access_metrics import aggregate
from .access_noise import noise_lookahead_estimate, preview_noise_summary

COLORS = {'vp': '#2568a0', 'vp_scaled': '#b87510', 'vp_scaled_accel': '#7b52a1',
          'dwvp': '#bf4145', 'dwpp': '#23845d', 'rpp': '#2568a0'}
LABELS = {'vp': 'Clipped VP', 'vp_scaled': 'Scaled VP', 'vp_scaled_accel': 'Scaled VP (vel. and acc.)',
          'dwvp': 'DWVP', 'dwpp': 'DWPP', 'rpp': 'RPP'}
STYLES = {'vp': '--', 'vp_scaled': '-.', 'vp_scaled_accel': ':', 'dwvp': '-', 'dwpp': (0, (5, 1, 1, 1)), 'rpp': '--'}
# Publication figures omit the auxiliary method; saved data and validation retain it.
HEADING_METHODS = ('vp', 'vp_scaled', 'dwvp')


def style():
    # Fail visibly if the specified font is absent; do not silently substitute.
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, stem):
    fig.tight_layout(pad=.45)
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


def panel(stem, curves, xlabel, ylabel, reference=None, vertical=None, xscale='linear', xticks=None):
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    for method, x, y in curves:
        # Put dashed RPP above DWVP so identical trajectories show both colors.
        ax.plot(x, y, color=COLORS[method], ls=STYLES[method], lw=1., zorder=3 if method=='rpp' else 2)
    if reference is not None:
        ax.axhline(reference, color='0.35', ls=':', lw=.7)
    if vertical is not None:
        ax.axvline(vertical, color='0.35', ls=':', lw=.7)
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale)
    if xticks is not None:
        ax.set_xticks(xticks, labels=[f'{x:.2f}' for x in xticks])
        ax.minorticks_off()
    ax.grid(alpha=.2, lw=.4)
    save(fig, stem)


def legend(stem, methods, ratio_boundary=False):
    fig = plt.figure(figsize=(5.4 if 'vp_scaled_accel' in methods or ratio_boundary else 3.7, .3))
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1., label=LABELS[m]) for m in methods]
    if ratio_boundary:
        handles.append(Line2D([], [], color='0.35', ls=':', lw=.7,
                              label=r'$\omega_{\max}/(a_\omega T)=2$'))
    fig.legend(handles=handles, loc='center', ncol=len(handles), frameon=False, borderaxespad=0)
    save(fig, stem)


def theory_curves(e0, lookahead, distance):
    """Ideal fixed-L orbits as functions of travelled arc length s.

    Omni: e=e0 exp(-s/L) while |e0|<L, equivalent to
    de/dx=-e/sqrt(L^2-e^2). PP is the small-error linear approximation.
    """
    if abs(e0) >= lookahead:
        raise ValueError('omnidirectional theory overlay requires |e0| < L')
    z = np.asarray(distance) / lookahead
    return e0*np.exp(-z), e0*np.exp(-z)*(np.cos(z)+np.sin(z))


def figures(output, rows, settings, config):
    rows = [r for r in rows if r['test']!='test2' or r['method'] in HEADING_METHODS]
    style()
    preview_noise_figures(output, rows, config)
    for test in ('test1', 'test2', 'test3'):
        chosen = [r for r in rows if r['test']==test and r['status']!='error' and r['part']!='acceleration']
        if not chosen:
            continue
        out = output/test
        legend(out/'legend', ('dwpp', 'vp', 'vp_scaled', 'dwvp') if test=='test1'
               else HEADING_METHODS if test=='test2' else ('rpp', 'dwvp'))
        groups = {}
        for row in chosen:
            key = row['value'] if test=='test3' else row['scenario']
            groups.setdefault(key, []).append(row)
        for name, group in groups.items():
            name = str(name).replace('=', '').replace(',', '_').replace('.', 'p')
            data = []
            for row in group:
                with np.load(output/'trials'/row['trial_id']/'trajectory.npz') as a:
                    data.append((row['method'], {k:a[k] for k in a}))
            if test=='test1':
                fig, ax = plt.subplots(figsize=(3.35, 2.15))
                for method, a in data:
                    mask = a['poses'][:, 0] <= settings['evaluation_end']
                    ax.plot(a['travel_distance'][mask], a['poses'][mask, 1], color=COLORS[method], ls=STYLES[method], lw=1.)
                if config.fixed_lookahead is not None and abs(group[0]['offset_m']) < config.fixed_lookahead:
                    s = data[0][1]['travel_distance']
                    omni, pp = theory_curves(group[0]['offset_m'], config.fixed_lookahead, s)
                    ax.plot(s, omni, color='0.3', ls=':', lw=.7)
                    ax.plot(s, pp, color='0.6', ls=':', lw=.7)
                ax.axhline(0, color='0.5', lw=.5)
                ax.set(xlabel='Travel distance [m]', ylabel='Signed lateral error [m]')
                ax.grid(alpha=.2, lw=.4)
                save(fig, out/(name+'_lateral'))
            if test=='test2':
                curves = []
                for method, a in data:
                    mask = a['poses'][:, 0] <= settings['evaluation_end']
                    curves.append((method, a['travel_distance'][mask], np.rad2deg(a['yaw_errors'][mask])))
                panel(out/(name+'_heading'), curves, 'Travel distance [m]', 'Heading error [deg]')
            if test in ('test2', 'test3'):
                for quantity, label in (('speed', 'Translation speed [m/s]'), ('yaw', 'Yaw rate [rad/s]')):
                    curves = [(m, a['times'][:-1], np.linalg.norm(a['applied'][:, :2], axis=1) if quantity=='speed' else a['applied'][:, 2]) for m, a in data]
                    reference = group[0].get('predicted_speed_m_s') if quantity=='speed' and test=='test2' else None
                    panel(out/(name+'_'+quantity), curves, 'Time [s]', label, reference)
            if test=='test3':
                curves = [(m, a['travel_distance'][:-1], np.linalg.norm(a['applied'][:, :2], axis=1)) for m, a in data]
                panel(out/(name+'_distance_speed'), curves, 'Travel distance [m]', 'Translation speed [m/s]')
        if test=='test3':
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(figsize=(3.35, 2.15))
            ax.plot([0, settings['path_length']], [0, 0], color='0.2', ls='--', lw=.8)
            for x, y, radius in settings['obstacles']:
                ax.add_patch(Circle((x, y), radius, color='0.45'))
                ax.add_patch(Circle((x, y), radius+config.robot_radius, fill=False, ls=':', ec='0.45', lw=.8))
            ax.set(xlabel='x [m]', ylabel='y [m]', xlim=(-.1, 4.1), ylim=(-.9, 1.))
            ax.set_aspect('equal')
            save(fig, out/'scene')
    ramps = [r for r in rows if r['test']=='test2' and r['part']=='nominal' and r['transition_length_m']>0 and r['status']!='error']
    if ramps:
        critical = config.translation_speed*(np.pi/2)/config.w_max
        out = output/'test2'
        for ratio in (False, True):
            curves=[]
            for method in HEADING_METHODS:
                series=sorted([r for r in ramps if r['method']==method],
                              key=lambda r: critical/r['transition_length_m'] if ratio else r['transition_length_m'])
                curves.append((method, [critical/r['transition_length_m'] if ratio else r['transition_length_m'] for r in series],
                               [r['eval_max_heading_error_deg'] for r in series]))
            panel(out/('max_heading_vs_rate_ratio' if ratio else 'max_heading_vs_length'), curves,
                  r'Required yaw rate / limit' if ratio else 'Orientation transition length [m]',
                  'Max. heading error [deg]', vertical=1. if ratio else critical)
        fig = plt.figure(figsize=(3.35, .3))
        fig.legend(handles=[Line2D([], [], color='0.35', ls=':', lw=.7,
                                  label=r'$\omega_{\max}\ell/(\pi/2)$')], loc='center', ncol=1, frameon=False)
        save(fig, out/'prediction_legend')
        fig = plt.figure(figsize=(3.35, .3))
        fig.legend(handles=[Line2D([], [], color='0.35', ls=':', lw=.7,
                                  label='Required yaw rate = limit')], loc='center', ncol=1, frameon=False)
        save(fig, out/'rate_limit_legend')
    acceleration = [r for r in rows if r['test']=='test2' and r['part']=='acceleration' and r['status']!='error']
    if acceleration:
        out = output/'test2'
        methods = HEADING_METHODS
        legend(out/'acceleration_ratio_legend', methods, ratio_boundary=True)
        for ell in settings['acceleration_transition_lengths']:
            group = [r for r in acceleration if r['parameter']=='acceleration_scale'
                     and r['transition_length_m']==ell and r['method'] in methods]
            if not group:
                continue
            ratio = lambda r: r['acceleration_time_w_s']/r['lookahead_time_s']
            stem = f'acceleration_ratio_ramp_{ell:g}'.replace('.', 'p')
            for metric, suffix, label in (('post_transition_heading_overshoot_deg', 'overshoot', 'Heading overshoot [deg]'),
                                          ('eval_heading_error_integral_deg_s', 'heading_integral', 'Heading error integral [deg s]'),
                                          ('eval_max_heading_error_deg', 'heading', 'Max. heading error [deg]')):
                curves = []
                for method in methods:
                    series = sorted([r for r in group if r['method']==method], key=ratio)
                    curves.append((method, [ratio(r) for r in series], [r[metric] for r in series]))
                panel(out/(stem+'_'+suffix), curves, r'$\omega_{\max}/(a_\omega T)$', label,
                      vertical=2., xscale='log', xticks=sorted({ratio(r) for r in group}))
        legend(out/'acceleration_time_series_legend', methods)
        for scale in (.25, 1.):
            group = [r for r in acceleration if r['parameter']=='acceleration_scale' and r['value']==scale
                     and r['transition_length_m']==settings['representative_rapid'] and r['method'] in methods]
            data = []
            for row in group:
                with np.load(output/'trials'/row['trial_id']/'trajectory.npz') as a:
                    data.append((row['method'], {k:a[k] for k in a}))
            stem = f"ramp_{settings['representative_rapid']:g}_acceleration_{scale:g}".replace('.', 'p')
            for quantity, label in (('speed', 'Translation speed [m/s]'), ('yaw', 'Yaw rate [rad/s]'),
                                    ('signed_heading', 'Signed heading error [deg]')):
                curves = []
                for method, a in data:
                    if quantity=='signed_heading':
                        mask = a['poses'][:, 0] <= settings['evaluation_end']
                        x = a['times'][mask]
                        y = np.rad2deg(a['signed_yaw_errors'][mask])
                    else:
                        mask = a['poses'][:-1, 0] <= settings['evaluation_end']
                        x = a['times'][:-1][mask]
                        y = (np.linalg.norm(a['applied'][:, :2], axis=1) if quantity=='speed' else a['applied'][:, 2])[mask]
                    curves.append((method, x, y))
                panel(out/(stem+'_'+quantity), curves, 'Time [s]', label,
                      reference=0. if quantity=='signed_heading' else None)
    chosen = [r for r in rows if r['test'] in ('test4', 'regulation-sweep', 'acceleration-sweep')]
    if not chosen:
        return
    out = output/chosen[0]['test']
    legend(out/'legend', ('dwpp', 'vp', 'vp_scaled', 'dwvp'))
    # Per-scenario, per-parameter panels preserve distinct units and sweep axes.
    sweep_groups = {}
    for r in chosen:
        if (r['part'] in ('b', 'd') or r['test']=='acceleration-sweep') and r['status']!='error':
            sweep_groups.setdefault((r['scenario'], r['parameter']), []).append(r)
    for (scene, parameter), group in sweep_groups.items():
        metric, ylabel = ('crossing_m', 'Crossing [m]') if scene=='offset' else ('eval_max_heading_error_deg', 'Max. heading error [deg]')
        if scene=='obstacles':
            metric, ylabel = 'mean_near_obstacle_speed_m_s', 'Near-obstacle mean speed [m/s]'
        xlabels = {'fixed_lookahead':'Lookahead distance [m]', 'lookahead_time':'Lookahead time [s]',
                   'acceleration_scale':'Acceleration multiplier', 'angular_acceleration_scale':'Yaw acceleration multiplier',
                   'cost_scaling_dist':'Cost distance [m]', 'cost_scaling_gain':'Cost gain',
                   'approach_distance':'Approach distance [m]', 'vp_translation_speed':'Desired translation speed [m/s]'}
        curves = []
        for method in ('dwpp', 'vp', 'vp_scaled', 'dwvp'):
            series = sorted([r for r in group if r['method']==method and r.get(metric) is not None], key=lambda r:r['value'])
            if series:
                curves.append((method, [r['value'] for r in series], [r[metric] for r in series]))
        panel(out/(scene+'_'+parameter), curves, xlabels[parameter], ylabel)
    noise = aggregate([r for r in chosen if r['test']=='test4' and r['part']=='c'], ('scenario', 'method', 'noise_xy_m'))
    if not noise:
        return
    for scene in ('offset', 'gradual', 'rapid'):
        metric, label = ('crossing_m', 'Crossing [m]') if scene=='offset' else ('eval_max_heading_error_deg', 'Max. heading error [deg]')
        fig, ax = plt.subplots(figsize=(3.35, 2.15))
        for method in ('dwpp', 'vp', 'vp_scaled', 'dwvp'):
            series = sorted([r for r in noise if r['scenario']==scene and r['method']==method], key=lambda r:r['noise_xy_m'])
            if series:
                ax.errorbar([r['noise_xy_m'] for r in series], [r[metric+'_mean'] for r in series],
                            yerr=[r[metric+'_std'] for r in series], color=COLORS[method], ls=STYLES[method], lw=1., capsize=2)
        ax.set(xlabel='Position noise standard deviation [m]', ylabel=label)
        ax.grid(alpha=.2, lw=.4)
        save(fig, out/(scene+'_noise'))


def preview_noise_figures(output, rows, config):
    grouped = [r for r in preview_noise_summary(rows) if r['parameter'] == 'fixed_lookahead']
    if not grouped:
        return
    out = output/'preview-noise'
    levels = sorted({(r['noise_xy_m'], r['noise_yaw_deg']) for r in grouped})
    colors = ('#333333', '#2568a0', '#23845d', '#b87510', '#bf4145')
    markers = ('o', 's', '^', 'D', 'v')
    handles = []
    for i, (xy, yaw) in enumerate(levels):
        handles.append(Line2D([], [], color=colors[i % len(colors)], marker=markers[i % len(markers)],
                              ms=3, lw=1., label=rf'$\sigma_{{xy}}={xy:g}$ m, $\sigma_\psi={yaw:g}^\circ$'))
    handles.append(Line2D([], [], color='0.35', ls=':', lw=.8,
                          label=rf'$L={noise_lookahead_estimate(config, 1.):g}\sigma_{{xy}}$ (same color)'))
    for scene in dict.fromkeys(r['scenario'] for r in grouped):
        fig, ax = plt.subplots(figsize=(3.35, 2.15))
        for i, (xy, yaw) in enumerate(levels):
            series = sorted((r for r in grouped if (r['scenario'], r['noise_xy_m'], r['noise_yaw_deg']) == (scene, xy, yaw)),
                            key=lambda r: r['value'])
            color = colors[i % len(colors)]
            ax.plot([r['value'] for r in series],
                    [r['travel_time_s_mean'] if r['travel_time_s_mean'] is not None else np.nan for r in series],
                    color=color, marker=markers[i % len(markers)], ms=3, lw=1.)
            ax.axvline(noise_lookahead_estimate(config, xy), color=color, ls=':', lw=.8)
        ax.set(xlabel='Fixed lookahead distance [m]', ylabel='Mean travel time [s]')
        ax.grid(alpha=.2, lw=.4)
        save(fig, out/(scene+'_travel_time_vs_fixed_lookahead'))
    fig = plt.figure(figsize=(11.5, .3))
    fig.legend(handles=handles, loc='center', ncol=len(handles), frameon=False, borderaxespad=0)
    save(fig, out/'noise_lookahead_legend')
