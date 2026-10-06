"""Refresh reported Test 2 curves from saved CSV/NPZ files, without simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from omnidirectional_dwvp.access_plotting import HEADING_METHODS, figures, panel
from omnidirectional_dwvp.access_studies import atomic_json, numerical_hash
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.studies import ROOT, code_hash


def archived_acceleration_figures(output, rows, existing):
    """Keep the archived multiplier axes/metrics, using the reported methods."""
    rows = [r for r in rows if r['part']=='acceleration' and r['status']!='error'
            and r['method'] in HEADING_METHODS]
    output.mkdir(parents=True, exist_ok=True)
    for parameter, ell in dict.fromkeys((r['parameter'], r['transition_length_m']) for r in rows):
        group = [r for r in rows if r['parameter']==parameter and r['transition_length_m']==ell]
        stem = f'{parameter}_ramp_{ell:g}'.replace('.', 'p')
        for metric, suffix, label in (('eval_max_heading_error_deg', 'heading', 'Max. heading error [deg]'),
                                      ('eval_heading_error_integral_deg_s', 'heading_integral', 'Heading error integral [deg s]'),
                                      ('eval_max_position_error_m', 'position', 'Max. position error [m]')):
            name = stem+'_'+suffix
            if not (existing/(name+'.pdf')).is_file():
                continue
            curves = []
            for method in HEADING_METHODS:
                series = sorted([r for r in group if r['method']==method], key=lambda r: r['value'])
                curves.append((method, [r['value'] for r in series], [r[metric] for r in series]))
            panel(output/name, curves,
                  'Acceleration multiplier' if parameter=='acceleration_scale' else 'Yaw acceleration multiplier', label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert args.output.is_dir() and not args.output.is_symlink()
    root = args.output.resolve()
    assert root.is_relative_to(ROOT)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and manifest['numerical_source_sha256']==numerical_hash()
    summary = root/'test2/summary.csv'
    summary_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
    with summary.open() as stream:
        rows = list(csv.DictReader(stream))
    # Check every CSV value against its saved typed summary before plotting.
    saved = {t['condition_id']: t['summary'] for t in manifest['trials'] if t['condition']['test']=='test2'}
    assert len(rows)==len(saved) and {r['condition_id'] for r in rows}==set(saved)
    numeric = ('transition_length_m', 'value', 'eval_max_heading_error_deg',
               'eval_heading_error_integral_deg_s', 'eval_max_position_error_m', 'predicted_speed_m_s')
    for row in rows:
        expected = saved[row['condition_id']]
        assert all(value==('' if expected.get(key) is None else str(expected[key])) for key, value in row.items())
        for key in numeric:
            row[key] = float(row[key]) if row[key] else None
    nominal = [r for r in rows if r['part']!='acceleration']
    stems = {'legend', 'max_heading_vs_length', 'max_heading_vs_rate_ratio'}
    stems.update(r['scenario'].replace('=', '').replace(',', '_').replace('.', 'p')+'_'+quantity
                 for r in nominal for quantity in ('heading', 'speed', 'yaw'))
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    regenerated = []
    with TemporaryDirectory(prefix='test2-figures-', dir=build) as temporary:
        stage = Path(temporary)
        (stage/'test2').mkdir()
        (stage/'trials').symlink_to(root/'trials', target_is_directory=True)
        figures(stage, nominal, manifest['settings'], Config(**manifest['config']))
        archive = Path('previous_acceleration_figures')
        archived_acceleration_figures(stage/'test2'/archive, rows, root/'test2'/archive)
        stems.update(str(p.relative_to(stage/'test2').with_suffix(''))
                     for p in (stage/'test2'/archive).glob('*.pdf'))
        # Leave unaffected figures, including boundary legends, byte-identical.
        for stem in sorted(stems):
            source = stage/'test2'/stem
            target = root/'test2'/stem
            if (target.with_suffix('.png').is_file() and target.with_suffix('.pdf').is_file()
                    and source.with_suffix('.png').read_bytes()==target.with_suffix('.png').read_bytes()):
                continue
            for suffix in ('.pdf', '.png'):
                shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))
                regenerated.append(str(target.with_suffix(suffix).relative_to(root)))
    assert hashlib.sha256(summary.read_bytes()).hexdigest()==summary_hash
    source_hash = code_hash()
    if regenerated or manifest['source_sha256']!=source_hash:
        previous_source = manifest['source_sha256']
        manifest.setdefault('simulation_source_sha256', previous_source)
        manifest['source_sha256'] = source_hash
        manifest['test2_figure_refresh'] = dict(previous_source_sha256=previous_source,
                                               source_sha256=source_hash, simulation_runs=0,
                                               methods=list(HEADING_METHODS),
                                               summary_sha256=summary_hash, regenerated_files=regenerated)
        atomic_json(root/'manifest.json', manifest)
    print(json.dumps(dict(simulation_runs=0, regenerated_files=regenerated), indent=2))


if __name__=='__main__':
    main()
