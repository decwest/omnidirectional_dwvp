"""Freeze independent DWPP oracle outputs from a local MIT source checkout."""
import argparse
import ast
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Path to dwpp/simulator/controllers.py')
    parser.add_argument('--output', type=Path, default=Path('fixtures/dwpp_reference_cases.json'))
    args = parser.parse_args()
    source = args.source.read_text()
    config_source = args.source.with_name('simulator.py').read_text()
    namespace = dict(math=math, np=np, dataclass=dataclass)
    tree = ast.parse(config_source)
    config_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='SimulationConfig')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), config_node], type_ignores=[])), str(args.source.with_name('simulator.py')), 'exec'), namespace)
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])), str(args.source), 'exec'), namespace)
    windows = [(0.22,0.,.6,-.6), (.2,.15,.1,-.1), (.03,0.,.4,.3), (.03,0.,-.3,-.4), (.2,.15,.4,.3), (.2,.15,-.3,-.4)]
    kernel=[]
    for window in windows:
        for curvature in (0., .0009, .001, -4., .5, 10., -10.):
            kernel.append(dict(window=window, curvature=curvature, expected=namespace['optimal_velocity_in_window'](window,curvature)))
    config=namespace['SimulationConfig'](max_linear_velocity=.22, max_angular_velocity=.6, min_angular_velocity=-.6,
              linear_acceleration=.22, linear_deceleration=.22, angular_acceleration=.6, angular_deceleration=.6,
              lookahead_time=.75, min_lookahead=.11, max_lookahead=.33, use_curvature_regulation=False,
              min_approach_velocity=.05/math.sqrt(2))
    path=np.c_[np.arange(0.,4.001,.005), np.zeros(801)]
    full=[]
    for fixed in (None,.165):
        for pose,current in [([0.,.1,0.],[0.,0.]), ([.6,.5,.2],[.2,-.2]), ([1.,-.05,-.1],[.15,.4]),
                             ([1.,0.,0.],[.18,.2]), ([3.6,.03,-.1],[.2,.1]), ([3.9,0.,0.],[.1,0.])]:
            c=replace(config,use_velocity_scaled_lookahead=fixed is None,fixed_lookahead=.165)
            command,nearest,curvature=namespace['compute_command'](path,pose,current,0,c)
            full.append(dict(pose=pose,current=current,fixed_lookahead=fixed,expected=command,curvature=curvature,nearest=nearest))
    result=dict(source_repository='decwest/dwpp', source_commit=subprocess.check_output(['git','-C',str(args.source.parent),'rev-parse','HEAD'],text=True).strip(),
                source_file='simulator/controllers.py', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                oracle='Functions executed directly from source AST; no local port imported.', kernel_cases=kernel, command_cases=full)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
