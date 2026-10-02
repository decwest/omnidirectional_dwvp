#!/usr/bin/env python3
"""Check the C++ runner's nine output scalars against shared JSON fixtures.

Runner stdin per case: desired3 physical_min3 physical_max3 acceleration3
current3 dt cap prefer_large. stdout: regulated_min3 regulated_max3 command3.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runner',help='Executable filename')
p.add_argument('--fixtures',type=Path,default=Path(__file__).resolve().parents[1]/'fixtures/solver_cases.json')
a=p.parse_args()
cases=json.loads(a.fixtures.read_text())['cases']
lines=[]
for c in cases:
    values=c['desired']+c['physical_min']+c['physical_max']+c['acceleration']+c['current']+[c['dt'],c['cap'],int(c['prefer_large'])]
    lines.append(' '.join(format(x,'.17g') for x in values))
run=subprocess.run([a.runner],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
output=run.stdout.strip().splitlines()
assert len(output)==len(cases),(len(output),len(cases),run.stderr)
for c,line in zip(cases,output):
    expected=c['expected_min']+c['expected_max']+c['expected_command']
    np.testing.assert_allclose(np.fromstring(line,sep=' '),expected,atol=1e-9,rtol=1e-9,err_msg=c['name'])
print(f'PASS: {len(cases)} shared Python/C++ box-and-command cases')
