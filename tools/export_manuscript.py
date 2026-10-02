#!/usr/bin/env python3
"""Copy generated manuscript assets without modifying manuscript prose."""
import argparse
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'results/paper')
p.add_argument('--destination',type=Path,required=True)
a=p.parse_args()
if not (a.source/'manifest.json').exists(): raise SystemExit('Run dwvp-study all before exporting.')
for study in ('mechanism','obstacles','sweeps'):
    destination=a.destination/study
    destination.mkdir(parents=True,exist_ok=True)
    for path in (a.source/study).iterdir():
        if path.suffix in {'.pdf','.png','.tex','.csv','.json'}:
            shutil.copy2(path,destination/path.name)
shutil.copy2(a.source/'manifest.json',a.destination/'manifest.json')
print(f'Exported generated assets to {a.destination}')
