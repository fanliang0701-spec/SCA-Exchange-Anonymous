"""Rebuild paper results locally from the four included rank archives."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PACKAGE / 'results')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SCA_OUTPUT=str(output), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    start = time.perf_counter()
    for script in ['reanalyze.py', 'revision_analysis.py'] + ([] if args.no_plots else ['plots.py']):
        subprocess.run([sys.executable, '-B', str(PACKAGE / 'analysis' / script)],
                       env=env, cwd=PACKAGE, check=True)
    result = json.loads((output / 'extended-analysis.json').read_text(encoding='utf-8'))
    fields = ['dataset', 'Q', 'method', 'A95', 'A95_sustained', 'Astrict', 'C', 'score_seconds_10000']
    with (output / 'configurations.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:g.get(k) for k in fields} for g in result['groups'])
    verify = subprocess.run([sys.executable, '-B', str(PACKAGE / 'scripts/verify.py'), '--output', str(output)],
                            env=env, cwd=PACKAGE, check=True)
    print(f'Finished in {time.perf_counter()-start:.1f} s. Results: {output}')


if __name__ == '__main__':
    main()
