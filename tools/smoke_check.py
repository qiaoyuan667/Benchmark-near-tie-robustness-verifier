#!/usr/bin/env python3
"""Exercise installed public commands without raw paper data or model calls."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New external directory for synthetic data, logs and reports.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Use an output directory outside the public repository')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    # Do not mask a broken package installation with a source-tree import path.
    env.pop('PYTHONPATH', None)
    python = sys.executable
    scripts = Path(sysconfig.get_path('scripts'))
    records = []

    def run(label, command):
        start = time.monotonic()
        log = output / f'{len(records):02d}-{label}.txt'
        record = dict(check=label, status='running')
        records.append(record)
        print(f'Checking {label}', flush=True)
        try:
            with log.open('w') as handle:
                completed = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT, timeout=600)
            record.update(status='pass' if completed.returncode == 0 else 'fail',
                          returncode=completed.returncode)
            if completed.returncode:
                raise RuntimeError(f'{label} failed; inspect {log.name} in the output directory')
        except BaseException:
            record['status'] = 'fail'
            raise
        finally:
            record['seconds'] = time.monotonic() - start
            (output / 'smoke_verification.json').write_text(json.dumps(dict(
                checks=records, remote_model_calls=0,
                scope='Installed CLI, synthetic maintainer, released statistics and figures; not full paper fitting'
            ), indent=2) + '\n')

    run('dependencies', [python, '-m', 'pip', 'check'])
    package = importlib.metadata.distribution('family-dif-benchmark-audit')
    for entry in sorted(package.entry_points, key=lambda e: e.name):
        if entry.group == 'console_scripts':
            executable = scripts / (entry.name + ('.exe' if os.name == 'nt' else ''))
            run(entry.name + '-help', [executable, '--help'])
    for tool in sorted((ROOT / 'tools').glob('*.py')):
        if tool.name != Path(__file__).name:
            if tool.name == 'generate_overview.py' and importlib.util.find_spec('matplotlib') is None:
                records.append(dict(check='generate_overview-help', status='skipped',
                                    reason='Optional schematic dependency absent; install requirements-figures.txt'))
                continue
            run(tool.stem + '-help', [python, tool, '--help'])
    run('synthetic-input', [python, ROOT / 'examples/make_synthetic_benchmark.py',
                            '--output', output / 'benchmark.npz'])
    audit = scripts / ('family-dif-audit.exe' if os.name == 'nt' else 'family-dif-audit')
    run('maintainer-quick', [audit, output / 'benchmark.npz', '--quick', '--output', output / 'quick'])
    run('maintainer-default', [audit, output / 'benchmark.npz', '--output', output / 'default'])
    run('maintainer-module', [python, '-m', 'family_dif_benchmark_audit.maintainer',
                              output / 'benchmark.npz', '--quick', '--output', output / 'module'])
    run('maintainer-api', [python, '-c',
        'from family_dif_benchmark_audit.maintainer import AuditConfig,load_benchmark,run_audit,write_audit; '
        'import sys; write_audit(run_audit(load_benchmark(sys.argv[1]), AuditConfig()), sys.argv[2])',
        output / 'benchmark.npz', output / 'api'])
    run('interface-agreement', [python, '-c',
        'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); '
        'read=lambda name: json.loads((p/name/"summary.json").read_text()); '
        'assert read("default")==read("api"); assert read("quick")==read("module")', output])
    run('content-replay', [python, '-m', 'family_dif_benchmark_audit.interpretation.content_replay',
                           'replay', '--labels', ROOT / 'results/content_audit/labels.csv',
                           '--reference-dir', ROOT / 'results/content_audit',
                           '--output-dir', output / 'content'])
    run('empirical-figures', [python, ROOT / 'tools/render_paper_figures.py',
                              '--output-dir', output / 'figures'])
    run('staged-command-plan', [python, ROOT / 'tools/reproduce.py', '--data-root', output / 'data',
                                '--stage', 'all', '--dry-run'])
    passed = sum(record['status'] == 'pass' for record in records)
    skipped = sum(record['status'] == 'skipped' for record in records)
    print(f'{passed} command checks passed; {skipped} optional checks skipped. '
          'No raw paper fitting or remote annotation was performed.')


if __name__ == '__main__':
    main()
