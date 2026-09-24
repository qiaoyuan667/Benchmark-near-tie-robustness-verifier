#!/usr/bin/env python3
"""Run independent, ordered paper-reproduction stages without changing results/."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('mmlu_pro', 'bbh', 'mmlu', 'hellaswag', 'winogrande')
STAGES = ('primary', 'summary', 'gap', 'population', 'stability', 'diagnostics', 'content', 'figures')
PACKAGE = 'family_dif_benchmark_audit.'


def commands(stage, data):
    def module(name, *args):
        return [sys.executable, '-m', PACKAGE + name, *map(str, args)]
    output = data / 'outputs'
    if stage == 'primary':
        return [module('ranking.mirt_primary', '--benchmark', key) for key in KEYS]
    modules = {'summary': 'ranking.summarize_v3', 'gap': 'robustness.gap_sensitivity_v3',
               'population': 'robustness.family_owner_v3', 'stability': 'interpretation.item_stability_v3'}
    if stage in modules:
        return [module(modules[stage])]
    if stage == 'diagnostics':
        return [module('diagnostics.' + name, '--primary-root', output,
                       '--output-dir', output / target) for name, target in
                [('discrimination', 'discrimination_matched'), ('within_family', 'within_family'),
                 ('capability_profile', 'capability_profile')]]
    if stage == 'content':
        return [module('interpretation.content_replay', 'replay',
                       '--labels', ROOT / 'results/content_audit/labels.csv',
                       '--reference-dir', ROOT / 'results/content_audit',
                       '--output-dir', output / 'content_audit_replay')]
    if stage == 'figures':
        return [[sys.executable, str(ROOT / 'tools/render_paper_figures.py'),
                 '--output-dir', str(output / 'paper_figures')]]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES + ('all',), required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rerun', action='store_true', help='Allow overwriting prior generated stage outputs')
    args = parser.parse_args()
    data = args.data_root.resolve()
    if data == ROOT or ROOT in data.parents:
        parser.error('Use a data-root outside the public repository to keep inputs and outputs separate')
    stages = STAGES if args.stage == 'all' else (args.stage,)
    env = dict(os.environ, FAMILY_DIF_PROJECT_ROOT=str(data), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for stage in stages:
        plan = commands(stage, data)
        if args.dry_run:
            print(json.dumps({'stage': stage, 'commands': plan}, indent=2))
            continue
        log = data / 'outputs/reproduction_log' / f'{stage}.json'
        if log.exists() and not args.rerun:
            parser.error(f'Stage {stage} was already attempted; inspect its log before using --rerun')
        if stage not in ('primary', 'content', 'figures'):
            for key in KEYS:
                if not (data / 'outputs' / f'{key}_family_ranking_impact/protocol.json').is_file():
                    parser.error('This stage requires all five completed primary runs')
        if stage == 'primary' and not args.rerun:
            if any((data / 'outputs' / f'{key}_family_ranking_impact').exists() for key in KEYS):
                parser.error('Primary output directories exist; inspect them before using --rerun')
        log.parent.mkdir(parents=True, exist_ok=True)
        record = dict(stage=stage, status='running', completed_commands=0)
        log.write_text(json.dumps(record, indent=2) + '\n')
        start = time.monotonic()
        try:
            for cmd in plan:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                record['completed_commands'] += 1
            record['status'] = 'complete'
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            log.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
