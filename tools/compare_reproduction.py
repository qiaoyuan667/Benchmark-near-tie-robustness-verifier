#!/usr/bin/env python3
"""Compare regenerated CSV fields with frozen public references, without changing either."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCATIONS = {
    'primary': 'v3_five_benchmark_mirt_primary',
    'gap_sensitivity': 'v3_ranking_gap_sensitivity',
    'population_robustness': 'v3_family_owner_robustness',
    'item_stability': 'v3_item_dif_signature_stability',
    'discrimination': 'discrimination_matched',
    'within_family': 'within_family',
    'capability_profile': 'capability_profile',
    'direct_mirt': 'direct_mirt',
}
SPECTRAL = tuple(name for name in LOCATIONS if name != 'direct_mirt')


def generated_path(relative, output):
    group = relative.parts[0]
    if len(relative.parts) > 2 and relative.parts[1] == 'per_benchmark':
        key = relative.parts[2]
        if group in ('primary', 'direct_mirt'):
            base = output if group == 'primary' else output / 'direct_mirt'
            return base / f'{key}_family_ranking_impact' / relative.name
        return output / LOCATIONS[group] / key / relative.name
    return output / LOCATIONS[group] / relative.name


def compare(reference, output, analyses):
    records = []
    for group in analyses:
        files = sorted((reference / group).rglob('*.csv'))
        if not files:
            raise ValueError(f'No reference CSV files for {group}')
        for ref in files:
            relative = ref.relative_to(reference)
            actual = generated_path(relative, output)
            record = dict(table=relative.as_posix())
            if not actual.is_file():
                record['status'] = 'missing'
            else:
                expected = frame = None
                try:
                    expected, frame = pd.read_csv(ref), pd.read_csv(actual)
                    missing = set(expected.columns) - set(frame.columns)
                    if missing:
                        raise AssertionError(f'Missing released columns: {sorted(missing)}')
                    # Compare released fields in original row order; private fitting
                    # outputs may have extra identifying columns not in the release.
                    frame = frame[expected.columns]
                    for column in expected:
                        if pd.api.types.is_integer_dtype(expected[column].dtype):
                            pd.testing.assert_series_equal(frame[column], expected[column],
                                                           check_dtype=False, check_exact=True)
                    pd.testing.assert_frame_equal(frame, expected, check_dtype=False,
                                                  check_exact=False, rtol=1e-12, atol=1e-12)
                    record.update(status='match', rows=len(frame), columns=len(expected.columns))
                except (AssertionError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                    # Do not include private identifiers or source paths in this report.
                    record.update(status='mismatch', reason=type(error).__name__)
                    if expected is not None and frame is not None:
                        record.update(expected_rows=len(expected), observed_rows=len(frame))
                        record['missing_columns'] = sorted(set(expected.columns) - set(frame.columns))
                        differing = []
                        for column in expected.columns.intersection(frame.columns):
                            try:
                                pd.testing.assert_series_equal(
                                    frame[column], expected[column], check_dtype=False,
                                    check_exact=pd.api.types.is_integer_dtype(expected[column].dtype),
                                    rtol=1e-12, atol=1e-12)
                            except AssertionError:
                                differing.append(column)
                        record['differing_columns'] = differing
            records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--analyses', nargs='+', choices=tuple(LOCATIONS), default=SPECTRAL)
    parser.add_argument('--report', type=Path, help='Optional new JSON report; existing files are refused')
    parser.add_argument('--allow-missing', action='store_true', help='Permit incomplete runs, never mismatches')
    args = parser.parse_args(argv)
    if args.report and args.report.exists():
        parser.error('Report exists; select a new report path')
    records = compare(ROOT / 'results', args.data_root.resolve() / 'outputs', args.analyses)
    counts = {s: sum(r['status'] == s for r in records) for s in ('match', 'mismatch', 'missing')}
    passed = counts['mismatch'] == 0 and (args.allow_missing or counts['missing'] == 0)
    report = dict(passed=passed, complete=counts['missing'] == 0, counts=counts,
                  tolerance=dict(rtol=1e-12, atol=1e-12, integer_columns='exact'), tables=records)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
