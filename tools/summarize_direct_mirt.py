#!/usr/bin/env python3
"""Recreate the direct-versus-spectral comparison from completed primary runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

LABELS = {'mmlu_pro': 'MMLU-Pro', 'bbh': 'BBH', 'mmlu': 'MMLU',
          'hellaswag': 'HellaSwag', 'winogrande': 'WinoGrande'}


def summarize(direct, spectral):
    rows, predictions = [], []
    for key, label in LABELS.items():
        newdir = direct / f'{key}_family_ranking_impact'
        olddir = spectral / f'{key}_family_ranking_impact'
        dimensions = pd.read_csv(newdir / 'dimension_selection.csv').set_index('audit_half')
        new = pd.read_csv(newdir / 'matched_random_anchor_control_summary.csv').set_index('metric')
        old = pd.read_csv(olddir / 'matched_random_anchor_control_summary.csv').set_index('metric')
        r, s = new.loc['close_cross_family_flip_rate'], old.loc['close_cross_family_flip_rate']
        rows.append(dict(benchmark=label, K='/'.join(str(int(dimensions.loc[h, 'selected_dimension']))
                        for h in ('discovery', 'validation')), reversal_percent=100*r.observed_low_dif,
                        random_percent=100*r.random_control_median,
                        excess_pp=100*(r.observed_low_dif-r.random_control_median),
                        p=r.one_sided_randomization_p,
                        spectral_excess_pp=100*(s.observed_low_dif-s.random_control_median),
                        tau=new.loc['kendall_tau_b', 'observed_low_dif']))
        cv = pd.read_csv(newdir / 'dimension_cv.csv')
        oldcv = pd.read_csv(olddir / 'dimension_cv.csv')
        paired = cv.merge(oldcv, on=['audit_half', 'dimension'], suffixes=('_direct', '_spectral'),
                          validate='one_to_one')
        if len(paired) != len(cv) or len(paired) != len(oldcv):
            raise ValueError(f'{label}: candidate grids differ')
        for name in ['n_fit_models', 'n_validation_models', 'n_calibration_items', 'n_evaluation_items']:
            if not paired[f'{name}_direct'].equals(paired[f'{name}_spectral']):
                raise ValueError(f'{label}: CV {name} differs')
        paired['benchmark'] = label
        predictions.append(paired)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    output = args.data_root.resolve() / 'outputs'
    summary, paired = summarize(output / 'direct_mirt', output)
    summary.to_csv(output / 'direct_mirt/summary.csv', index=False)
    paired.to_csv(output / 'direct_mirt/paired_cv_comparison.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
