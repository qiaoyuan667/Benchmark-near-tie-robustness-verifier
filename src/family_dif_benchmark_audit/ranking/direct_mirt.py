"""Repeat the primary audit with direct penalized joint logistic MIRT."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .._paths import PROJECT_ROOT
from ..core import spectral_mirt as core
from ..core.direct_mirt import fit
from . import mirt_primary as primary

RIDGE = .01


class DirectMIRT:
    def __init__(self, output: Path, max_steps=6000):
        self.output = output
        self.max_steps = max_steps
        self.call = 0
        output.mkdir(parents=True, exist_ok=True)

    def best_fit(self, y, k, seed, fixed=None, purpose='fit'):
        self.call += 1
        tag = f'{self.call:04d}_{purpose}_k{k}'
        candidates = []
        for initialization in range(2 if fixed is None else 1):
            result = fit(y, k, 'cpu', seed + initialization,
                         self.max_steps, ridge=RIDGE, fixed=fixed)
            a, b, t, info = result
            z = a.astype('float64') @ t.astype('float64').T + b[:, None]
            objective = float(np.mean(np.logaddexp(0, z) - y * z)
                              + .5 * RIDGE * np.square(t).sum(1).mean())
            if fixed is None:
                objective += float(.5 * RIDGE * np.square(a).sum(1).mean()
                                   + .00005 * np.square(b).mean())
            info['returned_objective'] = objective
            info['initialization_seed'] = seed + initialization
            candidates.append((objective, result))
        a, b, t, info = min(candidates, key=lambda x: x[0])[1]
        record = dict(purpose=purpose, k=k, shape=list(y.shape),
                      runs=[r[1][3] for r in candidates],
                      selected_seed=info['initialization_seed'])
        (self.output / f'{tag}.json').write_text(json.dumps(record, indent=2) + '\n')
        np.savez_compressed(self.output / f'{tag}.npz', a=a, b=b, theta=t)
        if not info['converged']:
            raise RuntimeError(f'{tag}: objective-plateau stopping criterion not met')
        print(f'{tag}: steps={info["epochs"]} objective={info["returned_objective"]:.7f}', flush=True)
        return a, b, t

    def choose_dimension(self, responses, owners, blueprint_sources, dimensions,
                         owner_seed, svd_seed, ridge):
        truth = np.asarray(responses, dtype=np.float32)
        val = core.nested_owner_split(owners, owner_seed)
        cal = core.stratified_calibration_mask(blueprint_sources, svd_seed)
        assert not set(np.asarray(owners)[val]) & set(np.asarray(owners)[~val])
        train = truth[:, ~val]
        feasible = [int(k) for k in dimensions if k <= min(train.shape)]
        if not feasible:
            raise ValueError('No feasible candidate dimensions')
        rows = []
        for k in feasible:
            a, b, t = self.best_fit(train, k, svd_seed, purpose='cv_train')
            _, _, v = self.best_fit(truth[cal][:, val], k, svd_seed,
                                   fixed=(a[cal], b[cal]), purpose='cv_coordinates')
            z = a[~cal].astype('float64') @ v.astype('float64').T + b[~cal, None]
            losses = core.binary_log_loss_by_model(truth[~cal][:, val], z)
            rows.append(dict(dimension=k, mean_log_loss=float(losses.mean()),
                se_across_validation_models=float(losses.std(ddof=1) / np.sqrt(len(losses))),
                n_fit_models=int((~val).sum()), n_validation_models=int(val.sum()),
                n_calibration_items=int(cal.sum()), n_evaluation_items=int((~cal).sum())))
            pd.DataFrame(rows).to_csv(self.output / f'cv_progress_seed{svd_seed}.csv', index=False)
            np.savez_compressed(self.output / f'cv_loss_seed{svd_seed}_k{k}.npz',
                                losses=losses, validation=val, calibration=cal)
        frame = pd.DataFrame(rows)
        best = frame.loc[frame.mean_log_loss.idxmin()]
        threshold = float(best.mean_log_loss + best.se_across_validation_models)
        selected = int(frame.loc[frame.mean_log_loss <= threshold, 'dimension'].min())
        frame['one_se_threshold'] = threshold
        frame['selected'] = frame.dimension.eq(selected)
        return selected, frame, dict(requested_dimensions=list(dimensions),
            feasible_dimensions=feasible, fitting_matrix_rank_bound=min(train.shape),
            best_dimension=int(best.dimension), selected_dimension=selected,
            one_se_threshold=threshold,
            validation_owner_count=len(np.unique(np.asarray(owners)[val])))

    def direct_offsets(self, responses, selected, dimension, seed, maximum_offset=6.):
        import torch

        truth = np.asarray(responses, dtype=np.float32)
        selected = np.asarray(selected, dtype=bool)
        a, b, t = self.best_fit(truth[selected], dimension, seed, purpose='anchor_train')
        center = t.mean(0)
        t = t - center
        y = torch.tensor(truth)
        theta = torch.tensor(t)
        n, m = truth.shape
        loadings = torch.nn.Parameter(torch.zeros(n, dimension))
        intercept = torch.nn.Parameter(torch.logit((y.mean(1) * m + .5) / (m + 1)))
        with torch.no_grad():
            loadings[selected] = torch.tensor(a)
            intercept[selected] = torch.tensor(b + a @ center)
        optimizer = torch.optim.Adam([loadings, intercept], lr=.03)
        history = []
        stable = 0
        start = time.perf_counter()
        for step in range(self.max_steps):
            optimizer.zero_grad()
            z = loadings @ theta.T + intercept[:, None]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                    + .5 * RIDGE * loadings.square().sum(1).mean()
                    + .00005 * intercept.square().mean())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite item fit')
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                value = float(loss.detach())
                stable = stable + 1 if history and abs(value - history[-1]['objective']) < 1e-6 else 0
                history.append(dict(step=step, objective=value))
                if stable >= 3:
                    break
        prefix = self.output / f'items_seed{seed}_k{dimension}_anchors{selected.sum()}'
        prefix.with_suffix('.json').write_text(json.dumps(dict(history=history,
            converged=stable >= 3, seconds=time.perf_counter() - start), indent=2) + '\n')
        if stable < 3:
            raise RuntimeError('Conditional item fit failed stopping criterion')
        aa, bb = loadings.detach().numpy(), intercept.detach().numpy()
        np.savez_compressed(prefix.with_suffix('.npz'), a=aa, b=bb, theta=t, selected=selected)
        offsets = np.clip(aa.astype('float64') @ t.astype('float64').T,
                          -maximum_offset, maximum_offset)
        return offsets, t, aa


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=tuple(primary.BENCHMARKS), required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reference-root', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    import torch
    if args.max_steps < 1 or args.threads < 1:
        parser.error('max-steps and threads must be positive')
    directory = args.output_dir or PROJECT_ROOT / 'outputs/direct_mirt' / f'{args.benchmark}_family_ranking_impact'
    if (directory / 'protocol.json').exists():
        raise RuntimeError('Completed output exists; choose a new output directory')
    backend = DirectMIRT(directory / 'optimization', args.max_steps)
    torch.set_num_threads(args.threads)
    settings = dict(estimator='direct penalized joint logistic MIRT', ridge=RIDGE,
        intercept_penalty=.00005, optimizer='Adam', learning_rate=.03,
        max_steps=args.max_steps, stopping='absolute training objective change <1e-6 at three 20-step checks',
        initializations=2, initialization_selection='smallest returned training penalized objective',
        coordinate_initialization='zero; conditional item parameters fixed', device='cpu', dtype='float32',
        threads=args.threads, offset_clip=[-6, 6], family_labels_in_ability_fit=False,
        coordinate_ridge_note='direct penalty 0.01; spectral coordinate ridge 1 is not algebraically transferable',
        torch_version=torch.__version__, numpy_version=np.__version__,
        paper_optimizer_settings=args.max_steps == 6000 and args.threads == 2,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory / 'direct_mirt_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    original_offsets, original_selection, original_argv = (
        core.fit_spectral_mirt_offsets, primary.choose_spectral_dimension, sys.argv)
    try:
        core.fit_spectral_mirt_offsets = backend.direct_offsets
        primary.choose_spectral_dimension = backend.choose_dimension
        sys.argv = ['primary', '--benchmark', args.benchmark, '--output-dir', str(directory)]
        primary.main()
    finally:
        core.fit_spectral_mirt_offsets = original_offsets
        primary.choose_spectral_dimension = original_selection
        sys.argv = original_argv
    for path in directory.iterdir():
        if path.suffix in {'.csv', '.json', '.md'}:
            text = path.read_text().replace('spectral_mirt_primary', 'direct_mirt_primary')
            text = text.replace('spectral MIRT', 'direct logistic MIRT').replace('spectral-MIRT', 'direct-MIRT')
            path.write_text(text)
    reference = args.reference_root / f'{args.benchmark}_family_ranking_impact'
    verified = reference.is_dir()
    if verified:
        for name in ['family_models.csv', 'owner_family_representatives.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(directory / name), pd.read_csv(reference / name))
        actual = json.loads((directory / 'protocol.json').read_text())
        expected = json.loads((reference / 'protocol.json').read_text())
        for key in ['primary_anchor_fraction', 'sensitivity_anchor_fractions', 'families',
                    'owner_cap', 'owner_disjoint_crossfit', 'ranking_population', 'dif_penalty',
                    'purification_rounds', 'close_pair_gap', 'bootstrap_replicates',
                    'matched_random_anchor_replicates', 'seeds']:
            if actual[key] != expected[key]:
                raise AssertionError(f'Protocol differs from spectral reference: {key}')
        if [r['sha256'] for r in actual['inputs']] != [r['sha256'] for r in expected['inputs']]:
            raise AssertionError('Input content differs from spectral reference')
    (directory / 'verification.json').write_text(json.dumps(
        dict(spectral_reference_available=verified, population_and_protocol_match=verified), indent=2) + '\n')
    print(f'COMPLETE {args.benchmark}; spectral reference checked: {verified}', flush=True)


if __name__ == '__main__':
    main()
