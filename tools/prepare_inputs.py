#!/usr/bin/env python3
"""Build the paper's numeric inputs from checksum-verified upstream files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORE_URL = 'https://huggingface.co/datasets/linggm/RouterEval/resolve/main/leaderboard_score.zip'
ITEM_URL = 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verified(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path.name}; supply it or use --download')
    if sha256(path) != expected:
        raise ValueError(f'Checksum mismatch for {path.name}; do not deserialize a changed upstream pickle')


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.partial')
        request = urllib.request.Request(url, headers={'User-Agent': 'family-dif-audit/1.0'})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as out:
            shutil.copyfileobj(source, out)
        temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--score-zip', type=Path)
    parser.add_argument('--mmlu-pro-parquet', type=Path)
    parser.add_argument('--download', action='store_true', help='Download approximately 88 MB from the documented upstream sources')
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / 'configs/upstream_inputs.json').read_text())
    score = args.score_zip or root / 'external_sources/leaderboard_score.zip'
    items = args.mmlu_pro_parquet or root / 'external_cache/mmlu_pro_test.parquet'
    if args.download:
        download(SCORE_URL, score)
        download(ITEM_URL, items)
    verified(score, spec['score_zip_sha256'])
    verified(items, spec['mmlu_pro_parquet_sha256'])
    extracted = root / 'external_sources/leaderboard_score'
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(score) as archive:
        for name, expected in spec['source_sha256'].items():
            candidates = [member for member in archive.namelist()
                          if member.rstrip('/').split('/')[-1] == name and not member.endswith('/')]
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one archive member for {name}')
            target = extracted / name
            if not target.exists():
                with archive.open(candidates[0]) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
            verified(target, expected)
    metadata = root / 'external_sources/item_layout.csv'
    layout = json.loads((ROOT / 'configs/routereval_item_layout.json').read_text())
    with metadata.open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['item_id', 'dataset', 'dataset_name'])
        writer.writeheader()
        position = 0
        for block in layout:
            if position != block['start']:
                raise ValueError('Noncontiguous item layout')
            for position in range(block['start'], block['stop']):
                writer.writerow(dict(item_id=position, dataset=block['dataset'], dataset_name=block['dataset_name']))
            position = block['stop']
    global_dir = root / 'outputs/routereval_raw_global'
    matrix = global_dir / 'all_response_matrix_q_by_m.npy'
    if not matrix.exists():
        subprocess.run([sys.executable, str(ROOT / 'tools/build_routereval_raw_global_matrix.py'),
            '--score-dir', str(extracted), '--prompts', str(metadata), '--out-dir', str(global_dir)], check=True)
        generated = global_dir / 'routereval_raw_50265x8577_q_by_m.npy'
        if generated.exists():
            generated.rename(matrix)
    verified(matrix, spec['expected_global_matrix_sha256'])
    response_dir = root / 'outputs/routereval_raw_matrices'
    if not (response_dir / 'mmlu_pro_response_matrix.npz').exists():
        generated = response_dir / 'mmlu_pro_raw_responses.npz'
        if not generated.exists():
            subprocess.run([sys.executable, str(ROOT / 'tools/export_routereval_raw_matrices.py'),
                '--score-dir', str(extracted), '--prompts', str(metadata), '--out-dir', str(response_dir)], check=True)
        shutil.copyfile(generated, response_dir / 'mmlu_pro_response_matrix.npz')
    # Check the arrays, independently of ZIP/NPZ serialization details.
    with np.load(response_dir / 'mmlu_pro_response_matrix.npz', allow_pickle=False) as archive:
        responses = archive['responses_q_by_m']
        if responses.shape != (12032, 1823) or not np.isin(responses, [0, 1]).all():
            raise ValueError('MMLU-Pro response shape or values differ from the paper input')
        response_hash = hashlib.sha256(np.asarray(responses, dtype=np.int8, order='C').tobytes()).hexdigest()
        model_hash = hashlib.sha256(('\n'.join(archive['model_ids'].astype(str)) + '\n').encode()).hexdigest()
        if response_hash != spec['mmlu_pro_responses_int8_sha256'] or model_hash != spec['mmlu_pro_model_order_utf8_sha256']:
            raise ValueError('MMLU-Pro response values or model ordering do not match the paper input')
    item_target = root / 'external_cache/mmlu_pro_test.parquet'
    item_target.parent.mkdir(parents=True, exist_ok=True)
    if items.resolve() != item_target.resolve():
        if item_target.exists():
            verified(item_target, spec['mmlu_pro_parquet_sha256'])
        else:
            shutil.copyfile(items, item_target)
    paths = [matrix, global_dir / 'routereval_raw_50265_items.csv',
             global_dir / 'routereval_raw_8577_models.csv',
             response_dir / 'mmlu_pro_response_matrix.npz', item_target]
    report = dict(upstream_verified=True, matrix_content_verified=True, mmlu_pro_content_and_order_verified=True,
                  files=[dict(path=str(p.relative_to(root)), sha256=sha256(p)) for p in paths])
    (root / 'input_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Inputs ready. Set FAMILY_DIF_PROJECT_ROOT to your data-root before paper commands.')
    print('The raw global matrix is approximately 1.7 GB; question text is not needed for numeric audits.')


if __name__ == '__main__':
    main()
