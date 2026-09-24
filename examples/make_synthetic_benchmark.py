"""Create a small illustrative benchmark with explicit owner/family metadata.

Usage: python examples/make_synthetic_benchmark.py --output demo.npz
Synthetic data test the interface, not scientific validity or effect recovery.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit


def make_benchmark(seed: int = 17, items: int = 512, owners: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    families = 3
    family_code = np.tile(np.arange(families), owners)
    owner_code = np.repeat(np.arange(owners), families)
    coordinates = rng.normal(size=(owners * families, 3))
    loadings = rng.normal(0, 0.7, size=(items, 3))
    intercepts = rng.normal(0, 0.6, size=items)
    residual_family = rng.normal(0, 0.55, size=(items, families))
    residual_family[items // 2:] = 0
    logits = intercepts[:, None] + loadings @ coordinates.T + residual_family[:, family_code]
    return {
        "responses": rng.binomial(1, expit(logits)).astype(np.int8),
        "model_id": np.asarray([f"model_{i:04d}" for i in range(len(family_code))]),
        "family": np.asarray([f"family_{i}" for i in family_code]),
        "owner": np.asarray([f"owner_{i:04d}" for i in owner_code]),
        "item_group": np.asarray([f"task_{i % 4}" for i in range(items)]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--items", type=int, default=512)
    parser.add_argument("--owners", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    if args.items < 32 or args.owners < 16:
        parser.error("this example requires at least 32 items and 16 owners")
    if args.output.suffix.lower() != ".npz":
        parser.error("output filename must end in .npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **make_benchmark(args.seed, args.items, args.owners))
    print("Synthetic benchmark written. It is illustrative, not an empirical paper result.")
