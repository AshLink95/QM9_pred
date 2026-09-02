"""Thin CLI: parse SIESTA/Wannier files -> cached dataset (CLAUDE.md §11)."""

import argparse

from data.dataset import build_dataset, save_cache
from training.train import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="+", help="one or more roots, each searched recursively")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="dataset.pkl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    examples, meta = build_dataset(args.data_dir, cfg["wannier"]["spins"])
    save_cache(args.out, examples, meta)
    print(f"cached {len(examples)} molecules -> {args.out}  (max_centers={meta['max_centers']})")


if __name__ == "__main__":
    main()
