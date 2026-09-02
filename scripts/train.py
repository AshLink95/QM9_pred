"""Thin CLI: config + cached dataset -> trained params (CLAUDE.md §11)."""

import argparse

from data.dataset import load_cache
from training.train import load_config, save_params, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="dataset.pkl")
    ap.add_argument("--out", default="params.msgpack")
    args = ap.parse_args()

    cfg = load_config(args.config)
    examples, meta = load_cache(args.dataset)
    # size the Wannier head to the dataset (§6, §9) rather than trusting the config guess
    cfg["wannier"]["max_centers"] = max(cfg["wannier"]["max_centers"], meta["max_centers"])
    params = train(cfg, examples)
    save_params(args.out, params)
    print(f"saved params -> {args.out}")


if __name__ == "__main__":
    main()
