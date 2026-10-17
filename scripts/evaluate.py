"""Thin CLI: accuracy benchmark — parsed truth vs model output (CLAUDE.md §8, §10, §11).

Default: the held-out TEST split of the same dataset.pkl the model was trained on (the split is
recomputed with the training config's split_seed/fractions, so it's the exact molecules training
never saw). Fresh set: --data-dir ROOT [ROOT ...] parses .fdf/.wout recursively (matched by
molecule id) and scores every parsed molecule. Logic: training/report.py:evaluate_checkpoint.

    python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack
    python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack \\
        --data-dir /path/to/fresh_parent
"""

import argparse

from training.report import evaluate_checkpoint
from training.train import load_config


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark a trained model: Wannier center/radius accuracy vs the .wout files.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="the config the model was TRAINED with (dims + split seed/fractions)")
    ap.add_argument("--params", default="params.msgpack")
    ap.add_argument("--dataset", default="dataset.pkl", help="default mode: training dataset")
    ap.add_argument("--subset", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--data-dir", nargs="+", metavar="ROOT",
                    help="fresh set: 1+ roots searched recursively for .fdf/.wout")
    ap.add_argument("--out-dir", default="accuracy_report")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    text, _ = evaluate_checkpoint(load_config(args.config), args.params, args.out_dir,
                                  dataset_path=args.dataset, subset=args.subset,
                                  data_dir=args.data_dir, batch_size=args.batch_size)
    print(text)
    print(f"\nwrote {args.out_dir}/: molecules.csv, centers.csv, summary.txt, plots/*.png")


if __name__ == "__main__":
    main()
