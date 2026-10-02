"""Thin CLI: accuracy benchmark — parsed truth vs model output (CLAUDE.md §8, §10, §11).

Default: the held-out TEST split of the same dataset.pkl the model was trained on (the split is
recomputed with the training config's seed/fractions, so it's the exact molecules training never
saw). Fresh set: --data-dir ROOT [ROOT ...] parses .fdf/.out/.wout recursively (matched by
molecule id) and scores every parsed molecule. Logic lives in training/report.py.

    python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack
    python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack \\
        --data-dir /path/to/fresh_parent
"""

import argparse

import jax
import numpy as np

from data.dataset import load_cache, split
from model.model import model_from_config
from training import report as R
from training.train import load_config, load_params


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark a trained model: energy + Wannier accuracy vs SIESTA outputs.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="the config the model was TRAINED with (dims + split seed/fractions)")
    ap.add_argument("--params", default="params.msgpack")
    ap.add_argument("--dataset", default="dataset.pkl", help="default mode: training dataset")
    ap.add_argument("--subset", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--data-dir", nargs="+", metavar="ROOT",
                    help="fresh set: 1+ roots searched recursively for .fdf/.out/.wout")
    ap.add_argument("--out-dir", default="accuracy_report")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--presence-threshold", type=float, default=0.5,
                    help="slots with presence above this count as predicted centers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spins = cfg["wannier"]["spins"]
    # slot count from the checkpoint itself, so shapes match what was trained
    cfg["wannier"]["max_centers"] = R.max_centers_from_checkpoint(args.params, len(spins))

    header, skipped = [], []
    if args.data_dir:
        examples, skipped = R.load_fresh(args.data_dir, spins)
        names = [e["id"] for e in examples]
        header.append(f"Fresh set: {', '.join(args.data_dir)}  ->  {len(examples)} molecules"
                      f" parsed, {len(skipped)} skipped")
    else:
        allex, _ = load_cache(args.dataset)
        t = cfg["train"]
        parts = dict(zip(("train", "val", "test"),
                         split(list(range(len(allex))), t["val_frac"], t["test_frac"],
                               t["seed"])))
        idx = range(len(allex)) if args.subset == "all" else parts[args.subset]
        examples = [allex[i] for i in idx]
        names = [f"pkl#{i}" for i in idx]
        header.append(f"{args.dataset}: subset '{args.subset}'  ->  {len(examples)} molecules")
    if not examples:
        raise SystemExit("no molecules to evaluate")

    model = model_from_config(cfg)
    template = model.init(jax.random.PRNGKey(0), np.asarray(examples[0]["z"]),
                          np.asarray(examples[0]["pos"]))
    try:
        params = load_params(args.params, template)
    except Exception as err:
        raise SystemExit(f"{args.params} doesn't match {args.config} ({err}). "
                         "Use the config the model was trained with.")

    preds = R.predict_all(model, params, examples, args.batch_size)
    ref = params["params"].get("atom_ref", {}).get("embedding")
    atom_ref = None if ref is None else np.asarray(ref)[:, 0]
    mol_rows, center_rows, summary, data = R.build_report(
        examples, preds, names, spins, args.presence_threshold, atom_ref)

    de, dc = R.rotation_drift(model, params, examples[0])
    header = [f"Accuracy report — model {args.params}"] + header
    text = R.format_summary(summary, header) + (
        f"\n\nSYMMETRY spot-check (random rotation, one molecule): energy drift {de:.2e} eV, "
        f"center drift {dc:.2e} A")
    if skipped:
        text += "\n\nSKIPPED (first 20):\n" + "\n".join(f"  {m}: {why}" for m, why in skipped[:20])
    R.write_outputs(args.out_dir, mol_rows, center_rows, text, data, spins)
    print(text)
    print(f"\nwrote {args.out_dir}/: molecules.csv, centers.csv, summary.txt, accuracy.png")


if __name__ == "__main__":
    main()
