"""Thin CLI: config + cached dataset -> trained params (CLAUDE.md §11).

Writes next to --out (default params.msgpack): the best parameters (--out itself), the full
resumable state (params.last.msgpack) and the best epoch's validation metrics (params.best.json).
Re-running the same command resumes. SIGTERM/SIGUSR1 (e.g. Slurm's time limit) saves the state
and exits cleanly.
"""

import argparse
import signal

from data.dataset import load_cache
from training.train import load_config, train_with_status


def main():
    ap = argparse.ArgumentParser(description="Train the model on a cached dataset.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="dataset.pkl")
    ap.add_argument("--out", default="params.msgpack", help="best parameters (resume key)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    examples, meta = load_cache(args.dataset)
    # size the Wannier head to the dataset (§6, §9) rather than trusting the config guess
    cfg["wannier"]["max_centers"] = max(cfg["wannier"]["max_centers"], meta["max_centers"])

    stop = {"flag": False}
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.update(flag=True))

    _, status = train_with_status(cfg, examples, ckpt_path=args.out,
                                  should_stop=lambda: stop["flag"])
    print(f"training {status}; best parameters -> {args.out}")


if __name__ == "__main__":
    main()
