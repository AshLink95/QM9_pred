"""Thin CLI for the parallel sweep (docs/sweeps.md). Logic: training/sweep.py.

    python -m scripts.sweep work    configs/sweep.yaml        # worker (what train-bunch.sh runs)
    python -m scripts.sweep status  configs/sweep.yaml        # table of all runs
    python -m scripts.sweep requeue configs/sweep.yaml RUN    # retry a failed run
    python -m scripts.sweep promote configs/sweep.yaml        # start stage 2 now
"""

import argparse
import signal

from training import sweep


def main():
    ap = argparse.ArgumentParser(description="Parallel training sweep with a lock-free queue.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (("work", "claim and train runs until none are left"),
                          ("status", "print a table of every run"),
                          ("promote", "create the stage-2 'final' run now")):
        sub.add_parser(cmd, help=helptext).add_argument("sweep")
    rq = sub.add_parser("requeue", help="move a failed run back to the queue")
    rq.add_argument("sweep"); rq.add_argument("run")
    args = ap.parse_args()

    if args.cmd == "work":
        stop = {"flag": False}                     # SIGTERM / SIGUSR1: save, re-queue, exit
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: stop.update(flag=True))
        sweep.work(args.sweep, stop)
    elif args.cmd == "status":
        print(sweep.status(args.sweep))
    elif args.cmd == "requeue":
        print(sweep.requeue(args.sweep, args.run))
    else:
        print(sweep.promote(args.sweep))


if __name__ == "__main__":
    main()
