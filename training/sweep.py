"""Parallel sweep runner: a lock-free run queue on a shared filesystem (docs/sweeps.md).

Many cluster jobs run the same worker (`scripts/train-bunch.sh` → `python -m scripts.sweep work`).
Each job repeatedly claims one run from the queue, trains it, writes its accuracy report, and
claims the next, until nothing is left. Jobs never block each other and never hold a lock:

  * a run's state is the folder its marker file sits in: queue/{queued,running,finished,failed}/;
  * creating a run = os.mkdir(run_dir)  — atomic, fails if it exists ⇒ never duplicated;
  * claiming a run = os.rename(queued/x, running/x) — atomic ⇒ exactly one job wins;
  * a run whose owner job died (squeue no longer lists it, or no heartbeat for stale_minutes) is
    moved back to queued/ the same way, and the next job RESUMES it from its last checkpoint;
  * on SIGTERM/SIGUSR1 (time limit) a job saves state and puts its run back in queued/.

Two stages: stage 1 = the runs listed in the sweep file (screening, with early stopping); when
all are done, the first worker to notice creates run `final` from the best stage-1 run's config
plus stage-2 overrides (a long run) — again via atomic mkdir, so it's created exactly once.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

STATES = ("queued", "running", "finished", "failed")
FINAL = "final"


# ------------------------------------------------------------------ sweep file / configs
def load_sweep(path):
    spec = yaml.safe_load(Path(path).read_text())
    for key in ("name", "base", "dataset", "runs"):
        if key not in spec:
            raise ValueError(f"sweep file {path} is missing '{key}'")
    names = [r["name"] for r in spec["runs"]]
    if len(set(names)) != len(names) or FINAL in names:
        raise ValueError(f"run names must be unique and not '{FINAL}' (reserved for stage 2)")
    return spec


def set_dotted(cfg, key, value):
    """set_dotted(cfg, "train.lr", 5e-4) → cfg["train"]["lr"] = 5e-4 (key must exist)."""
    *path, last = key.split(".")
    node = cfg
    for p in path:
        node = node[p]
    if last not in node:
        raise KeyError(f"override '{key}': no such config key")
    node[last] = value


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg


def stage1_config(spec, run):
    base = yaml.safe_load(Path(spec["base"]).read_text())
    return apply_overrides(apply_overrides(base, spec.get("stage1_overrides")),
                           run.get("overrides"))


def sweep_root(spec):
    return Path(spec.get("root", "runs")) / spec["name"]


# ------------------------------------------------------------------ the queue
class RunQueue:
    """Marker files under <root>/queue/<state>/<run>; run data under <root>/<run>/."""

    def __init__(self, root):
        self.root = Path(root)
        for st in STATES:
            (self.root / "queue" / st).mkdir(parents=True, exist_ok=True)

    def marker(self, state, name):
        return self.root / "queue" / state / name

    def names(self, state):
        return sorted(os.listdir(self.root / "queue" / state))

    def state_of(self, name):
        return next((st for st in STATES if self.marker(st, name).exists()), None)

    def add(self, name, cfg):
        """Create run `name` with this config and queue it. Atomic mkdir ⇒ only the first
        caller creates it; returns False if it already existed."""
        run_dir = self.root / name
        try:
            os.mkdir(run_dir)
        except FileExistsError:
            return False
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.marker("queued", name).touch()
        return True

    def repair_orphans(self, min_age_s=120):
        """A run dir with a config but no marker anywhere (its creator died between mkdir and
        touching the marker) is queued. O_EXCL keeps this race-free."""
        for d in self.root.iterdir():
            if d.name == "queue" or not (d / "config.yaml").exists():
                continue
            if self.state_of(d.name) is None and time.time() - d.stat().st_mtime > min_age_s:
                try:
                    os.close(os.open(self.marker("queued", d.name),
                                     os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    pass

    def move(self, name, src, dst):
        """Atomic state change; returns False if the marker was no longer in `src`."""
        try:
            os.rename(self.marker(src, name), self.marker(dst, name))
            return True
        except FileNotFoundError:
            return False

    def claim(self):
        """Claim the first queued run (sorted by name), or None. Losing a rename race just
        means another job took that run; try the next one."""
        for name in self.names("queued"):
            if self.move(name, "queued", "running"):
                return name
        return None


def owner_info():
    return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "host": socket.gethostname(), "pid": os.getpid(), "started": time.time()}


def owner_alive(owner, marker_mtime, stale_s):
    """Is the job that owns a running marker still alive? Ask Slurm when we know the job id;
    otherwise (no Slurm, or squeue failing) fall back to the heartbeat age."""
    job = (owner or {}).get("slurm_job_id")
    if job:
        try:
            r = subprocess.run(["squeue", "-h", "-j", str(job)], capture_output=True, text=True,
                               timeout=30)
            if r.returncode == 0:
                return bool(r.stdout.strip())
            if "Invalid job id" in r.stderr:      # finished jobs are reported this way
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return time.time() - marker_mtime < stale_s


def reclaim_stale(q, stale_s, alive=owner_alive):
    """Move runs whose owner died back to queued/ (they resume from their last checkpoint)."""
    me = (os.environ.get("SLURM_JOB_ID"), socket.gethostname(), os.getpid())
    for name in q.names("running"):
        owner_path = q.root / name / "owner.json"
        try:
            owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
            mtime = q.marker("running", name).stat().st_mtime
        except (FileNotFoundError, json.JSONDecodeError):
            continue                               # changed under us; look again next time
        if owner and (owner.get("slurm_job_id"), owner.get("host"), owner.get("pid")) == me:
            continue
        if not alive(owner, mtime, stale_s) and q.move(name, "running", "queued"):
            print(f"[sweep] reclaimed '{name}' (owner {owner and owner.get('slurm_job_id')} "
                  f"is gone) — it will resume from its last checkpoint", flush=True)


# ------------------------------------------------------------------ stage 2
def best_val(q, name):
    p = q.root / name / "params.best.json"
    return json.loads(p.read_text())["objective"] if p.exists() else float("inf")


def maybe_promote(q, spec, force=False):
    """Create the stage-2 run from the best finished stage-1 run, once all stage-1 runs are
    finished/failed (or immediately with force). Returns True if this call created it."""
    if (q.root / FINAL).exists():
        return False
    states = {r["name"]: q.state_of(r["name"]) for r in spec["runs"]}
    if not force and any(s not in ("finished", "failed") for s in states.values()):
        return False
    finished = [n for n, s in states.items() if s == "finished"]
    if not finished:
        return False
    winner = min(finished, key=lambda n: best_val(q, n))
    cfg = yaml.safe_load((q.root / winner / "config.yaml").read_text())
    cfg = apply_overrides(cfg, spec.get("stage2_overrides"))
    if q.add(FINAL, cfg):
        (q.root / FINAL / "promoted_from.txt").write_text(
            f"{winner} (best val objective {best_val(q, winner):.6f})\n")
        print(f"[sweep] stage 2: '{FINAL}' created from '{winner}'", flush=True)
        return True
    return False


# ------------------------------------------------------------------ one run
class _Tee:
    """Write to the job's stdout and to the run's train.log."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


def _heartbeat(marker, stop_event, every_s=60):
    while not stop_event.wait(every_s):
        try:
            os.utime(marker)
        except FileNotFoundError:
            return                                  # run moved/reclaimed; stop beating


def run_one(q, name, spec, examples, meta, stop):
    """Train one claimed run, report on the test split, and set its final state."""
    from training.report import evaluate_checkpoint
    from training.train import train_with_status

    run_dir = q.root / name
    (run_dir / "owner.json").write_text(json.dumps(owner_info()))
    beat_stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(q.marker("running", name), beat_stop),
                     daemon=True).start()
    tee = _Tee(run_dir / "train.log")
    sys.stdout = tee
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        cfg["wannier"]["max_centers"] = max(cfg["wannier"]["max_centers"], meta["max_centers"])
        print(f"[sweep] === run '{name}' on {socket.gethostname()} "
              f"(job {os.environ.get('SLURM_JOB_ID')}) ===", flush=True)
        _, status = train_with_status(cfg, examples, ckpt_path=run_dir / "params.msgpack",
                                      should_stop=lambda: stop["flag"])
        if status == "stopped":
            q.move(name, "running", "queued")
            print(f"[sweep] '{name}' stopped and re-queued; another job will resume it",
                  flush=True)
            return
        if status == "diverged":
            (run_dir / "error.txt").write_text("training diverged (non-finite loss)\n")
            q.move(name, "running", "failed")
            return
        _, summary = evaluate_checkpoint(cfg, run_dir / "params.msgpack", run_dir / "report",
                                         examples=examples, dataset_path=spec["dataset"])
        w = summary["wannier"]["all"]
        (run_dir / "result.json").write_text(json.dumps({
            "status": status, "test_energy_mae": summary["energy"]["mae"],
            "test_center_mae": w["center_mae"], "test_radius_mae": w["radius_mae"],
            "test_count_accuracy": w["count_accuracy"]}, indent=2))
        if not q.move(name, "running", "finished"):
            print(f"[sweep] WARNING: '{name}' was reclaimed by another job while running",
                  flush=True)
        print(f"[sweep] '{name}' finished ({status})", flush=True)
    except Exception:
        (run_dir / "error.txt").write_text(traceback.format_exc())
        q.move(name, "running", "failed")
        print(f"[sweep] '{name}' FAILED — see {run_dir / 'error.txt'}", flush=True)
    finally:
        beat_stop.set()
        sys.stdout = sys.__stdout__
        tee.f.close()


# ------------------------------------------------------------------ commands
def init(q, spec):
    """Create every stage-1 run that doesn't exist yet (safe to call from many jobs at once)."""
    for run in spec["runs"]:
        q.add(run["name"], stage1_config(spec, run))
    q.repair_orphans()


def work(spec_path, stop=None):
    """The worker loop: claim → train → report, until nothing is claimable."""
    from data.dataset import load_cache

    spec = load_sweep(spec_path)
    q = RunQueue(sweep_root(spec))
    init(q, spec)
    stop = stop if stop is not None else {"flag": False}
    stale_s = 60 * spec.get("stale_minutes", 30)
    examples, meta = load_cache(spec["dataset"])      # once per job, shared by all its runs
    print(f"[sweep] worker started: {len(examples)} molecules, queue {q.root}", flush=True)
    while not stop["flag"]:
        reclaim_stale(q, stale_s)
        name = q.claim()
        if name is None:
            if maybe_promote(q, spec):
                continue
            break
        print(f"[sweep] claimed '{name}'", flush=True)
        run_one(q, name, spec, examples, meta, stop)
    print("[sweep] worker exiting (nothing left to claim)" if not stop["flag"]
          else "[sweep] worker exiting (stop signal)", flush=True)


def status(spec_path):
    """One table of every run, best validation objective first."""
    spec = load_sweep(spec_path)
    q = RunQueue(sweep_root(spec))
    names = [r["name"] for r in spec["runs"]] + ([FINAL] if (q.root / FINAL).exists() else [])
    rows = []
    for n in names:
        d = q.root / n
        rd = lambda f: json.loads((d / f).read_text()) if (d / f).exists() else {}
        best, prog, res = rd("params.best.json"), rd("params.progress.json"), rd("result.json")
        owner = rd("owner.json")
        centers = best.get("center_mae", {})
        rows.append({
            "run": n, "state": q.state_of(n) or "-",
            "epoch": f"{prog.get('epoch', '-')}/{prog.get('epochs', '-')}",
            "val_obj": best.get("objective", float("inf")),
            "val_E": best.get("energy_mae"), "val_center": (sum(centers.values()) / len(centers)
                                                            if centers else None),
            "val_radius": best.get("radius_mae"), "test_E": res.get("test_energy_mae"),
            "test_center": res.get("test_center_mae"), "job": owner.get("slurm_job_id") or "-"})
    stage1 = [r for r in rows if r["run"] != FINAL]
    winner = min((r for r in stage1 if r["state"] == "finished"), key=lambda r: r["val_obj"],
                 default=None)
    f = lambda x: "-" if x is None or x == float("inf") else f"{x:.4f}"
    head = (f"{'run':<16}{'state':<10}{'epoch':>10}{'val obj':>10}{'val E':>9}{'val ctr':>9}"
            f"{'val rad':>9}{'test E':>9}{'test ctr':>9}  job")
    lines = [f"sweep '{spec['name']}' ({q.root})", head, "-" * len(head)]
    for r in sorted(stage1, key=lambda r: r["val_obj"]) + [r for r in rows if r["run"] == FINAL]:
        mark = " *" if winner and r["run"] == winner["run"] else ""
        lines.append(f"{r['run'] + mark:<16}{r['state']:<10}{r['epoch']:>10}{f(r['val_obj']):>10}"
                     f"{f(r['val_E']):>9}{f(r['val_center']):>9}{f(r['val_radius']):>9}"
                     f"{f(r['test_E']):>9}{f(r['test_center']):>9}  {r['job']}")
    lines.append("* = best finished stage-1 run (promoted to 'final'). E in eV, ctr/rad in Å.")
    return "\n".join(lines)


def requeue(spec_path, name):
    q = RunQueue(sweep_root(load_sweep(spec_path)))
    if not q.move(name, "failed", "queued"):
        raise SystemExit(f"'{name}' is not in failed/ (state: {q.state_of(name)})")
    (q.root / name / "error.txt").unlink(missing_ok=True)
    return f"'{name}' moved back to queued (it resumes from its last checkpoint)"


def promote(spec_path):
    spec = load_sweep(spec_path)
    q = RunQueue(sweep_root(spec))
    if maybe_promote(q, spec, force=True):
        return f"'{FINAL}' created and queued; any running worker or new job will train it"
    return f"nothing promoted ('{FINAL}' already exists, or no stage-1 run has finished)"
