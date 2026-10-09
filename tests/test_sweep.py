"""Lock-free run queue: no duplicates, exclusive claims, dead-owner reclaim, preemption,
exactly-once promotion, and a tiny sweep end to end."""

import copy
import json
import threading

import yaml

from data.dataset import save_cache
from tests._fake import WANNIER_CFG, molecules
from training import sweep

BASE = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": WANNIER_CFG,
    "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
    "train": {"lr": 5e-3, "batch_size": 4, "epochs": 2, "ckpt_every": 1, "seed": 0,
              "split_seed": 0, "val_frac": 0.25, "test_frac": 0.25, "patience": 0,
              "lr_schedule": "cosine", "warmup_epochs": 1, "lr_end_frac": 0.01},
}


def _spec(tmp_path, n_runs=3):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(BASE))
    save_cache(tmp_path / "ds.pkl", molecules(16), {"max_centers": 10, "spins": ["up", "down"]})
    spec = {"name": "t", "root": str(tmp_path / "runs"), "base": str(tmp_path / "base.yaml"),
            "dataset": str(tmp_path / "ds.pkl"), "stage1_overrides": {"train.epochs": 2},
            "stage2_overrides": {"train.epochs": 3},
            "runs": [{"name": f"r{i}", "overrides": {"train.seed": i}} for i in range(n_runs)]}
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path, spec


def test_init_is_idempotent(tmp_path):
    _, spec = _spec(tmp_path)
    q = sweep.RunQueue(sweep.sweep_root(spec))
    sweep.init(q, spec); sweep.init(q, spec)
    assert q.names("queued") == ["r0", "r1", "r2"]
    assert yaml.safe_load((q.root / "r1" / "config.yaml").read_text())["train"]["seed"] == 1


def test_concurrent_claims_are_exclusive(tmp_path):
    _, spec = _spec(tmp_path, n_runs=12)
    q = sweep.RunQueue(sweep.sweep_root(spec))
    sweep.init(q, spec)
    got, lock = [], threading.Lock()

    def worker():
        while (name := q.claim()) is not None:
            with lock:
                got.append(name)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert sorted(got) == sorted(f"r{i}" for i in range(12))   # each claimed exactly once
    assert q.names("queued") == [] and len(q.names("running")) == 12


def test_dead_owner_is_reclaimed_but_own_run_is_not(tmp_path):
    _, spec = _spec(tmp_path)
    q = sweep.RunQueue(sweep.sweep_root(spec))
    sweep.init(q, spec)
    dead, mine = q.claim(), q.claim()
    (q.root / dead / "owner.json").write_text(json.dumps({"slurm_job_id": "999", "pid": -1}))
    (q.root / mine / "owner.json").write_text(json.dumps(sweep.owner_info()))
    sweep.reclaim_stale(q, 60, alive=lambda owner, mtime, stale: False)
    assert q.state_of(dead) == "queued" and q.state_of(mine) == "running"


def test_promotion_happens_exactly_once(tmp_path):
    _, spec = _spec(tmp_path, n_runs=2)
    q = sweep.RunQueue(sweep.sweep_root(spec))
    sweep.init(q, spec)
    for name, obj in (("r0", 0.5), ("r1", 0.2)):
        q.move(name, "queued", "finished")
        (q.root / name / "params.best.json").write_text(json.dumps({"objective": obj}))
    results = []
    threads = [threading.Thread(target=lambda: results.append(sweep.maybe_promote(q, spec)))
               for _ in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert results.count(True) == 1 and q.state_of(sweep.FINAL) == "queued"
    cfg = yaml.safe_load((q.root / sweep.FINAL / "config.yaml").read_text())
    assert cfg["train"]["epochs"] == 3 and cfg["train"]["seed"] == 1   # from r1 (lower obj)


def test_stop_signal_requeues_run(tmp_path):
    _, spec = _spec(tmp_path, n_runs=1)
    q = sweep.RunQueue(sweep.sweep_root(spec))
    sweep.init(q, spec)
    name = q.claim()
    from data.dataset import load_cache
    ex, meta = load_cache(spec["dataset"])
    sweep.run_one(q, name, spec, ex, meta, {"flag": True})
    assert q.state_of(name) == "queued"
    assert (q.root / name / "params.last.msgpack").exists()   # resumes from here


def test_tiny_sweep_end_to_end(tmp_path):
    path, spec = _spec(tmp_path, n_runs=2)
    sweep.work(path)                       # trains r0, r1, then promotes and trains 'final'
    q = sweep.RunQueue(sweep.sweep_root(spec))
    assert q.names("finished") == ["final", "r0", "r1"]
    for name in ("r0", "r1", "final"):
        assert (q.root / name / "report" / "summary.txt").exists()
        assert json.loads((q.root / name / "result.json").read_text())["test_energy_mae"] > 0
    table = sweep.status(path)
    assert "final" in table and "r0" in table and " *" in table
