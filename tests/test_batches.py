"""Precomputed batches: fixed size, one atom count per batch, full coverage, reproducible."""

import numpy as np

from tests._fake import molecules
from training.train import epoch_batches, eval_batches, stack_buckets, steps_per_epoch

EX = molecules(40, seed=3)
BUCKETS = stack_buckets(EX, n_spins=2, kmax=10)
B = 4


def test_every_batch_is_full_and_single_size():
    for n, idx in epoch_batches(BUCKETS, B, np.random.default_rng(0)):
        assert len(idx) == B
        assert idx.max() < len(BUCKETS[n]["k"])            # top-ups come from the same bucket
        assert {len(EX[i]["z"]) for i in BUCKETS[n]["index"][idx]} == {n}


def test_every_molecule_seen_each_epoch():
    seen = set()
    for n, idx in epoch_batches(BUCKETS, B, np.random.default_rng(1)):
        seen |= set(BUCKETS[n]["index"][idx].tolist())
    assert seen == set(range(len(EX)))


def test_same_seed_same_batches():
    a = epoch_batches(BUCKETS, B, np.random.default_rng([0, 5]))
    b = epoch_batches(BUCKETS, B, np.random.default_rng([0, 5]))
    assert [(n, i.tolist()) for n, i in a] == [(n, i.tolist()) for n, i in b]
    assert len(a) == steps_per_epoch(BUCKETS, B)


def test_eval_batches_cover_each_molecule_once():
    real = [BUCKETS[n]["index"][idx[:r]] for n, idx, r in eval_batches(BUCKETS, B)]
    flat = np.concatenate(real).tolist()
    assert sorted(flat) == list(range(len(EX)))
