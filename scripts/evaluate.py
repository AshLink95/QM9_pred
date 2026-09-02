"""Thin CLI: load checkpoint -> metrics + symmetry suite (CLAUDE.md §8, §11)."""

import argparse

import jax
import numpy as np

from data.dataset import load_cache, split
from model.model import model_from_config
from symmetry import transforms as T
from training.loss import example_to_jax
from training.metrics import energy_mae, wannier_discrepancy
from training.train import load_config, load_params


def symmetry_report(apply_fn, params, ex):
    """Quick rotation/translation check on one molecule (full suite lives in tests/)."""
    z, pos = ex["z"], ex["pos"]
    base = apply_fn(params, z, pos)
    R = T.random_rotation(seed=0)
    rot = apply_fn(params, z, np.asarray(T.rotate(np.asarray(pos), R)))
    de = abs(float(rot["energy"]) - float(base["energy"]))
    dc = float(np.max(np.abs(np.asarray(rot["wannier"]["centers"])
                             - np.asarray(base["wannier"]["centers"]) @ R.T)))
    return {"rotation_energy_drift": de, "rotation_center_drift": dc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="dataset.pkl")
    ap.add_argument("--params", default="params.msgpack")
    args = ap.parse_args()

    cfg = load_config(args.config)
    examples, meta = load_cache(args.dataset)
    cfg["wannier"]["max_centers"] = max(cfg["wannier"]["max_centers"], meta["max_centers"])
    _, _, test = split(examples, cfg["train"]["val_frac"], cfg["train"]["test_frac"],
                       cfg["train"]["seed"])
    test_j = [example_to_jax(e) for e in test]

    model = model_from_config(cfg)
    params = model.init(jax.random.PRNGKey(0), test_j[0]["z"], test_j[0]["pos"])
    params = load_params(args.params, params)

    print(f"test energy MAE      = {energy_mae(model.apply, params, test_j):.4f} eV")
    print(f"test cloud L2        = {wannier_discrepancy(model.apply, params, test_j, cfg['wannier']['width_scale']):.4f}")
    print("symmetry             =", symmetry_report(model.apply, params, test_j[0]))


if __name__ == "__main__":
    main()
