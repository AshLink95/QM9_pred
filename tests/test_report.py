"""Accuracy report: outputs written, order preserved, fresh folders parsed, CLI end to end."""

import csv
import sys

import jax
import numpy as np
import yaml

from data.dataset import save_cache
from model.model import model_from_config
from training import report as R
from training.train import save_params

from tests._fake import WANNIER_CFG, molecules

CFG = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": WANNIER_CFG,
    "train": {"lr": 1e-3, "batch_size": 4, "epochs": 1, "seed": 0,
              "val_frac": 0.25, "test_frac": 0.25},
}


def _examples(n=9, seed=0):
    return molecules(n, seed)


def _model_params(ex):
    model = model_from_config(CFG)
    return model, model.init(jax.random.PRNGKey(0), ex[0]["z"], ex[0]["pos"])


def test_predict_all_keeps_input_order():
    ex = _examples()
    model, params = _model_params(ex)
    preds = R.predict_all(model, params, ex, batch_size=4)
    from data.electrons import centers_per_spin
    from model.heads import select_centers
    for e, p in zip(ex, preds):                  # same molecule's prediction, in input order
        k = centers_per_spin(e["z"], 2)
        own = select_centers(model.apply(params, e["z"], e["pos"])["wannier"], k)
        for s in range(2):
            np.testing.assert_allclose(np.sort(p["centers"][s][:k[s]], axis=0),
                                       np.sort(own["centers"][s][:k[s]], axis=0), atol=1e-4)


def test_report_outputs(tmp_path):
    ex = _examples()
    model, params = _model_params(ex)
    preds = R.predict_all(model, params, ex, batch_size=4)
    mol, cen, summary, data = R.build_report(ex, preds, [f"m{i}" for i in range(len(ex))],
                                             ["up", "down"])
    R.write_outputs(tmp_path, mol, cen, R.format_summary(summary), data, ["up", "down"])
    for f in ("molecules.csv", "centers.csv", "summary.txt", *(f"plots/{p}" for p in R.PLOTS)):
        assert (tmp_path / f).stat().st_size > 0
    rows = list(csv.DictReader(open(tmp_path / "molecules.csv")))
    assert [r["molecule"] for r in rows] == [f"m{i}" for i in range(len(ex))]
    n_true = sum(len(w["radii"]) for e in ex for w in e["wannier"])
    centers = list(csv.DictReader(open(tmp_path / "centers.csv")))
    # every true center appears exactly once (matched or missed)
    assert sum(r["status"] in ("matched", "missed") for r in centers) == n_true


FDF = """%block ChemicalSpeciesLabel
 1 6 C
 2 1 H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
 0 0 0 1
 0 0 1.1 2
%endblock AtomicCoordinatesAndAtomicSpecies
"""
WOUT = """ Final State
  WF centre and spread    1  (  0.000000,  0.000000,  0.500000 )     0.25000000
 Sum of centres and spreads ( x )
"""


def test_load_fresh_recursive_two_roots(tmp_path):
    for d in ("done/sub", "wout_files"):
        (tmp_path / d).mkdir(parents=True)
    for mid in ("CH_1", "CH_2"):
        (tmp_path / "done/sub" / f"{mid}.fdf").write_text(FDF)            # nested -> recursive
        for s in ("up", "down"):
            (tmp_path / "wout_files" / f"{mid}.manifold.valence.{s}.wout").write_text(WOUT)
    (tmp_path / "done" / "CH_9.fdf").write_text(FDF)                       # no .wout
    ex, skipped = R.load_fresh([tmp_path], ["up", "down"])
    assert [e["id"] for e in ex] == ["CH_1", "CH_2"]
    assert skipped == [("CH_9", "missing spin .wout")]
    assert "energy" not in ex[0] and len(ex[0]["wannier"][0]["radii"]) == 1


def test_cli_default_test_split(tmp_path, monkeypatch, capsys):
    ex = _examples(12)
    save_cache(tmp_path / "ds.pkl", ex, {"max_centers": 10, "spins": ["up", "down"]})
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(CFG))
    _, params = _model_params(ex)
    save_params(tmp_path / "p.msgpack", params)

    from scripts import evaluate
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--config", str(tmp_path / "cfg.yaml"), "--params",
        str(tmp_path / "p.msgpack"), "--dataset", str(tmp_path / "ds.pkl"),
        "--out-dir", str(tmp_path / "rep"), "--batch-size", "4"])
    evaluate.main()
    out = capsys.readouterr().out
    assert "subset 'test'  ->  3 molecules" in out              # 25% of 12
    assert all((tmp_path / "rep" / "plots" / p).exists() for p in R.PLOTS)
