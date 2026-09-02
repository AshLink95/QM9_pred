"""Batch prediction over a directory: only .fdf read, single JSON out, record shape.

Uses random-init params (values irrelevant — this checks the I/O plumbing, not accuracy)."""

import json

import jax
import numpy as np

from main import run_directory
from model.model import model_from_config
from training.train import load_config, save_params

FDF = """\
NumberOfAtoms 2
%block ChemicalSpeciesLabel
 1 6 C
 2 1 H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
 0.0 0.0 0.0 1
 0.0 0.0 1.1 2
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def test_run_directory_ignores_non_fdf_and_writes_json(tmp_path):
    (tmp_path / "mol1.fdf").write_text(FDF)
    (tmp_path / "mol2.fdf").write_text(FDF)
    (tmp_path / "notes.txt").write_text("ignore me")        # non-.fdf must be ignored
    (tmp_path / "mol1.out").write_text("siesta: Total = -1.0")  # sibling, not globbed as input

    cfg = load_config("configs/default.yaml")
    # a params file with the right structure (random init) so load_model can fill the template
    model = model_from_config(cfg)
    z, pos = np.array([6, 1]), np.zeros((2, 3))
    params_path = tmp_path / "params.msgpack"
    save_params(params_path, model.init(jax.random.PRNGKey(0), z, pos))

    out_path = tmp_path / "predictions.json"
    records = run_directory(tmp_path, cfg, out_path, params_path)

    data = json.loads(out_path.read_text())
    assert isinstance(data, list) and len(data) == 2        # notes.txt ignored, 2 fdf in
    inputs = {r["input"] for r in data}
    assert inputs == {"mol1.fdf", "mol2.fdf"}
    for r in data:
        assert "energy" in r and isinstance(r["energy"], float)
        assert set(r["wannier"]) == set(cfg["wannier"]["spins"])
        for spin_centers in r["wannier"].values():
            for c in spin_centers:                          # centers (if any) carry all 3 fields
                assert set(c) == {"position", "radius", "presence"}
                assert len(c["position"]) == 3
    assert records == data                                  # return value matches file
