"""Predict energy + Wannier centers from SIESTA `.fdf` geometry (no `.wout` needed — the model
infers Wannier positions from geometry alone; CLAUDE.md §1).

Single file  -> human-readable block on stdout.
Directory    -> predict every *.fdf and write ONE predictions.json (records keyed by input file).

    uv run python main.py molecule.fdf --params params.msgpack
    uv run python main.py path/to/fdf_dir --params params.msgpack --out predictions.json
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from data.electrons import centers_per_spin
from data.parse_siesta import parse_fdf
from model.heads import select_centers
from model.model import model_from_config
from training.report import nms_radius
from training.train import load_config, load_params


def load_model(cfg, params_path, z0, pos0):
    """Build the model and load trained params ONCE. Param shapes are independent of atom count,
    so the template from the first molecule applies to molecules of any size."""
    model = model_from_config(cfg)
    template = model.init(jax.random.PRNGKey(0), z0, pos0)
    return model, load_params(params_path, template)


def record_for(name, out, spins, z, nms=0.0):
    """One JSON record: {input, energy, wannier:{spin:[{position, radius}, ...]}}.
    Each spin's centers are its k top-scoring candidates, k = valence electrons / 2
    (data/electrons.py)."""
    k = centers_per_spin(z, len(spins))
    sel = select_centers(out["wannier"], k, nms)
    centers, radii = sel["centers"], sel["radii"]
    wannier = {spin: [{"position": [round(float(v), 6) for v in centers[s, j]],
                       "radius": round(float(radii[s, j]), 6)} for j in range(k[s])]
               for s, spin in enumerate(spins)}
    return {"input": name, "energy": round(float(out["energy"]), 6), "wannier": wannier}


def run_directory(input_dir, cfg, out_path, params_path):
    """Batch predict every *.fdf in a directory -> single JSON file.

    Snapshot semantics (so files added mid-run are ignored): list the *.fdf names first, load all
    their contents into memory, and only then run the model and write output.
    """
    input_dir = Path(input_dir)
    # 1) snapshot the filenames (glob ignores every non-.fdf file)
    names = sorted(p.name for p in input_dir.glob("*.fdf"))
    if not names:
        raise FileNotFoundError(f"no .fdf files in {input_dir}")

    # 2) load all geometry into memory before any model work
    mols, errors = [], {}
    for name in names:
        try:
            z, pos = parse_fdf(input_dir / name)
            mols.append((name, np.asarray(z), np.asarray(pos)))
        except Exception as e:  # a bad file becomes an error record, doesn't abort the batch
            errors[name] = str(e)

    # 3) build the model once, predict each in-memory molecule
    spins = cfg["wannier"]["spins"]
    records = []
    if mols:
        model, params = load_model(cfg, params_path, mols[0][1], mols[0][2])
        for name, z, pos in mols:
            records.append(record_for(name, model.apply(params, z, pos), spins, z,
                                      nms_radius(cfg)))
    records.extend({"input": name, "error": err} for name, err in errors.items())
    records.sort(key=lambda r: r["input"])

    # 4) write the single results file
    Path(out_path).write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} predictions -> {out_path}")
    return records


def run_single(fdf_path, cfg, params_path):
    """Single molecule -> human-readable block on stdout (unchanged behavior)."""
    z, pos = parse_fdf(fdf_path)
    z, pos = np.asarray(z), np.asarray(pos)
    model, params = load_model(cfg, params_path, z, pos)
    out = model.apply(params, z, pos)

    print(f"energy = {float(out['energy']):.4f} eV")
    spins = cfg["wannier"]["spins"]
    k = centers_per_spin(z, len(spins))
    sel = select_centers(out["wannier"], k, nms_radius(cfg))
    centers, radii = sel["centers"], sel["radii"]
    for s, spin in enumerate(spins):
        print(f"[{spin}] {k[s]} centers  (x y z  radius, Angstrom)")
        for c, r in zip(centers[s][:k[s]], radii[s][:k[s]]):
            print(f"    {c[0]:8.4f} {c[1]:8.4f} {c[2]:8.4f}   {r:.4f}")


def build_argparser():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Predict total energy + Wannier centers from SIESTA .fdf geometry. "
                    "Wannier positions are inferred from geometry alone (no .wout input).",
        epilog="examples:\n"
               "  uv run python main.py molecule.fdf --params params.msgpack\n"
               "  uv run python main.py fdf_dir/ --params params.msgpack --out predictions.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="a single .fdf file OR a directory of .fdf files")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="params.msgpack")
    ap.add_argument("--out", default="predictions.json",
                    help="output JSON file (directory mode only)")
    return ap


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    if Path(args.input).is_dir():
        run_directory(args.input, cfg, args.out, args.params)
    else:
        run_single(args.input, cfg, args.params)


if __name__ == "__main__":
    main()
