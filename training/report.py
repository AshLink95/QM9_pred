"""Accuracy report: parsed truth vs model output for energy + Wannier centers (CLAUDE.md §10).

Used by `scripts/evaluate.py`. Pure evaluation — nothing here is imported by training.

Pipeline: examples (from dataset.pkl or freshly parsed folders) -> `predict_all` (batched
forward pass) -> `build_report` (per-molecule + per-center rows, summary numbers via
training/metrics.py) -> `write_outputs` (CSV logs, summary.txt, accuracy.png).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from data.dataset import _index, _mol_id, load_molecule
from symmetry import transforms as T
from training.metrics import match_wannier, wannier_summary

_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def formula(z) -> str:
    """Hill-order formula from atomic numbers (C, H first, then alphabetical)."""
    c = Counter(_SYMBOL.get(int(q), f"Z{int(q)}") for q in z)
    order = [s for s in ("C", "H") if s in c] + sorted(s for s in c if s not in ("C", "H"))
    return "".join(s + (str(c[s]) if c[s] > 1 else "") for s in order)


# ------------------------------------------------------------------------ loading
def load_fresh(dirs, spins):
    """Parse a fresh data set: every .fdf under `dirs` (recursive), with its .out and per-spin
    .wout matched by molecule id across all roots. Returns (examples with "id", skipped)."""
    # ponytail: mirrors data/dataset.py:build_dataset only to keep each molecule's id without
    # changing dataset code (frozen this round). Fold back into build_dataset when the dataset
    # format is revisited.
    fdfs = sorted(f for d in dirs for f in Path(d).rglob("*.fdf"))
    outs, wouts = _index(dirs, "out"), _index(dirs, "wout")
    examples, skipped = [], []
    for f in fdfs:
        mid = _mol_id(f)
        if mid not in outs:
            skipped.append((mid, "no .out")); continue
        wout_by_spin = {}
        for spin in spins:
            match = [p for p in wouts.get(mid, []) if spin in p.name]
            if match:
                wout_by_spin[spin] = match[0]
        if len(wout_by_spin) < len(spins):
            skipped.append((mid, "missing spin .wout")); continue
        try:
            ex = load_molecule(f, outs[mid][0], wout_by_spin, spins)
        except Exception as err:                      # malformed / unconverged file
            skipped.append((mid, str(err))); continue
        ex["id"] = mid
        examples.append(ex)
    return examples, skipped


def max_centers_from_checkpoint(params_path, n_spins) -> int:
    """Wannier slot count baked into a checkpoint (the head's output width / n_spins), so the
    model is rebuilt with the shapes it was trained with regardless of the eval data."""
    raw = serialization.msgpack_restore(Path(params_path).read_bytes())
    return int(raw["params"]["WannierHead_0"]["w1"]["kernel"].shape[1]) // n_spins


# ------------------------------------------------------------------------ prediction
def predict_all(model, params, examples, batch_size=256):
    """Forward pass over all examples, batched by atom count; returns per-example outputs in
    INPUT ORDER. Every chunk is padded (by repetition) to `batch_size` and the extras dropped,
    so there's exactly one compiled shape per atom count."""
    fwd = jax.jit(lambda prm, z, p: jax.vmap(lambda zz, pp: model.apply(prm, zz, pp))(z, p))
    buckets = defaultdict(list)
    for i, e in enumerate(examples):
        buckets[len(e["z"])].append(i)
    out = [None] * len(examples)
    for idx in buckets.values():
        for k in range(0, len(idx), batch_size):
            chunk = idx[k:k + batch_size]
            padded = chunk + [chunk[0]] * (batch_size - len(chunk))
            res = jax.device_get(fwd(params,
                                     np.stack([examples[i]["z"] for i in padded]),
                                     np.stack([examples[i]["pos"] for i in padded])))
            w = res["wannier"]
            for j, i in enumerate(chunk):
                out[i] = {"energy": float(res["energy"][j]), "centers": w["centers"][j],
                          "radii": w["radii"][j], "presence": w["presence"][j]}
    return out


def rotation_drift(model, params, ex):
    """Symmetry spot-check on one molecule: rotate input, compare outputs (should be ~0)."""
    z, pos = np.asarray(ex["z"]), np.asarray(ex["pos"])
    base = model.apply(params, z, pos)
    R = T.random_rotation(seed=0)
    rot = model.apply(params, z, T.rotate(pos, R))
    return (abs(float(rot["energy"]) - float(base["energy"])),
            float(np.max(np.abs(np.asarray(rot["wannier"]["centers"])
                                - np.asarray(base["wannier"]["centers"]) @ R.T))))


# ------------------------------------------------------------------------ comparison
def _f(x, nd=6):
    return "" if x is None else round(float(x), nd)


def build_report(examples, preds, names, spins, thr, atom_ref=None):
    """Compare truth vs prediction. Returns (molecule_rows, center_rows, summary, data) where
    `data` holds the raw arrays the figure is drawn from."""
    mol_rows, center_rows = [], []
    by_spin = {s: [] for s in spins}
    e_true, e_pred, e_base = [], [], []
    r_true, r_pred = [], []                       # matched-pair radii, for the parity panel
    for ex, pr, name in zip(examples, preds, names):
        et, ep = float(ex["energy"]), pr["energy"]
        e_true.append(et); e_pred.append(ep)
        if atom_ref is not None:
            e_base.append(float(np.sum(atom_ref[np.asarray(ex["z"])])))
        row = {"molecule": name, "formula": formula(ex["z"]), "n_atoms": len(ex["z"]),
               "E_true_eV": _f(et), "E_pred_eV": _f(ep), "E_err_eV": _f(ep - et)}
        for s, spin in enumerate(spins):
            tc, tr = ex["wannier"][s]["centers"], ex["wannier"][s]["radii"]
            m = match_wannier(pr["centers"][s], pr["radii"][s], pr["presence"][s], tc, tr, thr)
            by_spin[spin].append(m)
            d = [p[2] for p in m["pairs"]]
            row |= {f"{spin}_n_true": m["n_true"], f"{spin}_n_pred": m["n_pred"],
                    f"{spin}_center_mae_A": _f(np.mean(d)) if d else "",
                    f"{spin}_radius_mae_A": _f(np.mean([abs(p[3]) for p in m["pairs"]]))
                    if d else ""}
            matched_t = {p[0] for p in m["pairs"]}
            matched_p = {p[1] for p in m["pairs"]}
            for t, p, dist, _ in m["pairs"]:
                r_true.append(float(tr[t])); r_pred.append(float(pr["radii"][s][p]))
                center_rows.append(_center_row(name, spin, "matched", tc[t], tr[t],
                                               pr, s, p, dist))
            for t in range(m["n_true"]):
                if t not in matched_t:
                    center_rows.append(_center_row(name, spin, "missed", tc[t], tr[t]))
            for p in m["kept"]:
                if p not in matched_p:
                    center_rows.append(_center_row(name, spin, "extra", None, None, pr, s, p))
        mol_rows.append(row)

    err = np.array(e_pred) - np.array(e_true)
    summary = {"n_molecules": len(examples), "presence_threshold": thr,
               "energy": _err_stats(err),
               "energy_baseline": _err_stats(np.array(e_base) - np.array(e_true))
               if e_base else None,
               "wannier": {spin: wannier_summary(ms) for spin, ms in by_spin.items()}}
    summary["wannier"]["all"] = wannier_summary([m for ms in by_spin.values() for m in ms])
    data = {"e_true": np.array(e_true), "e_pred": np.array(e_pred), "by_spin": by_spin,
            "radii_true": np.array(r_true), "radii_pred": np.array(r_pred)}
    return mol_rows, center_rows, summary, data


def _center_row(name, spin, status, tc, tr, pr=None, s=None, p=None, dist=None):
    row = {"molecule": name, "spin": spin, "status": status,
           "true_x": "", "true_y": "", "true_z": "", "true_radius": "",
           "pred_x": "", "pred_y": "", "pred_z": "", "pred_radius": "", "pred_presence": "",
           "dist_A": _f(dist)}
    if tc is not None:
        row |= {"true_x": _f(tc[0]), "true_y": _f(tc[1]), "true_z": _f(tc[2]),
                "true_radius": _f(tr)}
    if pr is not None:
        c = pr["centers"][s][p]
        row |= {"pred_x": _f(c[0]), "pred_y": _f(c[1]), "pred_z": _f(c[2]),
                "pred_radius": _f(pr["radii"][s][p]),
                "pred_presence": _f(pr["presence"][s][p], 4)}
    return row


def _err_stats(err):
    a = np.abs(err)
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "median_abs": float(np.median(a)), "p95_abs": float(np.percentile(a, 95)),
            "max_abs": float(a.max()), "bias": float(err.mean())} if len(err) else {}


# ------------------------------------------------------------------------ outputs
def format_summary(summary, header_lines=()):
    L = list(header_lines)
    e, b = summary["energy"], summary["energy_baseline"]
    L += ["", "ENERGY (eV)        MAE        RMSE       median|err|  p95|err|   max|err|   bias",
          f"  model        {e['mae']:10.4f} {e['rmse']:10.4f} {e['median_abs']:12.4f} "
          f"{e['p95_abs']:10.4f} {e['max_abs']:10.4f} {e['bias']:+9.4f}"]
    if b:
        L.append(f"  composition  {b['mae']:10.4f} {b['rmse']:10.4f} {b['median_abs']:12.4f} "
                 f"{b['p95_abs']:10.4f} {b['max_abs']:10.4f} {b['bias']:+9.4f}"
                 "   <- sum of per-element references only (what the network must beat)")
    L += ["", f"WANNIER CENTERS (Hungarian-matched; predicted = presence > "
              f"{summary['presence_threshold']})",
          "  spin   matched  center MAE  RMSE     median   p95      radius MAE  count acc"
          "   count error (pred-true): n"]
    for spin, w in summary["wannier"].items():
        brk = "  ".join(f"{k:+d}:{v}" for k, v in sorted(w["count_error_breakdown"].items()))
        L.append(f"  {spin:<6} {w['n_matched']:8d}  {w['center_mae']:9.4f}  "
                 f"{w['center_rmse']:7.4f}  {w['center_median']:7.4f}  {w['center_p95']:7.4f}  "
                 f"{w['radius_mae']:9.4f}  {100 * w['count_accuracy']:8.2f}%   {brk}")
    L += ["  (distances in Angstrom)", "",
          "Reference: Deep Wannier (Zhang et al., PRB 102, 041121, 2020) reports ~0.003-0.005 A",
          "RMSE for per-oxygen Wannier CENTROIDS in water/ice — an easier task (one chemistry,",
          "averaged centers); treat it as a lower bound, not a target."]
    return "\n".join(L)


def write_outputs(out_dir, mol_rows, center_rows, summary_text, data, spins):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("molecules.csv", mol_rows), ("centers.csv", center_rows)):
        with open(out / name, "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    (out / "summary.txt").write_text(summary_text + "\n")
    _figure(out / "accuracy.png", data, spins)


# ------------------------------------------------------------------------ figure
# Colors: validated categorical slots 1-2 (spin up / down), sequential blue ramp for density,
# recessive chrome — from the dataviz reference palette (light surface).
_SERIES = ["#2a78d6", "#eb6834"]
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_INK, _INK2, _MUTED, _GRID, _AXIS, _SURF = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                             "#c3c2b7", "#fcfcfb")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(_SURF)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=_MUTED); ax.set_ylabel(ylabel, color=_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(color=_GRID, linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)


def _cbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax)
    cb.set_label(label, color=_MUTED)
    cb.ax.tick_params(colors=_MUTED, labelsize=8)
    cb.outline.set_edgecolor(_AXIS)


def _beyond(n, unit_lim):
    """Axis-label suffix for values past the plotted range. They're dropped, not piled into the
    last bin; the label says how many, so it never collides with the data."""
    return f"   ({n} beyond {unit_lim} not shown)" if n else ""


def _figure(path, data, spins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    seq = LinearSegmentedColormap.from_list("seq_blue", _SEQ)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=_SURF)
    et, ep = data["e_true"], data["e_pred"]
    err = ep - et

    # 1. energy error vs energy (a parity plot hides eV-scale errors on a ~1e3 eV axis)
    ax, note = axs[0, 0], ""
    if len(et):
        lim = np.percentile(np.abs(err), 99.5) or 1.0
        keep = np.abs(err) <= lim
        hb = ax.hexbin(et[keep], err[keep], gridsize=60, cmap=seq, mincnt=1, norm=LogNorm(),
                       linewidths=0)
        ax.axhline(0, color=_MUTED, linewidth=1, linestyle="--")
        _cbar(fig, hb, ax, "molecules per cell")
        note = _beyond(int((~keep).sum()), f"±{lim:.3g} eV")
    _style(ax, "Energy error vs SIESTA energy", "SIESTA total energy (eV)" + note,
           "predicted − SIESTA (eV)")

    # 2. energy error histogram
    ax, note = axs[0, 1], ""
    if len(err):
        lim = np.percentile(np.abs(err), 99.5) or 1.0
        ax.hist(err, bins=80, range=(-lim, lim), color=_SERIES[0], edgecolor=_SURF,
                linewidth=0.5)
        ax.axvline(0, color=_MUTED, linewidth=1)
        ax.text(0.02, 0.95, f"MAE {np.abs(err).mean():.4f} eV\nbias {err.mean():+.4f} eV",
                transform=ax.transAxes, va="top", color=_INK2, fontsize=9)
        note = _beyond(int((np.abs(err) > lim).sum()), f"±{lim:.3g} eV")
    _style(ax, "Energy error (predicted − SIESTA)", "error (eV)" + note, "molecules")

    dists = {sp: np.array([p[2] for m in data["by_spin"][sp] for p in m["pairs"]])
             for sp in spins}
    # 3. center distance histogram per spin
    ax, note = axs[0, 2], ""
    alld = np.concatenate([d for d in dists.values() if len(d)] or [np.zeros(0)])
    if len(alld):
        top = np.percentile(alld, 99.5) or 1.0
        for k, sp in enumerate(spins):
            if len(dists[sp]):
                ax.hist(dists[sp], bins=80, range=(0, top), histtype="step", linewidth=2,
                        color=_SERIES[k], label=f"spin {sp}  (MAE {dists[sp].mean():.4f} Å)")
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK2)
        note = _beyond(int((alld > top).sum()), f"{top:.3g} Å")
    _style(ax, "Wannier center error (matched pairs)", "distance to true center (Å)" + note,
           "centers")

    # 4. count error per spin (grouped bars)
    ax = axs[1, 0]
    errs = {sp: Counter(m["n_pred"] - m["n_true"] for m in data["by_spin"][sp]) for sp in spins}
    keys = sorted(set().union(*errs.values())) if errs else []
    width = 0.8 / max(len(spins), 1)
    for k, sp in enumerate(spins):
        x = np.arange(len(keys)) + (k - (len(spins) - 1) / 2) * width
        ax.bar(x, [errs[sp].get(c, 0) for c in keys], width=width, color=_SERIES[k],
               edgecolor=_SURF, linewidth=2, label=f"spin {sp}")
    ax.set_xticks(np.arange(len(keys)), [f"{c:+d}" for c in keys])
    if keys:
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK2)
    _style(ax, "Center count error per molecule", "predicted − true count",
           "molecules")

    # 5. radius parity on matched pairs
    ax = axs[1, 1]
    tr, prr = data["radii_true"], data["radii_pred"]
    if len(tr):
        hb = ax.hexbin(tr, prr, gridsize=50, cmap=seq, mincnt=1, norm=LogNorm(), linewidths=0)
        lo, hi = min(np.min(tr), np.min(prr)), max(np.max(tr), np.max(prr))
        ax.plot([lo, hi], [lo, hi], color=_MUTED, linewidth=1, linestyle="--")
        _cbar(fig, hb, ax, "centers per cell")
    _style(ax, "Wannier radius: predicted vs true (matched)", "true radius (Å)",
           "predicted radius (Å)")

    # 6. per-molecule center MAE (both spins pooled)
    ax, note, title = axs[1, 2], "", "Per-molecule mean center error"
    per_mol = []
    for ms in zip(*[data["by_spin"][sp] for sp in spins]):
        d = [p[2] for m in ms for p in m["pairs"]]
        if d:
            per_mol.append(np.mean(d))
    if per_mol:
        per_mol = np.array(per_mol)
        top = np.percentile(per_mol, 99.5) or 1.0
        ax.hist(per_mol, bins=80, range=(0, top), color=_SERIES[0], edgecolor=_SURF,
                linewidth=0.5)
        title += f"  (median {np.median(per_mol):.4f} Å)"
        note = _beyond(int((per_mol > top).sum()), f"{top:.3g} Å")
    _style(ax, title, "mean distance (Å)" + note, "molecules")

    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=_SURF)
    plt.close(fig)
