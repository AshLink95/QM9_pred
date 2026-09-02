"""Graph construction: positions -> fully-connected edges, distances, relative vectors (§3).

Deterministic tensor prep, NO learning (CLAUDE.md §3 pipeline boundary). QM9 molecules are
tiny (<=~29 atoms) so we fully connect all ordered pairs i!=j — no cutoff, no sparsification,
no canonicalization. Distances are in Angstrom and are NOT normalized (§3).

Note (§3): the model may recompute distances/rel-vectors in-layer from `positions` so that
`jax.grad` can flow to positions (needed if forces ever become a target). This module gives
the deterministic *construction rule* (who connects to whom) and a convenience precompute.
"""

from __future__ import annotations

import numpy as np


def fully_connected_edges(n_atoms: int) -> np.ndarray:
    """All ordered pairs i!=j as edge_index[2, E] = (senders, receivers)."""
    idx = np.arange(n_atoms)
    send, recv = np.meshgrid(idx, idx, indexing="ij")
    mask = send != recv
    return np.stack([send[mask], recv[mask]], axis=0).astype(np.int32)


def build_graph(pos: np.ndarray) -> dict[str, np.ndarray]:
    """positions[N,3] -> {edge_index[2,E], rel_vec[E,3], distances[E]} (Angstrom)."""
    edge_index = fully_connected_edges(len(pos))
    i, j = edge_index
    rel_vec = pos[i] - pos[j]                       # equivariant (used only in equiv ops, §4)
    distances = np.linalg.norm(rel_vec, axis=-1)    # invariant scalar
    return {"edge_index": edge_index, "rel_vec": rel_vec, "distances": distances}
