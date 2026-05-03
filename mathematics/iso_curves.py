"""
Trace iso-lines of a scalar function g(x,y,z) on an explicit triangle mesh.

This is what the volume → mesh path of geometry nodes can NOT give you:
once we have explicit triangles and vertex positions, we can evaluate any
secondary scalar field g (e.g. atan2(y, x) for longitude) at the vertices,
find where g equals a chosen iso-value on each triangle, and stitch the
resulting segments into polylines.

Used together with marching_tetrahedra.py to draw meridians, parallels,
gradient lines, level curves of curvature, etc., on top of an implicit
surface.
"""

from __future__ import annotations

import numpy as np


def trace_iso_lines(verts, faces, g, levels):
    """
    Find polylines on the mesh where g(x,y,z) = c, for each c in ``levels``.

    Parameters
    ----------
    verts : ndarray (V, 3)
        Mesh vertex positions.
    faces : ndarray (F, 3) of int
        Triangle indices.
    g : callable
        Vectorized scalar field g(X, Y, Z) -> ndarray of same shape.
        For periodic functions like atan2, use ``trace_iso_lines_modular``
        instead so the discontinuity at ±π doesn't create spurious lines.
    levels : iterable of float
        Iso-values at which to draw curves.

    Returns
    -------
    list of list of ndarray
        Outer index: one entry per level.
        Inner index: list of polylines, each an ndarray of shape (N, 3).
        Closed loops have their first and last point repeated.
    """
    g_v = g(verts[:, 0], verts[:, 1], verts[:, 2]).astype(np.float64)
    return [_trace_one_level(verts, faces, g_v, c) for c in levels]


def trace_iso_lines_modular(verts, faces, g, levels, period=2 * np.pi):
    """
    Like ``trace_iso_lines`` but for *periodic* fields (e.g. atan2).

    Triangles whose vertex values span more than half the period are
    skipped — without this, the discontinuity at ±π in atan2 produces
    spurious meridians along the seam.

    Parameters
    ----------
    period : float
        The period of g (default 2π).
    """
    g_v = g(verts[:, 0], verts[:, 1], verts[:, 2]).astype(np.float64)
    g_v = np.mod(g_v + period / 2, period) - period / 2  # canonicalize to [-period/2, period/2)
    return [_trace_one_level(verts, faces, g_v, c, max_span=period / 2) for c in levels]


def _trace_one_level(verts, faces, g_v, c, max_span=None):
    """Build polylines for one iso-value."""
    # If the iso-level coincides with vertex values (very common when the
    # marching-tetrahedra grid is aligned with the level — e.g. tracing
    # z = 0 on a sphere extracted with z-aligned cells), the triangulation
    # has degenerate triangles whose three vertices are all on the level.
    # Those triangles plus their 1-vertex-on, 2-vertex-on neighbors
    # produce many zero-length segments that fragment the polyline.
    # Nudge the level by a tiny amount to avoid the alignment.
    rng = float(g_v.max() - g_v.min()) or 1.0
    n_on_level = int(np.sum(np.isclose(g_v, c, atol=rng * 1e-9)))
    if n_on_level > 0:
        c = c + rng * 1e-5

    g_shifted = g_v - c

    edge_pt = {}      # (i, j) -> segment-vertex index
    pos_pt = {}       # spatial bucket key -> segment-vertex index (fallback dedup)
    seg_verts = []    # list of (x, y, z)
    seg_edges = []    # list of (sva, svb) — segments

    # Position bucket: rounding scale chosen to merge essentially-coincident
    # points without merging genuinely distinct ones.  Use the smallest
    # representative edge-length in the mesh as the scale.
    box_extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    pos_tol = max(box_extent * 1e-7, 1e-10)

    def get_or_create_seg_vert(edge_key, pos):
        idx = edge_pt.get(edge_key)
        if idx is not None:
            return idx
        bucket = (round(pos[0] / pos_tol),
                  round(pos[1] / pos_tol),
                  round(pos[2] / pos_tol))
        idx = pos_pt.get(bucket)
        if idx is None:
            idx = len(seg_verts)
            seg_verts.append(tuple(pos))
            pos_pt[bucket] = idx
        edge_pt[edge_key] = idx
        return idx

    for face in faces:
        a, b, cc = int(face[0]), int(face[1]), int(face[2])
        ga, gb, gc = g_shifted[a], g_shifted[b], g_shifted[cc]

        if max_span is not None:
            # Skip wrap-around triangles for periodic g.
            spread = max(g_v[a], g_v[b], g_v[cc]) - min(g_v[a], g_v[b], g_v[cc])
            if spread > max_span:
                continue

        crossings = []
        for (i, j, gi, gj) in [(a, b, ga, gb), (b, cc, gb, gc), (cc, a, gc, ga)]:
            # Use >= so we don't double-count a vertex that sits exactly on c
            # (it would otherwise produce two crossings on adjacent edges).
            if (gi >= 0) != (gj >= 0):
                key = (i, j) if i < j else (j, i)
                t = gi / (gi - gj) if gi != gj else 0.5
                p = verts[i] + t * (verts[j] - verts[i])
                crossings.append(get_or_create_seg_vert(key, p))

        if len(crossings) == 2 and crossings[0] != crossings[1]:
            seg_edges.append((crossings[0], crossings[1]))
        # 0, 1, or 3 crossings: ignore.  3 happens when the iso-line passes
        # exactly through a vertex; the triangle on the other side will handle it.

    if not seg_edges:
        return []

    return _stitch(seg_verts, seg_edges)


def _stitch(seg_verts, seg_edges):
    """
    Stitch a soup of line segments into polylines by walking the graph.

    Each segment-vertex has degree 1 (polyline endpoint) or 2 (interior).
    We start from each degree-1 node first to capture open polylines,
    then any remaining cycles.
    """
    # Adjacency: list of neighbors for each segment-vertex.
    adj = [[] for _ in range(len(seg_verts))]
    for (a, b) in seg_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, prev=None):
        path = [start]
        cur = start
        last = prev
        while True:
            nbrs = [n for n in adj[cur] if n != last and edge_key(cur, n) not in visited_edges]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited_edges.add(edge_key(cur, nxt))
            path.append(nxt)
            last, cur = cur, nxt
            if cur == start:  # closed loop
                break
        return path

    polylines = []

    # Start with open polylines: nodes of odd degree.
    for v in range(len(seg_verts)):
        if len(adj[v]) == 1 and not all(edge_key(v, n) in visited_edges for n in adj[v]):
            path = walk(v)
            if len(path) >= 2:
                polylines.append(np.array([seg_verts[i] for i in path], dtype=np.float64))

    # Then closed loops.
    for v in range(len(seg_verts)):
        for n in adj[v]:
            if edge_key(v, n) in visited_edges:
                continue
            visited_edges.add(edge_key(v, n))
            path = [v, n]
            last, cur = v, n
            while True:
                nbrs = [m for m in adj[cur] if m != last and edge_key(cur, m) not in visited_edges]
                if not nbrs:
                    break
                nxt = nbrs[0]
                visited_edges.add(edge_key(cur, nxt))
                path.append(nxt)
                last, cur = cur, nxt
                if cur == path[0]:
                    break
            if len(path) >= 2:
                polylines.append(np.array([seg_verts[i] for i in path], dtype=np.float64))

    return polylines
