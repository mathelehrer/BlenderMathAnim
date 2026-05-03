"""
Marching tetrahedra: convert an implicit surface f(x,y,z) = level into
an explicit triangle mesh by sampling f on a uniform 3D grid, splitting
each cube into 6 tetrahedra, and emitting interpolated triangles per tet.

Why marching tetrahedra rather than the more common marching cubes?

  * No 256-entry lookup table — just three case branches per tet
    (1, 2, or 3 vertices on the negative side).
  * No ambiguous-face cases — each tet's iso-surface is unambiguous.
  * Roughly 1.5–2× more triangles than MC; topology still correct.

The output is a plain (verts, faces) pair — see iso_curves.py for the
companion utility that traces lines of constant coordinate functions
on the resulting mesh.
"""

from __future__ import annotations

import numpy as np

# the splitting of the cube into six tetrahedra is visualized in
# video_shapes.py/tetrahedra_marching

# Cube corner offsets, indexed 0..7
#   0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)
#   4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)
_CUBE_CORNERS = np.array([
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
], dtype=np.int8)

# Standard 6-tet split sharing the cube diagonal 0 -> 6.
# Each tet is given as 4 corner indices into _CUBE_CORNERS.
_TETS = (
    (0, 1, 5, 7),
    (0, 1, 7, 3),
    (0, 3, 7, 2),
    (0, 2, 7, 6),
    (0, 6, 7, 4),
    (5, 4, 0, 7),
)


def marching_tetrahedra(f, bounds, resolution, level=0.0):
    """
    Extract a triangle mesh of the iso-surface { f(x,y,z) = level }.

    Parameters
    ----------
    f : callable
        A vectorized function f(X, Y, Z) -> ndarray, accepting numpy arrays
        of identical shape and returning an array of the same shape.
        Example: ``lambda x, y, z: x*x + y*y + z*z - 1``
    bounds : tuple of (min, max) per axis, e.g. ``((-2,2),(-2,2),(-2,2))``
    resolution : int or (rx, ry, rz)
        Number of cells along each axis.  Number of grid samples is
        (rx+1) * (ry+1) * (rz+1).
    level : float
        Iso-value (default 0).

    Returns
    -------
    verts : ndarray, shape (V, 3), float64
    faces : ndarray, shape (F, 3), int32
        Triangle indices into ``verts``.

    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    rx, ry, rz = resolution
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    xs = np.linspace(xmin, xmax, rx + 1)
    ys = np.linspace(ymin, ymax, ry + 1)
    zs = np.linspace(zmin, zmax, rz + 1)

    # Vectorized sampling of f at every grid point.
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    F = f(X, Y, Z) - level  # shift so we look for zero crossings

    # The advantage of the vectorized data is speed. We avoid python loops to check every cube individually.
    # F contains the sample data of our function on a three-dimensional cubic grid (nxnxn)
    # the following F0, F1, ..., F7 store the values at the corresponding corners of the cubes (n-1)x(n-1)x(n-1)
    # Skip cells that have no sign change — by far the most common case.
    F0 = F[:-1, :-1, :-1]
    F1 = F[1:, :-1, :-1]
    F2 = F[:-1, 1:, :-1]
    F3 = F[1:, 1:, :-1]
    F4 = F[:-1, :-1, 1:]
    F5 = F[1:, :-1, 1:]
    F6 = F[:-1, 1:, 1:]
    F7 = F[1:, 1:, 1:]
    # compute the minimum value of F inside each cube
    Fmin = np.minimum.reduce([F0, F1, F2, F3, F4, F5, F6, F7])
    # compute the maximum value of F inside each cube
    Fmax = np.maximum.reduce([F0, F1, F2, F3, F4, F5, F6, F7])
    active = np.argwhere((Fmin <= 0) & (Fmax >= 0))  # (N, 3) cell indices

    verts = []
    faces = []
    # differentiate two kinds of faces from tetrahedron marching
    # 0: only one vertex of the tetrahedron has a different sign
    # 1: two vertices of teh tetrahedron have a different sign
    face_types = []

    # perform the tetrahedra marching for each active cube
    for (i, j, k) in active:
        # 8 corner positions and values for this cube.
        cv = (F0[i, j, k], F1[i, j, k], F2[i, j, k], F3[i, j, k],
              F4[i, j, k], F5[i, j, k], F6[i, j, k], F7[i, j, k])
        cp = tuple(
            (xs[i + co[0]], ys[j + co[1]], zs[k + co[2]])
            for co in _CUBE_CORNERS
        )
        for tet in _TETS:
            _emit_tet(cp, cv, tet, verts, faces)

    if not verts:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int32))

    # reduce closeby vertices that are less than 1/1000 of the
    # smallest extension of the bounding box
    return dedup_vertices(np.asarray(verts, dtype=np.float64),
                          np.asarray(faces, dtype=np.int32),
                          tol=min((xmax - xmin) / rx,
                                  (ymax - ymin) / ry,
                                  (zmax - zmin) / rz) * 1e-3)



def dedup_vertices(verts, faces, tol=1e-7):
    """
    Merge vertices that share a position (within ``tol``) and rewrite faces.

    Marching tetrahedra emits each edge crossing once per tetrahedron, so
    the raw output has 4–6× more vertices than necessary.  Without this
    merge, downstream iso-line tracing would fragment every polyline at
    every triangle, because adjacent triangles wouldn't share vertex
    indices.
    """

    if len(verts) == 0:
        return verts, faces
    keys = np.round(verts / tol).astype(np.int64)
    # Hash each row to a single integer index by lexicographic sorting.
    # due to transposition: the first row are all x-values, the second all y-values and the last all z-values
    order = np.lexsort(keys.T) # transposed version of keys
    sorted_keys = keys[order]
    # now equal vertices are next to each other
    # compare neighbours (i,i+1)
    diff = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    new_idx_sorted = np.concatenate([[0], np.cumsum(diff)])
    remap = np.empty(len(verts), dtype=np.int64)
    remap[order] = new_idx_sorted

    n_unique = int(new_idx_sorted[-1]) + 1
    unique_verts = np.empty((n_unique, 3), dtype=verts.dtype)
    unique_verts[remap] = verts  # last writer wins; all writers agree within tol
    new_faces = remap[faces]
    return unique_verts, new_faces


def _emit_tet(cp, cv, tet, verts, faces):
    """
        Emit triangles for the iso-surface inside one tetrahedron.

        :param cp: 8 corner positions
        :param cv: 8 corner values of the function for which the surface is sampled
        :param tet: corner indices of one of the 6 tetrahedra
        :param faces: the collection of all faces emitted so far

    """
    p = (cp[tet[0]], cp[tet[1]], cp[tet[2]], cp[tet[3]])
    v = (cv[tet[0]], cv[tet[1]], cv[tet[2]], cv[tet[3]])

    # Classify: which corners are on the negative side?
    inside = (v[0] < 0, v[1] < 0, v[2] < 0, v[3] < 0)
    n_in = sum(inside)
    if n_in == 0 or n_in == 4:
        return  # no intersection with the iso-surface

    if n_in == 1 or n_in == 3:
        # Single triangle through 3 edges from the lone vertex.
        if n_in == 1:
            a = inside.index(True)  # index on the inside side
            others = [i for i in range(4) if not inside[i]]
        else:
            a = inside.index(False)  # index on the outside side
            others = [i for i in range(4) if inside[i]]
        pts = []
        for o in others:
            # linearly interpolate the edge endpoints to find the vertex position
            pts.append(_lerp(p[a], p[o], v[a], v[o]))
        n = len(verts)  # define base index position for the new vertices
        verts.extend(pts)

        # Orient consistently: when n_in == 1 the lone-vertex side is "inside",
        # and we want the triangle normal to face outward.  Flip on n_in == 3.
        # we have to make sure that the vertices have the correct orientation for the tetrahedra
        # if the fourth (3) or second (1) is missing the indices are in the correct order
        # the correctly oriented tetrahedral faces are [(0, 1, 2), (1,0, 3), (0, 2, 3), (1,3,2)]
        # the zero and second face has the indices in order
        # the first and third face has two indices flipped

        if (n_in == 1 and a in [1, 3]) or (n_in == 3 and a in [0, 2]):
            faces.append((n, n + 1, n + 2))  # triangle is added as for the tetrahedron
        else:
            faces.append((n, n + 2, n + 1))  # triangle is added with flipped normal
        return

    # n_in == 2: a quadrilateral split into 2 triangles.
    ins = [i for i in range(4) if inside[i]]
    out = [i for i in range(4) if not inside[i]]
    a0, a1 = ins
    b0, b1 = out
    p00 = _lerp(p[a0], p[b0], v[a0], v[b0])
    p01 = _lerp(p[a0], p[b1], v[a0], v[b1])
    p11 = _lerp(p[a1], p[b1], v[a1], v[b1])
    p10 = _lerp(p[a1], p[b0], v[a1], v[b0])
    n = len(verts)
    verts.extend([p00, p01, p11, p10])

    # strongly depends on which two tetrahedral indices are negative
    # the combinatorics have been solved with trial and error
    if a0 < a1 < b0 < b1:
        faces.append((n, n + 2, n + 1))
        faces.append((n, n + 3, n + 2))
    if a0 <b0 < a1 < b1:
        faces.append((n, n + 1, n + 2))
        faces.append((n, n + 2, n + 3))
    if a0 < b0 < b1 < a1:
        faces.append((n, n + 2, n + 1))
        faces.append((n, n + 3, n + 2))
    if b0 < a0 < b1 < a1:
        faces.append((n, n + 1, n + 2))
        faces.append((n, n + 2, n + 3))
    if b0 < b1 < a0 < a1:
        faces.append((n, n + 2, n + 1))
        faces.append((n, n + 3, n + 2))
    if b0  < a0 < a1 < b1:
        faces.append((n, n + 2, n + 1))
        faces.append((n, n + 3, n + 2))


def _lerp(p_a, p_b, va, vb):
    """Linear interpolation along an edge to find where v = 0."""
    # va and vb are guaranteed to straddle zero (one is < 0, other >= 0).
    d = vb - va
    if d == 0:
        return p_a
    t = -va / d
    return (
        p_a[0] + t * (p_b[0] - p_a[0]),
        p_a[1] + t * (p_b[1] - p_a[1]),
        p_a[2] + t * (p_b[2] - p_a[2]),
    )


def parse_implicit_function(expr_or_callable):
    """
    Coerce a function spec to a vectorized callable f(X, Y, Z) -> array.

    Accepts either a Python callable (returned unchanged) or an infix
    expression string in the variables ``x``, ``y``, ``z``.  String form
    uses standard Python operators (``**`` for power) and the usual
    numpy-friendly names (``sin``, ``cos``, ``exp``, ``sqrt``, ``abs``,
    ``atan2``).

    >>> f = parse_implicit_function("x*x + y*y + z*z - 1")
    >>> import numpy as np
    >>> float(f(np.array(1.0), np.array(0.0), np.array(0.0)))
    0.0
    """
    if callable(expr_or_callable):
        return expr_or_callable
    expr = expr_or_callable.replace("^", "**")
    code = compile(expr, "<implicit>", "eval")
    safe = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "atan2": np.arctan2,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs,
        "min": np.minimum, "max": np.maximum,
        "pi": np.pi, "e": np.e,
    }

    def f(x, y, z):
        return eval(code, {"__builtins__": {}}, {**safe, "x": x, "y": y, "z": z})

    return f
