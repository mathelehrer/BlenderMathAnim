"""Compatible remeshing: one connectivity that fits two shapes.

The pre-pass ``docs/theory_morphing.tex`` asks for. A geometry-node graph can
pair the points of two shapes but it cannot build a mesh, so the pairing is
always a compromise: index order is arbitrary, nearest-point projection is
many-to-one, and either way the target's edges and faces never appear. The
fix the literature settles on is to stop pairing and start *remeshing* - put
both shapes on one mesh, so that interpolating vertex positions is correct by
construction and the connectivity is the same at every value of the morph.

This module does that the way Kent, Carlson and Parent do it for star-shaped
polyhedra (SIGGRAPH 1992): both shapes are written over a common spherical
domain, and the domain's own triangulation becomes the shared connectivity.
Where they merge the two projected topologies into an overlay, this resamples
instead - the sphere is tessellated once, at a resolution of the caller's
choosing, and each of its directions is traced onto both surfaces:

    p_A(d) = c_A + t_A(d) * d          p_B(d) = c_B + t_B(d) * d

with ``c`` the centre a shape is star-shaped about and ``t`` the distance to
its surface along ``d``. Resampling loses whatever detail falls between the
sphere's directions, which the overlay would have kept; in exchange it is
twenty lines of ray casting instead of spherical polygon clipping, it cannot
produce degenerate triangles, and the output resolution is a dial rather than
a consequence.

What comes out is two vertex arrays over one face array. Feed them to
``MorphNode`` as ``Geometry 1`` and ``Geometry 2`` with ``Match Nearest``
off: index pairing is then exactly right, because point *i* of the one *is*
point *i* of the other. Nothing about the node needs to change.

Limits, stated plainly:

- **Star-shaped is the assumption.** Every ray from ``c`` must meet the
  surface. Where several do, the outermost hit is taken, which is what makes
  a cone standing on a cylinder come out as its silhouette rather than as its
  internal seam. Where none does, the direction is a *miss* and falls back to
  the surface point lying most nearly along it; the count of misses is
  reported, and a shape with many of them is one this method does not fit.
- **Genus 0 only.** A sphere cannot parameterise a shape with a hole in it -
  a picture frame is a torus, and no amount of ray casting changes that. The
  Euler characteristic of each input is reported for exactly this reason.
- **The centre matters.** It defaults to the mean of the vertices, which is
  inside anything convex and most things that are not, but a shape whose
  vertex mean falls outside it (an L, a crescent) needs one passed in.

Usage::

    sm = supermesh_from_objects(frame_obj, arrow_obj, subdivisions=4)
    print(sm.report)
    mesh_a, mesh_b = sm.to_meshes()
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# The common domain
# ---------------------------------------------------------------------------
def icosphere(subdivisions=3):
    """A unit sphere as a subdivided icosahedron.

    Chosen over a UV sphere because its triangles are near enough uniform:
    the poles of a UV sphere would spend a quarter of the supermesh's points
    on two directions and starve the equator, and every one of those points
    costs the same at render time.

    :param subdivisions: how many times each triangle is split into four.
        0 gives 12 points, 1 gives 42, 2 gives 162, 3 gives 642, 4 gives 2562.
    :return: ``(directions, faces)`` - unit vectors and integer triangles.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)

    for _ in range(subdivisions):
        # every edge gains its midpoint; a dict keyed on the ordered pair of
        # endpoints is what keeps the two triangles sharing an edge sharing
        # that midpoint as well, so the sphere stays a closed surface
        midpoint = {}
        new_faces = []
        verts = list(verts)
        for a, b, c in faces:
            ab = _midpoint(verts, midpoint, a, b)
            bc = _midpoint(verts, midpoint, b, c)
            ca = _midpoint(verts, midpoint, c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(verts)
        faces = np.array(new_faces, dtype=int)

    verts = np.asarray(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    return verts, faces


def _midpoint(verts, cache, a, b):
    """Index of the midpoint of edge ``(a, b)``, created once and reused."""
    key = (a, b) if a < b else (b, a)
    if key not in cache:
        cache[key] = len(verts)
        verts.append((np.asarray(verts[a]) + np.asarray(verts[b])) / 2)
    return cache[key]


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------
def triangulate(faces):
    """Fan-triangulate a list of polygons.

    Blender's meshes are polygonal and the intersection test is not, so
    every n-gon becomes ``n - 2`` triangles around its first corner. Fine for
    the convex quads and triangles primitives produce; a wildly concave n-gon
    would need ear clipping, and none of the shapes this is aimed at have one.
    """
    tris = []
    for face in faces:
        for i in range(1, len(face) - 1):
            tris.append([face[0], face[i], face[i + 1]])
    return np.array(tris, dtype=int)


def surface_distances(vertices, tris, centre, directions, chunk=2048):
    """How far the surface is from ``centre`` along each direction.

    Moeller-Trumbore, vectorised over rays and triangles at once and chunked
    over the triangles so that the intermediate arrays stay a few tens of
    megabytes whatever the mesh size. The *farthest* hit is kept: a ray that
    passes through interior geometry on the way out (the cap where a cone
    sits on a cylinder, say) should land on the outside of the shape, which
    is the surface anybody looking at it sees.

    :return: ``(distances, hit)`` - distance along each direction, and a
        boolean saying whether that direction met the surface at all.
    """
    v0 = vertices[tris[:, 0]]
    edge1 = vertices[tris[:, 1]] - v0
    edge2 = vertices[tris[:, 2]] - v0

    best = np.zeros(len(directions))
    hit = np.zeros(len(directions), dtype=bool)
    eps = 1e-9

    for start in range(0, len(tris), chunk):
        stop = min(start + chunk, len(tris))
        e1, e2 = edge1[start:stop], edge2[start:stop]
        s = centre - v0[start:stop]                              # (m, 3)

        h = np.cross(directions[:, None, :], e2[None, :, :])     # (n, m, 3)
        a = np.einsum("mj,nmj->nm", e1, h)
        parallel = np.abs(a) < eps
        a = np.where(parallel, 1.0, a)                           # avoid 0/0
        f = 1.0 / a

        u = f * np.einsum("mj,nmj->nm", s, h)
        q = np.cross(s, e1)                                      # (m, 3)
        v = f * np.einsum("nj,mj->nm", directions, q)
        t = f * np.einsum("mj,mj->m", e2, q)[None, :]

        good = (~parallel) & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > eps)
        if not good.any():
            continue
        candidate = np.where(good, t, -np.inf)
        block_best = candidate.max(axis=1)
        block_hit = np.isfinite(block_best)
        update = block_hit & (block_best > best)
        best[update] = block_best[update]
        hit |= block_hit

    return best, hit


def _fallback_directions(vertices, centre, directions, missing):
    """Where a direction that misses the surface should land instead.

    The vertex whose own direction from the centre is closest to the one
    that missed - the surface point most nearly *that* way. It keeps the map
    defined everywhere and roughly continuous, which is all a fallback can
    do; a shape that needs it often is a shape this method does not fit, and
    the miss count says so.
    """
    rays = np.asarray(vertices) - centre
    lengths = np.linalg.norm(rays, axis=1)
    lengths[lengths == 0] = 1.0
    unit = rays / lengths[:, None]
    best = (directions[missing] @ unit.T).argmax(axis=1)
    return rays[best]


# ---------------------------------------------------------------------------
# The supermesh
# ---------------------------------------------------------------------------
class SuperMesh:
    """Two shapes written on one mesh.

    :ivar vertices_a: ``(n, 3)`` positions of the first shape.
    :ivar vertices_b: ``(n, 3)`` positions of the second, in the same order.
    :ivar faces: ``(m, 3)`` triangles, valid for both.
    :ivar report: what the construction had to say about its own assumptions.
    """

    def __init__(self, vertices_a, vertices_b, faces, report):
        self.vertices_a = vertices_a
        self.vertices_b = vertices_b
        self.faces = faces
        self.report = report

    def __len__(self):
        return len(self.vertices_a)

    def positions(self, t):
        """The mesh at morph parameter ``t``, for checking without blender."""
        return (1 - t) * self.vertices_a + t * self.vertices_b

    # -- blender -----------------------------------------------------------
    def to_meshes(self, names=("Supermesh A", "Supermesh B")):
        """Two blender meshes sharing one connectivity.

        Plug them into ``MorphNode`` as ``Geometry 1`` and ``Geometry 2``
        with ``Match Nearest`` off - index pairing is exact here, since the
        two meshes are the same mesh twice over.
        """
        from interface.ibpy import create_mesh          # blender only from here
        faces = [list(map(int, f)) for f in self.faces]
        return (create_mesh([list(map(float, v)) for v in self.vertices_a],
                            faces=faces, name=names[0]),
                create_mesh([list(map(float, v)) for v in self.vertices_b],
                            faces=faces, name=names[1]))

    def to_mesh_with_attribute(self, name="Supermesh", attribute="target_position"):
        """One mesh, with the second shape's positions as a point attribute.

        The single-object form: a ``Named Attribute`` node reads the
        destination straight off the geometry, so the morph needs no second
        input and no sampling at all.
        """
        mesh = self.to_meshes(names=(name, name + " (unused)"))[0]
        layer = mesh.attributes.new(attribute, 'FLOAT_VECTOR', 'POINT')
        layer.data.foreach_set("vector", np.asarray(self.vertices_b, dtype=float).ravel())
        mesh.update()
        return mesh


def build_supermesh(vertices_a, faces_a, vertices_b, faces_b,
                    subdivisions=3, centre_a=None, centre_b=None):
    """Remesh two shapes onto a common sphere tessellation.

    Pure numpy, so it can be checked without blender running.

    :param vertices_a: ``(n, 3)`` vertices of the first shape.
    :param faces_a: its polygons, as lists of vertex indices.
    :param vertices_b: ``(n, 3)`` vertices of the second shape.
    :param faces_b: its polygons.
    :param subdivisions: resolution of the common sphere - see
        :func:`icosphere`. The supermesh has that many points whatever the
        inputs carry, which is the point: it is *their* resolution that stops
        mattering.
    :param centre_a: the point the first shape is star-shaped about. Defaults
        to the mean of its vertices.
    :param centre_b: likewise for the second.
    :return: a :class:`SuperMesh`.
    """
    vertices_a = np.asarray(vertices_a, dtype=float)
    vertices_b = np.asarray(vertices_b, dtype=float)
    tris_a, tris_b = triangulate(faces_a), triangulate(faces_b)

    centre_a = vertices_a.mean(axis=0) if centre_a is None else np.asarray(centre_a, float)
    centre_b = vertices_b.mean(axis=0) if centre_b is None else np.asarray(centre_b, float)

    directions, faces = icosphere(subdivisions)

    out = []
    misses = []
    for vertices, tris, centre in ((vertices_a, tris_a, centre_a),
                                   (vertices_b, tris_b, centre_b)):
        distance, hit = surface_distances(vertices, tris, centre, directions)
        points = centre + distance[:, None] * directions
        if not hit.all():
            points[~hit] = centre + _fallback_directions(vertices, centre,
                                                         directions, ~hit)
        out.append(points)
        misses.append(int((~hit).sum()))

    report = {
        "points": len(directions),
        "faces": len(faces),
        "misses_a": misses[0], "misses_b": misses[1],
        "euler_a": _euler_characteristic(faces_a),
        "euler_b": _euler_characteristic(faces_b),
        "centre_a": tuple(centre_a), "centre_b": tuple(centre_b),
    }
    report["warnings"] = _warnings(report)
    return SuperMesh(out[0], out[1], faces, report)


def _euler_characteristic(faces):
    """``V - E + F`` of a polygon soup, as a genus-0 sanity check.

    A closed orientable surface has ``chi = 2 - 2g``, so 2 means a sphere and
    0 a torus. Only meaningful for a closed manifold - an open surface is not
    wrong, it just cannot be read this way, which is why this is reported
    rather than enforced.
    """
    edges = set()
    vertices = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            vertices.add(a)
            edges.add((a, b) if a < b else (b, a))
    return len(vertices) - len(edges) + len(faces)


def _warnings(report):
    """Plain sentences about whichever assumption the inputs did not meet."""
    out = []
    for shape in ("a", "b"):
        misses = report["misses_%s" % shape]
        if misses:
            out.append("shape %s: %d of %d directions missed the surface (%.0f%%) - "
                       "not star-shaped about its centre" %
                       (shape.upper(), misses, report["points"],
                        100 * misses / report["points"]))
        chi = report["euler_%s" % shape]
        if chi != 2:
            out.append("shape %s: Euler characteristic %d, not 2 - a sphere cannot "
                       "parameterise this (genus %s)" %
                       (shape.upper(), chi,
                        (2 - chi) // 2 if (2 - chi) % 2 == 0 else "?"))
    return out


# ---------------------------------------------------------------------------
# Blender glue
# ---------------------------------------------------------------------------
def mesh_arrays(obj, evaluated=True):
    """``(vertices, faces)`` of a blender object, modifiers and all.

    Evaluating first is what lets a geometry-node object be an input: the
    shapes worth morphing are usually built by a modifier, and the mesh in
    ``obj.data`` is whatever was there before it ran.
    """
    import bpy                                          # blender only from here
    if evaluated:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        target = obj.evaluated_get(depsgraph)
        mesh = target.to_mesh()
        vertices = np.array([v.co[:] for v in mesh.vertices], dtype=float)
        faces = [list(p.vertices) for p in mesh.polygons]
        target.to_mesh_clear()
    else:
        mesh = obj.data
        vertices = np.array([v.co[:] for v in mesh.vertices], dtype=float)
        faces = [list(p.vertices) for p in mesh.polygons]
    return vertices, faces


def supermesh_from_objects(obj_a, obj_b, subdivisions=3,
                           centre_a=None, centre_b=None, evaluated=True):
    """:func:`build_supermesh` for two blender objects.

    Vertices are read in each object's own local space, so an object moved
    by its transform morphs from where its mesh is, not from where it looks
    like it is. Apply the transform first if that is not what is wanted.
    """
    vertices_a, faces_a = mesh_arrays(obj_a, evaluated=evaluated)
    vertices_b, faces_b = mesh_arrays(obj_b, evaluated=evaluated)
    return build_supermesh(vertices_a, faces_a, vertices_b, faces_b,
                           subdivisions=subdivisions,
                           centre_a=centre_a, centre_b=centre_b)
