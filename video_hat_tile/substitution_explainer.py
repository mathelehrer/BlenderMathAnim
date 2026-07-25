"""
Manual (non-geometry-node) construction of the hat substitution animation.

The geometry-node ``HatTileSubstitutionModifier`` produces the final tiling in a
single static (or parametrically driven) mesh.  The *explainer* animation needs
each individual tile to move and rotate independently while the supertile is
assembled, so the geometry is built here from plain ``BObject`` meshes that can
be keyframed one by one.

Everything is a faithful Python port of
``video_hat_tile/data/substitution_data.txt`` (the Mathematica notebook):

* ``tile_vertices`` == ``TileVertices[{1,Sqrt[3]}]``.  The outline itself is no
  longer duplicated here: it delegates to ``hat_tile._hat_vertices14`` and only
  translates between the two modules' labelling conventions (see its docstring).
* ``substitute`` == the notebook ``substitute[tile1,tile2]`` (verified to
  reproduce ``H7Tile``/``H8Tile`` code and control points exactly).  This is
  pure control-point/code algebra and does not touch ``tile_vertices``.

A *tile* (a.k.a. cluster) is a dict ``{"cp": [6 control points], "code":
[(pos, dir, ref, pt), ...]}`` where ``dir`` is in units of 30 degrees, ``ref``
is the reflection flag and ``pt`` is the (1-indexed) pivot vertex.
"""
import os

from numpy import pi, sqrt, cos, sin, array

from interface.ibpy import create_mesh, Vector
from objects.hat_tile import hat_vertices_from_code
from objects.bobject import BObject

r3 = sqrt(3)


# ===========================================================================
# Notebook geometry / substitution algebra
# ===========================================================================

def tile_vertices(pos, direction, ref, pt, scale=1.0):
    """``TileVertices[{1,Sqrt[3]}][pos, dir, ref, pt]`` -> list of 14 (x, y).

    Thin wrapper around :func:`new_stuff.objects.hat_tile.hat_vertices_from_code`,
    which is the single source of truth for the hat outline and owns the
    translation between this module's notebook labelling (30-degree ``dir``,
    1-indexed ``pt``) and ``_hat_vertices14``'s own convention.

    Delegating also gives the reflected tiles the same winding as the
    unreflected ones, so every hat face carries a consistent normal -- the
    notebook's ``x -> -x`` mirror alone flipped the winding of every anti-hat.
    """
    return hat_vertices_from_code(pos, direction, ref, pt, scale=scale)


def _rot(units, v):
    """``RotationMatrix[units*Pi/6] . v``."""
    a = units * pi / 6.0
    c, s = cos(a), sin(a)
    return array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _add(*vs):
    acc = array([0.0, 0.0])
    for v in vs:
        acc = acc + array(v)
    return acc


def _rotate_cluster(code, r):
    out = []
    for pos, d, ref, pt in code:
        nd = (d - r) % 12 if ref else (d + r) % 12
        out.append((_rot(r, pos), nd, ref, pt))
    return out


def _place_cluster(pos, direction, code):
    """``placeCluster`` -- rotate the code by ``direction`` then translate."""
    return [(_add(p, pos), d, ref, pt) for p, d, ref, pt in _rotate_cluster(code, direction)]


def rotate_code(code, units):
    """Rotate a tile code about the origin by ``units`` * 30 degrees."""
    return _rotate_cluster(code, units)


def place_code(code, pos=(0, 0), direction=0):
    """Rotate a tile code by ``direction`` * 30 degrees, then translate to ``pos``."""
    return _place_cluster(pos, direction, code)


def placements(t1, t2):
    """Return the 7 placements ``(source_tile, position, dir)`` of ``substitute``.

    The first six form ``tile1`` (the H7-type supertile), all seven form
    ``tile2`` (the H8-type supertile).  ``source_tile`` is ``t1`` for the first
    placement (the central anchor) and ``t2`` for the rest.
    """
    cp = t1["cp"]
    q = t2["cp"]
    cp1, cp4, cp5, cp6 = cp[0], cp[3], cp[4], cp[5]
    q2, q3, q4 = q[1], q[2], q[3]
    return [
        (t1, array(cp1), 0),
        (t2, array(cp6), 4),
        (t2, array(cp5), 2),
        (t2, array(cp4), 0),
        (t2, _add(cp4, q3), 10),
        (t2, _add(cp4, q2), 8),
        (t2, _add(cp4, q4), 0),
    ]


def substitute(t1, t2):
    """``substitute[t1, t2]`` -> ``(tile1, tile2)`` (H7-type, H8-type)."""
    place = placements(t1, t2)
    code7, code8 = [], []
    for i, (src, pos, d) in enumerate(place):
        placed = _place_cluster(pos, d, src["code"])
        if i < 6:
            code7 += placed
        code8 += placed
    cp = t1["cp"]
    q = t2["cp"]
    cp1, cp4, cp5, cp6 = cp[0], cp[3], cp[4], cp[5]
    q2, q3, q4 = q[1], q[2], q[3]
    npt = q4
    common = [cp1,
              _add(_rot(8, npt), cp4, q2),
              _add(_rot(10, npt), cp4, q3)]
    tail = [_add(_rot(2, npt), cp5), _add(_rot(4, npt), cp6)]
    tile1 = {"cp": common + [_add(npt, cp4)] + tail, "code": code7}
    tile2 = {"cp": common + [_add(npt, cp4, q4)] + tail, "code": code8}
    return tile1, tile2


# ---- seed tiles and their iterated supertiles -----------------------------

H2_TILE = {"cp": [(0, 0), (-3 / 2, 5 * r3 / 2), (-3, 2 * r3), (-6, 0), (-9 / 2, -r3 / 2), (-3, 0)],
           "code": [((-9 / 2, -r3 / 2), 8, False, 3), ((-9 / 2, -r3 / 2), 2, True, 11)]}
H1_TILE = {"cp": [(0, 0), (-3 / 2, 5 * r3 / 2), (-3, 2 * r3), (-3, r3), (-9 / 2, -r3 / 2), (-3, 0)],
           "code": [((0, 0), 8, False, 7)]}

# H7 plays the role of H2, H8 the role of H1 in the next level (notebook order).
H7_TILE, H8_TILE = substitute(H2_TILE, H1_TILE)
SUPER_H7_TILE, SUPER_H8_TILE = substitute(H7_TILE, H8_TILE)
SUPER_SUPER_H7_TILE, SUPER_SUPER_H8_TILE = substitute(SUPER_H7_TILE, SUPER_H8_TILE)
SUPER_SUPER_SUPER_H7_TILE, SUPER_SUPER_SUPER_H8_TILE = substitute(SUPER_SUPER_H7_TILE, SUPER_SUPER_H8_TILE)


# ===========================================================================
# Blender geometry helpers
# ===========================================================================

def hat_color(ref):
    return "yellow" if ref else "blue"


def make_hat(hat, scale=1.0, solid=0.3, roughness=0.1, name="Hat"):
    """One hat tile as an extruded ``BObject`` coloured by its reflection flag."""
    pos, direction, ref, pt = hat
    verts2d = tile_vertices(pos, direction, ref, pt, scale=scale)
    verts3d = [(float(v[0]), float(v[1]), 0.0) for v in verts2d]
    mesh = create_mesh(verts3d, faces=[list(range(len(verts3d)))])
    return BObject(mesh=mesh, name=name, color=hat_color(ref), roughness=roughness,
                   solid=solid, shadow=False)


def make_cluster(tile, scale=1.0, solid=0.3, roughness=0.1, name="Cluster",
                 location=(0, 0, 0), rotation_units=0):
    """A cluster (group of hats) as a transformable container ``BObject``.

    The hats are built in the cluster's own frame (origin at control point 1);
    the returned container carries the placement transform, so moving/rotating
    it moves the whole cluster rigidly.
    """
    hats = [make_hat(hat, scale=scale, solid=solid, roughness=roughness,
                     name=name + f"_hat{i}")
            for i, hat in enumerate(tile["code"])]
    loc = location if len(location) == 3 else (location[0], location[1], 0.0)
    container = BObject(children=hats, name=name, no_material=True,
                        location=Vector(loc),
                        rotation_euler=[0, 0, rotation_units * pi / 6])
    container._hats = hats
    return container


def _tuple2vector(p):
    return Vector((float(p[0]), float(p[1]), 0.0))


def _center(cluster):
    center = Vector()
    codes = cluster["code"]
    for pos, _, _, _ in codes:
        center += _tuple2vector(pos)
    center /= len(codes)
    return center

# ===========================================================================
# Animation
# ===========================================================================

def _assemble(t1, t2, place, scale, solid, anchor_stage, ring_stage,
              t0, appear_t, move_t, stagger, rev = False, intershift = Vector()):
    """Build and animate one supertile assembly.

    ``place`` is the placement list (6 entries for an H7-type, 7 for H8-type).
    The first placement is the central anchor (a copy of ``t1``); the rest are
    copies of ``t2`` that fly in from staging positions and rotate into place.
    Returns ``(members, end_time)``.
    """
    members = []
    region = _tuple2vector(anchor_stage)

    center1 = _center(t1)

    # central anchor (the copy of the t1/H2 cluster)
    src, pos, d = place[0]
    anchor = make_cluster(src, scale=scale, solid=solid, name="A_anchor",
                           location=anchor_stage)
    anchor.appear(begin_time=t0, transition_time=appear_t)
    t_move = t0 + appear_t+0.5

    # surrounding copies of the t2 cluster
    ring = place[1:]
    end = t_move + move_t
    if rev:
        iterator = reversed(ring)
    else:
        iterator = ring
    for k, (src, pos, d) in enumerate(iterator):
        final = region + _tuple2vector(pos) * scale
        stage = Vector((ring_stage[0] - (3-k) * 7, ring_stage[1], 0.0))
        cluster = make_cluster(src, scale=scale, solid=solid, name=f"A_{k}",
                               location=stage)
        cluster.appear(begin_time=t0+0.1*k, transition_time=appear_t/2)
        if rev:
            cluster.change_hue(-0.1+0.2*k/5,begin_time=t0+0.1*k+appear_t/2,transition_time=appear_t/2)
        else:
            cluster.change_hue(-0.1+0.2*(4-k)/5,begin_time=t0+0.1*k+appear_t/2,transition_time=appear_t/2)
        begin = t_move + k * stagger
        # first move above or below the final destination depending on the location of the final anchor point
        if region[1]<final[1]:
            # move below
            intermediate = region+Vector([-5,10,0])
        else:
            intermediate = region+Vector([-5,-5,0])-intershift
        cluster.move_to(target_location=intermediate,begin_time=begin,transition_time=2*move_t/3)
        cluster.move_to(target_location=final, begin_time=begin+move_t/2, transition_time=move_t/3)
        cluster.rotate(rotation_euler=[0, 0, d * pi / 6], begin_time=begin, transition_time=0.75*move_t)
        members.append(cluster)
        end = max(end, begin + move_t)
    members.append(anchor)
    return members, end


def play_substitution(t1, t2, *, scale=1.0, solid=0.3,
                      seed1=(-10, 10), seed2=(15, 10),
                      anchor7=(-10, -7.5), anchor8=(15, -7.5),
                      t0=0.0, move_t=2.5, stagger=1,
                      relocate=True,rescale=1):
    """Full ``substitution_explainer`` choreography for the pair ``(t1, t2)``.

    1. show the two seed clusters ``t1`` (H2-type) and ``t2`` (H1-type),
    2. assemble the H7-type supertile (6 placements) at ``region7``,
    3. assemble the H8-type supertile (7 placements) at ``region8``,
    4. erase the seeds and move the two supertiles to the seed positions.

    Returns the end time.
    """
    place = placements(t1, t2)

    # --- 1. seeds -----------------------------------------------------------
    seed_t1 = make_cluster(t1, scale=scale, solid=solid, name="Seed_H2", location=seed1)
    seed_t2 = make_cluster(t2, scale=scale, solid=solid, name="Seed_H1", location=seed2)
    seed_t1.appear(begin_time=t0, transition_time=1.0)
    t0 = seed_t2.appear(begin_time=t0, transition_time=1.0) + 0.5

    # --- 2. H7-type supertile (placements 0..5) -----------------------------
    h7_members, t_h7 = _assemble(
        t1, t2, place[:6], scale, solid,
        anchor_stage=anchor7, ring_stage=Vector([seed2[0], seed2[1] - 10.0, 0.0]),
        t0=t0, appear_t=1.0, move_t=move_t, stagger=stagger,rev = True)

    # --- 3. H8-type supertile (placements 0..6), further down ---------------
    h8_members, t_h8 = _assemble(
        t1, t2, place[:7], scale, solid,
        anchor_stage=anchor8, ring_stage=Vector([seed2[0], seed2[1] - 10.0, 0]),
        t0=t_h7, appear_t=1.0, move_t=move_t, stagger=stagger, rev=False, intershift = Vector([10,0,0]))
    t0 = t_h8 + 0.5

    if not relocate:
        return t0

    # --- 4. erase seeds, move the supertiles to the seed positions ----------
    seed_t1.disappear(begin_time=t0, transition_time=1.0)
    seed_t2.disappear(begin_time=t0, transition_time=1.0)
    t0 += 1.0

    v_anchor7 = _tuple2vector(anchor7)
    delta7 = _tuple2vector(seed1) - v_anchor7
    v_anchor8 = _tuple2vector(anchor8)
    delta8 = _tuple2vector(seed2) - v_anchor8
    scalebox7 = BObject(children=h7_members,name="Scalebox_H7")
    scalebox7.appear(begin_time=t0, transition_time=0,children=False)
    scalebox8 = BObject(children=h8_members,name="Scalebox_H8")
    scalebox8.appear(begin_time=t0, transition_time=0,children=False)
    rs = rescale # rescale factor after composition
    for k,m in enumerate(h7_members):
        scalebox7.move(direction=delta7+(1-rs)*v_anchor7, begin_time=t0, transition_time=move_t)
        if k<5:
            m.change_hue(0.1-0.2*k/5,begin_time=t0,transition_time=0.5)
        scalebox7.rescale(rescale=rs,begin_time=t0,transition_time=move_t)
    for k,m in enumerate(h8_members):
        scalebox8.move(direction=delta8+(1-rs)*v_anchor8, begin_time=t0, transition_time=move_t)
        if k < 6:
            m.change_hue(0.1-0.2*(4-k)/5,begin_time=t0,transition_time=0.5)
        scalebox8.rescale(rescale=rs, begin_time=t0, transition_time=move_t)
    return t0 + move_t

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), "data", f"SUPER_SUPER_SUPER_H8.csv")
    with open(data_path,"w") as f:
        f.write("x,y,dir,ref,pt\n")
        for (x,y),dir,ref,pt in SUPER_SUPER_SUPER_H8_TILE['code']:
            f.write(f"{x},{y},{dir},{ref},{pt}\n")