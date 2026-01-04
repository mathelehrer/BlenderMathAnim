import itertools

import numpy as np

bl_info = {
    "name": "Regular Solids From 3–10 Objects",
    "author": "NumberCruncher",
    "version": (1, 1, 0),
    "blender": (5, 0, 1),
    "location": "View3D > Sidebar > Regular Solids",
    "category": "Object",
    "description": "Create a regular solid matching the n-gon count of selected objects (3-10)",
}

import bpy
from mathutils import Vector, Matrix
from bpy.props import (
    FloatVectorProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    BoolProperty,
    CollectionProperty,
)


# Face-sidedness for each  regular solid
SOLID_FACE_SIDES = {
    'TETRA': (3,),
    'OCTA': (3,),
    'ICOSA': (3,),
    'CUBE': (4,),
    'DODECA': (5,),
    'PRISM3': (3,4),
    'PRISM5': (4,5),
    'PRISM6': (4,6),
    'PRISM8': (4,8),
    'PRISM10': (4,10),
    'TRUNC_OCTA': (4, 6),
    'CUBOCTAHEDRON': (3, 4),
    'TRUNC_CUBOCTA': (4, 6, 8),
    'RHOMBICUBOCTA': (3, 4),
    'TRUNC_HEXA': (3, 8),
    'TRUNC_TETRA': (3, 6),
    'ICOSIDODECA': (3, 5),
    'TRUNC_ICOSA': (5, 6),
    'TRUNC_DODECA': (3, 10),
    'RHOMBICOSIDODECA': (3, 4, 5),
    'TRUNC_ICOSIDODECA': (4, 6, 10),

}
# Colors for each regular solid type (RGBA)
SOLID_COLORS = {
    'TETRA': (0,1,0, 1.0),
    'TRUNC_TETRA': (0,0.5,0,1),
    'OCTA':  (0,0.5,1, 1.0),
    'CUBOCTAHEDRON': (0,0.5,0.5, 1.0),
    'TRUNC_OCTA': (0, 0.25, 0.25, 1.0),
    'CUBE':  (0,0,1, 1.0),
    'TRUNC_HEXA': (0,0,0.5, 1.0),
    'RHOMBICUBOCTA': (0,0.5,0.25,1),
    'TRUNC_CUBOCTA': (0,0.25,0.25,1),
    'DODECA':(1,0, 0,1.0),
    'ICOSA': (1,0.5, 0,1.0),
    'ICOSIDODECA': (1,1,0, 1.0),
    'TRUNC_DODECA': (0.5,0,0, 1.0),
    'TRUNC_ICOSA': (0.5,0.25,0, 1.0),
    'RHOMBICOSIDODECA': (0.5,0,0.5, 1.0),
    'TRUNC_ICOSIDODECA': (1,0,1, 1.0),
    'PRISM3':(0.333,0.3333,0.3333,1.0),# grey
    'PRISM5':(0.2,0.2,0.2,1.0),
    'PRISM6':(0.167,0.167,0.167,1.0),
    'PRISM8':(0.1,0.1,0.1,1.0),
    'PRISM10':(0.05,0.05,0.05,1.0),
}


r5 = np.sqrt(5.0)
phi = (1.0 + r5) / 2.0

# ------------------------------------------------------------------------
# Solid data data
# ------------------------------------------------------------------------

def get_solid_data(solid_type: str):

    # --- Regular prisms  ---
    def make_prism(n: int):
        """
        Create a canonical n-gon prism centered at origin.
        - Bottom cap on z = -1, vertices 0..n-1 in CCW order when viewed from +z.
        - Top cap on z = +1, vertices n..2n-1 matching bottom order.
        Faces:
          - Bottom n-gon, Top n-gon, and n rectangular sides as quads.
        ref_indices: pick first three bottom vertices (0,1,2).
        """
        # use radius, such that side length is equal to 1
        r = 1.0 /2/ np.sin(np.pi / n)
        z0, z1 = -0.5, 0.5
        verts = []
        # bottom ring (0..n-1)
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            verts.append((r * np.cos(ang), r * np.sin(ang), z0))
        # top ring (n..2n-1)
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            verts.append((r * np.cos(ang), r * np.sin(ang), z1))

        # Faces
        # Bottom: keep order 0..n-1 (CCW from +z gives outward -z normal, acceptable)
        bottom = tuple(range(n-1, -1,-1))
        # Top: reverse to ensure outward +z normal when viewed from +z
        top = tuple(range(n, 2*n, 1))

        faces = [bottom, top]
        # Side quads
        for i in range(n):
            i_next = (i + 1) % n
            faces.append((i, i_next, n + i_next, n + i))

        return verts, faces

    if solid_type in {'PRISM3', 'PRISM_3'}:
        return make_prism(3)
    if solid_type in {'PRISM5', 'PRISM_5'}:
        return make_prism(5)
    if solid_type in {'PRISM6', 'PRISM_6'}:
        return make_prism(6)
    if solid_type in {'PRISM8', 'PRISM_8'}:
        return make_prism(8)
    if solid_type in {'PRISM10', 'PRISM_10'}:
        return make_prism(10)

    if solid_type == 'TETRA':
        verts = [
            (1.0,  1.0,  1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]
        faces = [
            (0, 2, 1),
            (0, 1, 3),
            (0, 3, 2),
            (1, 2, 3),
        ]
        return verts, faces

    if solid_type == 'CUBE':
        verts = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            (-1.0,  1.0,  1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0,  1.0),
            (1.0,  1.0, -1.0),
            (1.0,  1.0,  1.0),
        ]
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]

        return verts, faces

    if solid_type == 'OCTA':
        verts = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        faces = [
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ]
        return verts, faces

    if solid_type == 'ICOSA':
        phi = (1.0 + r5) / 2.0
        a = 1.0
        verts = [
            (0,  a,  phi),
            (0, -a,  phi),
            (0,  a, -phi),
            (0, -a, -phi),
            (a,  phi, 0),
            (-a, phi, 0),
            (a, -phi, 0),
            (-a, -phi, 0),
            (phi, 0,  a),
            (-phi, 0,  a),
            (phi, 0, -a),
            (-phi, 0, -a),
        ]
        faces = [
              (0, 8, 4),  (0, 4, 5),  (0, 5, 9),  (0, 9, 1),
            (1, 6, 8),  (1, 7, 6),  (1, 9, 7),  (3, 2, 10), (2, 3, 11),
            (5, 2, 11), (4, 2, 5),  (2, 4, 10 ),   (3, 7, 11),
            (3, 6, 7),  (3, 10, 6), (4,  8,10),
            (6, 10, 8), (9,5,11), (7,9,11),(0,1, 8)
        ]
        return verts, faces

    if solid_type == 'DODECA':
        phi = (1.0 + r5) / 2.0
        inv_phi = 1.0 / phi

        verts = []
        # (±1, ±1, ±1)
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    verts.append((x, y, z))

        # (0, ±1/φ, ±φ) and permutations
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                verts.append((0.0, s1 * inv_phi, s2 * phi))
                verts.append((s1 * inv_phi, s2 * phi, 0.0))
                verts.append((s1 * phi, 0.0, s2 * inv_phi))

        faces = [
            (8,4,15,9,0),
            (15,5,11,1,9),
            (4,16,19,5,15),
            (8,14,6,16,4),
            (0,10,2,14,8),
            (9,1,13,10,0),
            (7,18,12,3,17),
            (3,13,1,11,17),
            (12,2,10,13,3),
            (18,6,14,2,12),
            (7,19,16,6,18),
            (17,11,5,19,7),
        ]
        return verts, faces

    if solid_type in {'TRUNC_OCTA', 'CUBOCTAHEDRON', 'TRUNC_TETRA',
                      'TRUNC_CUBOCTA', 'RHOMBICUBOCTA', 'TRUNC_HEXA',
                      'ICOSIDODECA', 'TRUNC_ICOSA', 'TRUNC_DODECA',
                      'RHOMBICOSIDODECA', 'TRUNC_ICOSIDODECA'
                      }:
        verts = []
        normals = []
        phi = (1.0 + r5) / 2.0

        def get_cyclic_perms(coords):
            # coords is a tuple of 3
            # generate cyclic shifts: (x,y,z), (y,z,x), (z,x,y)
            # for each, generate sign flips
            res = set()
            for p in [(coords[0], coords[1], coords[2]),
                      (coords[1], coords[2], coords[0]),
                      (coords[2], coords[0], coords[1])]:
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        for s3 in (-1, 1):
                            res.add((p[0] * s1, p[1] * s2, p[2] * s3))
            return list(res)

        if solid_type == 'CUBOCTAHEDRON':
            # Vertices: Permutations of (±1, ±1, 0)
            for i in range(3):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        v = [0, 0, 0]
                        v[i] = 0
                        v[(i + 1) % 3] = s1
                        v[(i + 2) % 3] = s2
                        verts.append(tuple(v))

            # Normals: 8 triangles (±1, ±1, ±1), 6 squares (±1, 0, 0)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            for i in range(3):
                for s in (-1, 1):
                    n = [0, 0, 0]
                    n[i] = s
                    normals.append(tuple(n))

        elif solid_type == 'TRUNC_OCTA':
            # Vertices: Permutations of (0, ±1, ±2)
            for p in itertools.permutations((0, 1, 2)):
                nz_indices = [k for k, val in enumerate(p) if val != 0]
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        v = list(p)
                        v[nz_indices[0]] *= s1
                        v[nz_indices[1]] *= s2
                        verts.append(tuple(v))
            verts = list(set(verts))

            # Normals: 6 squares (±1, 0, 0), 8 hexagons (±1, ±1, ±1)
            for i in range(3):
                for s in (-1, 1):
                    n = [0, 0, 0]
                    n[i] = s
                    normals.append(tuple(n))
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))

        elif solid_type == 'TRUNC_TETRA':
            # Vertices: Permutations of (±1, ±1, ±3) with product > 0
            unique_coords = set()
            for p in itertools.permutations((1, 1, 3)):
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            if sx * sy * sz > 0:
                                unique_coords.add((p[0] * sx, p[1] * sy, p[2] * sz))
            verts = list(unique_coords)

            # Normals: (±1, ±1, ±1) cover both triangles and hexagons
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))

        elif solid_type == 'TRUNC_CUBOCTA':
            # Great Rhombicuboctahedron
            # Vertices: Permutations of (±1, ±(1+√2), ±(1+2√2))
            val0 = 1.0
            val1 = 1.0 + np.sqrt(2.0)
            val2 = 1.0 + 2.0 * np.sqrt(2.0)

            for p in itertools.permutations((val0, val1, val2)):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        for s3 in (-1, 1):
                            verts.append((p[0] * s1, p[1] * s2, p[2] * s3))
            verts = list(set(verts))

            # Normals:
            # 12 Squares (±1, ±1, 0)... but actually faces are 4,6,8
            # Faces are: 12 squares, 8 hexagons, 6 octagons
            # Octagons: along axes (±1, 0, 0)
            for i in range(3):
                for s in (-1, 1):
                    n = [0, 0, 0]
                    n[i] = s
                    normals.append(tuple(n))
            # Hexagons: corners (±1, ±1, ±1)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            # Squares: edges (±1, ±1, 0)
            for i in range(3):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        n = [0, 0, 0]
                        n[i] = 0
                        n[(i + 1) % 3] = s1
                        n[(i + 2) % 3] = s2
                        normals.append(tuple(n))

        elif solid_type == 'RHOMBICUBOCTA':
            # Small Rhombicuboctahedron
            # Vertices: Permutations of (±1, ±1, ±(1+√2))
            val0 = 1.0
            val1 = 1.0 + np.sqrt(2.0)

            for p in itertools.permutations((val0, val0, val1)):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        for s3 in (-1, 1):
                            verts.append((p[0] * s1, p[1] * s2, p[2] * s3))
            verts = list(set(verts))

            # Normals: 18 squares, 8 triangles
            # Squares type 1: (±1, 0, 0)
            for i in range(3):
                for s in (-1, 1):
                    n = [0, 0, 0]
                    n[i] = s
                    normals.append(tuple(n))
            # Squares type 2: (±1, ±1, 0) normalized? No, just direction matters for face finding
            for i in range(3):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        n = [0, 0, 0]
                        n[i] = 0
                        n[(i + 1) % 3] = s1
                        n[(i + 2) % 3] = s2
                        normals.append(tuple(n))
            # Triangles: (±1, ±1, ±1)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))

        elif solid_type == 'TRUNC_HEXA':
            # Truncated Cube
            # Vertices: Permutations of (±1, ±1, ±(√2 - 1))
            # Or typically (±1, ±1, ±(√2 - 1)) scaled.
            # Standard coords: (±ξ, ±1, ±1) where ξ = √2 - 1
            xi = np.sqrt(2.0) - 1.0
            for p in itertools.permutations((xi, 1.0, 1.0)):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        for s3 in (-1, 1):
                            verts.append((p[0] * s1, p[1] * s2, p[2] * s3))
            verts = list(set(verts))

            # Normals: 6 octagons (cube faces), 8 triangles (corners)
            # Octagons: (±1, 0, 0)
            for i in range(3):
                for s in (-1, 1):
                    n = [0, 0, 0]
                    n[i] = s
                    normals.append(tuple(n))
            # Triangles: (±1, ±1, ±1)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))

        elif solid_type == 'ICOSIDODECA':

            # Vertices: 30
            # Cyclic perms of (0, 0, ±phi) -> 6
            # Cyclic perms of (±1/2, ±phi/2, ±phi^2/2) -> 24
            # Note: phi^2 = phi + 1
            verts.extend(get_cyclic_perms((0.0, 0.0, phi)))
            verts.extend(get_cyclic_perms((0.5, phi / 2.0, (phi + 1.0) / 2.0)))
            verts = list(set(verts))

            # Normals: 12 Pentagons (Icosahedron verts), 20 Triangles (Dodecahedron verts)
            # Pentagons: Cyclic (0, ±1, ±phi)
            normals.extend(get_cyclic_perms((0.0, phi,-1)))
            # Triangles: (±1, ±1, ±1) and Cyclic (0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0, 1.0 / phi, phi)))

        elif solid_type == 'TRUNC_ICOSA':
            # Vertices: 60 (Soccer ball)
            # Cyclic perms of (0, ±1, ±3phi)
            # Cyclic perms of (±2, ±(1+2phi), ±phi)
            # Cyclic perms of (±1, ±(2+phi), ±2phi)

            verts.extend(get_cyclic_perms((0.0, 1.0, 3.0 * phi)))
            verts.extend(get_cyclic_perms((2.0, 1.0 + 2.0 * phi, phi)))
            verts.extend(get_cyclic_perms((1.0, 2.0 + phi, 2.0 * phi)))
            verts = list(set(verts))

            # Normals: 12 Pentagons (from Icosa verts), 20 Hexagons (from Icosa faces/Dodeca verts)
            # Pentagons: Cyclic (0, ±1, ±phi)
            normals.extend(get_cyclic_perms((0.0, -1.0, phi)))
            # Hexagons: (±1, ±1, ±1) and Cyclic (0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0,  phi, 1.0/phi)))

        elif solid_type == 'TRUNC_DODECA':
            # Vertices: 60
            # Cyclic perms of (0, ±1/phi, ±(2+phi))
            # Cyclic perms of (±1/phi, ±phi, ±2phi)
            # Cyclic perms of (±phi, ±2, ±(phi+1))
            verts.extend(get_cyclic_perms((0.0, 1.0 / phi, 2.0 + phi)))
            verts.extend(get_cyclic_perms((1.0 / phi, phi, 2.0 * phi)))
            verts.extend(get_cyclic_perms((phi, 2.0, phi + 1.0)))
            verts = list(set(verts))

            # Normals: 12 Decagons (from Dodeca faces/Icosa verts), 20 Triangles (from Dodeca verts)
            # Decagons: Cyclic (0, ±1, ±phi)
            normals.extend(get_cyclic_perms((0.0, 1.0, 1/phi)))
            # Triangles: (±1, ±1, ±1) and Cyclic (0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0, 1.0 / phi, phi)))

        elif solid_type == 'RHOMBICOSIDODECA':
            # Vertices: 60
            # Cyclic perms of (±1, ±1, ±phi^3)
            # Cyclic perms of (±phi^2, ±phi, ±2phi)
            # Cyclic perms of (±(2+phi), 0, ±phi^2)
            # Note: phi^3 = 2*phi + 1
            p3 = 2.0 * phi + 1.0
            p2 = phi + 1.0
            verts.extend(get_cyclic_perms((1.0, 1.0, p3)))
            verts.extend(get_cyclic_perms((p2, phi, 2.0 * phi)))
            verts.extend(get_cyclic_perms((2.0 + phi, 0.0, p2)))
            verts = list(set(verts))

            # Normals:
            # 12 Pentagons (Icosa verts direction): Cyclic (0, ±1, ±phi)
            normals.extend(get_cyclic_perms((0.0, phi, 1.0)))
            # 20 Triangles (Dodeca verts direction): (±1,±1,±1) + Cyclic(0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0, 1.0 / phi, phi)))
            # 30 Squares (Icosidodeca verts direction):
            # Cyclic (0, 0, ±phi) AND Cyclic (±1/2, ±phi/2, ±p2/2)
            normals.extend(get_cyclic_perms((0.0, 0.0, phi)))
            normals.extend(get_cyclic_perms((0.5, phi / 2.0, p2 / 2.0)))

        elif solid_type == 'TRUNC_ICOSIDODECA':
            # Great Rhombicosidodecahedron
            # Vertices: 120. All even permutations of:
            # (±1/phi, ±1/phi, ±(3+phi))
            # (±2/phi, ±phi, ±(1+2phi))
            # (±1/phi, ±phi^2, ±(3phi-1))
            # (±(2phi-1), ±2, ±(2+phi))
            # (±phi, ±3, ±2phi)

            # Using cyclic perms gives enough coverage because of the symmetry
            # (even perms of x,y,z include cyclic shifts).
            sets = [
                (1.0 / phi, 1.0 / phi, 3.0 + phi),
                (2.0 / phi, phi, 1.0 + 2.0 * phi),
                (1.0 / phi, phi * phi, 3.0 * phi - 1.0),
                (2.0 * phi - 1.0, 2.0, 2.0 + phi),
                (phi, 3.0, 2.0 * phi)
            ]
            for s in sets:
                verts.extend(get_cyclic_perms(s))
            verts = list(set(verts))

            # Normals: 12 Decagons, 20 Hexagons, 30 Squares
            # Same directions as Rhombicosidodecahedron
            # Decagons: Cyclic (0, ±1, ±phi)
            normals.extend(get_cyclic_perms((0.0,  phi, 1.0)))
            # Hexagons: (±1,±1,±1) + Cyclic(0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0, 1.0 / phi, phi)))
            # Squares: Cyclic (0, 0, ±phi) + Cyclic (±1/2, ±phi/2, ±p2/2)
            normals.extend(get_cyclic_perms((0.0, 0.0, phi)))
            normals.extend(get_cyclic_perms((0.5, phi / 2.0, (phi + 1.0) / 2.0)))

        # Build faces from normals
        faces = []
        verts_arr = np.array(verts)

        for n in normals:
            n_vec = np.array(n)
            dots = verts_arr @ n_vec
            max_val = np.max(dots)
            # Find vertices close to max plane
            tol = 1e-5
            indices = [i for i, d in enumerate(dots) if abs(d - max_val) < tol*max_val]

            # Sort vertices angularly
            u = np.array([0.0, 0.0, 1.0])
            if abs(n_vec[2]) > 0.9 * np.linalg.norm(n_vec):
                u = np.array([0.0, 1.0, 0.0])
            v_basis = np.cross(n_vec, u)
            v_basis /= np.linalg.norm(v_basis)
            u_basis = np.cross(v_basis, n_vec)
            u_basis /= np.linalg.norm(u_basis)

            face_verts = verts_arr[indices]
            center = np.mean(face_verts, axis=0)

            def get_angle(idx):
                vec = verts_arr[idx] - center
                x = np.dot(vec, u_basis)
                y = np.dot(vec, v_basis)
                return np.arctan2(y, x)

            sorted_indices = sorted(indices, key=get_angle)
            faces.append(tuple(sorted_indices))

        return verts, faces
    raise ValueError(f"Unknown solid_type: {solid_type}")


# ------------------------------------------------------------------------
# Similarity transform (fit solid so 3 canonical vertices match 3 points)
# ------------------------------------------------------------------------

def compute_similarity_transform(a0, a1, a2, p0, p1, p2):
    """
    Find scale s, rotation R, translation t so that
    R * (a_i * s) + t ~= p_i for i=0,1,2.
    """
    vA1 = a1 - a0
    vA2 = a2 - a0
    vP1 = p1 - p0
    vP2 = p2 - p0

    lenA1 = vA1.length
    lenP1 = vP1.length
    if lenA1 < 1e-8 or lenP1 < 1e-8:
        raise ValueError("Degenerate reference configuration")

    scale = lenP1 / lenA1

    uA = vA1.normalized()
    wA = vA1.cross(vA2)
    if wA.length < 1e-8:
        raise ValueError("Canonical reference points are collinear")
    wA.normalize()
    vA = wA.cross(uA)

    uP = vP1.normalized()
    wP = vP1.cross(vP2)
    if wP.length < 1e-8:
        raise ValueError("Target points are collinear")
    wP.normalize()
    vP = wP.cross(uP)

    RA = Matrix((uA, vA, wA)).transposed()
    RP = Matrix((uP, vP, wP)).transposed()



    R = RP @ RA.transposed()
    R.resize_4x4()

    a0_scaled = a0 * scale
    t = p0 - (R.to_3x3() @ a0_scaled)

    return scale, R, t


def apply_similarity_to_vertices(verts, scale, R, t):
    result = []
    for v in verts:
        v_vec = Vector(v) * scale
        v_world = R.to_3x3() @ v_vec + t
        result.append(v_world)
    return result


# ------------------------------------------------------------------------
# Operator: Create regular solid from stored points
# ------------------------------------------------------------------------

def solid_items_for_ngon(self, context):
    face_sides = getattr(self, "face_sides", 3)
    all_items = [
        ('TETRA', "Tetrahedron (triangles)", ""),
        ('OCTA', "Octahedron (triangles)", ""),
        ('ICOSA', "Icosahedron (triangles)", ""),
        ('CUBE', "Cube (quads)", ""),
        ('DODECA', "Dodecahedron (pentagons)", ""),
        ('TRUNC_OCTA', "Truncated Octahedron (squares, hexagons)", ""),
        ('CUBOCTAHEDRON', "Cuboctahedron (triangles, squares)", ""),
        ('TRUNC_TETRA', "Truncated Tetrahedron (triangles, hexagons)", ""),
        ('PRISM3', "Three prism (triangles, quads)", ""),
        ('PRISM5', "Five prism (quads, pentagons)", ""),
        ('PRISM6', "Six prism (quads, hexagons)", ""),
        ('PRISM8', "Eight prism (quads, octagons)", ""),
        ('PRISM10', "Ten prism (quads, decagons)", ""),
        ('TRUNC_CUBOCTA', "Great Rhombicuboctahedron (4, 6, 8)", ""),
        ('RHOMBICUBOCTA', "Rhombicuboctahedron (3, 4)", ""),
        ('TRUNC_HEXA', "Truncated Hexahedron (3, 8)", ""),
        ('ICOSIDODECA', "Icosidodecahedron (3, 5)", ""),
        ('TRUNC_ICOSA', "Truncated Icosahedron (5, 6)", ""),
        ('TRUNC_DODECA', "Truncated Dodecahedron (3, 10)", ""),
        ('RHOMBICOSIDODECA', "Rhombicosidodecahedron (3, 4, 5)", ""),
        ('TRUNC_ICOSIDODECA', "Truncated Icosidodecahedron (4, 6, 10)", ""),
    ]

    return [
        item for item in all_items
        if face_sides in SOLID_FACE_SIDES[item[0]]
        # if face_sides == (3) #in SOLID_FACE_SIDES[item[0]]
    ]

class PointItem(bpy.types.PropertyGroup):
    """Helper property group to store variable number of points in the operator"""
    vec: FloatVectorProperty(name="Location", size=3)

class OBJECT_OT_remove_spheres(bpy.types.Operator):
    """Remove spheres from selected objects"""
    bl_idname = "object.remove_spheres"
    bl_label = "Remove Spheres and Edges"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if "Sphere_Vert" in obj.name:
                obj.select_set(True)

        bpy.ops.object.delete()

        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if isinstance(mod,bpy.types.NodesModifier):
                    nodes = mod.node_group.nodes
                    curve_circle_node = nodes.get("Curve Circle")
                    curve_circle_node.inputs["Radius"].default_value=0


        return {'FINISHED'}


class OBJECT_OT_add_solid_from_selection(bpy.types.Operator):
    """Create a Platonic solid fitted through selected objects (3–10)"""
    bl_idname = "object.add_solid_from_selection"
    bl_label = "Add Regular Solid From Selection"
    bl_options = {'REGISTER', 'UNDO'}

    # We use a CollectionProperty to store dynamic points
    points: CollectionProperty(type=PointItem)

    # Keep these for reference or logic
    point_count: IntProperty(default=0)
    face_sides: IntProperty(default=3)

    solid_type: EnumProperty(
        name="Solid",
        description="Regular solid with faces that are n-gons",
        items=solid_items_for_ngon,
    )

    flip_face: BoolProperty(
        name="Flip Face",
        description="Flip the orientation of the solid relative to the reference face",
        default=False
    )

    offset: IntProperty(
        name="Offset",
        description="Offset the mapping of old and new vertices",
        default = 0,
        min = 0,
        max = 2
    )

    vertex_sphere_radius: FloatProperty(
        name="Vertex Sphere Radius",
        description="Radius of spheres to place at each vertex",
        default=0.1,
        min=0.001
    )

    def invoke(self, context, event):
        # Immediately grab selected objects
        objs = context.selected_objects
        if not (3 <= len(objs) <= 10):
            self.report({'ERROR'}, "Please select between 3 and 10 objects.")
            return {'CANCELLED'}

        # Sort objects by name to have a deterministic order, or keep selection order if possible?
        # Blender's selected_objects order isn't guaranteed, but let's just use it.
        # If needed, one could sort by name: objs.sort(key=lambda o: o.name)

        self.points.clear()

        locs = [obj.location.copy() for obj in objs]

        # create orthonormal basis
        center = sum(locs,Vector()) / len(locs)
        ref = locs[0]-center
        dic = {}
        ref2 = locs[1]-center

        n = ref.cross(ref2)
        if n.length<1.e-4: # two points might be collinear
            ref3 = locs[2] - center
            n = ref.cross(ref3)

        ref.normalize()
        n.normalize()
        u = ref
        v = n.cross(u)

        for i in range(len(locs)):
            x,y=u.dot(locs[i]-center),v.dot(locs[i]-center)
            dic[i] = np.arctan2(y,x)

        dic  = dict(sorted(dic.items(), key=lambda x: x[1]))
        self.report({'INFO'}, f"Points sorted by angle: {dic}")

        for key in dic.keys():
            item = self.points.add()
            item.vec = locs[key]

        self.point_count = len(objs)

        # Map count to face sides.
        if self.point_count in [3, 4, 5, 6, 8, 10]:
            self.face_sides = self.point_count
        else:
            # fall-back for the case of illegal point count
            self.face_sides = 3

        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "solid_type")
        layout.prop(self, "flip_face")
        layout.prop(self,"offset")
        layout.prop(self, "vertex_sphere_radius")
        layout.separator()
        # layout.label(text=f"Selected {self.point_count} locations:")
        # box = layout.box()
        # for i, item in enumerate(self.points):
        #     box.prop(item, "vec", text=f"Object {i+1}")

    def execute(self, context):
        if len(self.points) < 3:
            return {'CANCELLED'}

        # Use first 3 points to define the transform
        p0 = Vector(self.points[0].vec)
        p1 = Vector(self.points[1].vec)
        p2 = Vector(self.points[2].vec)

        # If flipped, swap p1 and p2 to invert normal direction of the base triangle
        if self.flip_face:
            p0,p1,p2=p2,p1,p0

        # cycle through the mapping points
        for o in range(self.offset):
            p0,p1,p2 = p1,p2,p0


        self.report({'INFO'}, f"{self.solid_type} is selected now.")

        try:
            verts_canon, faces= get_solid_data(self.solid_type)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        sel_face = None
        for face in faces:
            if len(face)==len(self.points):
                sel_face = face
                self.report({'INFO'}, f"Selected face: {sel_face}")
                break

        a0 = Vector(verts_canon[sel_face[0]])
        a1 = Vector(verts_canon[sel_face[1]])
        a2 = Vector(verts_canon[sel_face[2]])

        try:
            scale, R, t = compute_similarity_transform(a0, a1, a2, p0, p1, p2)
        except ValueError as e:
            self.report({'ERROR'}, f"Cannot fit solid to points: {e}")
            return {'CANCELLED'}

        verts_world = apply_similarity_to_vertices(verts_canon, scale, R, t)

        mesh = bpy.data.meshes.new(f"{self.solid_type}_FromSelection_Mesh")
        mesh.from_pydata(verts_world, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(f"{self.solid_type}_FromSelection", mesh)
        context.scene.collection.objects.link(obj)

        # --- Add Material ---
        mat_name = f"Mat_{self.solid_type}"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                color = SOLID_COLORS.get(self.solid_type, (0.8, 0.8, 0.8, 1.0))
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs["Roughness"].default_value = 0.1

        # --- Add Material ---
        mat_name = "Mat_Sphere"
        sphere_mat = bpy.data.materials.get(mat_name)
        if not sphere_mat:
            sphere_mat = bpy.data.materials.new(name=mat_name)
            sphere_mat.use_nodes = True
            bsdf = sphere_mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                color = (0.3,0,0,1.0)
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs["Roughness"].default_value = 0.1

        mat_edges = "Mat_Edges"
        edge_mat = bpy.data.materials.get(mat_edges)
        if not edge_mat:
            edge_mat = bpy.data.materials.new(name=mat_edges)
            edge_mat.use_nodes = True
            bsdf = edge_mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                color = (0.97,0.95,0.5,1.0)
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs["Roughness"].default_value = 0.1


        #  Add geometry nodes modifier to visualize the edges
        tree = bpy.data.node_groups.new(name="EdgeVisualizer", type='GeometryNodeTree')
        nodes = tree.nodes
        tree.interface.new_socket(name="Geo In", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket(name="Geo Out", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        ins =nodes.new("NodeGroupInput")
        ins.location = (-600,0)
        out = nodes.new("NodeGroupOutput")
        out.location = (400,0)
        mesh_to_curve = nodes.new(type="GeometryNodeMeshToCurve")
        mesh_to_curve.location=(-400,200)
        curve_to_mesh = nodes.new(type="GeometryNodeCurveToMesh")
        curve_to_mesh.location=(-200,200)
        curve_circle = nodes.new(type="GeometryNodeCurvePrimitiveCircle")
        curve_circle.location=(-400,0)
        curve_circle.inputs["Radius"].default_value=0.05
        curve_circle.inputs["Resolution"].default_value=4
        edge_mat_node = nodes.new(type="GeometryNodeSetMaterial")
        edge_mat_node.location=(0,200)
        edge_mat_node.inputs["Material"].default_value=edge_mat
        mat_node = nodes.new(type="GeometryNodeSetMaterial")
        mat_node.location=(0,-200)
        mat_node.inputs["Material"].default_value = mat
        join_geometry = nodes.new(type="GeometryNodeJoinGeometry")
        join_geometry.location=(200,0)
        tree.links.new(ins.outputs[0],mat_node.inputs[0])
        tree.links.new(mat_node.outputs[0],join_geometry.inputs[0])
        tree.links.new(ins.outputs[0],mesh_to_curve.inputs[0])
        tree.links.new(curve_circle.outputs[0],curve_to_mesh.inputs["Profile Curve"])
        tree.links.new(mesh_to_curve.outputs[0],curve_to_mesh.inputs[0])
        tree.links.new(curve_to_mesh.outputs[0],edge_mat_node.inputs[0])
        tree.links.new(edge_mat_node.outputs[0],join_geometry.inputs[0])
        tree.links.new(join_geometry.outputs[0],out.inputs[0])

        mod = obj.modifiers.new(name="EdgeVisualizer", type="NODES")
        mod.node_group = tree

        # add materials
        obj.data.materials.append(mat)
        obj.data.materials.append(edge_mat)


        # --- Add Spheres at Vertices ---
        if self.vertex_sphere_radius > 0:
            # Create a shared mesh for all spheres to be efficient
            bpy.ops.mesh.primitive_ico_sphere_add(
                radius=self.vertex_sphere_radius,subdivisions=1
            )
            sphere_obj = context.active_object
            sphere_mesh = sphere_obj.data
            # We don't need the initial object, just the mesh data
            bpy.data.objects.remove(sphere_obj)

            sphere_collection_name = f"{self.solid_type}_VertexSpheres"
            sphere_collection = bpy.data.collections.new(sphere_collection_name)
            context.scene.collection.children.link(sphere_collection)

            for i, v_co in enumerate(verts_world):
                v_sphere = bpy.data.objects.new(f"Sphere_Vert_{i}", sphere_mesh)
                v_sphere.location = v_co
                sphere_collection.objects.link(v_sphere)

                # Optional: parent spheres to the main solid
                v_sphere.parent = obj

                # Apply same material or a different one? Let's use the same for now or white
                v_sphere.data.materials.append(sphere_mat)

        # Select the new object
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        self.report(
            {'INFO'},
            f"{self.solid_type} created from selection."
        )
        return {'FINISHED'}


# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class VIEW3D_PT_solid_from_selection(bpy.types.Panel):
    """Panel in the 3D Viewport"""
    bl_label = "Regular Solid From Selection"
    bl_idname = "VIEW3D_PT_solid_from_selection"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Regular Solid"

    def draw(self, context):
        layout = self.layout

        # Check selection count
        n = len(context.selected_objects)
        if 3 <= n <= 10:
            layout.operator(
                OBJECT_OT_add_solid_from_selection.bl_idname,
                text=f"Create from {n} Objects"
            )
        else:
            layout.label(text="Select 3–10 objects", icon='INFO')

        layout.operator(OBJECT_OT_remove_spheres.bl_idname, text="Remove Spheres and Edges")


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    PointItem,
    OBJECT_OT_add_solid_from_selection,
    OBJECT_OT_remove_spheres,
    VIEW3D_PT_solid_from_selection,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()