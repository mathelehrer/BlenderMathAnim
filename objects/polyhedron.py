from __future__ import annotations
import itertools

import numpy as np
from anytree import RenderTree
from mathutils import Vector, Quaternion, Matrix

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import PolyhedronViewModifier
from interface import ibpy
from interface.ibpy import create_mesh, to_vector
from mathematics import geometry
from mathematics.geometry.field_extensions import QR
from mathematics.geometry.meshface import MeshFace
from mathematics.lin_alg.subspace import Subspace
from new_stuff.mathematics.unfolder import color_dict
from objects.bobject import BObject
from objects.face import Face

from objects.tex_bobject import SimpleTexBObject
from utils import kwargs
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

# ------------------------------------------------------------------------
# Solid data data
# ------------------------------------------------------------------------
r5 = np.sqrt(5)


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
        r = 1.0 / 2 / np.sin(np.pi / n)
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
        bottom = tuple(range(n - 1, -1, -1))
        # Top: reverse to ensure outward +z normal when viewed from +z
        top = tuple(range(n, 2 * n, 1))

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
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
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
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
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
            (0, a, phi),
            (0, -a, phi),
            (0, a, -phi),
            (0, -a, -phi),
            (a, phi, 0),
            (-a, phi, 0),
            (a, -phi, 0),
            (-a, -phi, 0),
            (phi, 0, a),
            (-phi, 0, a),
            (phi, 0, -a),
            (-phi, 0, -a),
        ]
        faces = [
            (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
            (1, 6, 8), (1, 7, 6), (1, 9, 7), (3, 2, 10), (2, 3, 11),
            (5, 2, 11), (4, 2, 5), (2, 4, 10), (3, 7, 11),
            (3, 6, 7), (3, 10, 6), (4, 8, 10),
            (6, 10, 8), (9, 5, 11), (7, 9, 11), (0, 1, 8)
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
            (8, 4, 15, 9, 0),
            (15, 5, 11, 1, 9),
            (4, 16, 19, 5, 15),
            (8, 14, 6, 16, 4),
            (0, 10, 2, 14, 8),
            (9, 1, 13, 10, 0),
            (7, 18, 12, 3, 17),
            (3, 13, 1, 11, 17),
            (12, 2, 10, 13, 3),
            (18, 6, 14, 2, 12),
            (7, 19, 16, 6, 18),
            (17, 11, 5, 19, 7),
            (3, 13, 1, 11, 17),
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
            normals.extend(get_cyclic_perms((0.0, 1.0, phi)))
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
            normals.extend(get_cyclic_perms((0.0, 1.0, phi)))
            # Hexagons: (±1, ±1, ±1) and Cyclic (0, ±1/phi, ±phi)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        normals.append((x, y, z))
            normals.extend(get_cyclic_perms((0.0, 1.0 / phi, phi)))

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
            normals.extend(get_cyclic_perms((0.0, 1.0, phi)))
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
            normals.extend(get_cyclic_perms((0.0, phi, 1.0)))
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
            indices = [i for i, d in enumerate(dots) if abs(d - max_val) < tol * max_val]

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


def compute_similarity_transform(a0, a1, a2, p0, p1, p2):
    """
    Find scale s, rotation R, translation t so that
    R * (a_i * s) + t ~= p_i for i=0,1,2.
    """
    vA1 = to_vector(a1) - to_vector(a0)
    vA2 = to_vector(a2) - to_vector(a0)
    if isinstance(p1, QR):
        vP1 = p1.real() - p0
        vP2 = p2.real() - p0
    else:
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

    a0_scaled = to_vector(a0) * scale
    t = p0 - (R.to_3x3() @ a0_scaled)

    return scale, R, t


def apply_similarity_to_vertices(verts, scale, R, t):
    result = []
    for v in verts:
        v_vec = Vector(v) * scale
        v_world = R.to_3x3() @ v_vec + t
        result.append(v_world)
    return result


class Polyhedron(BObject):
    """
    Create a polyhedron from vertices, edges and faces:
    """

    def __init__(self, vertices, faces, **kwargs):
        """
        :param vertices: vertices
        :param edges: edges in terms of vertices
        :param faces: faces in terms of vertices
        :param index_one: if True -> counting starts at one
        :param kwargs:
        """

        self.simple = get_from_kwargs(kwargs, 'simple', False)
        if self.simple:
            # short cut, when vertices and faces are given
            self.vertices = vertices
            if isinstance(faces[0],list):
                f = []
                for face in faces:
                    f.append(Face(self.vertices,face,Vector()))
                self.faces = f
            else:
                self.faces = faces
            super().__init__(mesh=create_mesh(vertices=vertices, faces=faces), **kwargs)
        else:
            self.counter = None  # dummy counter for growing the polyhedron
            self.root = None
            self.kwargs = kwargs
            self.faces = []
            self.vertices = vertices

            self.coordinate_system = self.get_from_kwargs('coordinate_system', None)
            if self.coordinate_system:  # adjust coordinates to the parenting coordinate system
                for i, v in enumerate(self.vertices):
                    self.vertices[i] = self.coordinate_system.coords2location(v)

            self.index_base = self.get_from_kwargs('index_base', 1)
            name = self.get_from_kwargs('name', 'Polyhedron')
            self.vertex_radius = self.get_from_kwargs('vertex_radius',
                                                      0.1)  # need to access the vertex radius to place word labels properly
            if 'name' in kwargs:
                kwargs.pop('name')

            # calculate the center of the polyhedron
            center = Vector([0, 0, 0])
            for vertex in vertices:
                center += Vector(vertex)
            center /= len(vertices)
            self.center = center

            objects = []

            location = self.get_from_kwargs('location', [0, 0, 0])
            if 'location' in kwargs:
                kwargs.pop('location')

            vertex_colors = self.get_from_kwargs('vertex_colors', ['example'])
            face_colors = self.get_from_kwargs('face_colors', ['drawing'])

            for i, f in enumerate(faces):
                face = Face(self.vertices, f, center, index=i, index_base=self.index_base,
                            colors=[vertex_colors, ['text'], face_colors], vertex_radius=self.vertex_radius, **kwargs)
                objects.append(face)
                self.faces.append(face)

            super().__init__(children=objects, name=name, location=location, **kwargs)

            if self.coordinate_system:
                self.coordinate_system.add_object(self)

    @classmethod
    def from_points(cls, vertices=None, solid_type="TETRA", **kwargs):
        src_vertices, faces = get_solid_data(solid_type)
        if vertices is None:
            bob = Polyhedron(src_vertices, faces, name=solid_type, simple=True, **kwargs)
        else:
            # transform source vertices to align polyhedron with the given vertices
            # find appropriate face
            flipped = get_from_kwargs(kwargs, 'flipped', False)
            if flipped:
                vertices = vertices[::-1]

            n = len(vertices)
            for face in faces:
                if len(face) == n:
                    break

            # align three consecutive points
            scale, rot, translation = compute_similarity_transform(src_vertices[face[0]], src_vertices[face[1]],
                                                                   src_vertices[face[2]], vertices[0], vertices[1],
                                                                   vertices[2])

            img_vertices = apply_similarity_to_vertices(src_vertices, scale, rot, translation)
            bob = Polyhedron(img_vertices, faces, name=solid_type, simple=True, **kwargs)
        return bob

    @classmethod
    def from_group(cls, group, start, eps=1.e-4, **kwargs):
        """
        The full polyhedron is generated from the group provided that a polyhedron is generated with all its edges
        having the same length :param group: :param start: :param eps: :return:
        """
        cls.group = group
        cls.start = start
        cls.EPS = eps

        vertices = []
        cls.word_vertex_dict = {}
        for element in group.elements:
            # ugly conversion between numpy and mathutils
            vertex = Vector((element.matrix @ np.matrix([start]).transpose()).transpose().tolist()[0])
            # calculate distance to all verticex in the list
            # only append vertex if it is different from all others
            different = True
            for v in vertices:
                diff = v - vertex
                if diff.dot(diff) < 0.1:
                    different = False
                    break

            if different:
                vertices.append(vertex)
                cls.word_vertex_dict[str(element)] = vertex

        # for v in vertices:
        #     print(v)

        # calculate edges
        min_distance = np.Infinity
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = (vertices[i] - vertices[j]).length
                if distance < min_distance:
                    min_distance = distance

        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = (vertices[i] - vertices[j]).length
                if np.abs(distance - min_distance) < cls.EPS:
                    edges.append(set([i, j]))

        # find at least three edges in a plane to form a face
        face_subspaces = set()
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1 = edges[i]
                e2 = edges[j]
                e12 = e1.union(e2)
                # two edges with common vertex
                if len(e12) == 3:
                    plane = Subspace(*tuple(e12), vertices, eps=cls.EPS)
                    for m in range(len(vertices)):
                        if m not in e12:
                            plane.add(m)
                    face_subspaces = face_subspaces.union({plane})

        face_subspaces = sorted(face_subspaces)
        faces = []
        for subspace in face_subspaces:
            faces.append(list(subspace.indices))

        return Polyhedron(vertices, faces, index_base=0, **kwargs)

    def __repr__(self):
        if self.group:
            f"{self.__class__.__name__}(group={self.group})"
        else:
            f"{self.__class__.__name__}(vertices={self.vertices})"

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):

        super().appear(begin_time=begin_time, transition_time=transition_time)
        if not self.simple:
            for face in self.faces:
                face.appear(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def grow(self, index=0, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, show_faces=True):
        super().appear(begin_time=begin_time, transition_time=0)  # needed for linking
        self.create_face_tree(index)
        self.counter = 0  # counts the number of faces that have been grown already, (unskillful hack with a global variable)
        max_level = self.max_level(self.root, 0)
        dt = transition_time / len(self.faces)  # time per face
        self.grow_recursively(self.root, begin_time=begin_time, transition_time=dt, show_faces=show_faces)
        return begin_time + transition_time

    def grow_without_faces(self, index=0, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        super().grow(index, begin_time, transition_time, faces=False)

    def grow_recursively(self, face_node, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, show_faces=True):
        face_node.grow(begin_time=begin_time + self.counter * transition_time, transition_time=transition_time)
        self.counter += 1
        for child in face_node.children:
            self.grow_recursively(child, begin_time=begin_time, transition_time=transition_time, show_faces=show_faces)

    def max_level(self, tree, level):
        if not tree.children:
            return level
        else:
            tmp = 0
            for child in tree.children:
                c_level = self.max_level(child, level + 1)
                if tmp < c_level:
                    tmp = c_level
            return tmp

    def create_face_tree(self, root_face_index):
        face_indices = []
        for i in range(len(self.faces)):
            face_indices.append(i)

        self.root = self.faces[root_face_index]
        face_indices.remove(root_face_index)
        self.root.parent = None

        current_level_faces = [self.root]
        while len(current_level_faces) > 0:
            next_level_faces = []
            for face in current_level_faces:
                to_be_removed = []
                for index in face_indices:
                    if face.is_neighbour_of(self.faces[index]):
                        neighbour = self.faces[index]
                        next_level_faces.append(neighbour)
                        neighbour.parent = face
                        to_be_removed.append(index)
                for tbr in to_be_removed:  # remove face indices that have been identified as neighbours
                    face_indices.remove(tbr)
            current_level_faces = next_level_faces.copy()

    def show_face_tree(self):
        for pre, fill, node in RenderTree(self.root):
            tree_str = u"%s%s" % (pre, str(node))
            print(tree_str.ljust(8))

    def unfold(self, fraction=1, begin_time=0, resolution=10, transition_time=OBJECT_APPEARANCE_TIME):
        dt = transition_time / resolution
        for child in self.root.children:
            angle, axis, center = child.get_unfolding_parameters(self.index_base)

            for r in range(resolution + 1):
                d_angle = angle / resolution * fraction
                alpha = d_angle * r
                quaternion_axis = axis * np.sin(alpha / 2)
                quaternion = Quaternion([np.cos(alpha / 2), *quaternion_axis[:]])
                translation = center - quaternion @ center  # translation that compensates the rotation around a
                # center different from the origin
                self.recursively_unfold(child, quaternion, translation, begin_time + r * dt, dt,
                                        fraction * r / resolution)
        return begin_time + transition_time

    def recursively_unfold(self, face, quaternion, translation, begin_time, transition_time, fraction):
        face.rotate(rotation_quaternion=quaternion, begin_time=begin_time, transition_time=transition_time)
        face.move_to(translation, begin_time=begin_time, transition_time=transition_time)

        for child in face.children:
            angle, axis, center = child.get_unfolding_parameters(self.index_base)

            # transform axis and center of rotation to the new values according to the transformation of the parents
            axis = quaternion @ axis  # adjust axis of rotation to the transformed face, only rotations need to be
            # considered, since the position of the axis is unimportant
            center = quaternion @ center + translation  # adjust the center of rotation
            alpha = angle * fraction
            quaternion2_axis = axis * np.sin(alpha / 2)
            quaternion2 = Quaternion([np.cos(alpha / 2), *quaternion2_axis[:]])

            # the local transformation of this child with respect to its parent is a rotation of quaternion2 around
            # the center center the transformation is composed with the transformation provided by the parent and
            # grand parents
            self.recursively_unfold(child, quaternion2 @ quaternion, center + quaternion2 @ (translation - center),
                                    begin_time, transition_time, fraction)

    def disappear_faces(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME
                        ):
        for face in self.faces:
            face.disappear_polygon(begin_time=begin_time, transition_time=transition_time)

    def write_word(self, word, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, modus='top', **kwargs):
        if modus == 'top':
            shift = Vector((0, 0, 1))
        if modus == 'right':
            shift = Vector((1, 0, 0))
        if modus == 'front':
            shift = Vector((0, -1, 0))

        if word in self.group.words:
            vertex = self.word_vertex_dict[word]
            location = vertex + self.vertex_radius * shift
            if self.coordinate_system:
                location = self.coordinate_system.coords2location(location)

            if 'scale' in kwargs:
                scale = kwargs['scale']
                kwargs.pop('scale')
            else:
                scale = 2

            if 'thickness' in kwargs:
                thickness = kwargs['thickness']
                kwargs.pop('thickness')
            else:
                thickness = 3

            bword = SimpleTexBObject(word, location=location, scale=scale, thickness=thickness,
                                     **kwargs)
            self.coordinate_system.add_object(bword)
            bword.write(begin_time=begin_time, transition_time=transition_time)

    def change_emission(self, from_value=0, to_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for face in self.faces:
            face.change_emission(from_value=from_value, to_value=to_value, begin_time=begin_time,
                                 transition_time=transition_time)


class PolyhedronWithModifier(BObject):
    def __init__(self, vertices, faces, **kwargs):
        self.vertices = vertices
        self.faces = faces
        self.kwargs = kwargs
        self.name = get_from_kwargs(kwargs, "name", "PolyhedronWithModifier")
        self.group = get_from_kwargs(kwargs, "group", None)
        self.signature = get_from_kwargs(kwargs, "signature", None)
        self.shape_key=0

        face_classes = get_from_kwargs(kwargs, 'face_classes', None)
        self.face_classes = face_classes
        # assign a default color that creates a face-size dependent color
        self.color = get_from_kwargs(kwargs, 'color', "color_dict")

        if self.color == "color_dict":
            colors = []
            for key in face_classes.keys():
                colors.append(color_dict[len(key)])

        super().__init__(mesh=create_mesh(vertices, faces=faces), name=self.name,color=self.color, **kwargs)

        def face2slot(raw_face):
            face=MeshFace(raw_face)
            for slot,conj_class in enumerate(face_classes.values()):
                if face in conj_class:
                    return slot

            return 0

        if self.color == "color_dict":
            for i, col in enumerate(colors):
                ibpy.set_material(self, get_texture(col, **kwargs), slot=i)
            ibpy.set_color_to_faces(self, lambda x: face2slot(list(x)))

            # initialize mixed colors
            for raw_face in self.faces:
                if 0 in raw_face:
                    face = MeshFace(raw_face)
                    slot = list(face_classes.keys()).index(face)
                    if len(face) in [3, 4, 5]:
                        ibpy.set_mixer(self, slot, value=1, begin_time=0)
                    else:
                        ibpy.set_mixer(self, slot, value=0, begin_time=0)

        modifier = PolyhedronViewModifier()
        self.add_mesh_modifier(type="NODES", node_modifier=modifier)

    def transform_colors(self, shape_key=1, face_classes={}, begin_time=0, transition_time=0):
        shape_keys = self.ref_obj.data.shape_keys
        self.shape_key=shape_key
        old_sk = [v.co for v in shape_keys.key_blocks[shape_key -1 ].data]
        new_sk = [v.co for v in shape_keys.key_blocks[shape_key].data]
        face_maps = {}
        for face_index, raw_face in enumerate(self.faces):
            if 0 in raw_face:
                face = MeshFace(raw_face)
                old_vertices = [old_sk[i] for i in face]
                new_vertices = [new_sk[i] for i in face]

                # delete close-by vertices
                unique_old_vertices = []
                unique_new_vertices = []

                for i in range(len(old_vertices)):
                    diff = (old_vertices[i] - old_vertices[i - 1]).length
                    if diff > 0.001:
                        unique_old_vertices.append(old_vertices[i])
                    diff = (new_vertices[i] - new_vertices[i - 1]).length
                    if diff > 0.001:
                        unique_new_vertices.append(new_vertices[i])

                face_maps[face] = (len(unique_old_vertices), len(unique_new_vertices))

        for key, (src, target) in face_maps.items():
            slot = list(face_classes.keys()).index(key)
            if src != target:
                if target in [3, 4, 5]:
                    ibpy.adjust_mixer(self, slot=slot, from_value=0, to_value=1, begin_time=begin_time,
                                      transition_time=transition_time)
                else:
                    ibpy.adjust_mixer(self, slot=slot, from_value=1, to_value=0, begin_time=begin_time,
                                      transition_time=transition_time)

        return begin_time + transition_time

    def copy(self) -> PolyhedronWithModifier:
        if self.group is not None and self.signature is not None:
            return PolyhedronWithModifier.from_group_signature(self.group,self.signature,name="CopyOf"+self.name,color=self.color,**self.kwargs)


        return PolyhedronWithModifier(self.vertices, self.faces, name="CopyOf" + self.name, **self.kwargs)

    @classmethod
    def from_solid_type(cls, solid_type, **kwargs):
        vertices, faces = get_solid_data(solid_type.upper())
        return cls(vertices, faces, **kwargs)

    @classmethod
    def from_group_signature(cls, group, signature,radius=None, **kwargs):
        g = group()
        vertices = g.get_real_point_cloud(signature)
        if radius:
            scale = radius/vertices[0].length
            vertices = [v*scale for v in vertices]
        faces = g.get_faces(signature)
        face_classes = g.get_faces_in_conjugacy_classes(signature)
        return cls( vertices, faces, group = group,signature = signature, face_classes=face_classes, **kwargs)
