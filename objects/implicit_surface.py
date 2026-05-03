"""
Implicit surface as an explicit Blender mesh, plus optional iso-line
overlays for any scalar field on the surface.

The mesh is built Python-side via marching tetrahedra (see
``mathematics.marching_tetrahedra``) so we end up with real vertices
and triangles you can index into — which lets us trace lines of
constant longitude / latitude / curvature / etc. on top, something
the geometry-nodes Volume-to-Mesh pipeline can not do.

Usage::

    surface = ImplicitSurface(
        f="x*x + y*y + z*z - 1",          # str or callable(X,Y,Z)
        bounds=((-1.5, 1.5),) * 3,
        resolution=64,
    )

    meridians = CoordinateLines(
        surface,
        g=lambda x, y, z: np.arctan2(y, x),
        levels=np.linspace(-np.pi, np.pi, 25)[1:-1],
        modular=True,                       # atan2 is periodic
        thickness=0.01,
        color="text",
    )
"""

from __future__ import annotations

import bpy
import numpy as np

from appearance.textures import get_texture
from interface import ibpy
from mathematics.iso_curves import trace_iso_lines, trace_iso_lines_modular
from mathematics.marching_tetrahedra import marching_tetrahedra, parse_implicit_function
from objects.bobject import BObject


class ImplicitSurface(BObject):
    """
    Mesh of the zero level-set ``f(x,y,z) = level`` extracted via
    marching tetrahedra.  The verts/faces arrays are kept on ``self``
    so other objects (e.g. CoordinateLines) can reuse the geometry.
    """

    def __init__(self,
                 f,
                 bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
                 resolution=64,
                 level=0.0,
                 name="ImplicitSurface",
                 **kwargs):
        self.f = parse_implicit_function(f)
        self.bounds = bounds
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        self.resolution = resolution
        self.level = level

        self.verts, self.faces = marching_tetrahedra(
            self.f, bounds, resolution, level=level)

        from interface.ibpy import create_mesh
        super().__init__(name=name, mesh=create_mesh(self.verts,faces=self.faces), **kwargs)


class CoordinateLines(BObject):
    """
    Iso-lines of a scalar field ``g(x,y,z)`` on an existing
    ``ImplicitSurface``, drawn as a single Blender Curve object with
    one POLY spline per polyline.  Set ``thickness`` > 0 to bevel
    the splines into tubes.
    """

    def __init__(self,
                 surface: ImplicitSurface,
                 g,
                 levels,
                 modular=False,
                 period=2 * np.pi,
                 thickness=0.01,
                 bevel_resolution=2,
                 name="CoordinateLines",
                 **kwargs):
        if modular:
            polys_per_level = trace_iso_lines_modular(
                surface.verts, surface.faces, g, levels, period=period)
        else:
            polys_per_level = trace_iso_lines(
                surface.verts, surface.faces, g, levels)
        polylines = [p for ps in polys_per_level for p in ps]

        curve_data = bpy.data.curves.new(name + "_curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 1
        if thickness > 0:
            curve_data.bevel_depth = thickness
            curve_data.bevel_resolution = bevel_resolution

        for poly in polylines:
            if len(poly) < 2:
                continue
            spline = curve_data.splines.new('POLY')
            spline.points.add(len(poly) - 1)  # one is created by default
            for pt, p in zip(spline.points, poly):
                pt.co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
            # Close the spline if first and last points coincide.
            if np.allclose(poly[0], poly[-1]):
                spline.use_cyclic_u = True

        obj = bpy.data.objects.new(name, curve_data)
        super().__init__(name=name, obj=obj, **kwargs)
        self.polylines = polylines
