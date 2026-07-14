from __future__ import annotations

import os
import sys
import types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class _Vector(list):
    def __init__(self, values=()):
        super().__init__(float(v) for v in values)

    def _coerce(self, other):
        if isinstance(other, (int, float)):
            return [other] * len(self)
        return list(other)

    def _binary(self, other, op):
        other_vals = self._coerce(other)
        return _Vector(op(a, b) for a, b in zip(self, other_vals))

    def __neg__(self):
        return _Vector(-v for v in self)

    def __add__(self, other):
        import operator
        return self._binary(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        import operator
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        import operator
        return _Vector(operator.sub(a, b) for a, b in zip(self._coerce(other), self))

    def __mul__(self, other):
        import operator
        return self._binary(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return _Vector(v / other for v in self)

    @property
    def x(self):
        return self[0] if len(self) > 0 else 0.0

    @property
    def y(self):
        return self[1] if len(self) > 1 else 0.0

    @property
    def z(self):
        return self[2] if len(self) > 2 else 0.0

    def copy(self):
        return _Vector(self)


_mathutils = types.ModuleType("mathutils")
_mathutils.Vector = _Vector
_mathutils.Matrix = _Vector
_mathutils.Quaternion = _Vector
_mathutils.Euler = _Vector
sys.modules.setdefault("mathutils", _mathutils)

project = "BlenderMathAnim"
author = "NumberCruncher"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autodoc_mock_imports = [
    "aud",
    "bgl",
    "bmesh",
    "blf",
    "bpy",
    "bpy_extras",
    "gpu",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_static_path = ["_static"]
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
add_module_names = False

