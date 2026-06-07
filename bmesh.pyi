"""Type stubs for Blender's ``bmesh`` module.

Like ``mathutils``, ``bmesh`` is a C extension that Blender (and the pip
``bpy`` package) injects into ``sys.modules`` at runtime; it has no source on
disk, so static analysers such as PyCharm cannot resolve ``import bmesh`` and
flag it red.

This stub exists purely so the IDE can resolve the module and the parts of its
API used in this code base. CPython never imports ``.pyi`` files, so it cannot
shadow the real builtin at runtime -- only the IDE / type checker reads it.
Keep it light; extend as needed.

The ``ops``/``types``/``utils``/``geometry`` submodules are typed as ``Any`` so
attribute access (e.g. ``bmesh.ops.subdivide_edges(...)``) resolves without
having to mirror Blender's full operator surface.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from mathutils import Vector

# Submodules (bmesh.ops, bmesh.types, ...). Typed permissively on purpose.
ops: Any
types: Any
utils: Any
geometry: Any


class BMVert:
    co: Vector
    normal: Vector
    index: int
    select: bool
    link_edges: Sequence["BMEdge"]
    link_faces: Sequence["BMFace"]


class BMEdge:
    verts: Sequence[BMVert]
    index: int
    select: bool
    link_faces: Sequence["BMFace"]


class BMFace:
    verts: Sequence[BMVert]
    edges: Sequence[BMEdge]
    normal: Vector
    index: int
    select: bool


class _BMElemSeq:
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int) -> Any: ...
    def __len__(self) -> int: ...
    def ensure_lookup_table(self) -> None: ...
    def index_update(self) -> None: ...
    def new(self, *args: Any, **kwargs: Any) -> Any: ...
    def remove(self, elem: Any) -> None: ...


class BMesh:
    verts: _BMElemSeq
    edges: _BMElemSeq
    faces: _BMElemSeq

    def from_mesh(self, mesh: Any, face_normals: bool = ..., use_shape_key: bool = ...) -> None: ...
    def to_mesh(self, mesh: Any) -> None: ...
    def from_object(self, obj: Any, depsgraph: Any, cage: bool = ...) -> None: ...
    def transform(self, matrix: Any, filter: Any = ...) -> None: ...
    def normal_update(self) -> None: ...
    def select_flush(self, select: bool) -> None: ...
    def select_flush_mode(self) -> None: ...
    def calc_volume(self, signed: bool = ...) -> float: ...
    def copy(self) -> "BMesh": ...
    def clear(self) -> None: ...
    def free(self) -> None: ...


def new(use_operators: bool = ...) -> BMesh: ...
def from_edit_mesh(mesh: Any) -> BMesh: ...
def update_edit_mesh(mesh: Any, loop_triangles: bool = ..., destructive: bool = ...) -> None: ...
