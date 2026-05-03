"""Crystallographic space groups in 3D.

A space group is an extension 1 -> T -> G -> K -> 1 where T is the lattice of
translations and K is one of the 32 crystallographic point groups. The 230
distinct extensions (up to affine equivalence) are the space groups.

This module represents elements as affine maps (R, t) acting on fractional
coordinates: (R, t) . x = R x + t (mod 1) for orbits inside the unit cell, or
without the mod for orbits across multiple cells.

The canonical input format is the symmetry-operation string used by the
International Tables for Crystallography (e.g. "x,y+1/2,-z+1/2"), parsed into
a (3x3 integer rotation, 3-vector translation) pair. A small built-in table of
representative space groups covers all 7 crystal systems and both symmorphic
and non-symmorphic cases. Extending to all 230 is a matter of adding more
operation tables (e.g. from the Bilbao Crystallographic Server).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

import numpy as np


_TOKEN = re.compile(r"([+-]?)\s*(?:(\d+)\s*/\s*(\d+)|(\d+(?:\.\d+)?)|([xyz]))")


def parse_symmetry_op(op: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one ITC symmetry-operation string into (R, t).

    Examples:
        "x,y,z"            -> (I, 0)
        "-x,y+1/2,-z+1/2"  -> ([[-1,0,0],[0,1,0],[0,0,-1]], [0, 1/2, 1/2])
        "y,x,z+1/4"        -> ([[0,1,0],[1,0,0],[0,0,1]], [0, 0, 1/4])
    """
    parts = [p.strip() for p in op.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Symmetry op must have 3 components, got: {op}")
    R = np.zeros((3, 3), dtype=int)
    t = np.zeros(3, dtype=float)
    axis_index = {"x": 0, "y": 1, "z": 2}
    for row, expr in enumerate(parts):
        for sign_s, num, den, dec, axis in _TOKEN.findall(expr):
            sign = -1 if sign_s == "-" else 1
            if axis:
                R[row, axis_index[axis]] += sign
            elif num:
                t[row] += sign * float(Fraction(int(num), int(den)))
            elif dec:
                t[row] += sign * float(dec)
    return R, t


@dataclass(frozen=True)
class SymmetryOp:
    """One affine isometry (R, t) in fractional coordinates."""
    R: np.ndarray  # 3x3 int
    t: np.ndarray  # 3-vector float in [0, 1)

    @classmethod
    def from_string(cls, op: str) -> "SymmetryOp":
        R, t = parse_symmetry_op(op)
        return cls(R=R, t=t % 1.0)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.R @ x + self.t

    def compose(self, other: "SymmetryOp") -> "SymmetryOp":
        return SymmetryOp(R=self.R @ other.R, t=(self.R @ other.t + self.t) % 1.0)

    def is_pure_translation(self) -> bool:
        return np.array_equal(self.R, np.eye(3, dtype=int))

    def is_identity(self) -> bool:
        return self.is_pure_translation() and np.allclose(self.t, 0)


# The 14 Bravais centerings (in addition to lattice translations [1,0,0] etc.)
CENTERING_VECTORS: dict[str, list[tuple[float, float, float]]] = {
    "P": [(0, 0, 0)],
    "A": [(0, 0, 0), (0, 0.5, 0.5)],
    "B": [(0, 0, 0), (0.5, 0, 0.5)],
    "C": [(0, 0, 0), (0.5, 0.5, 0)],
    "I": [(0, 0, 0), (0.5, 0.5, 0.5)],
    "F": [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    "R": [(0, 0, 0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3)],  # obverse setting
}


@dataclass
class SpaceGroup:
    """A 3D crystallographic space group.

    Stores the generating coset representatives (one per element of the point
    group K). The full set of operations modulo lattice is built by closure;
    full orbits in space are obtained by tiling with lattice translations.
    """
    number: int
    hm_symbol: str          # Hermann-Mauguin (international) symbol
    crystal_system: str     # triclinic | monoclinic | ... | cubic
    centering: str          # one of P A B C I F R
    op_strings: list[str]
    is_symmorphic: bool = False
    description: str = ""
    _ops: list[SymmetryOp] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        seeds = [SymmetryOp.from_string(s) for s in self.op_strings]
        for c in CENTERING_VECTORS[self.centering]:
            shift = np.array(c, dtype=float)
            for op in seeds:
                self._ops.append(SymmetryOp(R=op.R, t=(op.t + shift) % 1.0))
        self._ops = self._close_under_composition(self._ops)

    @staticmethod
    def _close_under_composition(seed_ops: list[SymmetryOp]) -> list[SymmetryOp]:
        seen: dict[tuple, SymmetryOp] = {}

        def key(op: SymmetryOp) -> tuple:
            return tuple(op.R.flatten().tolist()) + tuple(np.round(op.t % 1.0, 6).tolist())

        frontier = list(seed_ops)
        for op in seed_ops:
            seen[key(op)] = op
        while frontier:
            new_frontier = []
            for a in frontier:
                for b in seed_ops:
                    c = a.compose(b)
                    k = key(c)
                    if k not in seen:
                        seen[k] = c
                        new_frontier.append(c)
            frontier = new_frontier
        return list(seen.values())

    @property
    def operations(self) -> list[SymmetryOp]:
        return self._ops

    @property
    def order_in_cell(self) -> int:
        """Number of distinct operations modulo the lattice (the multiplicity)."""
        return len(self._ops)

    def orbit(self, fractional_point: Iterable[float],
              cells: tuple[int, int, int] = (1, 1, 1),
              tol: float = 1e-6) -> np.ndarray:
        """Generate all symmetry-equivalent points within `cells` unit cells.

        The motif point is given in fractional coordinates (typically in [0,1)).
        Returns an (N, 3) array of fractional coordinates with duplicates removed.
        """
        x0 = np.array(list(fractional_point), dtype=float)
        in_cell = []
        seen_keys = set()
        for op in self._ops:
            p = op.apply(x0) % 1.0
            k = tuple(np.round(p / tol).astype(int))
            if k not in seen_keys:
                seen_keys.add(k)
                in_cell.append(p)
        in_cell = np.array(in_cell)
        nx, ny, nz = cells
        shifts = np.array([(i, j, k)
                           for i in range(nx) for j in range(ny) for k in range(nz)])
        all_pts = (in_cell[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        return all_pts

    def to_cartesian(self, fractional_pts: np.ndarray,
                     lattice: np.ndarray) -> np.ndarray:
        """Convert (N, 3) fractional coords to Cartesian using a 3x3 lattice basis (rows = a, b, c)."""
        return fractional_pts @ lattice


# --------------------------------------------------------------------------
# Built-in catalogue of representative space groups
#
# Operations are coset representatives over the centering — composition with
# centering vectors and closure happens in SpaceGroup.__post_init__.
# Operation strings follow the standard ITC conventions; verify against the
# Bilbao Crystallographic Server when extending.
# --------------------------------------------------------------------------

_CATALOG: dict[int, SpaceGroup] = {}


def _register(sg: SpaceGroup) -> None:
    _CATALOG[sg.number] = sg


_register(SpaceGroup(
    number=1, hm_symbol="P1", crystal_system="triclinic", centering="P",
    op_strings=["x,y,z"],
    is_symmorphic=True,
    description="Trivial group: only lattice translations.",
))

_register(SpaceGroup(
    number=2, hm_symbol="P-1", crystal_system="triclinic", centering="P",
    op_strings=["x,y,z", "-x,-y,-z"],
    is_symmorphic=True,
    description="Inversion only. Most common space group for inorganic salts.",
))

_register(SpaceGroup(
    number=14, hm_symbol="P2_1/c", crystal_system="monoclinic", centering="P",
    op_strings=["x,y,z", "-x,y+1/2,-z+1/2", "-x,-y,-z", "x,-y+1/2,z+1/2"],
    is_symmorphic=False,
    description="2_1 screw + c-glide. The most common space group for organic molecules (~36% of CSD).",
))

_register(SpaceGroup(
    number=19, hm_symbol="P2_12_12_1", crystal_system="orthorhombic", centering="P",
    op_strings=[
        "x,y,z",
        "-x+1/2,-y,z+1/2",
        "-x,y+1/2,-z+1/2",
        "x+1/2,-y+1/2,-z",
    ],
    is_symmorphic=False,
    description="Three perpendicular 2_1 screw axes. Most common space group for chiral molecules (proteins).",
))

_register(SpaceGroup(
    number=123, hm_symbol="P4/mmm", crystal_system="tetragonal", centering="P",
    op_strings=[
        "x,y,z", "-x,-y,z", "-y,x,z", "y,-x,z",
        "-x,y,z", "x,-y,z", "y,x,z", "-y,-x,z",
        "-x,-y,-z", "x,y,-z", "y,-x,-z", "-y,x,-z",
        "x,-y,-z", "-x,y,-z", "-y,-x,-z", "y,x,-z",
    ],
    is_symmorphic=True,
    description="Symmorphic tetragonal holohedry. Affine Coxeter base I_2(4) x A_1.",
))

_register(SpaceGroup(
    number=136, hm_symbol="P4_2/mnm", crystal_system="tetragonal", centering="P",
    op_strings=[
        "x,y,z",
        "-x,-y,z",
        "-y+1/2,x+1/2,z+1/2",
        "y+1/2,-x+1/2,z+1/2",
        "-x+1/2,y+1/2,-z+1/2",
        "x+1/2,-y+1/2,-z+1/2",
        "y,x,-z",
        "-y,-x,-z",
        "-x,-y,-z",
        "x,y,-z",
        "y+1/2,-x+1/2,-z+1/2",
        "-y+1/2,x+1/2,-z+1/2",
        "x+1/2,-y+1/2,z+1/2",
        "-x+1/2,y+1/2,z+1/2",
        "-y,-x,z",
        "y,x,z",
    ],
    is_symmorphic=False,
    description="Rutile (TiO_2) structure. 4_2 screw + n-glide.",
))

_register(SpaceGroup(
    number=194, hm_symbol="P6_3/mmc", crystal_system="hexagonal", centering="P",
    op_strings=[
        "x,y,z",
        "-y,x-y,z",
        "-x+y,-x,z",
        "-x,-y,z+1/2",
        "y,-x+y,z+1/2",
        "x-y,x,z+1/2",
        "y,x,-z",
        "x-y,-y,-z",
        "-x,-x+y,-z",
        "-y,-x,-z+1/2",
        "-x+y,y,-z+1/2",
        "x,x-y,-z+1/2",
        "-x,-y,-z",
        "y,-x+y,-z",
        "x-y,x,-z",
        "x,y,-z+1/2",
        "-y,x-y,-z+1/2",
        "-x+y,-x,-z+1/2",
        "-y,-x,z",
        "-x+y,y,z",
        "x,x-y,z",
        "y,x,z+1/2",
        "x-y,-y,z+1/2",
        "-x,-x+y,z+1/2",
    ],
    is_symmorphic=False,
    description="Graphite, hcp metals (Mg, Zn). 6_3 screw axis along c.",
))

_register(SpaceGroup(
    number=221, hm_symbol="Pm-3m", crystal_system="cubic", centering="P",
    op_strings=[
        # 24 rotations of O ...
        "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
        "z,x,y", "z,-x,-y", "-z,-x,y", "-z,x,-y",
        "y,z,x", "-y,z,-x", "y,-z,-x", "-y,-z,x",
        "y,x,-z", "-y,-x,-z", "y,-x,z", "-y,x,z",
        "x,z,-y", "-x,z,y", "-x,-z,-y", "x,-z,y",
        "z,y,-x", "z,-y,x", "-z,y,x", "-z,-y,-x",
        # ... times inversion to get 48 = |O_h|
        "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
        "-z,-x,-y", "-z,x,y", "z,x,-y", "z,-x,y",
        "-y,-z,-x", "y,-z,x", "-y,z,x", "y,z,-x",
        "-y,-x,z", "y,x,z", "-y,x,-z", "y,-x,-z",
        "-x,-z,y", "x,-z,-y", "x,z,y", "-x,z,-y",
        "-z,-y,x", "-z,y,-x", "z,-y,-x", "z,y,x",
    ],
    is_symmorphic=True,
    description="Perovskite (SrTiO_3, CaTiO_3). Symmorphic cubic holohedry; affine Coxeter base B_3.",
))

_register(SpaceGroup(
    number=225, hm_symbol="Fm-3m", crystal_system="cubic", centering="F",
    op_strings=_CATALOG[221].op_strings,  # same point group, F-centering
    is_symmorphic=True,
    description="Rock salt (NaCl), copper (FCC metals). Symmorphic; F-centering quadruples multiplicity.",
))

_register(SpaceGroup(
    number=227, hm_symbol="Fd-3m", crystal_system="cubic", centering="F",
    op_strings=[
        "x,y,z",
        "-x+3/4,-y+1/4,z+1/2",
        "-x+1/4,y+1/2,-z+3/4",
        "x+1/2,-y+3/4,-z+1/4",
        "z,x,y",
        "z+1/2,-x+3/4,-y+1/4",
        "-z+3/4,-x+1/4,y+1/2",
        "-z+1/4,x+1/2,-y+3/4",
        "y,z,x",
        "-y+1/4,z+1/2,-x+3/4",
        "y+1/2,-z+3/4,-x+1/4",
        "-y+3/4,-z+1/4,x+1/2",
        "y+3/4,x+1/4,-z+1/2",
        "-y,-x,-z",
        "y+1/4,-x+1/2,z+3/4",
        "-y+1/2,x+3/4,z+1/4",
        "x+3/4,z+1/4,-y+1/2",
        "-x+1/2,z+3/4,y+1/4",
        "-x,-z,-y",
        "x+1/4,-z+1/2,y+3/4",
        "z+3/4,y+1/4,-x+1/2",
        "z+1/4,-y+1/2,x+3/4",
        "-z,-y,-x",
        "-z+1/2,y+3/4,x+1/4",
        "-x,-y,-z",
        "x+1/4,y+3/4,-z+1/2",
        "x+3/4,-y+1/2,z+1/4",
        "-x+1/2,y+1/4,z+3/4",
        "-z,-x,-y",
        "-z+1/2,x+1/4,y+3/4",
        "z+1/4,x+3/4,-y+1/2",
        "z+3/4,-x+1/2,y+1/4",
        "-y,-z,-x",
        "y+3/4,-z+1/2,x+1/4",
        "-y+1/2,z+1/4,x+3/4",
        "y+1/4,z+3/4,-x+1/2",
        "-y+1/4,-x+3/4,z+1/2",
        "y,x,z",
        "-y+3/4,x+1/2,-z+1/4",
        "y+1/2,-x+1/4,-z+3/4",
        "-x+1/4,-z+3/4,y+1/2",
        "x+1/2,-z+1/4,-y+3/4",
        "x,z,y",
        "-x+3/4,z+1/2,-y+1/4",
        "-z+1/4,-y+3/4,x+1/2",
        "-z+3/4,y+1/2,-x+1/4",
        "z,y,x",
        "z+1/2,-y+1/4,-x+3/4",
    ],
    is_symmorphic=False,
    description="Diamond, silicon, spinel. d-glide planes (the 'd' in Fd-3m).",
))

_register(SpaceGroup(
    number=229, hm_symbol="Im-3m", crystal_system="cubic", centering="I",
    op_strings=_CATALOG[221].op_strings,  # same point group, I-centering
    is_symmorphic=True,
    description="BCC metals (alpha-Fe, W), CsCl. Symmorphic; I-centering doubles multiplicity.",
))


def get(number: int) -> SpaceGroup:
    """Return the catalogued space group, or raise KeyError."""
    if number not in _CATALOG:
        raise KeyError(
            f"Space group #{number} not in built-in catalogue. "
            f"Available: {sorted(_CATALOG.keys())}. "
            f"Add it by registering operation strings from the Bilbao Crystallographic Server."
        )
    return _CATALOG[number]


def available() -> list[int]:
    return sorted(_CATALOG.keys())
