from mathematics.geometry.coxeter_group import CoxeterGroup3D
from mathematics.geometry.field_extensions import QR, FMatrix, FVector

COXA3_SEEDS = {
    "TETRA": FVector.parse("[1, 0, -1/2*r2]"),
    "OCTA": FVector.parse("[0, 0,r2]"),
    "TRUNC_TETRA": FVector.parse("[1, 0, -3/2*r2]"),
    "CUBOCTA": FVector.parse("[1,1,-r2]"),
    "TRUNC_OCTA": FVector.parse("[1, 1,-2*r2]"),
}

COXA3_SIGNATURES = {
    "TETRA": (1, 0, 0),
    "OCTA": (0, 0, 1),
    "TRUNC_TETRA": (1, 0, -1),
    "CUBOCTA": (1, 1, 0),
    "TRUNC_OCTA": (1, 1, -1)
}

COXA3_TYPES = {
    (1, 0, 0): "TETRA",
    (0, 0, 1): "OCTA",
    (1, 0, -1): "TRUNC_TETRA",
    (1, 1, 0): "CUBOCTA",
    (1, 1, -1): "TRUNC_OCTA"
}

_DEFAULT_PATH = "../mathematics/geometry/data/"


class CoxA3(CoxeterGroup3D):
    def __init__(self, path=None):
        self.size=24
        zero = QR.from_integers(0, 1, 0, 1, 2, "r2")
        one = QR.from_integers(1, 1, 0, 1, 2, "r2")
        half = QR.from_integers(1, 2, 0, 1, 2, "r2")

        normals = [
            FVector([one, zero, zero]),
            FVector([zero, one, zero]),
            FVector([half, half, QR.from_integers(0, 1, 1, 2, 2, "r2")]),
        ]

        identity = FMatrix([[one, zero, zero], [zero, one, zero], [zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]

        super().__init__("coxA3", normals, generators,
                         path=path or _DEFAULT_PATH,
                         seeds=COXA3_SEEDS, types=COXA3_TYPES)
