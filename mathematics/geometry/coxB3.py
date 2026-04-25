from mathematics.geometry.coxeter_group import CoxeterGroup3D
from mathematics.geometry.field_extensions import QR, FMatrix, FVector

COXB3_CD_LABELS = {
    "o3o4x": [0, 1, 0],
    "x3o4o": [1, 0, 0],
    "o3x4o": [0, 0, 1],
    "x3x4o": [1, 0, -1],
    "x3o4x": [1, 1, 0],
    "o3x4x": [1, 1, 1],
    "x3x4x": [1, 1, -1]
}

COXB3_SEEDS = {
    "OCTA": FVector.parse("[1, 0, -1]"),
    "CUBE": FVector.parse("[0, 1,-r2]"),
    "TRUNC_OCTA": FVector.parse("[1, 0, -3]"),
    "TRUNC_CUBE": FVector.parse("[1, 1, 1 - r2]"),
    "CUBOCTA": FVector.parse("[0, 0, 2]"),
    "RHOMBICUBOCTA": FVector.parse("[1, 1, -1 - r2]"),
    "TRUNC_CUBOCTA": FVector.parse("[1, 1, -3 - r2]")
}

COXB3_SIGNATURES = {
    "OCTA": (1, 0, 0),
    "CUBE": (0, 1, 0),
    "TRUNC_OCTA": (1, 0, -1),
    "TRUNC_CUBE": (1, 1, 1),
    "CUBOCTA": (0, 0, 1),
    "RHOMBICUBOCTA": (1, 1, 0),
    "TRUNC_CUBOCTA": (1, 1, -1)
}

COXB3_TYPES = {
    (1, 0, 0): "OCTA",
    (0, 1, 0): "CUBE",
    (0, 0, 1): "CUBOCTA",
    (1, 0, -1): "TRUNC_OCTA",
    (1, 1, 1): "TRUNC_CUBE",
    (1, 1, 0): "RHOMBICUBOCTA",
    (1, 1, -1): "TRUNC_CUBOCTA"
}

_DEFAULT_PATH = "../mathematics/geometry/data/"


class CoxB3(CoxeterGroup3D):
    def __init__(self, path=None):
        self.size=48
        zero = QR.from_integers(0, 1, 0, 1, 2, "r2")
        one = QR.from_integers(1, 1, 0, 1, 2, "r2")
        half = QR.from_integers(1, 2, 0, 1, 2, "r2")

        normals = [
            FVector([one, zero, zero]),
            FVector([zero, one, zero]),
            FVector([half, QR.from_integers(0, 1, 1, 2, 2, "r2"), half]),
        ]

        identity = FMatrix([[one, zero, zero], [zero, one, zero], [zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]

        super().__init__("coxB3", normals, generators,
                         path=path or _DEFAULT_PATH,
                         seeds=COXB3_SEEDS, types=COXB3_TYPES)
