from mathematics.groups.coxeter_group import CoxeterGroup3D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector

COXB3_SIGNATURES = {
    "OCTA": [1, 0, 0],
    "CUBE": [0, 1, 0],
    "TRUNC_OCTA": [1, 0, -1],
    "TRUNC_CUBE": [1, 1, 1],
    "CUBOCTA": [0, 0, 1],
    "RHOMBICUBOCTA": [1, 1, 0],
    "TRUNC_CUBOCTA": [1, 1, -1],
    "o3o4x": [0, 1, 0],
    "x3o4o": [1, 0, 0],
    "o3x4o": [0, 0, 1],
    "o3x4x": [1, 1, 1],
    "x3x4o": [1, 0, -1],
    "x3o4x": [1, 1, 0],
    "x3x4x": [1, 1, -1],
}


COXB3_SEEDS = {
    "x3o4o": FVector.parse("[1, 0, -1]"),
    "o3o4x": FVector.parse("[0, 1,-r2]"),
    "x3x4o": FVector.parse("[1, 0, -3]"),
    "o3x4x": FVector.parse("[1, 1, 1 - r2]"),
    "o3x4o": FVector.parse("[0, 0, 2]"),
    "x3o4x": FVector.parse("[1, 1, -1 - r2]"),
    "x3x4x": FVector.parse("[1, 1, -3 - r2]")
}

COXB3_NAMES = {
    "OCTA":"x3o4o",
    "CUBE":"o3o4x",
    "CUBOCTA":"o3x4o",
    "TRUNC_OCTA":"x3x4o",
    "TRUNC_CUBE":"o3x4x",
    "RHOMBICUBOCTA":"x3o4x",
    "TRUNC_CUBOCTA":"x3x4x"
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
                         seeds=COXB3_SEEDS,
                         signatures=COXB3_SIGNATURES,
                         name_dict=COXB3_NAMES)
