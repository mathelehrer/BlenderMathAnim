from mathematics.groups.coxeter_group import CoxeterGroup3D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector

COXA3_SIGNATURES = {
    # for compatability (has to come first)
    "TETRA": [0, 0, 1],
    "OCTA": [0, 1, 0],
    "TRUNC_TETRA": [1, 0, -1],
    "CUBOCTA": [1, 1, 0],
    "TRUNC_OCTA": [1, 1, -1],
    "SNUB_TETRA": [1, 1, -1],
    "o3o3x": [0, 0, 1],
    "x3o3o": [0, 0, 1],
    "o3x3o": [0, 1, 0],
    "o3x3x": [1, 0, -1],
    "x3x3o": [1, 0, -1],
    "x3o3x": [1, 1, 0],
    "x3x3x": [1, 1, -1],
}

COXA3_SEEDS = {
    "x3o3o": FVector.parse("[1, 0, -1/2*r2]"),
    "o3o3x": FVector.parse("[1, 0, -1/2*r2]"),
    "o3x3o": FVector.parse("[0, 0,r2]"),
    "x3x3o": FVector.parse("[1, 0, -3/2*r2]"),
    "o3x3x": FVector.parse("[1, 0, -3/2*r2]"),
    "x3o3x": FVector.parse("[1,1,-r2]"),
    "x3x3x": FVector.parse("[1, 1,-2*r2]"),
    # "s3s3s": FVector.parse("[1, 1,-2*r2]"),
}

COXA3_NAMES = {
    "TETRA": "x3o3o",
    "OCTA": "o3x3o",
    "TRUNC_TETRA": "x3x3o",
    "CUBOCTA": "x3o3x",
    "TRUNC_OCTA": "x3x3x",
    "SNUB_TETRA": "s3s3s",
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
                         seeds=COXA3_SEEDS, signatures=COXA3_SIGNATURES,name_dict=COXA3_NAMES)
